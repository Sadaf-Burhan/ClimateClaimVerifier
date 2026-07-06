"""
Stage 4b — GATED image modality (Week 7 multimodal).

Vision is a *precision intervention*, NOT a per-post feature. Cheap text+metadata
signals decide first; the vision model fires ONLY on edge-case posts that (a) have an
image and (b) aren't already resolved by metadata — the conspiracy/ambiguous claims
that drag precision down. On those, a real on-the-ground photo supports a genuine
claim (a precision save), while a cartoon/meme leans satire or fabrication.

The vision output is another SIGNAL FOR THE READER, never a truth verdict — the model
is told to describe only what is visible and NOT to judge the claim true or false.

    uv run python -m climate_verifier.pipeline.vision --gate-and-analyze
    uv run python -m climate_verifier.pipeline.vision            # dry run: list what would escalate
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

import ollama
import requests
import yaml

from .evidence import extract_citations, is_official   # reuse the metadata guards

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


_VISION_PROMPT = """You are describing an image attached to a social-media post so a reader can judge the post.

Rules:
1. Describe ONLY what is visibly in the image. Do NOT use outside knowledge.
2. Do NOT decide whether the post's claim is true or false — that is the reader's job, not yours.
3. Classify the image type and whether it appears to depict the specific event the claim describes.

Post claim: {claim}

Return ONLY this JSON (no markdown fences, no extra text):
{{"image_type": "real_photo|meme_or_cartoon|screenshot|infographic|ai_suspected|other",
  "depicts_claim": "yes|partial|no|unrelated",
  "description": "<one short sentence, only what is visible>"}}"""

_VALID_TYPES = {"real_photo", "meme_or_cartoon", "screenshot", "infographic", "ai_suspected", "other"}
_VALID_DEPICTS = {"yes", "partial", "no", "unrelated"}


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _correct(raw: str, model: str) -> dict | None:
    """Reformat malformed vision output to the schema with a cheap TEXT model (never sees the image)."""
    try:
        resp = ollama.chat(model=model, options={"temperature": 0}, messages=[{
            "role": "user",
            "content": ('Reformat this into ONLY the JSON '
                        '{"image_type":"real_photo|meme_or_cartoon|screenshot|infographic|ai_suspected|other",'
                        '"depicts_claim":"yes|partial|no|unrelated","description":"..."} '
                        "with no markdown or extra text:\n\n" + str(raw))}])
        return _parse_json(resp["message"]["content"])
    except Exception:
        return None


def normalize(out: dict | None) -> dict | None:
    """Clamp a parsed vision dict to the schema's allowed values."""
    if not out:
        return None
    it = str(out.get("image_type", "other")).lower()
    dp = str(out.get("depicts_claim", "unrelated")).lower()
    return {
        "image_type": it if it in _VALID_TYPES else "other",
        "depicts_claim": dp if dp in _VALID_DEPICTS else "unrelated",
        "description": str(out.get("description", ""))[:300],
    }


def analyze_image(image_url: str, claim: str, model: str, corrector: str, timeout: int = 20) -> dict | None:
    """Download the image and run the vision model -> normalized signal dict, or None on failure."""
    try:
        r = requests.get(image_url, timeout=timeout)
        r.raise_for_status()
        img = r.content
    except Exception:
        return None
    try:
        resp = ollama.chat(model=model, options={"temperature": 0}, messages=[{
            "role": "user", "content": _VISION_PROMPT.format(claim=(claim or "")[:400]), "images": [img]}])
        content = resp["message"]["content"]
        return normalize(_parse_json(content) or _correct(content, corrector))
    except Exception:
        return None


def vision_reader_note(vs: dict | None) -> str:
    """Derived reader phrase from a vision signal — suggestive, never a verdict."""
    if not vs:
        return ""
    it, dp = vs.get("image_type"), vs.get("depicts_claim")
    if it == "real_photo" and dp in ("yes", "partial"):
        return ("Image: a real on-the-ground photo consistent with the claim — supports a genuine "
                "observation (open it to confirm).")
    if it == "meme_or_cartoon":
        return "Image: cartoon/meme imagery — leans satire or opinion, not evidence of the event."
    if it == "ai_suspected":
        return "Image: appears synthetic / AI-generated — treat as a fabrication cue."
    if it == "screenshot":
        return "Image: a screenshot of a source — treat as a citation to open and verify."
    return (f"Image: {vs.get('description', '')}").strip()


def gate_edge_cases(db_path: str, cfg: dict, limit: int | None = None) -> list[dict]:
    """
    Select edge-case posts to escalate to vision: a claim, with an image, in a
    precision-weak category, NOT already resolved by metadata (official author /
    credible link / reshare-of-official). Highest reach first, capped.
    """
    ev, vz = cfg.get("evidence", {}), cfg.get("vision", {})
    official = ev.get("official_sources", [])
    cats = vz.get("gate_categories", ["conspiracy", "sensationalist"])
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    qmarks = ",".join("?" * len(cats))
    rows = con.execute(f"""
        SELECT p.post_id, p.text, p.author, p.image_url, p.keyword_category,
               p.reshare_of_author, p.external_url,
               (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p JOIN classifications c ON p.post_id = c.post_id
        WHERE p.has_image = 1 AND c.has_claim = 1 AND p.source = 'bluesky'
              AND p.keyword_category IN ({qmarks})
              AND (p.vision_signal IS NULL OR p.vision_signal = '')
        ORDER BY engagement DESC
    """, cats).fetchall()
    con.close()

    edge = []
    for r in rows:
        if is_official(r["author"], "", official):                       # official account
            continue
        if r["reshare_of_author"] and is_official(r["reshare_of_author"], "", official):  # reshares official
            continue
        cites = extract_citations((r["text"] or "") + " " + (r["external_url"] or ""),
                                  ev.get("citation_domains", []))
        if any(c["credible"] for c in cites):                            # links a credible source
            continue
        edge.append(dict(r))
    cap = limit or vz.get("max_images", 50)
    return edge[:cap]


def gate_and_analyze(db_path: str, cfg: dict | None = None, limit: int | None = None) -> dict:
    """Gate the edge cases, run vision on each, and persist the signal to posts.vision_signal."""
    cfg = cfg or _load_cfg()
    vz = cfg.get("vision", {})
    model = vz.get("model", "qwen2.5-vl")
    corrector = vz.get("corrector_model", "qwen2.5:3b")
    timeout = vz.get("download_timeout", 20)
    edge = gate_edge_cases(db_path, cfg, limit)
    con = sqlite3.connect(db_path)
    analyzed = 0
    for r in edge:
        vs = analyze_image(r["image_url"], r["text"] or "", model, corrector, timeout)
        if vs:
            vs["note"] = vision_reader_note(vs)
            con.execute("UPDATE posts SET vision_signal = ? WHERE post_id = ?",
                        (json.dumps(vs), r["post_id"]))
            analyzed += 1
    con.commit()
    con.close()
    return {"gated": len(edge), "analyzed": analyzed}


def main():
    ap = argparse.ArgumentParser(description="Gated image modality for edge-case posts.")
    ap.add_argument("--gate-and-analyze", action="store_true",
                    help="run the vision model on gated edge cases and store the signal")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of edge cases")
    args = ap.parse_args()
    cfg = _load_cfg()
    db = cfg["storage"]["db_path"]
    if args.gate_and_analyze:
        res = gate_and_analyze(db, cfg, args.limit)
        print(f"Vision gating: {res['gated']} edge cases → {res['analyzed']} analyzed.")
    else:  # dry run — show what WOULD escalate, without calling the model
        edge = gate_edge_cases(db, cfg, args.limit)
        print(f"{len(edge)} edge-case posts would be escalated to vision:")
        for r in edge[:20]:
            print(f"  [{r['keyword_category']}] {(r['text'] or '')[:70]}")


if __name__ == "__main__":
    main()
