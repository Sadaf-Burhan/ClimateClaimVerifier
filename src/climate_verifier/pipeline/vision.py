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
import base64
import io
import json
import re
import sqlite3
from pathlib import Path

import ollama
import requests
import yaml
from PIL import Image

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


def _encode_jpeg(raw: bytes) -> str | None:
    """Re-encode arbitrary image bytes to a base64 JPEG string for Ollama. Ollama's image
    loader (stb_image) cannot decode WebP (Bluesky's CDN) or many PNG variants — feeding it
    raw yields a blank image and a canned description. Round-tripping through PIL to JPEG is
    what makes the model actually see the picture."""
    try:
        buf = io.BytesIO()
        Image.open(io.BytesIO(raw)).convert("RGB").save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _fetch_jpeg(image_url: str, timeout: int) -> bytes | None:
    """Download an image and re-encode to JPEG. Bluesky's CDN serves WebP, which Ollama's
    image loader (stb_image) cannot decode — feeding it raw WebP silently yields a blank
    image and a canned description. Converting to JPEG is what makes the model actually see
    the picture. A browser User-Agent avoids the CDN rejecting the default one."""
    try:
        r = requests.get(image_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        if not r.headers.get("content-type", "").startswith("image/") or len(r.content) < 500:
            return None
        buf = io.BytesIO()
        Image.open(io.BytesIO(r.content)).convert("RGB").save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        return None


def analyze_image(image_url: str, claim: str, model: str, corrector: str, timeout: int = 20) -> dict | None:
    """Download + re-encode the image and run the vision model -> normalized signal dict, or None."""
    img = _fetch_jpeg(image_url, timeout)
    if img is None:
        return None
    try:
        b64 = base64.b64encode(img).decode()
        resp = ollama.chat(model=model, options={"temperature": 0}, messages=[{
            "role": "user", "content": _VISION_PROMPT.format(claim=(claim or "")[:400]), "images": [b64]}])
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


# ── Week 7 image-INPUT path ─────────────────────────────────────────────────────
# Distinct from the GATED vision above (which classifies the image of a KNOWN Bluesky post).
# Here a user uploads a SCREENSHOT of off-platform content (X/FB/IG/WhatsApp, a meme, an
# infographic) and the claim itself is read OUT of the image, then fed to the SAME pipeline as
# the Bluesky-link path. This is a degraded-fidelity coverage path: the transcription is
# best-effort OCR and there is no canonical post to open, so its signals are lower-confidence.
_EXTRACT_PROMPT = """You are extracting structured information from a SCREENSHOT or IMAGE of a social-media post or infographic about climate or weather, so a fact-checking tool can assess the claim it carries. The image is off-platform (X/Twitter, Facebook, Instagram, WhatsApp, a meme, or an infographic).

Your job is TRANSCRIPTION and OBSERVATION, never judgement. Follow every rule:
1. TRANSCRIBE LITERALLY. For claim_text, author_handle and the engagement counts, copy EXACTLY what is written in the image. Do NOT paraphrase, complete, correct, translate or invent anything.
2. NEVER GUESS. If a field is not clearly legible in the image, return null for it. A hallucinated handle or engagement number is the worst possible error — when unsure, use null.
3. claim_text = the main factual statement shown in the image (the headline/claim on a card, the body of the post). Transcribe the primary readable text. null only if there is no legible text at all.
4. INFER platform ONLY from visible branding (an X/Twitter, Bluesky, Facebook, Instagram, TikTok, Reddit or WhatsApp logo / UI). null if there is no clear branding.
5. engagement = the like / repost / reply counts VISIBLE in the image, as integers. Any count not shown = null. Do NOT sum, estimate or convert.
6. image_type describes the MAIN visual the post is built around (the central photo, chart, cartoon or AI-generated image the claim concerns) — NOT the fact that this whole upload is itself a screenshot of a post. Use "screenshot" ONLY when that central visual is a screenshot of another app or website.
7. Describe ONLY what is visibly in the image. Do NOT decide whether the claim is true or false — that is the reader's job, not yours.

Return ONLY this JSON (no markdown fences, no extra text):
{"claim_text": "<transcribed main text, or null>",
 "has_readable_text": true,
 "image_type": "real_photo|meme_or_cartoon|screenshot|infographic|ai_suspected|other",
 "depicts_claim": "yes|partial|no|unrelated",
 "author_handle": "<@handle exactly as shown, or null>",
 "platform": "<x|twitter|bluesky|facebook|instagram|tiktok|reddit|whatsapp|other, or null>",
 "engagement": {"likes": <int or null>, "reposts": <int or null>, "replies": <int or null>},
 "visible_citation": "<any source name or URL shown in the image, or null>",
 "description": "<one short sentence, only what is visible>"}"""

_PLATFORMS = {"x", "twitter", "bluesky", "facebook", "instagram", "threads", "tiktok",
              "reddit", "whatsapp", "telegram", "youtube", "linkedin", "mastodon", "truth social"}


def _clean_str(v) -> str | None:
    """Literal-transcription guard: return the trimmed string, or None for any
    not-legible / placeholder value (the model is told to null these, but it slips)."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "n/a", "na", "unknown", "unclear", "not visible",
                              "not legible", "illegible"):
        return None
    return s


def _clean_int(v) -> int | None:
    """Parse a transcribed count to an int, tolerating '1,234' and '1.2k'/'3M' social shorthand.
    None (never a guessed 0) when it isn't a clean number — a hallucinated count is the worst failure."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip().lower().replace(",", "")
    if not s or s in ("null", "none"):
        return None
    mult = 1
    if s and s[-1] in ("k", "m"):
        mult, s = (1000 if s[-1] == "k" else 1_000_000), s[:-1]
    try:
        return int(float(s) * mult)
    except Exception:
        return None


def normalize_extraction(out: dict | None) -> dict | None:
    """Clamp a parsed extraction dict to the schema — allowed enums, literal-null strings,
    integer-or-null engagement. Keeps the transcription honest (null over guess)."""
    if not out:
        return None
    it = str(out.get("image_type", "other")).lower()
    dp = str(out.get("depicts_claim", "unrelated")).lower()
    eng = out.get("engagement") if isinstance(out.get("engagement"), dict) else {}
    platform = _clean_str(out.get("platform"))
    if platform:
        platform = platform.lower()
    claim = _clean_str(out.get("claim_text"))
    handle = _clean_str(out.get("author_handle"))
    cite = _clean_str(out.get("visible_citation"))
    eng_vals = [_clean_int(eng.get(k)) for k in ("likes", "reposts", "replies")]
    # A result with NOTHING read — no claim, no handle, no citation, no engagement — is not a
    # finding, it's a failed extraction (usually the corrector echoing its own template). Returning
    # it would look like "the image has no text" and get a real post rejected by the social-post
    # gate, so fail honestly instead and let the caller say "couldn't read the image".
    if claim is None and handle is None and cite is None and not any(v is not None for v in eng_vals):
        return None
    hrt = out.get("has_readable_text")
    if hrt is None:
        hrt = claim is not None
    # Can't have "readable text" while having read no text — the model asserts this pair
    # inconsistently, and downstream gates on it.
    if claim is None:
        hrt = False
    return {
        "claim_text": claim,
        "has_readable_text": bool(hrt),
        "image_type": it if it in _VALID_TYPES else "other",
        "depicts_claim": dp if dp in _VALID_DEPICTS else "unrelated",
        "author_handle": _clean_str(out.get("author_handle")),
        "platform": platform,
        "engagement": {
            "likes": _clean_int(eng.get("likes")),
            "reposts": _clean_int(eng.get("reposts")),
            "replies": _clean_int(eng.get("replies")),
        },
        "visible_citation": _clean_str(out.get("visible_citation")),
        "description": str(out.get("description", ""))[:300],
    }


def _correct_extraction(raw: str, model: str) -> dict | None:
    """Reformat malformed extraction output to the schema with a cheap TEXT model (never sees
    the image). It only reshapes the JSON it is given — it must not add facts.

    The placeholders below are deliberately NOT valid JSON (`<...>`). An earlier version showed a
    template that was itself parseable with every value already null and has_readable_text=true —
    so when the 3B corrector simply ECHOED the template (which small models do), _parse_json
    accepted it and we silently produced a confident all-null extraction: no claim, no engagement,
    but has_readable_text=true. That looked like "the image is unreadable" and got real posts
    REJECTED by the social-post gate. Now an echo fails to parse and returns None, so the caller
    reports an honest extraction failure instead of a fabricated empty result."""
    try:
        resp = ollama.chat(model=model, options={"temperature": 0}, messages=[{
            "role": "user",
            "content": ("Reformat the text below into ONLY this JSON shape. Copy values VERBATIM from "
                        "the text; use null for anything genuinely absent. Do NOT invent values and do "
                        "NOT copy the placeholders — replace every <...> with a real value or null.\n"
                        '{"claim_text": <transcribed text or null>, "has_readable_text": <true or false>, '
                        '"image_type": <one of: real_photo|meme_or_cartoon|screenshot|infographic|ai_suspected|other>, '
                        '"depicts_claim": <one of: yes|partial|no|unrelated>, "author_handle": <handle or null>, '
                        '"platform": <platform or null>, "engagement": {"likes": <int or null>, '
                        '"reposts": <int or null>, "replies": <int or null>}, '
                        '"visible_citation": <source or null>, "description": <one short sentence>}\n'
                        "No markdown, no extra text. Text to reformat:\n\n" + str(raw))}])
        return _parse_json(resp["message"]["content"])
    except Exception:
        return None


def extract_from_image(image: bytes | str, model: str, corrector: str,
                       timeout: int = 30) -> dict | None:
    """Week 7 image-INPUT path: OCR + structure a user-uploaded screenshot of an off-platform
    climate claim into the extraction schema (claim_text, image_type, author_handle, platform,
    engagement, visible_citation, …). Accepts raw image bytes (a Streamlit upload) or a file
    path. Returns the normalized dict, or None if the image can't be read / the model fails.
    Same model as the gated edge-case vision — one model, two entry points."""
    raw = bytes(image) if isinstance(image, (bytes, bytearray)) else None
    if raw is None:
        try:
            raw = Path(image).read_bytes()
        except Exception:
            return None
    b64 = _encode_jpeg(raw)
    if b64 is None:
        return None
    try:
        resp = ollama.chat(model=model, options={"temperature": 0}, messages=[{
            "role": "user", "content": _EXTRACT_PROMPT, "images": [b64]}])
        content = resp["message"]["content"]
        return normalize_extraction(_parse_json(content) or _correct_extraction(content, corrector))
    except Exception:
        return None


def looks_like_social_post(extracted: dict) -> bool:
    """The scanner assesses the REACH-vs-support of social-media posts, so the image-input path's
    one hard entry requirement is visible engagement. An upload with no like/repost/comment count
    isn't a post we can assess (reach is undefined) — bare photos, satellite/comparison images,
    infographics, memes and illustrations fail this gate and are rejected with guidance."""
    eng = (extracted or {}).get("engagement") or {}
    return any(eng.get(k) is not None for k in ("likes", "reposts", "replies"))


def _citation_as_url(citation: str) -> str:
    """A visible_citation is often a BARE domain the vision model read off the image
    ('climate.us', 'vacancybridge.com'). The downstream citation detector (extract_citations)
    only recognizes http/www URLs, so a bare domain would never be credited. Normalize a
    domain-looking token to a URL. Leave source NAMES ('The New York Times', 'NOAA') and
    already-qualified URLs untouched — a name has no domain to match on."""
    c = (citation or "").strip()
    if not c or c.lower().startswith(("http://", "https://", "www.")):
        return c
    if re.match(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(/\S*)?$", c.lower()):   # domain[/path], no spaces
        return "https://" + c
    return c                                                              # a name, not a domain


def screenshot_signal_inputs(extracted: dict) -> dict:
    """Thin adapter: map an extract_from_image() result onto the inputs assess_claim() expects,
    so the SAME pipeline (classify → region-aware RAG → reader signal) runs on an uploaded
    screenshot. Sums the visible engagement counts into one int, packs the image classification
    into the vision dict the signal builder already understands (with a reader note), null-fills
    followers, and tags source='uploaded_screenshot' (an unverified social source)."""
    eng = extracted.get("engagement") or {}
    engagement = sum(v for v in (eng.get("likes"), eng.get("reposts"), eng.get("replies"))
                     if isinstance(v, int))
    vs = {
        "image_type": extracted.get("image_type", "other"),
        "depicts_claim": extracted.get("depicts_claim", "unrelated"),
        "description": extracted.get("description", ""),
    }
    vs["note"] = vision_reader_note(vs)
    return {
        "claim_text": extracted.get("claim_text") or "",
        "engagement": engagement,
        "source": "uploaded_screenshot",
        "followers": 0,
        "author": extracted.get("author_handle") or "",
        "external_url": _citation_as_url(extracted.get("visible_citation") or ""),
        "vision": vs,
    }


def gate_edge_cases(db_path: str, cfg: dict, limit: int | None = None) -> list[dict]:
    """
    Select posts to escalate to vision: a claim, with an image, NOT already resolved by
    metadata (official author / credible link / reshare-of-official). A text
    misclassification (the precision problem) can occur in ANY category, so the gate is
    NOT category-restricted — the image is escalated wherever text alone was uncertain and
    a picture could correct it. `vision.gate_categories` may still narrow it (empty = all).
    Highest reach first, capped.
    """
    ev, vz = cfg.get("evidence", {}), cfg.get("vision", {})
    official = ev.get("official_sources", [])
    cats = vz.get("gate_categories") or []                      # empty/None => all categories
    cat_clause = f"AND p.keyword_category IN ({','.join('?' * len(cats))})" if cats else ""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(f"""
        SELECT p.post_id, p.text, p.author, p.image_url, p.keyword_category,
               p.reshare_of_author, p.external_url,
               (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p JOIN classifications c ON p.post_id = c.post_id
        WHERE p.has_image = 1 AND c.has_claim = 1 AND p.source = 'bluesky'
              {cat_clause}
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
