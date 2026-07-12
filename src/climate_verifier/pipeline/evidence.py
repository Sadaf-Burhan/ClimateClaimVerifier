"""
Stage 4: Evidence Matching  (Week 6 — Semantic Search / RAG)

For each classified CLAIM, retrieve the most similar GDELT news articles from a
ChromaDB vector store and produce an *evidence-proximity* signal: does published
news cover a similar event?

This is NOT a truth verdict. It is a reader signal. Combined with reach
(engagement) and source context it surfaces the **reach-vs-support** mismatch —
a claim spreading widely with no news backing from an unverified source is the
misinformation red flag the reader should evaluate. The system never says "false".

Design:
  - Evidence corpus = GDELT news articles (`source='gdelt'`), excluding any post
    flagged `in_eval_set`/`in_train_set` (held out from the classifier).
  - Query = a classified claim (usually a Bluesky post).
  - Embedding model = `all-MiniLM-L6-v2` (same as Week 2), cosine space.
  - similarity = 1 - cosine_distance; proximity tier by config thresholds.

Build the index, then query:
  uv run python -m climate_verifier.pipeline.evidence --build
  uv run python -m climate_verifier.pipeline.evidence --claim "HAARP is causing the Alberta floods"
"""

import json
import re
import sqlite3
from pathlib import Path

import chromadb
import ollama
import yaml
from chromadb.utils import embedding_functions

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
COLLECTION = "gdelt_evidence"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _iso_date(created_at: str) -> str:
    """Normalize a stored timestamp to YYYY-MM-DD. GDELT uses the compact
    'YYYYMMDDTHHMMSSZ' form (a bare [:10] slice yields '20260605T2'); ISO
    timestamps already start with the date."""
    s = created_at or ""
    if len(s) >= 8 and s[:8].isdigit():          # GDELT compact 20260605T221500Z
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]                                # ISO 2026-06-05T... -> 2026-06-05


class ClimateEvidenceStore:
    """Persistent ChromaDB collection of GDELT news articles, queried by claim text."""

    def __init__(self, chroma_path: str, embed_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embed_model
            ),
            metadata={"hnsw:space": "cosine"},  # distances are cosine → similarity = 1 - distance
        )

    def count(self) -> int:
        return self.collection.count()

    def build_index(self, db_path: str) -> int:
        """(Re)index leak-free GDELT articles. Idempotent — upsert by post_id."""
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cols = [r[1] for r in con.execute("PRAGMA table_info(posts)").fetchall()]
        # exclude held-out posts from the retrieval pool (leakage guard)
        guard = ""
        if "in_eval_set" in cols:
            guard += " AND (in_eval_set IS NULL OR in_eval_set = 0)"
        if "in_train_set" in cols:
            guard += " AND (in_train_set IS NULL OR in_train_set = 0)"
        rows = con.execute(
            f"SELECT post_id, text, author, keyword_category, created_at "
            f"FROM posts WHERE source = 'gdelt'{guard}"
        ).fetchall()
        con.close()

        rows = [r for r in rows if (r["text"] or "").strip()]
        if rows:
            self.collection.upsert(
                ids=[r["post_id"] for r in rows],
                documents=[r["text"] for r in rows],
                metadatas=[{
                    "domain":   r["author"] or "",
                    "url":      r["post_id"],
                    "category": r["keyword_category"] or "",
                    "date":     _iso_date(r["created_at"]),
                } for r in rows],
            )
        return self.collection.count()

    def evidence_for_claim(self, claim_text: str, k: int = 5,
                           high: float = 0.60, low: float = 0.40) -> dict:
        """
        Returns {proximity, tier, matches}:
          proximity — top cosine similarity to any GDELT article (0-1)
          tier      — HIGH (news covers a similar event) / LOW / NONE (no corroboration)
          matches   — top-k articles [{title, domain, url, date, similarity}]
        """
        n = self.collection.count()
        if n == 0 or not (claim_text or "").strip():
            return {"proximity": 0.0, "tier": "NONE", "matches": []}
        res = self.collection.query(query_texts=[claim_text], n_results=min(k, n))
        sims = [round(1 - d, 3) for d in res["distances"][0]]
        matches = [{
            "title":      doc,
            "domain":     m.get("domain", ""),
            "url":        m.get("url", ""),
            "date":       m.get("date", ""),
            "similarity": s,
        } for doc, m, s in zip(res["documents"][0], res["metadatas"][0], sims)]
        top = sims[0] if sims else 0.0
        tier = "HIGH" if top >= high else ("LOW" if top >= low else "NONE")
        return {"proximity": top, "tier": tier, "matches": matches}


# Re-ranking pass (Module 6 "Advanced RAG"): dense retrieval finds topically-similar
# news, but topical overlap is NOT corroboration. The LLM re-reads the claim against the
# retrieved articles and judges whether any describes the SAME specific event.
# Safety guardrails baked into the prompt:
#   - judge ONLY from the provided article titles (no outside knowledge),
#   - NEVER decide whether the claim is true/false (that is the human reader's job),
#   - cite the supporting article number so the reader can open and review the source,
#   - "none" is scoped to the retrieved set — absence is not proof the event didn't happen.
_CORROBORATION_PROMPT = """You decide whether any of the NEWS ARTICLES listed below reports the SAME specific event, mechanism, place, or measurement that a claim asserts.

STRICT RULES — follow every one:
1. Judge ONLY from the article titles listed below. Do NOT use any outside knowledge about the claim, the events, HAARP, chemtrails, the weather, or the world. If it is not in the listed titles, it does not exist for this task.
2. Do NOT decide whether the claim is true or false — never. Your ONLY job is to report whether a listed article describes the same specific thing. Truth is judged by the human reader, not you.
3. An article corroborates ONLY if its own title reports the same specific event/mechanism the claim asserts — not merely the same topic, region, or weather in general. When in doubt, choose the WEAKER verdict.
4. If you answer "corroborated" or "partial" you MUST cite the exact article number whose title supports it, and your reason must name what that article reports.

Claim: "{claim}"

News articles (the ONLY evidence you may use):
{articles}

Return JSON only: {{"verdict": "corroborated" | "partial" | "none", "article": <the supporting article number, or 0 if none>, "reason": "<name what the cited article reports; under 15 words>"}}
- corroborated: a listed article reports the same specific event/mechanism as the claim.
- partial: a listed article covers the same topic or region but NOT the specific claim.
- none: no listed article reports the specific event — only loose topical overlap, or nothing."""


def corroboration_check(claim_text: str, matches: list[dict], model: str) -> dict:
    """LLM re-rank: does any retrieved article corroborate the SPECIFIC claim?
    Returns {verdict: corroborated|partial|none, article: int, reason: str}."""
    if not matches:
        return {"verdict": "none", "article": 0, "reason": "no candidate articles"}
    listing = "\n".join(f'{i}. [{m["domain"]}] {m["title"][:120]}' for i, m in enumerate(matches, 1))
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user",
                       "content": _CORROBORATION_PROMPT.format(claim=claim_text[:400], articles=listing)}],
            format={"type": "object",
                    "properties": {"verdict": {"type": "string"},
                                   "article": {"type": "integer"},
                                   "reason": {"type": "string"}},
                    "required": ["verdict", "article", "reason"]},
            options={"temperature": 0.0, "num_predict": 100},
        )
        m = re.search(r"\{.*\}", resp["message"]["content"], re.DOTALL)
        d = json.loads(m.group()) if m else {}
        verdict = str(d.get("verdict", "none")).lower()
        if verdict not in ("corroborated", "partial", "none"):
            verdict = "none"
        return {"verdict": verdict, "article": int(d.get("article", 0) or 0),
                "reason": str(d.get("reason", ""))}
    except Exception as e:
        return {"verdict": "none", "article": 0, "reason": f"check error: {e}"}


_URL_RE = re.compile(r"https?://[^\s)\]>]+|www\.[^\s)\]>]+", re.I)


def extract_citations(text: str, credible_domains: list[str]) -> list[dict]:
    """Links the post cites itself -> [{url, domain, credible}]. A self-citation means the
    post supplies its own evidence (which the reader can review) — the opposite of the
    'no support at all' misinformation pattern (e.g. a post linking a sciencedaily study)."""
    cites = []
    for raw in _URL_RE.findall(text or ""):
        url = raw.rstrip(".,);]")
        dom = re.sub(r"^https?://", "", url, flags=re.I).split("/")[0].lower()
        dom = dom[4:] if dom.startswith("www.") else dom
        credible = any(dom == d or dom.endswith("." + d) for d in credible_domains)
        cites.append({"url": url, "domain": dom, "credible": credible})
    return cites


def is_official(author: str, gdelt_domain: str, official: list[str]) -> bool:
    """Conservative allowlist match — exact or dotted-suffix on the handle/domain, so an
    'altgov' lookalike (altcdc.altgov.info) does NOT match a real agency (nws.noaa.gov)."""
    for c in ((author or "").lower(), (gdelt_domain or "").lower()):
        if not c:
            continue
        for o in (x.lower() for x in official):
            if c == o or c.endswith("." + o):
                return True
    return False


_FORECAST_RE = re.compile(
    r"\b(warn(?:ing|s|ed)?|forecast|expected|about to|braces? for|on track to|"
    r"this (?:weekend|week)|upcoming|will (?:hit|bring|be|cause)|advisory|watch|outlook|"
    r"heading (?:for|toward)|set to|days? ahead|week ahead)\b", re.I)


def looks_like_forecast(text: str) -> bool:
    """Heuristic: does the claim describe a FUTURE/predicted event (a warning/forecast)?
    Such claims can't be corroborated as having happened — news reports the warning, not the event."""
    return bool(_FORECAST_RE.search(text or ""))


def build_reader_signal(retrieval: dict, corro: dict, engagement: int, source: str,
                        followers: int = 0, domain: str = "", author: str = "",
                        citations: list[dict] | None = None, official: bool = False,
                        vision: dict | None = None, forecast: bool = False, high_reach: int = 50) -> dict:
    """
    Plain-language, *suggestive* reader signal from the corroboration verdict, reach,
    source context, self-citations, official-source status, and (for gated edge cases) an
    image signal. The reach-vs-support red flag is NOT raised for official sources, posts
    citing their own credible source, or edge cases whose image is a real on-the-ground
    photo of the event (a precision save) — those are legitimate reasons a real post lacks
    news corroboration. A cartoon/meme/AI image instead reinforces the flag. Never asserts truth.
    """
    citations = citations or []
    verdict = corro["verdict"]
    matches = retrieval["matches"]
    n = len(matches)
    art = corro.get("article", 0)
    cited = matches[art - 1] if verdict != "none" and 1 <= art <= n else None
    credible_cite = any(c["credible"] for c in citations)
    # A post that LINKS an official source (e.g. an NWS advisory) is *resharing* that
    # source — credibility travels with the origin, not the resharer — so it is treated
    # as official even from an unverified account.
    reshared_official = any(c.get("official") for c in citations)
    treated_official = official or reshared_official
    # Vision (edge cases only): a real on-the-ground photo of the event supports a genuine
    # claim → precision save (clears the flag). Cartoon/meme/AI imagery does not clear it.
    vision_supports = bool(vision and vision.get("image_type") == "real_photo"
                           and vision.get("depicts_claim") in ("yes", "partial"))

    if verdict == "corroborated" and cited:
        evidence_phrase = (f"A retrieved news article appears to report this event ({cited['domain']}) — "
                           "open the source to confirm it actually says so.")
    elif verdict == "partial" and cited:
        evidence_phrase = f"Retrieved news covers the topic but not this specific claim ({cited['domain']})."
    else:
        evidence_phrase = (f"None of the {n} retrieved news articles report this specific event. "
                           "Absence here is not proof it did not happen — the retrieved news set is limited.")

    if official:
        src_phrase = f"Source: a verified official source ({author or domain})."
    elif reshared_official and source == "bluesky":
        off_dom = next((c["domain"] for c in citations if c.get("official")), "")
        src_phrase = (f"Source: an unverified account, but it reshares an official source ({off_dom}) — "
                      "credibility credited to the origin, not the account.")
    elif source == "bluesky":
        src_phrase = f"Source: an unverified social account ({followers:,} followers)."
    else:
        src_phrase = f"Source: news domain {domain}." if domain else "Source: a news article."

    cite_phrase = ""
    if citations:
        doms = ", ".join(sorted({c["domain"] for c in citations}))
        cite_phrase = ("The post cites its own source (" + doms + ") — "
                       + ("a credible domain; review it." if credible_cite else "review the linked source."))

    reach_phrase = f"Reach: {engagement:,} engagements." if engagement else "Reach: low engagement."

    # Red flag ONLY when the reach-vs-support mismatch is real: high reach, no corroboration,
    # unverified social account, NOT official (directly or via reshare), NO credible cite,
    # and NOT rescued by a real on-the-ground photo of the event.
    red_flag = (engagement >= high_reach and verdict == "none"
                and source == "bluesky" and not treated_official and not credible_cite
                and not vision_supports)

    parts = [evidence_phrase, src_phrase]
    if cite_phrase:
        parts.append(cite_phrase)
    parts.append(reach_phrase)
    if vision and vision.get("note"):
        parts.append(vision["note"])
    if red_flag:
        parts.append("High reach but no corroboration in the retrieved news, from an unverified source "
                     "with no cited evidence — worth a closer look; this is the pattern of "
                     "misinformation amplification.")
    if forecast:
        # Frame the whole reading: a future warning can't be reported as having happened.
        parts.insert(0, "This reads as a FORECAST / WARNING about a *future* event — published news can "
                        "corroborate that the warning was issued, but a future event cannot be reported as "
                        "having already happened. Weigh the warning, not an occurrence.")
    return {"summary": " ".join(parts), "bullets": parts, "red_flag": red_flag, "verdict": verdict,
            "proximity": retrieval["proximity"], "reason": corro.get("reason", ""),
            "cited": cited, "official": official, "self_cited": bool(citations),
            "credible_cite": credible_cite, "reshared_official": reshared_official,
            "treated_official": treated_official, "vision": vision,
            "vision_supports": vision_supports, "forecast": forecast}


def assess_claim(store: "ClimateEvidenceStore", claim_text: str, engagement: int = 0,
                 source: str = "bluesky", followers: int = 0, domain: str = "",
                 author: str = "", vision: dict | None = None, cfg: dict | None = None) -> dict:
    """Full Stage-4 assessment: retrieve → corroborate (LLM re-rank) → reader signal,
    factoring self-citations, official-source status, and (edge cases) an image signal
    into the red flag."""
    cfg = cfg or _load_cfg()
    ev = cfg.get("evidence", {})
    retrieval = store.evidence_for_claim(claim_text, k=ev.get("top_k", 5),
                                         high=ev.get("high_proximity", 0.60),
                                         low=ev.get("low_proximity", 0.40))
    corro = corroboration_check(claim_text, retrieval["matches"], model=cfg["model"]["name"])
    official_list = ev.get("official_sources", [])
    citations = extract_citations(claim_text, ev.get("citation_domains", []))
    for c in citations:                                  # does the link reshare an official source?
        c["official"] = is_official(c["domain"], "", official_list)
    official = is_official(author, domain, official_list)
    forecast = looks_like_forecast(claim_text)
    signal = build_reader_signal(retrieval, corro, engagement, source, followers=followers,
                                 domain=domain, author=author, citations=citations,
                                 official=official, vision=vision, forecast=forecast,
                                 high_reach=ev.get("high_reach", 50))
    return {"retrieval": retrieval, "corroboration": corro, "citations": citations,
            "official": official, "vision": vision, "signal": signal}


def assess_db_claims(store: "ClimateEvidenceStore", db_path: str, limit: int = 10,
                     cfg: dict | None = None) -> list[dict]:
    """Assess the top classified claims by engagement (the highest-reach claims —
    where a reach-vs-support mismatch matters most). One LLM call per claim."""
    cfg = cfg or _load_cfg()
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT p.text, p.source, p.author, p.author_followers, p.vision_signal,
               (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p JOIN classifications c ON p.post_id = c.post_id
        WHERE c.has_claim = 1
        ORDER BY engagement DESC
        LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    out = []
    for r in rows:
        try:
            vision = json.loads(r["vision_signal"]) if r["vision_signal"] else None
        except Exception:
            vision = None
        a = assess_claim(store, r["text"], engagement=r["engagement"], source=r["source"],
                         followers=r["author_followers"] or 0, author=r["author"] or "",
                         domain=r["author"] if r["source"] == "gdelt" else "", vision=vision, cfg=cfg)
        out.append({"text": r["text"], "author": r["author"], "source": r["source"],
                    "engagement": r["engagement"], **a})
    return out


def get_store() -> ClimateEvidenceStore:
    cfg = _load_cfg()
    ev = cfg.get("evidence", {})
    return ClimateEvidenceStore(
        chroma_path=ev.get("chroma_path", "data/chroma_evidence"),
        embed_model=cfg["embedding"]["model_name"],
    )


def main():
    import argparse
    cfg = _load_cfg()
    ev = cfg.get("evidence", {})
    parser = argparse.ArgumentParser(description="Evidence matching — GDELT news corroboration for claims.")
    parser.add_argument("--build", action="store_true", help="(re)build the GDELT evidence index")
    parser.add_argument("--claim", type=str, help="a claim to match against news evidence")
    parser.add_argument("--engagement", type=int, default=0, help="engagement count (for the reach-vs-support red flag)")
    args = parser.parse_args()

    store = get_store()
    if args.build:
        n = store.build_index(cfg["storage"]["db_path"])
        print(f"Evidence index built: {n} GDELT articles in ChromaDB ({ev.get('chroma_path')}).")
    if args.claim:
        a = assess_claim(store, args.claim, engagement=args.engagement, cfg=cfg)
        print(f"\nClaim: {args.claim}")
        print(f"Retrieved (top proximity {a['retrieval']['proximity']:.3f}):")
        for m in a["retrieval"]["matches"]:
            print(f"  {m['similarity']:.3f}  [{m['domain']}]  {m['title'][:75]}")
        print(f"Corroboration: {a['corroboration']['verdict'].upper()} — {a['corroboration']['reason']}")
        print(f"Red flag: {a['signal']['red_flag']}")
        print(f"READER SIGNAL: {a['signal']['summary']}")
    if not args.build and not args.claim:
        parser.print_help()


if __name__ == "__main__":
    main()
