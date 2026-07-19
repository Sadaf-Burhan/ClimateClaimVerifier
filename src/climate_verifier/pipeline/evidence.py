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

from climate_verifier.pipeline.geo import extract_location, with_location

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


def _date_int(created_at: str) -> int:
    """YYYYMMDD integer for the hard date-window `where` filter, or 0 if unparseable.
    Numeric so ChromaDB can range-compare ($gte/$lte)."""
    iso = _iso_date(created_at)                  # -> 'YYYY-MM-DD'
    digits = iso.replace("-", "")
    return int(digits) if len(digits) == 8 and digits.isdigit() else 0


def _url_domain(url: str) -> str:
    """Bare registrable-ish domain of a URL: strip scheme, path, and a leading www."""
    dom = re.sub(r"^https?://", "", (url or ""), flags=re.I).split("/")[0].lower().strip()
    return dom[4:] if dom.startswith("www.") else dom


def _domain_is_credible(dom: str, credible_domains: list[str]) -> bool:
    """Exact or dotted-suffix match against the credible/official allowlist (same rule as
    extract_citations / is_official) — so 'x.theguardian.com' matches but 'theguardian.com.evil' can't."""
    return any(dom == d or dom.endswith("." + d) for d in credible_domains)


def _norm_url(url: str) -> str:
    """Loose URL equality key: lowercase, drop scheme, the query string / fragment (tracking
    params like ?utm_source, &CMP differ per reposter for the SAME article), and a trailing slash."""
    u = re.sub(r"^https?://", "", (url or "").strip(), flags=re.I).lower()
    u = u.split("#", 1)[0].split("?", 1)[0]
    return u[:-1] if u.endswith("/") else u


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
        """(Re)index leak-free evidence: GDELT articles, plus — when
        `evidence.index_credible_citations` — the CREDIBLE articles posts self-cite, so a
        post's own linked source can surface as REPORTED rather than a topical near-miss.
        Idempotent — upsert by id (GDELT keyed by post_id, citations by 'cite::<url>')."""
        # Cite docs are keyed by normalized URL (which can change) and must vanish if the flag is
        # turned off — upsert alone can't delete, so clear all prior cite docs and rebuild fresh.
        try:
            self.collection.delete(where={"cite": True})
        except Exception:
            pass
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

        # Region-aware index: derive a coarse location (headline place-name, else the domain —
        # e.g. weatherbc.com -> British Columbia) and embed it INTO the document so dense
        # retrieval prefers same-region news. `location`/`date_int` also live in metadata.
        docs, metas, ids = [], [], []
        for r in rows:
            if not (r["text"] or "").strip():
                continue
            loc = extract_location(r["text"], r["author"] or "")
            docs.append(with_location(r["text"], loc))
            metas.append({
                "domain":   r["author"] or "",
                "url":      r["post_id"],
                "category": r["keyword_category"] or "",
                "date":     _iso_date(r["created_at"]),
                "date_int": _date_int(r["created_at"]),
                "location": loc,
                "headline": r["text"],          # clean title (document carries the loc suffix)
                "cite":     False,
            })
            ids.append(r["post_id"])

        # Credible self-citations: a post's external link on a citation/official domain IS
        # evidence the reader can open, so index its headline too. Deduped by URL, leak-guarded.
        ev = _load_cfg().get("evidence", {})
        if ev.get("index_credible_citations", False) and "external_url" in cols:
            credible = list(ev.get("citation_domains", [])) + list(ev.get("official_sources", []))
            crows = con.execute(
                f"SELECT external_url, external_title, keyword_category, created_at "
                f"FROM posts WHERE source = 'bluesky' AND external_url IS NOT NULL AND external_url != '' "
                f"AND external_title IS NOT NULL AND external_title != ''{guard}"
            ).fetchall()
            seen = set()
            for r in crows:
                url, title = (r["external_url"] or "").strip(), (r["external_title"] or "").strip()
                key = _norm_url(url)
                if not url or not title or key in seen or not _domain_is_credible(_url_domain(url), credible):
                    continue
                seen.add(key)
                loc = extract_location(title, _url_domain(url))
                docs.append(with_location(title, loc))
                metas.append({
                    "domain":   _url_domain(url),
                    "url":      url,
                    "category": r["keyword_category"] or "",
                    "date":     _iso_date(r["created_at"]),
                    "date_int": _date_int(r["created_at"]),
                    "location": loc,
                    "headline": title,
                    "cite":     True,           # a credible source a post links, not a GDELT pull
                })
                ids.append("cite::" + key)
        con.close()

        if ids:
            # ChromaDB caps a single add/upsert (~5461 rows — the SQLite variable limit), so
            # upsert in batches under the client's max instead of one giant call.
            try:
                max_batch = self.client.get_max_batch_size()
            except Exception:
                max_batch = 5000
            step = max(1, min(max_batch, 5000))
            for i in range(0, len(ids), step):
                self.collection.upsert(
                    ids=ids[i:i + step],
                    documents=docs[i:i + step],
                    metadatas=metas[i:i + step],
                )
        return self.collection.count()

    def evidence_for_claim(self, claim_text: str, k: int = 5,
                           high: float = 0.60, low: float = 0.40,
                           claim_date: str = "", date_window_days: int = 0) -> dict:
        """
        Region- and time-aware retrieval. Returns {proximity, tier, location, matches}:
          proximity — top cosine similarity to any GDELT article (0-1)
          tier      — HIGH (news covers a similar event) / LOW / NONE (no corroboration)
          location  — the location derived from the claim and folded into the query
          matches   — top-k articles [{title, domain, url, date, location, similarity}]

        The claim's location is embedded INTO the query (same `with_location` form as the
        index) so same-region news ranks higher — a soft nudge, no pruning. `date_window_days`
        > 0 adds a HARD `where` filter to articles within +/-N days of the claim's date
        (dates are reliable, unlike best-effort location); 0 (default) disables it, because a
        live pasted post is dated NOW while the corpus is from its ingestion window — a date
        filter would then drop every article. Enable it only when claim and corpus are
        temporally aligned (e.g. batch-assessing same-era DB posts).
        """
        n = self.collection.count()
        if n == 0 or not (claim_text or "").strip():
            return {"proximity": 0.0, "tier": "NONE", "location": "", "matches": []}
        loc = extract_location(claim_text)
        query = with_location(claim_text, loc)

        where = None
        if date_window_days > 0:
            ci = _date_int(claim_date)
            if ci:
                lo, hi = ci - date_window_days, ci + date_window_days
                where = {"$and": [{"date_int": {"$gte": lo}}, {"date_int": {"$lte": hi}}]}

        res = self.collection.query(query_texts=[query], n_results=min(k, n), where=where)
        # a `where` filter can return fewer than k (or zero) — guard the empty case
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        sims = [round(1 - d, 3) for d in dists]
        matches = [{
            "title":      m.get("headline", doc),   # clean title; doc carries the loc suffix
            "domain":     m.get("domain", ""),
            "url":        m.get("url", ""),
            "date":       m.get("date", ""),
            "location":   m.get("location", ""),
            "similarity": s,
            "cite":       bool(m.get("cite")),       # True = a credible article a post self-cites
        } for doc, m, s in zip(docs, metas, sims)]
        top = sims[0] if sims else 0.0
        tier = "HIGH" if top >= high else ("LOW" if top >= low else "NONE")
        # Relevance floor: only surface articles at least topically close (>= low). Below that,
        # a "match" is noise — a different-region, different-topic headline that only ranked top
        # because nothing better exists. Showing it (e.g. a 0.26 UK article for a BC claim) is
        # misleading, so we return an empty set and the signal says "no relevant coverage found".
        matches = [m for m in matches if m["similarity"] >= low]
        return {"proximity": top, "tier": tier, "location": loc, "matches": matches}


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


def retrieval_only_verdict(retrieval: dict) -> dict:
    """Verdict from region-aware retrieval ALONE (when `use_llm_rerank` is off).

    Honest by construction: proximity is topical+regional similarity, NOT event-level
    confirmation, so a strong match becomes "partial" (open the source to confirm the
    specific event) and never "corroborated" — only the LLM re-read can claim that. A weak
    match is "none". This keeps the reach-vs-support red flag firing when nothing matches,
    without the extra LLM call the region-aware retrieval was built to make unnecessary."""
    tier = retrieval.get("tier", "NONE")
    if tier == "HIGH" and retrieval.get("matches"):
        return {"verdict": "partial", "article": 1,
                "reason": "retrieved news covers the same topic/region, but that is not the same as reporting "
                          "this specific claim — open it to check"}
    return {"verdict": "none", "article": 0,
            "reason": "no retrieved news is a close match to this specific claim"}


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


# Shared negative guard for BOTH reframes below. A framing that would otherwise defuse the red flag
# (first-person testimony, or a future warning) does NOT excuse a causal/mechanistic/conspiratorial
# assertion — the "casualness is a trap" rule. Those ARE exactly the claims a red flag targets, so
# "I saw them spraying chemtrails" and "they're warning us about the geoengineering agenda" must both
# still flag. Guarding forecast as well as eyewitness is what makes the forecast suppressor safe.
_CONSPIRACY_RE = re.compile(
    r"\b(chemtrail|haarp|geoengineer|cloud seed|weather (?:manipulat|control|modif)|"
    r"hoax|cover.?up|coverup|conspiracy|hiding|they'?re (?:spray|hiding|doing)|government|"
    r"agenda|spraying|caused by|because of|proof|exposed|depopulat|deep state|"
    r"scientists (?:hiding|lying)|man.?made|deliberate)\b", re.I)

_FORECAST_RE = re.compile(
    r"\b(warn(?:ing|s|ed)?|forecast|expected|about to|braces? for|on track to|"
    r"this (?:weekend|week)|upcoming|will (?:hit|bring|be|cause)|advisory|watch|outlook|"
    r"heading (?:for|toward)|set to|days? ahead|week ahead)\b", re.I)


def looks_like_forecast(text: str) -> bool:
    """Heuristic: does the claim describe a FUTURE/predicted event (a warning/forecast)?
    Such claims can't be corroborated as having happened — news reports the warning, not the event,
    so GDELT-silence is uninformative and the reach-vs-support mismatch does not apply.

    Conspiracy-guarded, exactly like `looks_like_eyewitness`: the forecast vocabulary is broad
    ("warning", "expected", "about to"), so without this guard a conspiracy post that merely uses
    the word "warning" would defuse its own red flag."""
    t = text or ""
    if _CONSPIRACY_RE.search(t):
        return False
    return bool(_FORECAST_RE.search(t))


# First-person, locally-anchored eyewitness observation: the poster reports a condition in
# their OWN surroundings ("smoke over my neighbourhood"). National news never covers one
# person's street, so GDELT-silence is uninformative — not the reach-vs-support red flag.
_FIRST_PERSON_RE = re.compile(r"\b(i|i'?m|my|we|we'?re|our|me|us)\b", re.I)
_LOCAL_ANCHOR_RE = re.compile(
    r"\bmy (neighbou?rhood|street|town|city|area|block|backyard|back yard|yard|window|"
    r"house|home|sky|skies|view|street|region|county)\b|"
    r"\b(right here|out(?:side)? my window|overhead|above (?:me|us|my)|"
    r"in my (?:area|town|city|neighbou?rhood|region)|where i (?:live|am))\b", re.I)


def looks_like_eyewitness(text: str) -> bool:
    """Heuristic: a first-person observation of the poster's OWN local surroundings, and NOT a
    causal/conspiratorial assertion. Such a claim is testimony news won't corroborate by nature —
    absence of a news match is expected, so it should be reframed, not red-flagged."""
    t = text or ""
    if _CONSPIRACY_RE.search(t):
        return False
    return bool(_FIRST_PERSON_RE.search(t) and _LOCAL_ANCHOR_RE.search(t))


def build_reader_signal(retrieval: dict, corro: dict, engagement: int, source: str,
                        followers: int = 0, domain: str = "", author: str = "",
                        citations: list[dict] | None = None, official: bool = False,
                        vision: dict | None = None, forecast: bool = False, high_reach: int = 50,
                        eyewitness: bool = False, topic_proximity: float = 0.50) -> dict:
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

    # DISPLAY status (decoupled from the red flag). TOPIC MATCH needs a REAL bar (`topic_proximity`),
    # not merely clearing the display floor — a ~0.42 neighbour (e.g. an Alaska-fishing headline for a
    # Florida-orange claim) is noise, so it reads as NO MATCH with the nearest headline shown, clearly
    # labelled as only loosely related.
    proximity = retrieval.get("proximity", 0.0)
    if verdict == "corroborated":
        news_status = "REPORTED"
    elif proximity >= topic_proximity:
        news_status = "TOPIC MATCH"
    else:
        news_status = "NO MATCH"

    # Region-mismatch demotion: the claim names a specific region, but every TOPIC MATCH is from a
    # DIFFERENT region (e.g. an Arizona flash-flood claim matched to Pennsylvania/Tennessee flash
    # floods). Same topic, not the same event → demote to NO MATCH and name the other regions.
    claim_region = retrieval.get("location") or ""
    claim_state = claim_region.split(",")[0].strip().lower() if "," in claim_region else ""
    region_mismatch = False
    other_regions = []
    if claim_state and news_status == "TOPIC MATCH":
        same = any(claim_state in (m.get("location") or "").lower() for m in matches)
        other_regions = sorted({m["location"] for m in matches
                                if m.get("location") and claim_state not in m["location"].lower()})
        if not same and other_regions:
            news_status = "NO MATCH"
            region_mismatch = True

    if verdict == "corroborated" and cited and corro.get("self_cite"):
        evidence_phrase = (f"The credible source this post cites ({cited['domain']}) reports this "
                           "exact claim — open the linked article to confirm it says so.")
    elif verdict == "corroborated" and cited:
        evidence_phrase = (f"A retrieved news article appears to report this event ({cited['domain']}) — "
                           "open the source to confirm it actually says so.")
    elif verdict == "partial" and cited:
        evidence_phrase = f"Retrieved news covers the topic but not this specific claim ({cited['domain']})."
    elif credible_cite or reshared_official:
        # The post supplies its OWN credible/official source — that linked article IS the evidence.
        # An independent news search is a bonus, so its absence is not a strike and must not read
        # like one. Lead with the source the post already provides, not "None retrieved".
        src_kind = "an official source" if reshared_official else "its own credible source"
        credible_doms = ", ".join(sorted({c["domain"] for c in citations if c.get("credible") or c.get("official")}))
        evidence_phrase = (f"This post links {src_kind} ({credible_doms}) — that linked article is its "
                           "supporting evidence; open it to verify. An independent news search found no "
                           "additional coverage of this specific event, which does not weaken the cited source.")
    elif region_mismatch:
        rlist = ", ".join(r.split(",")[0] for r in other_regions[:3])
        evidence_phrase = (f"Retrieved news covers the same topic but from OTHER regions ({rlist}) — not "
                           f"{claim_region.split(',')[0]}, so it likely does not report your specific event. "
                           "No same-region match was found; the corpus is limited, so this isn't proof either way.")
    elif news_status == "TOPIC MATCH":
        evidence_phrase = (f"Retrieved news covers this topic but none report this specific claim "
                           f"({n} related article{'s' if n != 1 else ''} below to judge). Absence of an exact "
                           "match is not proof it did not happen — the retrieved news set is limited.")
    elif n > 0:
        evidence_phrase = (f"No retrieved news is a real match to this claim — the nearest "
                           f"{n} headline{'s' if n != 1 else ''} below {'are' if n != 1 else 'is'} only loosely "
                           "related (weak similarity), not about this claim. Absence is not proof it did not "
                           "happen; the news corpus is limited.")
    else:
        evidence_phrase = ("No published news in the retrieved set is even topically close to this claim — "
                           "no relevant coverage was found. Absence here is not proof it did not happen; "
                           "the news corpus is limited.")

    # An unverified social source — a Bluesky account OR an uploaded off-platform screenshot.
    # Both carry no platform verification, so the reach-vs-support red flag and the unverified-
    # account wording apply to each (the ONE downstream change for the Week-7 image-input path).
    social = source in ("bluesky", "uploaded_screenshot")
    if official:
        src_phrase = f"Source: a verified official source ({author or domain})."
    elif reshared_official and social:
        off_dom = next((c["domain"] for c in citations if c.get("official")), "")
        src_phrase = (f"Source: an unverified account, but it reshares an official source ({off_dom}) — "
                      "credibility credited to the origin, not the account.")
    elif source == "uploaded_screenshot":
        src_phrase = ("Source: an uploaded screenshot of off-platform content — the original account is "
                      "unverified and the text was read by OCR (best-effort, degraded fidelity).")
    elif source == "bluesky":
        src_phrase = f"Source: an unverified social account ({followers:,} followers)."
    else:
        src_phrase = f"Source: news domain {domain}." if domain else "Source: a news article."

    # When the post cites a credible/official source the evidence_phrase above already leads with it,
    # so only add a citation line here for NON-credible self-citations (avoid saying it twice).
    cite_phrase = ""
    if citations and not (credible_cite or reshared_official):
        doms = ", ".join(sorted({c["domain"] for c in citations}))
        cite_phrase = f"The post links a source ({doms}) — review the linked source."

    reach_phrase = f"Reach: {engagement:,} engagements." if engagement else "Reach: low engagement."

    # A first-person report of the poster's OWN surroundings that news can't corroborate by
    # nature — GDELT-silence is uninformative here, so it does NOT count as the reach-vs-support
    # mismatch. Only defuses when there is genuinely no corroboration (verdict none); if news DID
    # match, the normal reading stands.
    eyewitness_defused = eyewitness and verdict == "none"
    # A FORECAST/warning about a future event cannot have been reported as having happened, so news
    # silence is uninformative here — the same logic that defuses eyewitness testimony. This was
    # always the documented design ("the flag is not raised for ... forecasts") but the flag never
    # actually checked it; the signal eval (scripts/eval_signal.py) caught the false alarm.
    # `looks_like_forecast` is conspiracy-guarded, so "they're warning us about chemtrails" still flags.
    forecast_defused = forecast and verdict == "none"

    # Red flag ONLY when the reach-vs-support mismatch is real: high reach, no corroboration,
    # unverified social account, NOT official (directly or via reshare), NO credible cite,
    # NOT rescued by a real on-the-ground photo, NOT a hyperlocal eyewitness observation, and
    # NOT a warning about something that hasn't happened yet.
    red_flag = (engagement >= high_reach and verdict == "none"
                and social and not treated_official and not credible_cite
                and not vision_supports and not eyewitness_defused
                and not forecast_defused)

    has_related = n > 0

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
    if eyewitness_defused:
        # Frame the whole reading: testimony about the poster's own surroundings, which national
        # news does not cover — so "no news match" is expected here, not a warning sign.
        parts.insert(0, "This reads as a FIRST-PERSON EYEWITNESS observation of the poster's own "
                        "surroundings — national news doesn't cover a single neighbourhood, so the "
                        "absence of a news match is expected here, not a red flag. Judge it as "
                        "testimony; an on-the-ground photo is the relevant evidence, not a news article.")
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
            "vision_supports": vision_supports, "forecast": forecast,
            "eyewitness": eyewitness_defused, "news_status": news_status, "has_related": has_related,
            "region_mismatch": region_mismatch, "other_regions": other_regions}


def assess_claim(store: "ClimateEvidenceStore", claim_text: str, engagement: int = 0,
                 source: str = "bluesky", followers: int = 0, domain: str = "",
                 author: str = "", vision: dict | None = None, cfg: dict | None = None,
                 claim_date: str = "", external_url: str = "",
                 retrieval: dict | None = None) -> dict:
    """Full Stage-4 assessment: retrieve (region/time-aware) → corroborate → reader signal,
    factoring self-citations, official-source status, and (edge cases) an image signal
    into the red flag. The LLM corroboration re-rank is optional (`evidence.use_llm_rerank`):
    when off, the verdict comes from region-aware retrieval alone.

    `retrieval` overrides the store lookup with a fixed result (and then `store` may be None). The
    signal eval needs this: the red flag depends on what the GDELT corpus happens to hold, which
    changes daily, so "expected red flag" is only stable ground truth when the retrieval side is
    held constant. Pinning it to the no-corroboration case isolates the SOURCE suppressors
    (official / credible cite / vision save / eyewitness) — i.e. the actual
    "precision is recovered downstream" claim."""
    cfg = cfg or _load_cfg()
    ev = cfg.get("evidence", {})
    if retrieval is None:
        retrieval = store.evidence_for_claim(claim_text, k=ev.get("top_k", 5),
                                             high=ev.get("high_proximity", 0.60),
                                             low=ev.get("low_proximity", 0.40),
                                             claim_date=claim_date,
                                             date_window_days=ev.get("date_window_days", 0))
    if ev.get("use_llm_rerank", False):
        corro = corroboration_check(claim_text, retrieval["matches"], model=cfg["model"]["name"])
    else:
        corro = retrieval_only_verdict(retrieval)
    # Credible self-citation → REPORTED: if the post links a credible article and retrieval's top
    # match IS that exact article (indexed via evidence.index_credible_citations), the post's own
    # cited source reports this — a stronger signal than a topical GDELT neighbour, so surface it.
    if ev.get("index_credible_citations", False) and external_url and retrieval.get("matches"):
        top = retrieval["matches"][0]
        if (top.get("cite") and _norm_url(top.get("url", "")) == _norm_url(external_url)
                and top.get("similarity", 0) >= ev.get("high_proximity", 0.60)):
            corro = {"verdict": "corroborated", "article": 1, "self_cite": True,
                     "reason": f"the credible source this post cites ({top.get('domain', '')}) reports this"}
    official_list = ev.get("official_sources", [])
    # Scan the post text AND its embed/link (external_url) for citations — a post that *shares* a
    # credible article via a Bluesky embed card cites its source just as much as one that pastes the
    # URL inline; without this it would be wrongly flagged "no cited evidence".
    cite_text = (claim_text or "") + ((" " + external_url) if external_url else "")
    citations = extract_citations(cite_text, ev.get("citation_domains", []))
    for c in citations:                                  # does the link reshare an official source?
        c["official"] = is_official(c["domain"], "", official_list)
    official = is_official(author, domain, official_list)
    forecast = looks_like_forecast(claim_text)
    eyewitness = looks_like_eyewitness(claim_text)
    signal = build_reader_signal(retrieval, corro, engagement, source, followers=followers,
                                 domain=domain, author=author, citations=citations,
                                 official=official, vision=vision, forecast=forecast,
                                 high_reach=ev.get("high_reach", 50), eyewitness=eyewitness,
                                 topic_proximity=ev.get("topic_proximity", 0.50))
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
        SELECT p.text, p.source, p.author, p.author_followers, p.vision_signal, p.created_at,
               p.external_url,
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
                         domain=r["author"] if r["source"] == "gdelt" else "", vision=vision, cfg=cfg,
                         claim_date=r["created_at"] or "", external_url=r["external_url"] or "")
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
        if a["retrieval"].get("location"):
            print(f"Claim location (folded into query): {a['retrieval']['location']}")
        print(f"Retrieved (top proximity {a['retrieval']['proximity']:.3f}):")
        for m in a["retrieval"]["matches"]:
            loc = f"  <{m['location']}>" if m.get("location") else ""
            print(f"  {m['similarity']:.3f}  [{m['domain']}]  {m['title'][:70]}{loc}")
        print(f"Corroboration: {a['corroboration']['verdict'].upper()} — {a['corroboration']['reason']}")
        print(f"Red flag: {a['signal']['red_flag']}")
        print(f"READER SIGNAL: {a['signal']['summary']}")
    if not args.build and not args.claim:
        parser.print_help()


if __name__ == "__main__":
    main()
