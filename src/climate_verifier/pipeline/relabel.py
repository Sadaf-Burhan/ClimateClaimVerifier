"""
Admin relabel layer — the human half of the evidence-nominated eval-set loop.

END USERS never relabel; they only see the runtime "possibly a missed claim" hint. The
ADMIN (site owner), behind the Maintenance tab's auth gate, reviews evidence-nominated
candidates and relabels. A confirmed relabel makes TWO writes because they are different stores:

  1. an **admin-override label** on the post (`classifications.admin_label`) that the app PREFERS
     over the model's — so users see the corrected label on refresh; and
  2. an append to the eval **benchmark** (`claim_eval.csv`) so the drift measurement / eventual
     classifier improve.

Nominations run both directions:
  - OPINION that looks like a CLAIM — strong evidence (REPORTED/TOPIC MATCH), official source,
    or a credible self-citation. (High value: these are the costly false negatives.)
  - CLAIM that looks like an OPINION — no related coverage, not official, no credible cite.
    (Lower-confidence / noisier, since the headline-only corpus means real claims often lack a
    match — shown as a separate, clearly-labelled section.)
"""

import csv
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from climate_verifier.pipeline.evidence import assess_claim, extract_citations, is_official
from climate_verifier.pipeline.topic_filter import is_na_relevant


def _norm_words(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())


def verbatim_headline_domain(text: str, external_title: str, external_url: str,
                            credible_domains: list[str]) -> str:
    """A news outlet posting its OWN story: the post's words ARE the headline of the credible
    article it links. Returns that domain, or "" if it isn't the case.

    Why this is a nomination signal and NOT a classifier input: the headline and the post text are
    the SAME string, so there is no extra information to give the classifier — it already read those
    exact words. This is *provenance*: a credible outlet republishing its own headline is reporting,
    not commentary, which makes a CLAIM label likely. It stays a NOMINATION because the inference is
    only probabilistic — op-ed headlines are verbatim headlines too ("Why we must act on climate" is
    a real Guardian headline and an opinion). Evidence nominates; the human disposes.
    """
    t = _norm_words(text)
    if not t or not (external_title or "").strip():
        return ""
    # Drop a trailing site-name suffix ("… | Extreme heat - Bytes Europe") before matching.
    core = _norm_words(re.split(r"\s+[-|]\s+", external_title)[0])
    if len(core.split()) < 4:                    # too short to be a distinctive headline
        return ""
    if core not in t:                            # the headline must appear verbatim in the post
        return ""
    dom = re.sub(r"^https?://", "", (external_url or ""), flags=re.I).split("/")[0].lower()
    dom = dom[4:] if dom.startswith("www.") else dom
    if not dom:
        return ""
    return dom if any(dom == d or dom.endswith("." + d) for d in credible_domains) else ""


def ensure_admin_columns(conn: sqlite3.Connection) -> None:
    """Add the admin-override columns to `classifications` (idempotent)."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(classifications)")}
    if "admin_label" not in have:      # NULL = not reviewed; 1 = admin says claim; 0 = admin says opinion
        conn.execute("ALTER TABLE classifications ADD COLUMN admin_label INTEGER")
    if "admin_labeled_at" not in have:
        conn.execute("ALTER TABLE classifications ADD COLUMN admin_labeled_at TEXT")
    conn.commit()


def get_relabel_candidates(store, db_path: str, cfg: dict, scan_limit: int = 100) -> dict:
    """Scan the top classified Bluesky posts (by engagement) and nominate label mismatches.
    Skips posts already admin-reviewed (`admin_label` set). Returns
    {opinion_to_claim: [...], claim_to_opinion: [...]}."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    ensure_admin_columns(con)
    rows = con.execute("""
        SELECT p.post_id, p.text, p.author, p.author_followers, p.external_url, p.external_title,
               p.keyword_category, c.has_claim, c.admin_label,
               (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p JOIN classifications c ON p.post_id = c.post_id
        WHERE p.source = 'bluesky'
        ORDER BY engagement DESC LIMIT ?
    """, (scan_limit,)).fetchall()
    con.close()

    credible = cfg.get("evidence", {}).get("citation_domains", [])
    to_claim, to_opinion = [], []
    for r in rows:
        if r["admin_label"] is not None:          # already reviewed by the admin
            continue
        if not is_na_relevant(r["text"] or ""):
            continue                              # out of scope — see the note in get_signal_candidates
        a = assess_claim(store, r["text"], engagement=r["engagement"], source="bluesky",
                         followers=r["author_followers"] or 0, author=r["author"] or "",
                         domain="", cfg=cfg, external_url=r["external_url"] or "")
        s = a["signal"]
        # STRONG signals only — a bare TOPIC MATCH fires on almost any climate opinion (topical
        # overlap is expected), so it's too weak to nominate on. Require news that actually REPORTS
        # the event, an official source, or a credible self-citation.
        why = []
        if s["news_status"] == "REPORTED":
            why.append("news reports this event")
        if s.get("treated_official"):
            why.append("official source")
        # A credible outlet posting its own headline verbatim: a sharper reason than the generic
        # "credible cited source" it would otherwise fire as, so it replaces it rather than stacking.
        vdom = verbatim_headline_domain(r["text"], r["external_title"], r["external_url"], credible)
        if vdom:
            why.append(f"post text IS the headline of its own linked article ({vdom}) — reporting, "
                       "not commentary")
        elif s.get("credible_cite"):
            why.append("credible cited source")
        cand = {"post_id": r["post_id"], "text": r["text"], "author": r["author"] or "",
                "engagement": r["engagement"], "keyword_category": r["keyword_category"] or "",
                "external_url": r["external_url"] or "", "has_claim": r["has_claim"],
                "news_status": s["news_status"], "why": ", ".join(why),
                "verbatim_headline": vdom}
        if r["has_claim"] == 0 and why:                       # OPINION but evidence contradicts
            to_claim.append(cand)
        elif r["has_claim"] == 1 and not why and s["news_status"] == "NO MATCH":  # CLAIM, no support
            to_opinion.append(cand)
    return {"opinion_to_claim": to_claim, "claim_to_opinion": to_opinion}


def get_signal_candidates(db_path: str, cfg: dict, store=None, limit: int = 60) -> dict:
    """The BENCHMARK-GROWTH lane: sweep the WHOLE corpus with CHEAP deterministic signals and
    nominate contradicted OPINIONs ranked by SIGNAL STRENGTH, not reach.

    Why a second lane: `get_relabel_candidates` ranks by engagement and caps the scan, because it
    calls assess_claim (a vector search) per post. That's correct for the RED-FLAG product — a
    reach-vs-support mismatch only matters at reach — but wrong for growing the eval set, since the
    classifier's blind spots don't correlate with likes. Measured on the corpus: 65 contradicted
    opinions exist and only 3 are high-reach enough for the engagement-ranked scan to ever see
    (e.g. 10 of 11 verbatim-headline cases sit at engagement ranks 429-2759).

    This lane is affordable over all posts because every signal here is pure text/metadata — no
    retrieval, no LLM (~0.02s for 2.8k posts). Only the small STRONG shortlist is then enriched with
    news_status, which is the one signal that does need the store.

    Returns {strong, weak} — tiered, because they are NOT equally trustworthy:
      strong — verbatim credible headline, or an official account: an outlet posting its own story
               is reporting, not commentary. Review these.
      weak   — a credible citation alone. "Here's a great Guardian piece, so depressing" links a
               credible source and is CORRECTLY an opinion. Noisy: a backlog, not a queue.
    """
    ev = cfg.get("evidence", {})
    official_list = ev.get("official_sources", [])
    credible = ev.get("citation_domains", [])
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    ensure_admin_columns(con)
    rows = con.execute("""
        SELECT p.post_id, p.text, p.author, p.external_url, p.external_title, p.keyword_category,
               c.has_claim, (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p JOIN classifications c ON p.post_id = c.post_id
        WHERE p.source = 'bluesky' AND c.has_claim = 0 AND c.admin_label IS NULL
    """).fetchall()
    con.close()

    strong, weak = [], []
    for r in rows:
        text, url = r["text"] or "", r["external_url"] or ""
        # SCOPE GATE. The system is North-America-scoped by design (README / project_description's
        # "Scope Decision"), and both production classify prompts say so. Nominating an out-of-scope
        # post invites a relabel to CLAIM — which is how 5 UK rows got into the benchmark, where the
        # NA-scoped classifier correctly answers "not a NA claim" and is scored as a false negative
        # for it. A UK headline IS a claim; it just isn't this system's claim. Cheap check, and it
        # keeps the leak from re-entering via the queue while the leaked posts age out (14d retention).
        if not is_na_relevant(text):
            continue
        vdom = verbatim_headline_domain(text, r["external_title"], url, credible)
        cites = extract_citations(text + " " + url, credible)
        if vdom:
            tier, why = "strong", (f"post text IS the headline of its own linked article ({vdom}) "
                                   "— reporting, not commentary")
        elif is_official(r["author"], "", official_list):
            tier, why = "strong", f"official source (@{r['author']})"
        elif any(c["credible"] for c in cites):
            doms = ", ".join(sorted({c["domain"] for c in cites if c["credible"]}))
            tier, why = "weak", (f"links a credible source ({doms}) — but sharing an article with a "
                                 "vibes caption is legitimately an OPINION, so judge carefully")
        else:
            continue
        cand = {"post_id": r["post_id"], "text": text, "author": r["author"] or "",
                "engagement": r["engagement"], "keyword_category": r["keyword_category"] or "",
                "external_url": url, "has_claim": r["has_claim"],
                "news_status": "not checked", "why": why}
        (strong if tier == "strong" else weak).append(cand)

    # Engagement is only a TIE-BREAK here (show the most-seen first) — never a filter, which is the
    # entire point of this lane.
    strong.sort(key=lambda c: -c["engagement"])
    weak.sort(key=lambda c: -c["engagement"])
    strong, weak = strong[:limit], weak[:limit]

    if store is not None:                      # enrich only the small strong list with news_status
        for c in strong:
            try:
                a = assess_claim(store, c["text"], engagement=c["engagement"], source="bluesky",
                                 author=c["author"], cfg=cfg, external_url=c["external_url"])
                c["news_status"] = a["signal"]["news_status"]
            except Exception:
                pass
    return {"strong": strong, "weak": weak}


def set_admin_label(db_path: str, post_id: str, admin_label: int) -> None:
    """Write the admin override (1 = claim, 0 = opinion) + timestamp. The app prefers this over
    the model's `has_claim`, so users see the corrected label on refresh."""
    con = sqlite3.connect(db_path)
    ensure_admin_columns(con)
    con.execute("UPDATE classifications SET admin_label = ?, admin_labeled_at = ? WHERE post_id = ?",
                (admin_label, datetime.now(timezone.utc).isoformat(), post_id))
    con.commit()
    con.close()


def eval_post_types(csv_path: str) -> list[str]:
    """Distinct `post_type` "thought" categories currently in the eval CSV, sorted. Read FRESH on
    every call (no caching) so the admin dropdown always reflects the real file — never appends a
    duplicate/typo'd category or works off a stale list."""
    p = Path(csv_path)
    if not p.exists():
        return []
    seen = set()
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = (r.get("post_type") or "").strip()
            if t:
                seen.add(t)
    return sorted(seen)


def append_to_eval_csv(csv_path: str, post_text: str, expected_label: str, keyword_category: str = "",
                       post_type: str = "", notes: str = "") -> None:
    """Append a corrected (post, label, thought) row to the eval benchmark. Matches the header:
    post_text, expected_label, keyword_category, post_type, notes."""
    p = Path(csv_path)
    text = " ".join((post_text or "").split())          # flatten newlines for a clean CSV cell
    with open(p, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([text, expected_label, keyword_category,
                                (post_type or "real_data").strip(),
                                (notes or "admin-confirmed relabel").strip()])


def apply_relabel(db_path: str, csv_path: str, post_id: str, post_text: str, corrected_label: str,
                  keyword_category: str = "", post_type: str = "", notes: str = "") -> None:
    """Confirm a relabel: BOTH writes — the admin override (users see it) and the eval-CSV append
    with the admin-chosen `post_type` (thought) + `notes`."""
    set_admin_label(db_path, post_id, 1 if corrected_label == "claim" else 0)
    append_to_eval_csv(csv_path, post_text, corrected_label, keyword_category, post_type, notes)


def mark_reviewed_ok(db_path: str, post_id: str, model_label: int) -> None:
    """Reject a nomination = the model was right. Stamp admin_label = model's label so it leaves
    the queue and the displayed label is unchanged. No eval-CSV append."""
    set_admin_label(db_path, post_id, model_label)
