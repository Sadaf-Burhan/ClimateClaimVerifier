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
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from climate_verifier.pipeline.evidence import assess_claim, extract_citations, is_official
from climate_verifier.pipeline.topic_filter import is_na_relevant

# The admin overrides LOG — the source of truth for human label judgments, and git-tracked.
#
# Why this file exists at all: `classifications.admin_label` used to be the only home for an
# override, and `ingested.db` is STATE that gets replaced wholesale on every Colab round-trip. So
# every relabel was irreplaceable human judgment stored in the most disposable file in the project,
# and the round-trip silently ate it: 5 of the 14 relabels committed on 2026-07-15 came back with
# admin_label NULL, and because the nomination queries filter `admin_label IS NULL`, those posts
# re-entered the review queue as though they had never been judged. An uncommitted relabel died
# outright — CSV row and override both.
#
# So overrides live HERE (append-only, one JSON per line, git-tracked, last line wins per post_id)
# and the DB column becomes a derived CACHE, rebuilt by `apply_overrides`. That matches the ops
# rule already in force elsewhere: logs accumulate, state gets versioned. A human decision is a
# log entry, not state — it can never be recomputed, so it must never live only in a file we throw
# away. Append-only (rather than rewriting a dict) keeps every change of mind in git history and
# makes concurrent appends safe.
ADMIN_OVERRIDES_PATH = Path("data/admin_overrides.jsonl")


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


# An opinion section republishing its own headline is NOT reporting — "Carbon capture is vital…"
# from the Guardian's Letters page is a verbatim credible headline AND an opinion. These patterns
# catch the common opinion sections so the provenance override skips them (they stay the
# classifier's / a human's call). Measured: this is exactly the case where a blunt "verbatim ->
# claim" rule would manufacture a false positive.
_OPED_TITLE_RE = re.compile(r"\|\s*(letters?|opinion|comment is free|comment|editorial|analysis|voices)\s*$", re.I)
_OPED_URL_RE = re.compile(r"/(commentisfree|opinion|editorial|letters?|voices|analysis|comment)(/|$)", re.I)


def is_oped(external_title: str, external_url: str) -> bool:
    """True if the linked article is an opinion-section piece (Letters / Comment is free / Opinion /
    Editorial / Analysis / Voices) — detected from the title suffix or the URL path."""
    return bool(_OPED_TITLE_RE.search(external_title or "") or _OPED_URL_RE.search(external_url or ""))


def provenance_override(text: str, external_title: str, external_url: str,
                        credible_domains: list[str]) -> bool:
    """The deterministic provenance rule: a post whose text IS the verbatim headline of its linked
    CREDIBLE, NON-opinion article is a CLAIM — a news outlet reporting its own story.

    Why deterministic and downstream, NOT a classifier-prompt rule: measured head-to-head, adding a
    headline rule to the prompt perturbed ~25 unrelated classifications and dropped the frozen gold
    recall, because a 3B model is globally sensitive to prompt edits. This rule fires ONLY on the
    handful of verbatim-headline rows and never touches the prompt, so every other post — and all of
    gold — is byte-identical. The op-ed guard is what keeps it from flipping Letters/Comment pages."""
    if is_oped(external_title, external_url):
        return False
    return bool(verbatim_headline_domain(text, external_title, external_url, credible_domains))


def apply_provenance_labels(db_path: str, cfg: dict) -> dict:
    """Product path: stamp the provenance override onto the DB so the dashboard shows CLAIM, the
    gate stops discarding these posts, and the signal sweep stops nominating them for hand-relabel.

    Overwrites the classifier's `has_claim` (recording why in `reason`) ONLY for Bluesky posts the
    override fires on that a human has NOT already judged (`admin_label IS NULL` — human disposes
    over everything). Idempotent. The eval is unaffected: it reads the frozen CSV and applies the
    same rule itself, so this DB stamp and the eval never diverge. Returns {overridden}."""
    credible = cfg.get("evidence", {}).get("citation_domains", [])
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    ensure_admin_columns(con)
    rows = con.execute("""
        SELECT p.post_id, p.text, p.external_title, p.external_url
        FROM posts p JOIN classifications c ON p.post_id = c.post_id
        WHERE p.source = 'bluesky' AND c.admin_label IS NULL AND c.has_claim = 0
              AND p.external_title IS NOT NULL AND p.external_title != ''
    """).fetchall()
    n = 0
    for r in rows:
        if provenance_override(r["text"], r["external_title"], r["external_url"] or "", credible):
            con.execute(
                "UPDATE classifications SET has_claim = 1, reason = ? WHERE post_id = ?",
                (f"provenance override: verbatim headline of the linked credible article", r["post_id"]))
            n += 1
    con.commit()
    con.close()
    return {"overridden": n}


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


def load_overrides(path: Path | str = ADMIN_OVERRIDES_PATH) -> dict[str, dict]:
    """Read the overrides log, newest-wins per post_id. Returns {post_id: {admin_label, ts, source}}.

    Last line wins because the log is append-only: changing your mind appends a new line rather
    than rewriting history, so git keeps the whole trail. Malformed lines are skipped, never
    raised — a corrupt line must not take the app down with it."""
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec.get("post_id") is not None and rec.get("admin_label") is not None:
                out[rec["post_id"]] = rec          # later line overwrites earlier = newest wins
        except json.JSONDecodeError:
            continue
    return out


def record_override(post_id: str, admin_label: int, source: str = "relabel",
                    path: Path | str = ADMIN_OVERRIDES_PATH, ts: str | None = None) -> dict:
    """Append one override to the git-tracked log. This is the write that MUST survive; the DB
    column is only a cache of it."""
    rec = {
        "post_id": post_id,
        "admin_label": int(admin_label),
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def apply_overrides(db_path: str, path: Path | str = ADMIN_OVERRIDES_PATH) -> dict:
    """Replay the overrides log onto the DB's cache column. Idempotent — run it after ANY import
    that replaces `ingested.db` (a Colab export, a backup restore), or the human judgments are
    gone and every reviewed post floods back into the nomination queue.

    Returns {applied, missing}: `missing` counts overrides whose post is no longer in the corpus,
    which is expected and harmless — the Bluesky refresher expires posts after 14 days, while the
    log keeps the decision forever."""
    overrides = load_overrides(path)
    if not overrides:
        return {"applied": 0, "missing": 0}
    con = sqlite3.connect(db_path)
    ensure_admin_columns(con)
    applied = missing = 0
    for pid, rec in overrides.items():
        cur = con.execute(
            "UPDATE classifications SET admin_label = ?, admin_labeled_at = ? WHERE post_id = ?",
            (int(rec["admin_label"]), rec.get("ts"), pid))
        if cur.rowcount:
            applied += 1
        else:
            missing += 1
    con.commit()
    con.close()
    return {"applied": applied, "missing": missing}


def set_admin_label(db_path: str, post_id: str, admin_label: int,
                    source: str = "relabel", overrides_path: Path | str = ADMIN_OVERRIDES_PATH) -> None:
    """Record an admin override (1 = claim, 0 = opinion).

    Writes the git-tracked LOG first, then the DB cache. Order matters: if the process dies between
    the two, the log still holds the judgment and `apply_overrides` restores the cache. The reverse
    order would lose it on the next round-trip, which is the bug this whole file exists to fix."""
    rec = record_override(post_id, admin_label, source=source, path=overrides_path)
    con = sqlite3.connect(db_path)
    ensure_admin_columns(con)
    con.execute("UPDATE classifications SET admin_label = ?, admin_labeled_at = ? WHERE post_id = ?",
                (int(admin_label), rec["ts"], post_id))
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


EVAL_FIELDS = ["post_text", "expected_label", "keyword_category", "post_type", "notes",
               "external_title", "external_url"]


def append_to_eval_csv(csv_path: str, post_text: str, expected_label: str, keyword_category: str = "",
                       post_type: str = "", notes: str = "",
                       external_title: str = "", external_url: str = "") -> None:
    """Append a corrected row to the eval benchmark. `external_title`/`external_url` are the linked
    article's provenance — frozen INTO the row so the eval's verbatim-headline override is
    reproducible and never has to re-read the live DB (a post aging out must not change the score)."""
    p = Path(csv_path)
    text = " ".join((post_text or "").split())          # flatten newlines for a clean CSV cell
    with open(p, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([text, expected_label, keyword_category,
                                (post_type or "real_data").strip(),
                                (notes or "admin-confirmed relabel").strip(),
                                " ".join((external_title or "").split()),
                                (external_url or "").strip()])


def apply_relabel(db_path: str, csv_path: str, post_id: str, post_text: str, corrected_label: str,
                  keyword_category: str = "", post_type: str = "", notes: str = "") -> None:
    """Confirm a relabel: BOTH writes — the admin override (users see it) and the eval-CSV append
    with the admin-chosen `post_type` (thought) + `notes`. The linked article's title/url are
    captured from the DB and frozen into the eval row so the provenance override stays reproducible."""
    set_admin_label(db_path, post_id, 1 if corrected_label == "claim" else 0)
    ext_title = ext_url = ""
    try:
        con = sqlite3.connect(db_path)
        row = con.execute("SELECT external_title, external_url FROM posts WHERE post_id = ?",
                          (post_id,)).fetchone()
        con.close()
        if row:
            ext_title, ext_url = row[0] or "", row[1] or ""
    except Exception:
        pass
    append_to_eval_csv(csv_path, post_text, corrected_label, keyword_category, post_type, notes,
                       external_title=ext_title, external_url=ext_url)


def mark_reviewed_ok(db_path: str, post_id: str, model_label: int) -> None:
    """Reject a nomination = the model was right. Stamp admin_label = model's label so it leaves
    the queue and the displayed label is unchanged. No eval-CSV append.

    Tagged `reviewed_ok` in the log so "I judged this and the model was right" stays distinguishable
    from "I corrected this" — it has no eval-CSV row to recover it from, so the log is its ONLY
    record."""
    set_admin_label(db_path, post_id, model_label, source="reviewed_ok")


def main():
    """Admin-override CLI. The cheap escape hatch for "I copied ingested.db back and my labels are
    gone": `maintenance` also replays the log, but only as part of classify/vision/reindex/evaluate,
    which is far too expensive to run just to restore a column."""
    import argparse
    import sys
    import yaml
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    db_default = cfg["storage"]["db_path"]

    parser = argparse.ArgumentParser(
        description="Admin label overrides — the git-tracked source of truth for human judgments.")
    parser.add_argument("--apply", action="store_true",
                        help="replay the overrides log onto the DB (run after ANY import that "
                             "replaces ingested.db)")
    parser.add_argument("--status", action="store_true",
                        help="compare the log against the DB and report any drift")
    parser.add_argument("--db", default=db_default, help=f"database path (default: {db_default})")
    parser.add_argument("--log", default=str(ADMIN_OVERRIDES_PATH),
                        help=f"overrides log (default: {ADMIN_OVERRIDES_PATH})")
    args = parser.parse_args()

    if args.apply:
        res = apply_overrides(args.db, args.log)
        print(f"Applied {res['applied']} admin override(s) to {args.db}."
              + (f" {res['missing']} post(s) are no longer in the corpus (expired — harmless)."
                 if res["missing"] else ""))
        return
    if args.status:
        log = load_overrides(args.log)
        con = sqlite3.connect(args.db)
        ensure_admin_columns(con)
        in_db = {r[0]: r[1] for r in con.execute(
            "SELECT post_id, admin_label FROM classifications WHERE admin_label IS NOT NULL")}
        con.close()
        # Drift in either direction is a bug worth naming: an override only in the DB is one the log
        # never captured (it will die on the next round-trip); a mismatch means the cache is stale.
        only_db = [p for p in in_db if p not in log]
        stale = [p for p, r in log.items() if p in in_db and in_db[p] != int(r["admin_label"])]
        print(f"log: {len(log)} override(s)  ·  db: {len(in_db)} override(s)")
        print(f"  in the DB but NOT in the log : {len(only_db)}"
              + ("  <- will be LOST on the next ingested.db replace" if only_db else ""))
        print(f"  label disagrees with the log : {len(stale)}"
              + ("  <- run --apply" if stale else ""))
        if not only_db and not stale:
            print("  in sync ✅")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
