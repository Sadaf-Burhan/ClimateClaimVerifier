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
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from climate_verifier.pipeline.evidence import assess_claim


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
        SELECT p.post_id, p.text, p.author, p.author_followers, p.external_url, p.keyword_category,
               c.has_claim, c.admin_label,
               (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p JOIN classifications c ON p.post_id = c.post_id
        WHERE p.source = 'bluesky'
        ORDER BY engagement DESC LIMIT ?
    """, (scan_limit,)).fetchall()
    con.close()

    to_claim, to_opinion = [], []
    for r in rows:
        if r["admin_label"] is not None:          # already reviewed by the admin
            continue
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
        if s.get("credible_cite"):
            why.append("credible cited source")
        cand = {"post_id": r["post_id"], "text": r["text"], "author": r["author"] or "",
                "engagement": r["engagement"], "keyword_category": r["keyword_category"] or "",
                "external_url": r["external_url"] or "", "has_claim": r["has_claim"],
                "news_status": s["news_status"], "why": ", ".join(why)}
        if r["has_claim"] == 0 and why:                       # OPINION but evidence contradicts
            to_claim.append(cand)
        elif r["has_claim"] == 1 and not why and s["news_status"] == "NO MATCH":  # CLAIM, no support
            to_opinion.append(cand)
    return {"opinion_to_claim": to_claim, "claim_to_opinion": to_opinion}


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
