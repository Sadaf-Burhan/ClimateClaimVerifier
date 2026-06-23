"""
After approving claim_eval_candidates.csv and appending to claim_eval.csv,
run this script to mark those posts as in_eval_set=1 in the DB.

Posts with in_eval_set=1 are excluded from ChromaDB population so they
cannot be retrieved as few-shot examples during RAG evaluation — preventing
data leakage.

Usage:
  uv run python scripts/flag_eval_posts.py
"""

import csv
import sqlite3
from pathlib import Path

DB_PATH       = Path("data/ingested.db")
EVAL_CSV      = Path("data/claim_eval.csv")

con = sqlite3.connect(DB_PATH)

# Add column if it doesn't exist yet
con.execute("""
    ALTER TABLE posts ADD COLUMN in_eval_set INTEGER NOT NULL DEFAULT 0
""") if "in_eval_set" not in [
    r[1] for r in con.execute("PRAGMA table_info(posts)").fetchall()
] else None

# Match eval posts to DB.
# Primary: source_post_id column (set by sample_eval_candidates.py).
# Fallback: exact text match (for rows appended before the column existed).
with open(EVAL_CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

real_rows = [r for r in rows if r.get("post_type") == "real_data"]

flagged = 0
for r in real_rows:
    post_id = r.get("source_post_id", "").strip()
    if post_id:
        cur = con.execute("UPDATE posts SET in_eval_set=1 WHERE post_id=?", (post_id,))
    else:
        cur = con.execute("UPDATE posts SET in_eval_set=1 WHERE text=?", (r["post_text"].strip(),))
    flagged += cur.rowcount

con.commit()
con.close()
print(f"Flagged {flagged} posts as in_eval_set=1 in {DB_PATH}")
print("These will be excluded from ChromaDB population queries.")
