"""
Sample real posts from ingested.db as candidates to extend claim_eval.csv.

Design:
  - Equal total per category: POSTS_PER_CATEGORY (default 20)
  - Equal claim/opinion split within every category (50/50)
  - Synthetic posts already in claim_eval.csv count toward the quota
  - Excludes posts already in claim_eval.csv (exact text match)
  - Saves candidates to data/claim_eval_candidates.csv for manual review

After you approve the candidates and append them to claim_eval.csv, run:
  uv run python scripts/flag_eval_posts.py
to mark those posts as in_eval_set=1 in the DB so they are excluded
from ChromaDB population (prevents data leakage in RAG evaluation).

Usage:
  uv run python scripts/sample_eval_candidates.py
"""

import csv
import random
import sqlite3
from pathlib import Path

RANDOM_SEED         = 42
POSTS_PER_CATEGORY  = 20        # target total per category (synthetic + real)
DB_PATH             = Path("data/ingested.db")
EXISTING_EVAL_CSV   = Path("data/claim_eval.csv")
OUTPUT_CSV          = Path("data/claim_eval_candidates.csv")

CATEGORIES = ["scientific", "extreme_events", "sensationalist", "conspiracy", "combinations"]
TARGET_PER_LABEL = POSTS_PER_CATEGORY // 2   # 10 claims + 10 opinions per category

# ── Load existing eval set ────────────────────────────────────────────────────
with open(EXISTING_EVAL_CSV, newline="", encoding="utf-8") as f:
    existing_rows = list(csv.DictReader(f))
existing_texts = {r["post_text"].strip() for r in existing_rows}

synthetic_counts: dict[tuple, int] = {}
for row in existing_rows:
    key = (row["keyword_category"], row["expected_label"])
    synthetic_counts[key] = synthetic_counts.get(key, 0) + 1

print(f"Target: {POSTS_PER_CATEGORY} per category  ({TARGET_PER_LABEL} claims + {TARGET_PER_LABEL} opinions)\n")
print(f"{'Category':<20} {'claim synth':>12} {'opinion synth':>14} {'claim needed':>13} {'opinion needed':>15}")
print("-" * 76)
for cat in CATEGORIES:
    c = synthetic_counts.get((cat, "claim"), 0)
    o = synthetic_counts.get((cat, "opinion"), 0)
    print(f"{cat:<20} {c:>12} {o:>14} {max(0, TARGET_PER_LABEL - c):>13} {max(0, TARGET_PER_LABEL - o):>15}")
print()

# ── Sample from DB ────────────────────────────────────────────────────────────
con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row
random.seed(RANDOM_SEED)

candidates = []
sampled_ids = []   # track post_ids for leakage-prevention flagging

for cat in CATEGORIES:
    for label, has_claim_val in [("claim", 1), ("opinion", 0)]:
        already_have = synthetic_counts.get((cat, label), 0)
        need = max(0, TARGET_PER_LABEL - already_have)
        if need == 0:
            continue

        rows = con.execute("""
            SELECT p.post_id, p.text, p.keyword_category, p.source, c.reason
            FROM posts p
            JOIN classifications c ON p.post_id = c.post_id
            WHERE p.keyword_category = ?
              AND c.has_claim = ?
        """, (cat, has_claim_val)).fetchall()

        # exclude already-in-eval, deduplicate by text
        pool = [r for r in rows if r["text"].strip() not in existing_texts]
        pool = list({r["text"]: r for r in pool}.values())
        random.shuffle(pool)
        sampled = pool[:need]

        if len(sampled) < need:
            print(f"  WARNING {cat}/{label}: needed {need}, only {len(sampled)} available")

        for row in sampled:
            sampled_ids.append(row["post_id"])
            candidates.append({
                "post_text":        row["text"],
                "expected_label":   label,
                "keyword_category": row["keyword_category"],
                "post_type":        "real_data",
                "source_post_id":   row["post_id"],   # kept separate — used by flag_eval_posts.py
                "notes":            f"source={row['source']} | {row['reason'][:100]}",
            })

con.close()

# ── Write candidates CSV ──────────────────────────────────────────────────────
fieldnames = ["post_text", "expected_label", "keyword_category", "post_type", "source_post_id", "notes"]
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(candidates)

# Summary table
print(f"\nSampled {len(candidates)} real posts -> {OUTPUT_CSV}\n")
from collections import Counter
counts = Counter((r["keyword_category"], r["expected_label"]) for r in candidates)
print(f"{'Category':<20} {'claim real':>11} {'opinion real':>13}")
print("-" * 46)
for cat in CATEGORIES:
    c = counts.get((cat, "claim"), 0)
    o = counts.get((cat, "opinion"), 0)
    print(f"{cat:<20} {c:>11} {o:>13}")

print(f"""
Next steps:
  1. Review {OUTPUT_CSV}
     Correct any wrong labels (check the notes column).
     Keep 'source_post_id' — flag_eval_posts.py uses it to mark posts in the DB.
  2. Append approved rows to {EXISTING_EVAL_CSV}
  3. Run:  uv run python scripts/flag_eval_posts.py
     This marks the sampled posts in the DB (prevents data leakage in RAG).
""")
