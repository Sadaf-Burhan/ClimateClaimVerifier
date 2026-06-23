"""
Import classification results from Google Colab back into the local database.

After downloading from Colab, place the file at tmp/colab/classifications_result.csv
then run:
  uv run python scripts/import_from_colab.py
"""

import sqlite3
import csv
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "src" / "climate_verifier" / "config.yaml"
SRC_PATH = Path(__file__).parent.parent / "tmp" / "colab" / "classifications_result.csv"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

db_path = cfg["storage"]["db_path"]

if not SRC_PATH.exists():
    print(f"Error: {SRC_PATH} not found.")
    print("Download classifications_result.csv from Colab and place it in tmp/colab/")
    raise SystemExit(1)

conn = sqlite3.connect(db_path)
conn.execute("""
    CREATE TABLE IF NOT EXISTS classifications (
        post_id       TEXT PRIMARY KEY,
        has_claim     INTEGER,
        reason        TEXT,
        classified_at TEXT
    )
""")

imported = skipped = 0
with open(SRC_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cur = conn.execute("""
            INSERT OR IGNORE INTO classifications (post_id, has_claim, reason, classified_at)
            VALUES (?, ?, ?, ?)
        """, (row["post_id"], int(row["has_claim"]), row["reason"], row["classified_at"]))
        if cur.rowcount:
            imported += 1
        else:
            skipped += 1

conn.commit()
conn.close()

print(f"Done — {imported} imported, {skipped} already existed (skipped).")
print(f"Database: {db_path}")
