"""
Export unclassified posts to CSV for GPU classification in Google Colab.

Run before uploading to Colab:
  uv run python scripts/export_for_colab.py
"""

import sqlite3
import csv
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "src" / "climate_verifier" / "config.yaml"
OUT_DIR = Path(__file__).parent.parent / "tmp" / "colab"
OUT_PATH = OUT_DIR / "unclassified_posts.csv"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

db_path = cfg["storage"]["db_path"]

conn = sqlite3.connect(db_path)
rows = conn.execute("""
    SELECT p.post_id, p.text
    FROM posts p
    LEFT JOIN classifications c ON p.post_id = c.post_id
    WHERE c.post_id IS NULL
""").fetchall()
conn.close()

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["post_id", "text"])
    w.writerows(rows)

print(f"Exported {len(rows)} unclassified posts → {OUT_PATH}")
print("Upload tmp/colab/unclassified_posts.csv to Google Colab.")
