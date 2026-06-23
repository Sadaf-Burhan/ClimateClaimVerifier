"""
Week 2 — Tokenisation and Embeddings

Embeds social media posts using sentence-transformers (all-MiniLM-L6-v2).

Model choice: switched from course baseline nomic-embed-text (Ollama) to
all-MiniLM-L6-v2 (sentence-transformers). Reason: higher MTEB scores on Semantic
Textual Similarity tasks — the primary use case here is measuring whether two
climate posts describe the same event. Also runs fully offline without a running
Ollama server, so the ingestion scheduler and dashboard do not need two services.
"""

import csv
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as _cosine

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return unit-normalised embedding matrix (n × 384) for a list of texts."""
    return _get_model().encode(texts, normalize_embeddings=True)


def similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity in [-1, 1] between two texts."""
    vecs = embed([text_a, text_b])
    return float(_cosine(vecs[0:1], vecs[1:2])[0][0])


def eval_pairs(csv_path: str) -> dict:
    """
    Runs embedding quality evaluation against labeled pairs.

    CSV must have columns: text_a, text_b, should_be_similar, pair_type.

    Returns:
        {
            "pairs": list of dicts with keys text_a, text_b, should_be_similar,
                     pair_type, score,
            "similar_mean": float,
            "dissimilar_mean": float,
            "separation": float,   # similar_mean - dissimilar_mean; higher is better
            "n_similar": int,
            "n_dissimilar": int,
        }
    """
    pairs = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            score = similarity(row["text_a"], row["text_b"])
            pairs.append({
                "text_a":            row["text_a"],
                "text_b":            row["text_b"],
                "should_be_similar": row["should_be_similar"].strip().lower() == "true",
                "pair_type":         row["pair_type"],
                "score":             round(score, 4),
            })

    similar    = [p["score"] for p in pairs if     p["should_be_similar"]]
    dissimilar = [p["score"] for p in pairs if not p["should_be_similar"]]

    return {
        "pairs":           pairs,
        "similar_mean":    round(float(np.mean(similar)),    4) if similar    else 0.0,
        "dissimilar_mean": round(float(np.mean(dissimilar)), 4) if dissimilar else 0.0,
        "separation":      round(
            (float(np.mean(similar)) if similar else 0.0)
            - (float(np.mean(dissimilar)) if dissimilar else 0.0),
            4,
        ),
        "n_similar":    len(similar),
        "n_dissimilar": len(dissimilar),
    }


def category_similarity_stats(db_path: str, limit_per_category: int = 20) -> dict:
    """
    Computes mean intra-category cosine similarity for each keyword_category.

    Fetches up to `limit_per_category` posts per category from the database,
    embeds them, and computes the mean pairwise similarity (upper triangle).

    Returns {category: mean_similarity}. Categories with < 2 posts are skipped.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    categories = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT keyword_category FROM posts WHERE keyword_category IS NOT NULL"
        ).fetchall()
    ]

    results = {}
    for cat in sorted(categories):
        rows = conn.execute(
            """SELECT text FROM posts
               WHERE keyword_category = ?
                 AND text IS NOT NULL
                 AND text != ''
               LIMIT ?""",
            (cat, limit_per_category),
        ).fetchall()
        texts = [r["text"] for r in rows]

        if len(texts) < 2:
            continue

        vecs = embed(texts)
        sim_matrix = _cosine(vecs)
        n = len(texts)
        upper_tri = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        results[cat] = round(float(np.mean(upper_tri)), 4)

    conn.close()
    return results
