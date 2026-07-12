"""
GDELT ingestion — fetches news articles for each keyword.
Free, no auth required. Respects rate limits with delay + retry.
"""

import time
import requests
from datetime import datetime, timezone

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_articles(keyword: str, days_back: int, delay: float, retries: int,
                   timeout: int = 30) -> list[dict]:
    """
    Query GDELT for recent articles matching keyword.
    Returns a list of normalised dicts ready for storage.

    `timeout` is the per-request HTTP timeout (seconds). Broad single-word queries can hang
    GDELT; callers doing many queries (the demand-driven top-up) pass a shorter timeout so a
    slow query fails fast and the run moves on instead of blocking for timeout*retries.
    """
    params = {
        "query":      keyword,
        "mode":       "artlist",
        "maxrecords": 250,
        "timespan":   f"{days_back}d",
        "format":     "json",
    }

    data = {}
    for attempt in range(retries):
        try:
            resp = requests.get(GDELT_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = (attempt + 1) * 20
                print(f"  GDELT rate limited — waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            print(f"  GDELT error on attempt {attempt+1}: {e}")
            time.sleep(delay)
            data = {}

    articles = []
    for art in data.get("articles", []):
        articles.append({
            "source":           "gdelt",
            "keyword":          keyword,
            "post_id":          art.get("url", ""),
            "author":           art.get("domain", ""),
            "author_followers": 0,   # not applicable for news articles
            "text":             art.get("title", ""),
            "created_at":       art.get("seendate", ""),
            "likes":            0,
            "reposts":          0,
            "replies":          0,
            "quotes":           0,
            "ingested_at":      datetime.now(timezone.utc).isoformat(),
        })

    time.sleep(delay)  # polite delay after every successful call
    return articles
