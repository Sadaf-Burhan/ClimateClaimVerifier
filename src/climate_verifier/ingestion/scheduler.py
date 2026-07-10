"""
Scheduler — runs one full ingestion cycle every N hours.
Guards against redundant ingestion: if less than the configured interval
has passed since the last run, the cycle is skipped entirely.
The Streamlit classifier always reads from the existing database and is
completely independent — it can be run as many times as needed.

Run directly:  uv run python -m climate_verifier.ingestion.scheduler
"""

import yaml
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone, timedelta

from climate_verifier.ingestion.bluesky import fetch_posts
from climate_verifier.ingestion.gdelt import fetch_articles
from climate_verifier.ingestion.store import save, hours_since_last_ingestion, set_last_ingestion_time
from climate_verifier.pipeline.topic_filter import filter_posts

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def flatten_keywords(keywords: dict) -> list[tuple[str, str]]:
    """
    Flattens the categorised keyword dict into a list of (keyword, category) tuples.
    e.g. {"scientific": ["climate change"], "conspiracy": ["climate hoax"]}
         → [("climate change", "scientific"), ("climate hoax", "conspiracy")]
    The category travels with the keyword so it gets stored in the database
    and can later be used as a signal in the credibility scoring stage.
    """
    flat = []
    for category, kw_list in keywords.items():
        for kw in kw_list:
            flat.append((kw, category))
    return flat


def run_ingestion_cycle(sources=None, force=False):
    """Run one ingestion cycle.

    sources : which data sources to fetch — subset of ["bluesky", "gdelt"]; default both.
              Split across machines when needed: GDELT's API throttles/times-out on shared
              IPs (e.g. Colab), so run Bluesky on Colab and GDELT locally, both merging into
              the same DB.
    force   : bypass the 24h interval guard (for manual/partial-source runs).
    """
    cfg           = load_config()
    sources       = sources or ["bluesky", "gdelt"]
    db_path       = cfg["storage"]["db_path"]
    interval_hrs  = cfg["ingestion"]["interval_hours"]
    bsky_lim      = cfg["ingestion"]["bluesky_limit"]
    gdelt_days    = cfg["ingestion"]["gdelt_days_back"]
    gdelt_del     = cfg["ingestion"]["gdelt_delay_seconds"]
    gdelt_ret     = cfg["ingestion"]["gdelt_retries"]

    # ── Guard: skip if not enough time has passed ─────────────────────────────
    elapsed = hours_since_last_ingestion(db_path)
    if not force and elapsed < interval_hrs:
        remaining = interval_hrs - elapsed
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"Skipping ingestion — only {elapsed:.1f}h since last run "
            f"(next run in ~{remaining:.1f}h)"
        )
        return

    # ── Run the cycle ─────────────────────────────────────────────────────────
    keywords = flatten_keywords(cfg["ingestion"]["keywords"])
    since = (datetime.now(timezone.utc) - timedelta(days=gdelt_days)).isoformat()
    print(
        f"\n[{datetime.now(timezone.utc).isoformat()}] Starting ingestion cycle "
        f"— {len(keywords)} keywords across {len(cfg['ingestion']['keywords'])} categories"
        f"\n  Sources: {', '.join(sources)} — fetching last {gdelt_days}d (since {since})"
    )
    total         = 0
    total_dropped = 0

    for kw, category in keywords:
        if "bluesky" in sources:
            print(f"  [{category}] Bluesky → '{kw}'")
            posts = fetch_posts(kw, limit=bsky_lim, since=since)
            for p in posts:
                p["keyword_category"] = category
            posts, dropped = filter_posts(posts)
            total_dropped += dropped
            total += save(posts, db_path)

        if "gdelt" in sources:
            print(f"  [{category}] GDELT   → '{kw}'")
            articles = fetch_articles(kw, days_back=gdelt_days,
                                      delay=gdelt_del, retries=gdelt_ret)
            for a in articles:
                a["keyword_category"] = category
            articles, dropped = filter_posts(articles)
            total_dropped += dropped
            total += save(articles, db_path)

    # ── Record completion time ────────────────────────────────────────────────
    set_last_ingestion_time(db_path)
    print(
        f"  Done — {total} new records saved, "
        f"{total_dropped} irrelevant posts dropped "
        f"out of {total + total_dropped} total fetched."
    )


if __name__ == "__main__":
    cfg      = load_config()
    interval = cfg["ingestion"]["interval_hours"]

    # On startup, attempt a cycle — the guard inside will skip it if too soon
    run_ingestion_cycle()

    # Schedule recurring runs at the configured interval
    scheduler = BlockingScheduler()
    scheduler.add_job(run_ingestion_cycle, "interval", hours=interval)
    print(f"\nScheduler running — checking every {interval} hours. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("Scheduler stopped.")
