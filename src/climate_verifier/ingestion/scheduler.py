"""
Scheduler — runs one full ingestion cycle every N hours.
Guards against redundant ingestion: if less than the configured interval
has passed since the last run, the cycle is skipped entirely.
The Streamlit classifier always reads from the existing database and is
completely independent — it can be run as many times as needed.

Run directly:  uv run python -m climate_verifier.ingestion.scheduler
"""

import re
import sqlite3
import yaml
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone, timedelta

from climate_verifier.ingestion.bluesky import fetch_posts
from climate_verifier.ingestion.gdelt import fetch_articles
from climate_verifier.ingestion.store import save, hours_since_last_ingestion, set_last_ingestion_time
from climate_verifier.pipeline.topic_filter import filter_posts
from climate_verifier.pipeline.geo import extract_location

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# ── Demand-driven evidence top-up ─────────────────────────────────────────────
# The fixed keyword list ingested climate *jargon* ("methane bomb"), not the actual
# events people post about — so a real "B.C. wildfires" claim had zero news to match.
# This lets the CLAIMS themselves drive GDELT: bucket each claim to a bounded
# (region, subject) topic key, dedup, and fetch news FOR THAT topic. Bounded key =
# a small, well-defined query set, so the >=2 / <=100 floor-and-cap counts mean something.

# Subject buckets: first matching variant -> the canonical GDELT subject term. Ordered so a
# specific subject (wildfire) is chosen before a co-occurring generic one (heat).
_SUBJECTS = [
    ("wildfire",          ["wildfire", "forest fire", "bushfire", "fire season", "wildfires"]),
    ("flood",             ["flood", "flooding", "flash flood", "deluge"]),
    ("tornado",           ["tornado", "twister", "funnel cloud"]),
    ("hurricane",         ["hurricane", "typhoon", "tropical storm", "cyclone"]),
    ("atmospheric river", ["atmospheric river"]),
    ("heat wave",         ["heat dome", "heatwave", "heat wave", "record heat", "extreme heat"]),
    ("drought",           ["drought", "megadrought"]),
    ("winter storm",      ["blizzard", "snowstorm", "winter storm", "ice storm"]),
    ("smoke air quality", ["smoke", "air quality", "haze"]),
    ("sea level rise",    ["sea level", "ice sheet", "glacier", "permafrost"]),
]


def _subject_term(text: str) -> str:
    """The dominant weather/climate subject in a claim -> a canonical GDELT query term."""
    low = (text or "").lower()
    for canon, variants in _SUBJECTS:
        if any(v in low for v in variants):
            return canon
    return ""


def build_topic_query(text: str, fallback_keyword: str = "") -> tuple[str, str]:
    """Bucket a claim to a bounded (query, topic_key) for demand-driven GDELT.

    query     — what we ask GDELT: "<region> <subject>" (e.g. "British Columbia wildfire"),
                so results are already topic- AND region-relevant (no post-hoc filtering).
    topic_key — the bounded bucket the floor/cap counts against: "region|subject".
    Region comes from geo.extract_location (the specific place, not the outlet); subject from
    the text, else the ingestion keyword that brought the post in. Returns ("","") if neither."""
    loc = extract_location(text)
    region = loc.split(",")[0].strip() if loc else ""      # "British Columbia, Canada" -> "British Columbia"
    subject = _subject_term(text) or (fallback_keyword or "").strip()
    if not subject and not region:
        return "", ""
    query = " ".join(x for x in (region, subject) if x).strip()
    topic_key = f"{region.lower()}|{subject.lower()}"
    return query, topic_key


def _article_age_days(created_at: str, now: datetime) -> float | None:
    """Age of an article in days from its stored date. Handles GDELT compact
    'YYYYMMDDTHHMMSSZ' and ISO timestamps. Returns None if unparseable (treated as recent —
    we don't drop an article just because its date is missing)."""
    s = (created_at or "").strip()
    try:
        if len(s) >= 8 and s[:8].isdigit():                 # GDELT compact 20260710T...
            d = datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), tzinfo=timezone.utc)
        else:                                               # ISO 2026-07-10T...
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
        return (now - d).total_seconds() / 86400
    except Exception:
        return None


def _is_recent(created_at: str, max_age_days: int, now: datetime) -> bool:
    """DATE RELEVANCY guard — keep only articles at most `max_age_days` old (unknown age = keep)."""
    age = _article_age_days(created_at, now)
    return age is None or age <= max_age_days


def topup_evidence_for_claims(db_path: str, days_back: int = 7, delay: float = 4.0,
                              retries: int = 2, min_recent: int = 2,
                              max_age_days: int = 45, max_topics: int = 60) -> int:
    """Read the stored CLAIMS, bucket each to a bounded (region, subject) topic, and top up
    GDELT evidence until the topic has `min_recent` RECENT articles.

    No cap on articles-per-topic — recency and URL-dedup are the bounds, not a fixed number.
    Every fetched article passes a `max_age_days` DATE-RELEVANCY guard, so stale news can never
    enter the corpus. The floor is recency-based (counts only articles within `max_age_days`),
    so a topic self-refreshes as its coverage ages out. Existing articles are never deleted.
    Dedups topics across claims; per-run `max_topics` cap is logged, not silent."""
    now = datetime.now(timezone.utc)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT p.text, p.keyword FROM posts p
        JOIN classifications c ON p.post_id = c.post_id
        WHERE c.has_claim = 1 AND p.source = 'bluesky'
    """).fetchall()

    # Dedup claims -> bounded topics; remember one representative query per topic_key.
    topics: dict[str, str] = {}
    for r in rows:
        query, key = build_topic_query(r["text"], r["keyword"] or "")
        if key and key not in topics:
            topics[key] = query

    # How many RECENT (<= max_age_days) articles does a topic already have?
    def recent_count(query: str) -> int:
        rs = con.execute("SELECT created_at FROM posts WHERE source='gdelt' AND keyword = ?",
                         (query,)).fetchall()
        return sum(1 for x in rs if _is_recent(x["created_at"], max_age_days, now))

    pending = [(k, q) for k, q in topics.items() if recent_count(q) < min_recent]
    print(f"  Demand-driven top-up: {len(topics)} distinct topics from {len(rows)} claims; "
          f"{len(pending)} below the {min_recent}-recent-article floor "
          f"(recency guard: <= {max_age_days}d).")
    if len(pending) > max_topics:
        print(f"  Capping this run at {max_topics} topics (skipping {len(pending)-max_topics} — "
              f"re-run to continue). Skipped: {[q for _, q in pending[max_topics:]][:5]}...")
        pending = pending[:max_topics]

    inserted = 0
    for key, query in pending:
        # short per-request timeout: a broad query that hangs fails fast and the run continues
        arts = fetch_articles(query, days_back=days_back, delay=delay, retries=retries, timeout=15)
        fresh = [a for a in arts if _is_recent(a.get("created_at", ""), max_age_days, now)]
        dropped_old = len(arts) - len(fresh)
        for a in fresh:                            # tag by topic so the floor counts are stable
            a["keyword"] = query
            a["keyword_category"] = key.split("|")[-1] or "extreme_events"
        # No filter_posts() here — the query is region+subject-targeted, and the topic filter
        # drops relevant articles (e.g. "Smoke turns sky ... orange" has no listed weather term).
        # The targeted query IS the relevance guarantee; recency is the only extra gate.
        n = save(fresh, db_path)
        inserted += n
        stale = f", {dropped_old} too old" if dropped_old else ""
        print(f"    [{query}] fetched {len(arts)}{stale}, +{n} new")
    con.close()
    print(f"  Top-up done — {inserted} new GDELT articles across {len(pending)} topics.")
    return inserted


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

    # ── Demand-driven evidence top-up ─────────────────────────────────────────
    # Let the accumulated CLAIMS' topics+regions pull targeted GDELT news, so the corpus
    # covers what people actually post about (not just the fixed keyword list). Cheap when
    # coverage is already fresh — only topics below the recency floor are re-queried. Then
    # rebuild the evidence index so the new articles are retrievable.
    ing = cfg["ingestion"]
    if "gdelt" in sources and ing.get("topup_enabled", True):
        try:
            added = topup_evidence_for_claims(
                db_path, days_back=ing.get("topup_days_back", 7),
                delay=gdelt_del / 2, retries=gdelt_ret,
                min_recent=ing.get("topup_min_recent", 2),
                max_age_days=ing.get("max_article_age_days", 45),
                max_topics=ing.get("topup_max_topics", 60))
            if added:
                from climate_verifier.pipeline.evidence import get_store  # lazy: heavy import
                n = get_store().build_index(db_path)
                print(f"  Evidence index rebuilt — {n} GDELT articles now retrievable.")
        except Exception as e:
            print(f"  Evidence top-up skipped: {e}")

    # ── Record completion time ────────────────────────────────────────────────
    set_last_ingestion_time(db_path)
    print(
        f"  Done — {total} new records saved, "
        f"{total_dropped} irrelevant posts dropped "
        f"out of {total + total_dropped} total fetched."
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingestion — cycle, demand-driven top-up, or scheduler.")
    parser.add_argument("--topup", action="store_true",
                        help="demand-driven GDELT top-up: let the stored claims' topic+location "
                             "drive what news we fetch (>=2 / <=100 per topic, never expired)")
    parser.add_argument("--days-back", type=int, default=None, help="override GDELT window for --topup")
    args = parser.parse_args()

    cfg      = load_config()
    ing      = cfg["ingestion"]

    if args.topup:
        topup_evidence_for_claims(
            cfg["storage"]["db_path"],
            days_back=args.days_back or ing.get("topup_days_back", 7),
            delay=ing["gdelt_delay_seconds"] / 2,
            retries=ing["gdelt_retries"],
            min_recent=ing.get("topup_min_recent", 2),
            max_age_days=ing.get("max_article_age_days", 45),
            max_topics=ing.get("topup_max_topics", 60),
        )
        raise SystemExit

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
