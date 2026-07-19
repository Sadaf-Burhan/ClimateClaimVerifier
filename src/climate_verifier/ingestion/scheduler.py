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

from climate_verifier.ingestion.bluesky import fetch_posts, check_posts_exist
from climate_verifier.ingestion.gdelt import fetch_articles
from climate_verifier.ingestion.store import (save, hours_since_last_ingestion, set_last_ingestion_time,
                                              old_bluesky_post_ids, oldest_bluesky_post_ids, delete_posts)
from climate_verifier.pipeline.topic_filter import filter_posts
from climate_verifier.pipeline.geo import extract_location, with_location
from climate_verifier.health import update_health

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


def valid_evidence_topics(db_path: str, cfg: dict) -> set[str]:
    """The topic-query set the GDELT corpus is allowed to hold: every config keyword, plus the
    (region, subject) query each stored CLAIM buckets to. A GDELT article whose `keyword` is
    outside this set is an orphan — the claim/keyword that pulled it is gone, so it's dead weight."""
    valid = {kw for kw, _ in flatten_keywords(cfg["ingestion"]["keywords"])}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT p.text, p.keyword FROM posts p "
            "JOIN classifications c ON p.post_id = c.post_id "
            "WHERE c.has_claim = 1 AND p.source = 'bluesky'"
        ).fetchall()
    finally:
        con.close()
    for r in rows:
        q, _ = build_topic_query(r["text"], r["keyword"] or "")
        if q:
            valid.add(q)
    return valid


def prune_gdelt_evidence(db_path: str, keep: int = 2, max_age_days: int = 45,
                         valid_topics: set[str] | None = None, dry_run: bool = False) -> dict:
    """Cap the GDELT evidence corpus so it stays small and relevant (scalability guard).

    Three deletions, in order — never touches Bluesky posts or held-out eval/train rows:
      1. STALE  — articles older than `max_age_days` (same recency rule the top-up uses).
      2. ORPHAN — articles whose topic (`keyword`) isn't in `valid_topics` (a live claim topic
                  or a config keyword). Skipped when `valid_topics` is None.
      3. CAP    — per remaining topic, keep only the `keep` articles most RELEVANT to the topic,
                  scored by cosine of each article's region-aware embedding to the topic query
                  (the same `with_location` form retrieval uses); delete the rest.

    Returns {before, stale, orphan, capped, kept, after}. `dry_run` counts without deleting."""
    now = datetime.now(timezone.utc)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cols = [r[1] for r in con.execute("PRAGMA table_info(posts)").fetchall()]
    guard = ""
    if "in_eval_set" in cols:
        guard += " AND (in_eval_set IS NULL OR in_eval_set = 0)"
    if "in_train_set" in cols:
        guard += " AND (in_train_set IS NULL OR in_train_set = 0)"
    rows = con.execute(
        f"SELECT post_id, text, author, keyword, created_at FROM posts "
        f"WHERE source = 'gdelt'{guard}"
    ).fetchall()
    con.close()
    before = len(rows)

    # 1) STALE
    stale = [r["post_id"] for r in rows if not _is_recent(r["created_at"], max_age_days, now)]
    dropped = set(stale)
    live = [r for r in rows if r["post_id"] not in dropped]

    # 2) ORPHAN
    orphan = []
    if valid_topics is not None:
        orphan = [r["post_id"] for r in live if (r["keyword"] or "") not in valid_topics]
        dropped |= set(orphan)
        live = [r for r in live if r["post_id"] not in dropped]

    # 3) CAP per topic — keep the `keep` most relevant to the topic query
    from collections import defaultdict
    by_topic: dict[str, list] = defaultdict(list)
    for r in live:
        by_topic[r["keyword"] or ""].append(r)

    capped, kept = [], 0
    fat = {t: a for t, a in by_topic.items() if len(a) > keep}
    kept += sum(len(a) for t, a in by_topic.items() if len(a) <= keep)
    if fat:
        import numpy as np
        from climate_verifier.pipeline.embedder import embed  # lazy: loads the model
        for topic, arts in fat.items():
            docs = [with_location(a["text"] or "", extract_location(a["text"] or "", a["author"] or ""))
                    for a in arts]
            qvec = embed([with_location(topic, extract_location(topic))])[0]
            scores = embed(docs) @ qvec                      # unit-normalised -> dot == cosine
            keep_idx = set(int(i) for i in np.argsort(scores)[::-1][:keep])
            for i, a in enumerate(arts):
                if i in keep_idx:
                    kept += 1
                else:
                    capped.append(a["post_id"])

    to_delete = stale + orphan + capped
    if not dry_run and to_delete:
        delete_posts(db_path, to_delete)
    return {"before": before, "stale": len(stale), "orphan": len(orphan),
            "capped": len(capped), "kept": kept, "after": before - len(to_delete)}


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


def refresh_corpus(db_path: str, cfg: dict, dry_run: bool = False) -> dict:
    """End-of-cycle refresher. Keeps the Bluesky corpus fresh and free of dead links:

      1. AGE EXPIRY — remove Bluesky posts older than `retention_days` (live OR deleted). A user
         can always paste an old post's link to assess it live, so a short window is safe.
      2. AVAILABILITY SWEEP — of the remaining (bounded by `verify_batch`, oldest first),
         remove any that no longer exist on Bluesky (deleted by their author).

    Never touches GDELT (the evidence corpus) or held-out eval/train posts. `dry_run` reports
    counts without deleting. Writes a `refresh` health heartbeat."""
    st = cfg.get("storage", {})
    removed_old = removed_gone = 0

    retention = st.get("retention_days", 0)
    if retention and retention > 0:
        old_ids = old_bluesky_post_ids(db_path, retention)
        removed_old = len(old_ids)
        verb = "would expire" if dry_run else "expiring"
        print(f"  Refresh: {verb} {removed_old} Bluesky posts older than {retention}d")
        if old_ids and not dry_run:
            delete_posts(db_path, old_ids)

    if st.get("verify_availability", True):
        uris = oldest_bluesky_post_ids(db_path, st.get("verify_batch", 500))
        if uris:
            existing = check_posts_exist(uris)          # fail-safe: keeps a batch on API error
            gone = [u for u in uris if u not in existing]
            verb = "would remove" if dry_run else "removing"
            print(f"  Refresh: availability-checked {len(uris)}; {verb} {len(gone)} no longer on Bluesky")
            if gone and not dry_run:
                removed_gone = delete_posts(db_path, gone)

    if not dry_run:
        update_health("refresh", ok=True, removed_old=removed_old, removed_gone=removed_gone)
    return {"removed_old": removed_old, "removed_gone": removed_gone}


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

    try:
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
    except Exception as e:
        # record the glitch so the app's health banner turns red, then re-raise so the
        # OS scheduler / CI sees a non-zero exit and fires its own failure alert
        update_health("ingestion", ok=False, sources=sources, error=f"{type(e).__name__}: {e}",
                      saved_before_error=total)
        raise

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
            # Cap the corpus so GDELT can't grow without bound: keep only the most-relevant
            # articles per topic, and drop stale/orphaned ones. Runs even when the top-up added
            # nothing, so accumulated bloat still gets swept.
            pruned = 0
            if ing.get("evidence_cap_enabled", True):
                r = prune_gdelt_evidence(
                    db_path, keep=ing.get("evidence_max_per_topic", 2),
                    max_age_days=ing.get("max_article_age_days", 45),
                    valid_topics=valid_evidence_topics(db_path, cfg))
                pruned = r["stale"] + r["orphan"] + r["capped"]
                print(f"  Evidence GC — kept {r['after']} articles "
                      f"(dropped {r['stale']} stale, {r['orphan']} orphan, {r['capped']} over-cap).")
            if added or pruned:
                from climate_verifier.pipeline.evidence import get_store  # lazy: heavy import
                n = get_store().build_index(db_path)
                print(f"  Evidence index rebuilt — {n} GDELT articles now retrievable.")
        except Exception as e:
            print(f"  Evidence top-up / GC skipped: {e}")

    # ── Corpus refresher: expire old posts + drop ones deleted on Bluesky ─────
    if cfg.get("storage", {}).get("refresh_enabled", True):
        try:
            refresh_corpus(db_path, cfg)
        except Exception as e:
            print(f"  Corpus refresh skipped: {e}")

    # ── Record completion time + health heartbeat ─────────────────────────────
    set_last_ingestion_time(db_path)
    update_health("ingestion", ok=True, sources=sources,
                  counts={"saved": total, "dropped": total_dropped})
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
    parser.add_argument("--once", action="store_true",
                        help="run ONE ingestion cycle and exit (for an OS scheduler / cron); "
                             "exits non-zero on failure so the scheduler's alert fires")
    parser.add_argument("--force", action="store_true", help="bypass the 24h interval guard")
    parser.add_argument("--refresh", action="store_true",
                        help="run ONLY the corpus refresher (expire old + remove deleted Bluesky posts)")
    parser.add_argument("--prune-evidence", action="store_true",
                        help="run ONLY the GDELT evidence GC: keep the N most-relevant articles per "
                             "topic (config: evidence_max_per_topic), drop stale + orphaned, rebuild index")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --refresh or --prune-evidence: report what WOULD be removed, delete nothing")
    args = parser.parse_args()

    cfg      = load_config()
    ing      = cfg["ingestion"]

    if args.refresh:
        r = refresh_corpus(cfg["storage"]["db_path"], cfg, dry_run=args.dry_run)
        print(f"Refresh {'(dry-run) ' if args.dry_run else ''}done — "
              f"expired {r['removed_old']}, removed-deleted {r['removed_gone']}.")
        raise SystemExit

    if args.prune_evidence:
        dbp = cfg["storage"]["db_path"]
        r = prune_gdelt_evidence(
            dbp, keep=ing.get("evidence_max_per_topic", 2),
            max_age_days=ing.get("max_article_age_days", 45),
            valid_topics=valid_evidence_topics(dbp, cfg), dry_run=args.dry_run)
        print(f"Evidence GC {'(dry-run) ' if args.dry_run else ''}— "
              f"{r['before']} → {r['after']} articles "
              f"(stale {r['stale']}, orphan {r['orphan']}, over-cap {r['capped']}).")
        if not args.dry_run and (r["before"] != r["after"]):
            from climate_verifier.pipeline.evidence import get_store
            n = get_store().build_index(dbp)
            print(f"Evidence index rebuilt — {n} GDELT articles now retrievable.")
        raise SystemExit

    if args.once:
        import sys
        try:
            run_ingestion_cycle(force=args.force)
        except Exception as e:
            print(f"Ingestion cycle FAILED: {type(e).__name__}: {e}")
            sys.exit(1)          # non-zero → OS scheduler / CI raises its own failure alert
        raise SystemExit(0)

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
