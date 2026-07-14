"""
Storage layer — saves ingested posts/articles to a SQLite database.
Also tracks ingestion run timestamps so the scheduler can enforce
the minimum interval between runs (no redundant ingestion).
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Canonical post columns — drives both the INSERT and the per-post default fill,
# so adding a field means editing this one list (+ _NEW_COLUMNS for migration).
POST_COLUMNS = [
    "post_id", "source", "keyword", "keyword_category", "author", "author_followers",
    "text", "created_at", "likes", "reposts", "replies", "quotes", "ingested_at",
    # --- Week 7 multimodal rebuild: media / provenance / author signals ---
    "has_image", "image_url", "image_alt",            # image METADATA (universal, cheap; no download)
    "reshare_of_author", "reshare_of_uri",            # quote-post provenance (post.embed record)
    "external_url", "external_title",                 # linked source (post.embed external)
    "author_bio", "author_post_count", "author_created_at",  # from the same get_profiles call
    "vision_signal",                                  # DERIVED later, edge-cases only (JSON); NULL at ingest
]

# Columns added after the original schema — ALTER-ed onto existing DBs on open.
_NEW_COLUMNS = {
    "has_image": "INTEGER DEFAULT 0", "image_url": "TEXT", "image_alt": "TEXT",
    "reshare_of_author": "TEXT", "reshare_of_uri": "TEXT",
    "external_url": "TEXT", "external_title": "TEXT",
    "author_bio": "TEXT", "author_post_count": "INTEGER DEFAULT 0",
    "author_created_at": "TEXT", "vision_signal": "TEXT",
}

# Numeric columns default to 0 when a post dict omits them (e.g. GDELT has no engagement/media).
_INT_DEFAULT = {"author_followers", "likes", "reposts", "replies", "quotes",
                "has_image", "author_post_count"}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any post columns missing from an existing DB (idempotent)."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(posts)")}
    for name, decl in _NEW_COLUMNS.items():
        if name not in have:
            conn.execute(f"ALTER TABLE posts ADD COLUMN {name} {decl}")


def _get_conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            post_id          TEXT PRIMARY KEY,
            source           TEXT,
            keyword          TEXT,
            keyword_category TEXT,
            author           TEXT,
            author_followers INTEGER DEFAULT 0,
            text             TEXT,
            created_at       TEXT,
            likes            INTEGER DEFAULT 0,
            reposts          INTEGER DEFAULT 0,
            replies          INTEGER DEFAULT 0,
            quotes           INTEGER DEFAULT 0,
            ingested_at      TEXT,
            has_image        INTEGER DEFAULT 0,
            image_url        TEXT,
            image_alt        TEXT,
            reshare_of_author TEXT,
            reshare_of_uri   TEXT,
            external_url     TEXT,
            external_title   TEXT,
            author_bio       TEXT,
            author_post_count INTEGER DEFAULT 0,
            author_created_at TEXT,
            vision_signal    TEXT
        )
    """)
    _migrate(conn)   # bring older DBs up to the current schema

    # Metadata table — key/value store for pipeline state
    # Used to track when ingestion last ran so we never ingest more
    # frequently than the configured interval, even across restarts.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    return conn


def get_last_ingestion_time(db_path: str) -> datetime | None:
    """
    Returns the UTC datetime of the last completed ingestion cycle,
    or None if ingestion has never run.
    """
    try:
        conn = _get_conn(db_path)
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'last_ingested_at'"
        ).fetchone()
        conn.close()
        if row:
            return datetime.fromisoformat(row[0])
        return None
    except Exception:
        return None


def set_last_ingestion_time(db_path: str) -> None:
    """Records the current UTC time as the last completed ingestion cycle."""
    conn = _get_conn(db_path)
    conn.execute("""
        INSERT INTO metadata (key, value) VALUES ('last_ingested_at', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """, (datetime.now(timezone.utc).isoformat(),))
    conn.commit()
    conn.close()


def hours_since_last_ingestion(db_path: str) -> float:
    """
    Returns how many hours have passed since the last ingestion run.
    Returns infinity if ingestion has never run (so first run always proceeds).
    """
    last = get_last_ingestion_time(db_path)
    if last is None:
        return float("inf")
    elapsed = datetime.now(timezone.utc) - last
    return elapsed.total_seconds() / 3600


def _heldout_guard(conn: sqlite3.Connection) -> str:
    """SQL fragment that EXCLUDES held-out eval/train posts from deletion (leakage guard) —
    only for columns that exist on this DB."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(posts)")}
    g = ""
    if "in_eval_set" in have:
        g += " AND (in_eval_set IS NULL OR in_eval_set = 0)"
    if "in_train_set" in have:
        g += " AND (in_train_set IS NULL OR in_train_set = 0)"
    return g


def delete_posts(db_path: str, post_ids: list[str]) -> int:
    """Delete the given posts AND their classifications. Returns posts removed.
    Caller is responsible for the held-out guard when selecting `post_ids`."""
    if not post_ids:
        return 0
    conn = _get_conn(db_path)
    removed = 0
    for i in range(0, len(post_ids), 400):
        chunk = post_ids[i:i + 400]
        q = ",".join("?" * len(chunk))
        conn.execute(f"DELETE FROM classifications WHERE post_id IN ({q})", chunk)
        cur = conn.execute(f"DELETE FROM posts WHERE post_id IN ({q})", chunk)
        removed += cur.rowcount
    conn.commit()
    conn.close()
    return removed


def old_bluesky_post_ids(db_path: str, retention_days: int) -> list[str]:
    """Bluesky post_ids older than retention_days (ISO created_at compares lexicographically),
    excluding held-out eval/train posts. Undated posts are kept (NULL fails the comparison)."""
    conn = _get_conn(db_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    guard = _heldout_guard(conn)
    rows = conn.execute(
        f"SELECT post_id FROM posts WHERE source = 'bluesky' AND created_at IS NOT NULL "
        f"AND created_at < ?{guard}", (cutoff,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def oldest_bluesky_post_ids(db_path: str, limit: int) -> list[str]:
    """Oldest Bluesky post_ids (for the availability sweep), excluding held-out eval/train posts."""
    conn = _get_conn(db_path)
    guard = _heldout_guard(conn)
    rows = conn.execute(
        f"SELECT post_id FROM posts WHERE source = 'bluesky'{guard} "
        f"ORDER BY created_at ASC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def save(posts: list[dict], db_path: str) -> int:
    """
    Insert posts into the database, skipping duplicates.
    Returns the number of new rows inserted.
    """
    if not posts:
        return 0

    conn = _get_conn(db_path)  # opens (or creates) the SQLite DB
    cols = ", ".join(POST_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in POST_COLUMNS)
    sql = f"INSERT OR IGNORE INTO posts ({cols}) VALUES ({placeholders})"
    inserted = 0
    for p in posts:
        # Fill any column the source didn't provide (e.g. GDELT has no media/embed fields)
        # so the named-param insert never KeyErrors; vision_signal stays NULL until Stage 4.
        row = {c: p.get(c, 0 if c in _INT_DEFAULT else None) for c in POST_COLUMNS}
        try:  # if a row with the same post_id already exists it skips instead of throwing error
            conn.execute(sql, row)
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except Exception as e:
            print(f"  DB insert error: {e}")
    conn.commit()
    conn.close()
    return inserted
