"""
Storage layer — saves ingested posts/articles to a SQLite database.
Also tracks ingestion run timestamps so the scheduler can enforce
the minimum interval between runs (no redundant ingestion).
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


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
            ingested_at      TEXT
        )
    """)

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


def save(posts: list[dict], db_path: str) -> int:
    """
    Insert posts into the database, skipping duplicates.
    Returns the number of new rows inserted.
    """
    if not posts:
        return 0

    conn = _get_conn(db_path)  # opens (or creates) the SQLite DB
    inserted = 0
    for p in posts:
        try:  # if a row with the same post_id already exists it skips instead of throwing error
            conn.execute("""
                INSERT OR IGNORE INTO posts
                    (post_id, source, keyword, keyword_category, author, author_followers, text,
                     created_at, likes, reposts, replies, quotes, ingested_at)
                VALUES
                    (:post_id, :source, :keyword, :keyword_category, :author, :author_followers, :text,
                     :created_at, :likes, :reposts, :replies, :quotes, :ingested_at)
            """, p)
            # count the actual inserts
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except Exception as e:
            print(f"  DB insert error: {e}")
    conn.commit()
    conn.close()
    return inserted
