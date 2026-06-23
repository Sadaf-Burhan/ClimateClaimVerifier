"""
Bluesky ingestion — fetches posts for each keyword using the atproto SDK.
An atproto SDK is a set of developer tools used to build applications on the
AT Protocol (the decentralized social network foundation powering platforms like Bluesky).
It handles complex network interactions, cryptographic identity, data schemas, and API requests
so developers can easily build custom clients, feeds, and bots.
Requires BLUESKY_HANDLE and BLUESKY_APP_PASSWORD in .env
"""

import os
import requests as _requests
from datetime import datetime, timezone
from atproto import Client
from atproto_client.exceptions import ModelError
from dotenv import load_dotenv

load_dotenv()

_client = None

def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client()
        _client.login(
            os.environ["BLUESKY_HANDLE"],
            os.environ["BLUESKY_APP_PASSWORD"],
        )
    return _client


_PROFILE_BATCH = 25  # Bluesky get_profiles accepts max 25 DIDs per call


def _batch_follower_counts(client: Client, dids: list[str]) -> dict[str, int]:
    """
    Fetch follower counts for a list of DIDs in batches of 25.
    Returns a dict of {did: followers_count}.
    One batch call instead of one API call per post — ~25x faster.
    """
    counts: dict[str, int] = {}
    for i in range(0, len(dids), _PROFILE_BATCH):
        batch = dids[i:i + _PROFILE_BATCH]
        try:
            resp = client.app.bsky.actor.get_profiles({"actors": batch})
            for profile in resp.profiles:
                counts[profile.did] = profile.followers_count or 0
        except Exception:
            for did in batch:
                counts[did] = 0
    return counts


def _fetch_posts_raw(client: Client, params: dict, keyword: str) -> list[dict]:
    """
    Raw HTTP fallback when the atproto SDK fails to parse a response
    (e.g. Bluesky added a new embed type the installed SDK doesn't know about).
    Parses JSON directly — bypasses pydantic validation entirely.
    """
    url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
    headers = {"Authorization": f"Bearer {client._session.access_jwt}"}
    try:
        r = _requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  Bluesky raw HTTP fallback also failed for '{keyword}': {e}")
        return []

    raw = data.get("posts", [])
    dids = [p.get("author", {}).get("did", "") for p in raw]
    follower_map = _batch_follower_counts(client, dids)

    posts = []
    for post in raw:
        record = post.get("record", {})
        author = post.get("author", {})
        did = author.get("did", "")
        posts.append({
            "source":           "bluesky",
            "keyword":          keyword,
            "post_id":          post.get("uri", ""),
            "author":           author.get("handle", ""),
            "author_followers": follower_map.get(did, 0),
            "text":             record.get("text", ""),
            "created_at":       record.get("createdAt", ""),
            "likes":            post.get("likeCount", 0) or 0,
            "reposts":          post.get("repostCount", 0) or 0,
            "replies":          post.get("replyCount", 0) or 0,
            "quotes":           post.get("quoteCount", 0) or 0,
            "ingested_at":      datetime.now(timezone.utc).isoformat(),
        })
    return posts


def fetch_posts(keyword: str, limit: int, since: str | None = None) -> list[dict]:
    """
    Search Bluesky for posts matching keyword.
    Captures full engagement metrics and author follower count.
    Follower counts are fetched in batches (25 DIDs per call) rather than
    one call per post — reduces API calls by ~25x.
    `since` is an ISO 8601 UTC string — only posts after that timestamp
    are returned, avoiding re-fetching already-ingested content.
    Returns a list of normalised dicts ready for storage.
    """
    client = _get_client()
    params = {"q": keyword, "limit": limit, "sort": "latest"}
    if since:
        params["since"] = since

    try:
        resp = client.app.bsky.feed.search_posts(params)
        raw_posts = resp.posts
    except ModelError:
        print(f"  Bluesky SDK parse error for '{keyword}' — falling back to raw HTTP")
        return _fetch_posts_raw(client, params, keyword)

    # Batch follower lookup — 4 calls for 100 posts instead of 100 calls
    dids = [p.author.did for p in raw_posts]
    follower_map = _batch_follower_counts(client, dids)

    posts = []
    for post in raw_posts:
        posts.append({
            "source":           "bluesky",
            "keyword":          keyword,
            "post_id":          post.uri,
            "author":           post.author.handle,
            "author_followers": follower_map.get(post.author.did, 0),
            "text":             post.record.text,
            "created_at":       post.record.created_at,
            "likes":            post.like_count or 0,
            "reposts":          post.repost_count or 0,
            "replies":          post.reply_count or 0,
            "quotes":           post.quote_count or 0,
            "ingested_at":      datetime.now(timezone.utc).isoformat(),
        })
    return posts


def fetch_trending_topics() -> list[str]:
    """
    Returns current trending topic labels from Bluesky.
    These can be used to dynamically supplement the keyword list
    with topics that are organically spiking right now.
    """
    client = _get_client()
    try:
        resp = client.app.bsky.unspecced.get_trending_topics({})
        return [topic.topic for topic in resp.topics]
    except Exception:
        return []
