"""
Bluesky ingestion — fetches posts for each keyword using the atproto SDK.
An atproto SDK is a set of developer tools used to build applications on the
AT Protocol (the decentralized social network foundation powering platforms like Bluesky).
It handles complex network interactions, cryptographic identity, data schemas, and API requests
so developers can easily build custom clients, feeds, and bots.
Requires BLUESKY_HANDLE and BLUESKY_APP_PASSWORD in .env
"""

import os
import re
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
_POST_BATCH = 25     # app.bsky.feed.getPosts accepts max 25 URIs per call


def check_posts_exist(uris: list[str]) -> set[str]:
    """Return the subset of Bluesky post URIs that STILL EXIST (getPosts omits deleted ones).

    Batched 25/call. FAIL-SAFE: if a batch call errors (network/transient), those URIs are
    treated as existing (returned), so a transient failure can never cause the refresher to
    delete real posts. Only posts confirmed absent by a successful call are excluded."""
    if not uris:
        return set()
    client = _get_client()
    existing: set[str] = set()
    for i in range(0, len(uris), _POST_BATCH):
        batch = [u for u in uris[i:i + _POST_BATCH] if u]
        if not batch:
            continue
        try:
            resp = client.app.bsky.feed.get_posts({"uris": batch})
            existing.update(p.uri for p in (getattr(resp, "posts", None) or []))
        except Exception as e:
            print(f"  availability check error (keeping this batch): {e}")
            existing.update(batch)   # fail-safe — assume present on error
    return existing

_EMPTY_PROFILE = {"followers": 0, "bio": "", "posts_count": 0, "created_at": ""}


def _batch_profiles(client: Client, dids: list[str]) -> dict[str, dict]:
    """
    Fetch author profiles for a list of DIDs in batches of 25 — one batch call
    instead of one per post (~25x fewer calls). Returns {did: {followers, bio,
    posts_count, created_at}}. bio/posts_count/account age are behavioural
    reliability signals captured at no extra cost (same call as follower count).
    """
    out: dict[str, dict] = {}
    for i in range(0, len(dids), _PROFILE_BATCH):
        batch = dids[i:i + _PROFILE_BATCH]
        try:
            resp = client.app.bsky.actor.get_profiles({"actors": batch})
            for pr in resp.profiles:
                out[pr.did] = {
                    "followers":   getattr(pr, "followers_count", 0) or 0,
                    "bio":         getattr(pr, "description", "") or "",
                    "posts_count": getattr(pr, "posts_count", 0) or 0,
                    "created_at":  getattr(pr, "created_at", None) or getattr(pr, "indexed_at", "") or "",
                }
        except Exception:
            for did in batch:
                out.setdefault(did, dict(_EMPTY_PROFILE))
    return out


def _acc(obj, *keys):
    """Nested attr/dict getter — defensive across SDK objects and raw dicts; None on any miss."""
    for k in keys:
        if obj is None:
            return None
        obj = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
    return obj


def _extract_embed(embed) -> dict:
    """
    Pull media + provenance from a Bluesky post embed (SDK object OR raw dict).
    Handles images / external / record (quote) / recordWithMedia. Defensive:
    any unknown or mixed embed shape degrades to empty rather than raising —
    ingestion must never crash on a new embed type.
    """
    out = {"has_image": 0, "image_url": None, "image_alt": None,
           "external_url": None, "external_title": None,
           "reshare_of_author": None, "reshare_of_uri": None}
    if embed is None:
        return out
    try:
        # recordWithMedia nests images/external under .media; plain embeds have them directly.
        media = _acc(embed, "media") or embed
        images = _acc(media, "images")
        img = images[0] if images else None
        if img is not None:
            out["has_image"] = 1
            out["image_url"] = _acc(img, "fullsize") or _acc(img, "thumb")
            out["image_alt"] = _acc(img, "alt")
        ext = _acc(media, "external")
        if ext is not None:
            out["external_url"] = _acc(ext, "uri")
            out["external_title"] = _acc(ext, "title")
        # Quote/reshare: record#view has author/uri directly; recordWithMedia nests it one deeper.
        rec = _acc(embed, "record")
        inner = _acc(rec, "record") or rec
        if inner is not None:
            out["reshare_of_author"] = _acc(inner, "author", "handle")
            out["reshare_of_uri"] = _acc(inner, "uri")
    except Exception:
        pass
    return out


def _extract_facet_link(record) -> str | None:
    """The first inline rich-text LINK in a post body carries its FULL url in a facet — the body
    text only shows a truncated display string ('www.nyc.gov/site/em/abou…'). Return that uri so a
    post whose link is an inline facet (NOT an embed card) still gets a working, complete
    external_url. Any feature carrying a `uri` is a link facet (mentions carry `did`, tags a `tag`).
    Defensive across SDK objects and raw dicts; None on any miss."""
    facets = _acc(record, "facets")
    if not facets:
        return None
    try:
        for f in facets:
            for feat in (_acc(f, "features") or []):
                uri = _acc(feat, "uri")
                if uri:
                    return uri
    except Exception:
        pass
    return None


def _embed_and_links(embed, record) -> dict:
    """`_extract_embed` plus a facet-link fallback: when the post has no embed-card external link,
    fill `external_url` from the first inline link facet (whose uri is the full, untruncated URL)."""
    out = _extract_embed(embed)
    if not out.get("external_url"):
        out["external_url"] = _extract_facet_link(record)
    return out


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
    profiles = _batch_profiles(client, dids)

    posts = []
    for post in raw:
        record = post.get("record", {})
        author = post.get("author", {})
        did = author.get("did", "")
        pr = profiles.get(did, _EMPTY_PROFILE)
        posts.append({
            "source":            "bluesky",
            "keyword":           keyword,
            "post_id":           post.get("uri", ""),
            "author":            author.get("handle", ""),
            "author_followers":  pr["followers"],
            "author_bio":        pr["bio"],
            "author_post_count": pr["posts_count"],
            "author_created_at": pr["created_at"],
            "text":              record.get("text", ""),
            "created_at":        record.get("createdAt", ""),
            "likes":             post.get("likeCount", 0) or 0,
            "reposts":           post.get("repostCount", 0) or 0,
            "replies":           post.get("replyCount", 0) or 0,
            "quotes":            post.get("quoteCount", 0) or 0,
            "ingested_at":       datetime.now(timezone.utc).isoformat(),
            **_embed_and_links(post.get("embed"), record),
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

    # Batch profile lookup — 4 calls for 100 posts instead of 100 calls
    dids = [p.author.did for p in raw_posts]
    profiles = _batch_profiles(client, dids)

    posts = []
    for post in raw_posts:
        pr = profiles.get(post.author.did, _EMPTY_PROFILE)
        posts.append({
            "source":            "bluesky",
            "keyword":           keyword,
            "post_id":           post.uri,
            "author":            post.author.handle,
            "author_followers":  pr["followers"],
            "author_bio":        pr["bio"],
            "author_post_count": pr["posts_count"],
            "author_created_at": pr["created_at"],
            "text":              post.record.text,
            "created_at":        post.record.created_at,
            "likes":             post.like_count or 0,
            "reposts":           post.repost_count or 0,
            "replies":           post.reply_count or 0,
            "quotes":            post.quote_count or 0,
            "ingested_at":       datetime.now(timezone.utc).isoformat(),
            **_embed_and_links(getattr(post, "embed", None), getattr(post, "record", None)),
        })
    return posts


def _url_to_at_uri(client: Client, url: str) -> str | None:
    """Convert a bsky.app post URL to an at:// URI. Resolves a handle to a DID if needed."""
    url = (url or "").strip()
    if url.startswith("at://"):
        return url
    m = re.search(r"bsky\.app/profile/([^/]+)/post/([^/?#\s]+)", url)
    if not m:
        return None
    actor, rkey = m.group(1), m.group(2)
    did = actor
    if not actor.startswith("did:"):
        try:
            did = client.com.atproto.identity.resolve_handle({"handle": actor}).did
        except Exception:
            return None
    return f"at://{did}/app.bsky.feed.post/{rkey}"


def fetch_post_by_url(url: str) -> dict | None:
    """
    Fetch a SINGLE Bluesky post by its bsky.app URL (or at:// URI) and return the same
    normalized dict shape as fetch_posts — text, engagement counts, author + follower/profile
    signals, and any image/reshare/external embed. Lets the app auto-populate a pasted post's
    metadata instead of asking the user to type likes/reposts by hand. None if it can't resolve.
    """
    client = _get_client()
    at_uri = _url_to_at_uri(client, url)
    if not at_uri:
        return None
    try:
        resp = client.app.bsky.feed.get_posts({"uris": [at_uri]})
        posts = getattr(resp, "posts", None) or []
        if not posts:
            return None
        post = posts[0]
        pr = _batch_profiles(client, [post.author.did]).get(post.author.did, dict(_EMPTY_PROFILE))
        return {
            "source":            "bluesky",
            "keyword":           "",
            "keyword_category":  "user_submitted",
            "post_id":           post.uri,
            "author":            post.author.handle,
            "author_followers":  pr["followers"],
            "author_bio":        pr["bio"],
            "author_post_count": pr["posts_count"],
            "author_created_at": pr["created_at"],
            "text":              getattr(post.record, "text", "") or "",
            "created_at":        getattr(post.record, "created_at", "") or "",
            "likes":             post.like_count or 0,
            "reposts":           post.repost_count or 0,
            "replies":           post.reply_count or 0,
            "quotes":            post.quote_count or 0,
            "ingested_at":       datetime.now(timezone.utc).isoformat(),
            **_embed_and_links(getattr(post, "embed", None), getattr(post, "record", None)),
        }
    except Exception:
        return None


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
