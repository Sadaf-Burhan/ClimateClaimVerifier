"""
Stage 3: Claim Presence Classifier

Uses Ollama to determine whether a social media post contains a verifiable
factual claim about a climate or extreme weather event.

Posts are classified in batches: several posts share one prompt, so the
few-shot instructions are paid for once per batch instead of once per post.
Ollama's structured-output mode (`format` = JSON schema) guarantees parseable
JSON, and reasons are kept short — on CPU, output tokens dominate runtime.

Output per post:
  {"has_claim": true,  "reason": "Specific verifiable claim about ..."}
  {"has_claim": false, "reason": "Opinion with no verifiable ..."}

Posts with has_claim=false are rejected here with a plain-language explanation.
Posts with has_claim=true proceed to the evidence retrieval stage.
"""

import json
import re
import sqlite3
import ollama
from datetime import datetime, timezone
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Posts per LLM request. Larger batches amortize the prompt better but give
# small models more room to drift; 16 stays well inside the 4096-token context.
LLM_BATCH_SIZE = 16

# Posts longer than this are truncated in the prompt to bound token cost.
MAX_POST_CHARS = 500

KEEP_ALIVE = "15m"  # keep the model loaded between batches

# Task definition + persona live in the system role (classify_v3 pattern).
# The few-shot examples and the JSON format example stay in the USER message
# (classify_v3 keeps them there via prompt_v2) — the model needs the format
# example right before generation. "Think first" enables chain-of-thought,
# which surfaces as the `thought` field in the structured output.
_SYSTEM_PROMPT = """You are a climate claim detector. For each social media post, decide whether it contains a verifiable factual claim about a climate or extreme weather event in North America.

A CLAIM is a specific assertion that could be checked against external records — a named event, a measurement, an official warning, an attributed study, or a specific cause-and-effect assertion. A claim does NOT need to be true: a false or conspiratorial assertion is still a CLAIM if it is specific and checkable. Emotional or hostile tone does not matter if the post contains at least one checkable assertion.
An OPINION has nothing checkable: feelings, jokes, sarcasm, personal experiences from the poster's own life, rhetorical questions, vague predictions, or general political viewpoints.
Never judge whether a post is TRUE — judge only whether it makes a checkable assertion.
Checkability is not the same as evidence. A specific assertion is a CLAIM even if the post gives no proof, no source, and no supporting data. "No evidence provided" is NEVER a reason to call something an OPINION — only the absence of a specific, checkable assertion is.

For each post, first note in `thought` what (if anything) is checkable, then decide."""


def _batch_schema() -> dict:
    """JSON schema passed to Ollama's structured-output mode.
    `thought` is first so the model reasons before committing to has_claim
    (chain-of-thought in the structured output, classify_v3 pattern)."""
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":        {"type": "integer"},
                        "thought":   {"type": "string"},
                        "has_claim": {"type": "boolean"},
                        "reason":    {"type": "string"},
                    },
                    "required": ["id", "thought", "has_claim", "reason"],
                },
            }
        },
        "required": ["results"],
    }


def build_batch_prompt(post_texts: list[str]) -> str:
    """User message: few-shot examples + the JSON format example + the posts.
    Task definition is supplied separately in the system role."""
    numbered = "\n".join(
        f'{i}. "{text[:MAX_POST_CHARS]}"' for i, text in enumerate(post_texts, 1)
    )
    return f"""Example input:
1. "Calgary set a new June rainfall record of 124mm on Tuesday"
2. "Honestly I can't even deal with this heat anymore"
3. "Volcanoes release more CO2 than all human activity combined"
4. "Climate policy is just politicians chasing votes"
5. "So angry right now, the river gauge here just hit major flood stage"
6. "My basement flooded again, this town is cursed"
7. "Cloud seeding planes triggered the flash floods in Dubai last week"
8. "They know what's really causing this but they will never admit it"

Example output:
{{"results": [
  {{"id": 1, "thought": "Named city and a specific record measurement — checkable", "has_claim": true, "reason": "Verifiable rainfall record at named location"}},
  {{"id": 2, "thought": "Just an emotional reaction", "has_claim": false, "reason": "Emotional reaction, nothing checkable"}},
  {{"id": 3, "thought": "Specific claim, no evidence given but still checkable", "has_claim": true, "reason": "Checkable assertion even though false"}},
  {{"id": 4, "thought": "A general political viewpoint, nothing specific", "has_claim": false, "reason": "General viewpoint, nothing specific to check"}},
  {{"id": 5, "thought": "Emotional but cites a flood measurement", "has_claim": true, "reason": "Emotional tone but checkable flood measurement"}},
  {{"id": 6, "thought": "Personal experience, not externally checkable", "has_claim": false, "reason": "Personal experience, not externally checkable"}},
  {{"id": 7, "thought": "Specific mechanism and event — checkable even if false", "has_claim": true, "reason": "Specific mechanism and event — checkable even if false"}},
  {{"id": 8, "thought": "Vague accusation, no specific assertion", "has_claim": false, "reason": "Vague accusation with no specific assertion to check"}}
]}}

Rules:
- Return JSON only: exactly one result per post, in the same order.
- Keep each thought and reason under 8 words.

Now classify these posts:
{numbered}"""


def classify_batch(post_texts: list[str], model: str = "gemma2:2b") -> list[dict]:
    """
    Classify a batch of posts in a single LLM request.
    Returns one {"has_claim": bool, "reason": str} dict per input post,
    in input order. Posts the model skipped fall back to single-post mode.
    """
    if not post_texts:
        return []

    results: list[dict | None] = [None] * len(post_texts)
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": build_batch_prompt(post_texts)},
            ],
            format=_batch_schema(),
            options={
                "temperature": 0.0,
                # thought field roughly doubles output tokens per post
                "num_predict": 96 * len(post_texts) + 128,
            },
            keep_alive=KEEP_ALIVE,
        )
        data = json.loads(response["message"]["content"])
        for item in data.get("results", []):
            idx = int(item.get("id", 0)) - 1
            if 0 <= idx < len(post_texts) and results[idx] is None:
                results[idx] = {
                    "has_claim": bool(item.get("has_claim", False)),
                    "reason":    str(item.get("reason", "")),
                }
    except Exception:
        pass  # fall through to single-post fallback for unfilled slots

    for i, r in enumerate(results):
        if r is None:
            results[i] = classify(post_texts[i], model=model)
    return results


def build_prompt(post_text: str) -> str:
    return f"""You are a climate claim detector. Your job is to decide whether a social media post contains a verifiable factual claim about a climate or extreme weather event in North America.

A CLAIM is a specific assertion that could be checked against external records — a named event, a measurement, an official warning, an attributed study, or a specific cause-and-effect assertion. A claim does NOT need to be true: a false or conspiratorial assertion is still a CLAIM if it is specific and checkable. Emotional or hostile tone does not matter if the post contains at least one checkable assertion.
An OPINION has nothing checkable: feelings, jokes, sarcasm, personal experiences from the poster's own life, rhetorical questions, vague predictions, or general political viewpoints.
Never judge whether a post is TRUE — judge only whether it makes a checkable assertion.

Examples:
Post: "Calgary set a new June rainfall record of 124mm on Tuesday"
Output: {{"has_claim": true, "reason": "Verifiable rainfall record at a named location."}}

Post: "Honestly I can't even deal with this heat anymore"
Output: {{"has_claim": false, "reason": "Emotional reaction with no verifiable factual content."}}

Post: "Volcanoes release more CO2 than all human activity combined"
Output: {{"has_claim": true, "reason": "Checkable assertion even though false."}}

Post: "Climate policy is just politicians chasing votes"
Output: {{"has_claim": false, "reason": "General viewpoint, nothing specific to check."}}

Post: "So angry right now, the river gauge here just hit major flood stage"
Output: {{"has_claim": true, "reason": "Emotional tone but checkable flood measurement."}}

Rules:
- Return JSON only.
- Keep the reason under 8 words.

Now classify this post:
Post: "{post_text[:MAX_POST_CHARS]}"
Output:"""


def classify(post_text: str, model: str = "gemma2:2b") -> dict:
    """
    Classify a single post (used as fallback when a batch result is missing).
    Returns {"has_claim": bool, "reason": str}.
    Falls back to {"has_claim": False, "reason": "..."} on failure.
    """
    prompt = build_prompt(post_text)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format={
                "type": "object",
                "properties": {
                    "has_claim": {"type": "boolean"},
                    "reason":    {"type": "string"},
                },
                "required": ["has_claim", "reason"],
            },
            options={"temperature": 0.0, "num_predict": 128},
            keep_alive=KEEP_ALIVE,
        )
        raw = response["message"]["content"].strip()

        # Extract JSON from response — model may add surrounding text
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "has_claim": bool(data.get("has_claim", False)),
                "reason":    str(data.get("reason", "")),
            }
        return {"has_claim": False, "reason": f"Could not parse model output: {raw[:100]}"}

    except Exception as e:
        return {"has_claim": False, "reason": f"Classifier error: {e}"}


def classify_pending(db_path: str, model: str, batch_size: int = 50,
                     llm_batch_size: int = LLM_BATCH_SIZE):
    """
    Generator — classifies unclassified posts from the database in LLM batches
    of `llm_batch_size`, but still yields one progress dict per post:
    {"done": int, "total": int, "post_id": str, "has_claim": bool, "reason": str}

    Uses a separate `classifications` table so ingestion and classification
    remain independent and the pipeline can be re-run without re-ingesting.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)

    # Fetch posts that have not been classified yet
    rows = conn.execute("""
        SELECT p.post_id, p.text, p.likes, p.reposts, p.replies, p.quotes
        FROM posts p
        LEFT JOIN classifications c ON p.post_id = c.post_id
        WHERE c.post_id IS NULL
        LIMIT ?
    """, (batch_size,)).fetchall()

    total = len(rows)
    done = 0
    for start in range(0, total, llm_batch_size):
        chunk = rows[start:start + llm_batch_size]
        results = classify_batch([r["text"] for r in chunk], model=model)
        now = datetime.now(timezone.utc).isoformat()
        for row, result in zip(chunk, results):
            conn.execute("""
                INSERT OR IGNORE INTO classifications (post_id, has_claim, reason, classified_at)
                VALUES (?, ?, ?, ?)
            """, (row["post_id"], int(result["has_claim"]), result["reason"], now))
        conn.commit()  # one commit per LLM batch
        for row, result in zip(chunk, results):
            done += 1
            yield {"done": done, "total": total, "post_id": row["post_id"], **result}

    conn.close()


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Creates all pipeline tables if they don't exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            post_id         TEXT PRIMARY KEY,
            has_claim       INTEGER,
            reason          TEXT,
            classified_at   TEXT
        )
    """)
    conn.commit()


def get_top_claims(db_path: str, limit: int = 10) -> list[dict]:
    """
    Returns top N classified claims, ranked by total engagement
    (likes + reposts + replies + quotes).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)
    rows = conn.execute("""
        SELECT p.text, p.author, p.source, p.keyword, p.keyword_category,
               p.likes, p.reposts, p.replies, p.quotes, p.author_followers,
               p.created_at, c.reason,
               (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p
        JOIN classifications c ON p.post_id = c.post_id
        WHERE c.has_claim = 1
        ORDER BY engagement DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_top_opinions(db_path: str, limit: int = 10) -> list[dict]:
    """
    Returns top N rejected (opinion) posts, ranked by engagement.
    High-engagement opinions are worth showing — they spread even though
    they contain no verifiable claim.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)
    rows = conn.execute("""
        SELECT p.text, p.author, p.source, p.keyword_category,
               p.likes, p.reposts, p.replies, p.quotes, p.author_followers,
               p.created_at, c.reason,
               (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p
        JOIN classifications c ON p.post_id = c.post_id
        WHERE c.has_claim = 0
        ORDER BY engagement DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(db_path: str) -> dict:
    """Returns pipeline counts for the dashboard header."""
    conn = sqlite3.connect(db_path)
    _ensure_tables(conn)
    stats = {}
    try:
        stats["total_ingested"]   = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        stats["total_classified"] = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
        stats["total_claims"]     = conn.execute("SELECT COUNT(*) FROM classifications WHERE has_claim=1").fetchone()[0]
        stats["total_opinions"]   = conn.execute("SELECT COUNT(*) FROM classifications WHERE has_claim=0").fetchone()[0]
    except Exception:
        stats = {"total_ingested": 0, "total_classified": 0, "total_claims": 0, "total_opinions": 0}
    conn.close()
    return stats


def main():
    """
    Headless classification of pending posts — preferred over the dashboard
    button for large backlogs (no browser session to keep alive). Commits per
    LLM batch, so it is safe to interrupt and resume; already-classified posts
    are never redone.

    Run from the project root:
      uv run python -m climate_verifier.pipeline.claim_classifier
      uv run python -m climate_verifier.pipeline.claim_classifier --limit 100
    """
    import argparse
    import yaml

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Classify pending posts from the database.")
    parser.add_argument("--model", default=cfg["model"]["name"],
                        help="Ollama model (default: model.name from config.yaml)")
    parser.add_argument("--limit", type=int, default=1_000_000,
                        help="max posts to classify this run (default: all pending)")
    args = parser.parse_args()

    db_path = cfg["storage"]["db_path"]
    llm_batch_size = cfg["model"].get("llm_batch_size", LLM_BATCH_SIZE)

    claims = opinions = 0
    for u in classify_pending(db_path, model=args.model, batch_size=args.limit,
                              llm_batch_size=llm_batch_size):
        if u["has_claim"]:
            claims += 1
            label = "CLAIM  "
        else:
            opinions += 1
            label = "opinion"
        print(f"[{u['done']}/{u['total']}] {label} {u['reason'][:70]}", flush=True)

    print(f"Done: {claims} claims, {opinions} opinions, {claims + opinions} total.")


if __name__ == "__main__":
    main()
