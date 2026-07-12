"""
Health / heartbeat signal for the in-house maintenance layer.

The pipeline runs in two places — local daily ingestion, and a Colab GPU notebook for
classify / evaluate / vision — and neither is watched live. Each stage records its last
outcome here so the external-facing app can show a freshness banner ("ingestion ok 3h ago",
"classification failed") instead of silently serving stale or half-updated data.

`data/health.json` is a small dict keyed by stage:
  { "ingestion":     {"ts": "...", "ok": true,  "counts": {...}},
    "classification":{"ts": "...", "ok": false, "error": "..."} }

No secrets, no network — just a file. Stages call `update_health(...)`; the app reads it
with `load_health(...)` and `stage_status(...)` to decide banner colour and staleness.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

HEALTH_PATH = Path("data/health.json")


def update_health(stage: str, ok: bool, path: Path | str = HEALTH_PATH, **extra) -> dict:
    """Record the last outcome of a maintenance `stage` (e.g. 'ingestion', 'classification',
    'evaluation', 'vision'). Merges into the existing file so other stages are preserved.
    Any extra kwargs (counts, error, duration_s) are stored verbatim."""
    p = Path(path)
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[stage] = {"ts": datetime.now(timezone.utc).isoformat(), "ok": bool(ok), **extra}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data[stage]


def load_health(path: Path | str = HEALTH_PATH) -> dict:
    """Read the whole health map. Returns {} if it doesn't exist yet."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def age_hours(ts: str) -> float | None:
    """Hours since an ISO timestamp, or None if unparseable."""
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - d).total_seconds() / 3600
    except Exception:
        return None


def stage_status(stage: dict, stale_after_hours: float = 48.0) -> str:
    """Traffic-light for a stage record: 'fail' (last run errored), 'stale' (too long since a
    good run), or 'ok'. Drives the app banner colour."""
    if not stage or not stage.get("ok", False):
        return "fail"
    age = age_hours(stage.get("ts", ""))
    if age is not None and age > stale_after_hours:
        return "stale"
    return "ok"
