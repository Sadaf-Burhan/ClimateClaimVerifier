"""
Climate Claim Scanner — Streamlit Dashboard

Trust Checker (the user-facing product): check a **Bluesky** climate/weather post and get the
  signals to judge it — claim vs opinion, news corroboration (RAG) with credible sources, source
  context, and a red flag for the reach-vs-support mismatch. It SURFACES signals; it never says a
  post is true/false — the reader concludes.
Claim Classifier (Week 1) · Embedding Analysis (Week 2) · Evidence Matching (Week 6) ·
Base-vs-Adapter (Week 5): method / analysis tabs.

Run:  uv run streamlit run app.py
"""

import csv
import io
import json
import re
import sqlite3
import subprocess
import streamlit as st
import yaml
import pandas as pd
from pathlib import Path

from climate_verifier.pipeline.claim_classifier import (
    classify,
    classify_batch,
    classify_lean,
    get_stats,
)
from climate_verifier.pipeline.evaluate import (
    load_eval_set,
    run_eval,
    compute_metrics,
    load_eval_history,
)
from climate_verifier.health import load_health, stage_status, age_hours
from climate_verifier.pipeline.embedder import (
    similarity,
    eval_pairs,
    category_similarity_stats,
)
from climate_verifier.pipeline.evidence import get_store, assess_claim, assess_db_claims
from climate_verifier.pipeline.vision import (
    extract_from_image, screenshot_signal_inputs, looks_like_social_post,
)
from climate_verifier.pipeline.relabel import (
    get_relabel_candidates, get_signal_candidates, apply_relabel, mark_reviewed_ok, eval_post_types,
)
from climate_verifier.ingestion.bluesky import fetch_post_by_url
from climate_verifier.ingestion.store import get_last_ingestion_time, hours_since_last_ingestion

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("src/climate_verifier/config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

DB_PATH        = cfg["storage"]["db_path"]
MODEL          = cfg["model"]["name"]
LLM_BATCH_SIZE = cfg["model"].get("llm_batch_size", 16)
# Week 5 comparison: the LoRA adapter, registered in Ollama as a GGUF model.
ADAPTER_MODEL  = cfg["model"].get("adapter_name", "qwen2.5-3b-claim-lora")
EMBED_MODEL    = cfg["embedding"]["model_name"]
VISION_MODEL      = cfg["vision"]["model"]                          # qwen2.5vl:7b — the image-input path (GPU)
VISION_CORRECTOR  = cfg["vision"].get("corrector_model", "qwen2.5:3b")
EVAL_CSV       = Path(cfg["evaluation"]["claim_eval_csv"])          # DYNAMIC — relabels append here
GOLD_EVAL_CSV  = Path(cfg["evaluation"].get("gold_eval_csv", "data/claim_eval_gold.csv"))  # FROZEN control
CLAIM_RECALL_TARGET = float(cfg["evaluation"]["claim_recall_target"])
PAIRS_CSV      = Path("data/embedding_pairs.csv")

# ── Trust Checker helpers ───────────────────────────────────────────────────────
@st.cache_resource
def _trust_store():
    return get_store()


_SORT_SQL = {"engagement": "engagement DESC",
             "followers": "p.author_followers DESC",
             "recent": "p.created_at DESC"}
_SORT_LABEL = {"engagement": "Most engagement", "followers": "Most followers", "recent": "Most recent"}


def _load_posts(db_path: str, has_claim: int, limit: int = 25, sort_by: str = "engagement") -> list[dict]:
    """Top classified Bluesky posts (claims or opinions), ranked by the chosen criterion, with the
    fields the trust panel needs (incl. post_id for the Bluesky link and any stored vision signal)."""
    order = _SORT_SQL.get(sort_by, _SORT_SQL["engagement"])
    con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
    # Prefer the admin override (admin_label) over the model's has_claim, so admin relabels show to
    # users. COALESCE tolerates DBs without the column (older schema) via a guard below.
    has_admin = "admin_label" in {r[1] for r in con.execute("PRAGMA table_info(classifications)")}
    label_expr = "COALESCE(c.admin_label, c.has_claim)" if has_admin else "c.has_claim"
    rows = con.execute(f"""
        SELECT p.post_id, p.text, p.author, p.author_followers, p.vision_signal,
               p.keyword_category, p.created_at, p.external_url, c.reason,
               {label_expr} AS has_claim,
               (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p JOIN classifications c ON p.post_id = c.post_id
        WHERE {label_expr} = ? AND p.source = 'bluesky'
        ORDER BY {order} LIMIT ?
    """, (has_claim, limit)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def _bsky_url(post_id: str) -> str:
    """at://did:plc:xxx/app.bsky.feed.post/rkey -> https://bsky.app/profile/did/post/rkey"""
    try:
        parts = (post_id or "").replace("at://", "").split("/")
        return f"https://bsky.app/profile/{parts[0]}/post/{parts[-1]}" if len(parts) >= 3 else ""
    except Exception:
        return ""


def _norm_url(u: str) -> str:
    return re.sub(r"^https?://", "", (u or "").strip()).removeprefix("www.").rstrip("/").lower()


def _linkify_post(text: str, external_url: str = "") -> str:
    """Bluesky embeds a TRUNCATED display URL in the post text (e.g. `www.cbc.ca/radio/whaton…`),
    which 404s when clicked. If we have the full embed URL and the truncated token is a prefix of it,
    replace that token with a working markdown link to the full URL."""
    ext = (external_url or "").strip()
    m = re.search(r"(https?://[^\s]+|www\.[^\s]+)", text or "")
    if not ext.startswith("http") or not m:
        return text
    token = m.group(0).rstrip(".…")                     # drop trailing … / .
    if _norm_url(ext).startswith(_norm_url(token)[:12]):  # same URL, just truncated
        return text.replace(m.group(0), f"[{m.group(0)}]({ext})")
    return text


_OTHER_PLATFORMS = ["facebook.com", "fb.com", "twitter.com", "x.com", "instagram.com",
                    "tiktok.com", "reddit.com", "youtube.com", "youtu.be", "threads.net",
                    "mastodon", "linkedin.com", "t.me"]


def _non_bluesky_url(text: str) -> str:
    """If the pasted text points at a non-Bluesky platform, return that platform name (else '').
    Used to prompt the user that the scanner works on Bluesky posts only."""
    t = (text or "").lower()
    if "bsky.app" in t or "bsky.social" in t:      # explicitly Bluesky → fine
        return ""
    for p in _OTHER_PLATFORMS:
        if p in t:
            return p.split(".")[0] if "." in p else p
    return ""


_MISINFO_TIPS = """
Wording that should make you look closer (these are *cues to check*, not proof of anything):
- **Vague conspiracy** — "they're hiding the truth", "wake up", with no specific who / what / where.
- **Unfalsifiable** — phrased so that no evidence could ever disprove it.
- **Urgency & shouting** — "BREAKING", "PROOF INSIDE", "share before it's deleted".
- **No source** — a strong factual claim with nothing to click through to.
- **Cherry-picked stat** — one number used to wave away a broader trend.

None of these make a post false — they mean *verify before you trust or share*. Open the original post
and the news sources below and judge for yourself.
"""


_HEALTH_ICON = {"ok": "🟢", "stale": "🟡", "fail": "🔴"}


def _age_label(ts: str) -> str:
    a = age_hours(ts or "")
    if a is None:
        return "?"
    return f"{a:.0f}h ago" if a < 48 else f"{a/24:.0f}d ago"


def _render_health_sidebar():
    """Compact maintenance-health readout in the sidebar: last outcome + age per stage.
    Reads data/health.json — written by the local ingestion cycle and the Colab notebook —
    so a stale corpus or a failed run is visible instead of silently serving old data."""
    health = load_health()
    st.sidebar.markdown("**⚙️ Maintenance health**")
    if not health:
        st.sidebar.caption("No runs recorded yet.")
        return
    for stage in ("ingestion", "classification", "evaluation", "vision", "refresh"):
        rec = health.get(stage)
        if not rec:
            continue
        status = stage_status(rec)
        line = f"{_HEALTH_ICON.get(status, '⚪')} {stage}: {_age_label(rec.get('ts', ''))}"
        if status == "fail" and rec.get("error"):
            line += f" — {str(rec['error'])[:38]}"
        st.sidebar.caption(line)


def _eval_set_status() -> dict:
    """Benchmark freshness — three numbers that SHOULD agree and silently don't:
      local_n     — rows in the working-copy `claim_eval.csv` (what your relabels wrote)
      committed_n — rows in the COMMITTED csv (what Colab clones and actually evaluates)
      last_eval_n — `n` on the newest drift-log entry (what the last eval really ran on)
    A relabel only reaches the drift chart once it is BOTH committed AND re-evaluated on the GPU."""
    local_n = committed_n = last_eval_n = None
    try:
        with open(EVAL_CSV, newline="", encoding="utf-8") as f:
            local_n = sum(1 for _ in csv.DictReader(f))
    except Exception:
        pass
    try:
        out = subprocess.run(["git", "show", f"HEAD:{EVAL_CSV.as_posix()}"],
                             capture_output=True, text=True, encoding="utf-8", timeout=5)
        if out.returncode == 0:
            committed_n = sum(1 for _ in csv.DictReader(io.StringIO(out.stdout)))
    except Exception:
        pass                       # not a git checkout / git unavailable — just skip this check
    hist = load_eval_history()
    if hist:
        try:
            last_eval_n = int(hist[-1].get("n") or 0) or None
        except Exception:
            pass
    return {"local_n": local_n, "committed_n": committed_n, "last_eval_n": last_eval_n}


def _render_eval_freshness():
    """Is the drift number on screen actually measured on the CURRENT benchmark? A relabel appends
    to `claim_eval.csv` immediately, but the eval only re-runs on the Colab GPU pass, and that pass
    only ever sees the COMMITTED csv. Both gaps are invisible from the UI — surface them."""
    s = _eval_set_status()
    local_n, committed_n, last_n = s["local_n"], s["committed_n"], s["last_eval_n"]
    if not local_n:
        return
    msgs = []
    if last_n and local_n != last_n:
        msgs.append(f"📊 The benchmark now has **{local_n}** labeled posts, but the last evaluation ran "
                    f"on **{last_n}** — **{abs(local_n - last_n)} label(s) added since**. The numbers "
                    "below are stale for the current set: **re-run the Colab GPU maintenance pass** to "
                    "refresh them (evaluation is GPU-only; the app can't run it).")
    if committed_n is not None and local_n != committed_n:
        msgs.append(f"⚠️ **{local_n - committed_n} relabel(s) are not committed.** Colab clones the "
                    f"benchmark from git, so it would evaluate **{committed_n}** rows — your newest "
                    "labels won't count until you commit + push `data/claim_eval.csv`.")
    if msgs:
        st.warning("\n\n".join(msgs))
    elif last_n:
        st.success(f"✅ Benchmark in sync — {local_n} labeled posts, matching the last evaluation.")


def _render_drift():
    """Model-drift panel, split by eval set — the whole point is that these two are NOT one series:
      GOLD    (frozen)  — data constant, so a move = the MODEL or ENVIRONMENT changed. The control.
      DYNAMIC (growing) — relabels append, so a move = model change + distribution change = concept drift.
    Reading them together is what made the old single chart uninterpretable. Reads
    data/eval_history.jsonl (one line per set per maintenance eval)."""
    hist = load_eval_history()
    if not hist:
        st.info("📈 **Drift log is empty.** Each evaluation run (the Colab GPU maintenance pass) appends "
                "one line per eval set — recall / precision / FN / FP / n / model digest — so you can "
                "watch for classifier drift over time.")
        return
    df = pd.DataFrame(hist)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return
    # Entries logged before the gold/dynamic split ran on the frozen 100 — that IS the gold set.
    if "eval_set" not in df.columns:
        df["eval_set"] = "gold"
    df["eval_set"] = df["eval_set"].fillna("gold")

    gold, dyn = df[df["eval_set"] == "gold"], df[df["eval_set"] == "dynamic"]
    latest = df.iloc[-1]
    target = float(latest.get("target", CLAIM_RECALL_TARGET) or CLAIM_RECALL_TARGET)

    st.markdown("#### 📈 Model drift — evaluation history")
    st.caption(f"Last evaluated **{_age_label(str(latest['ts']))}** on `{latest.get('model','?')}` "
               f"· {len(gold)} gold run(s) · {len(dyn)} dynamic run(s).")

    # ── GOLD: the control. Any movement here is the model/env, because the data never changes. ──
    st.markdown("##### 🥇 Gold set — the frozen control")
    if gold.empty:
        st.caption("No gold runs logged yet.")
    else:
        g, gp = gold.iloc[-1], (gold.iloc[-2] if len(gold) > 1 else gold.iloc[-1])
        multi = len(gold) > 1
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Recall (CLAIM)", f"{g['recall']:.3f}",
                  delta=f"{g['recall']-gp['recall']:+.3f}" if multi else None)
        c2.metric("Precision", f"{g['precision']:.3f}",
                  delta=f"{g['precision']-gp['precision']:+.3f}" if multi else None)
        c3.metric("FN / FP", f"{int(g['false_negatives'])} / {int(g['false_positives'])}",
                  help="False negatives (missed claims — costly) vs false positives (opinions surfaced — cheap)")
        c4.metric("n (frozen)", int(g.get("n", 0) or 0),
                  help="The gold set never grows — if this number moves, something is wrong.")
        c5.metric("Meets target", "✅" if bool(g.get("meets_target")) else "❌", help=f"recall ≥ {target:.0%}")
        if len(gold) > 1:
            st.line_chart(gold.set_index("ts")[["recall", "precision"]])
        st.caption("**The data here never changes**, so any move is the **model or the environment** — "
                   "a silent model re-pull, a new Ollama runtime, different hardware, or the classifier's "
                   "known nondeterminism. This is your ± jitter band: small wobble = noise; a sustained "
                   "step = a real regression. Check `model_digest` in the raw log before calling it drift.")

    # ── DYNAMIC: grows with the relabel loop, so it answers a different question. ──
    st.markdown("##### 📚 Dynamic set — grows with the relabel loop")
    if dyn.empty:
        st.caption("No dynamic runs logged yet.")
    else:
        d, dp = dyn.iloc[-1], (dyn.iloc[-2] if len(dyn) > 1 else dyn.iloc[-1])
        multi = len(dyn) > 1
        n_now, n_prev = int(d.get("n", 0) or 0), int(dp.get("n", 0) or 0)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Recall (CLAIM)", f"{d['recall']:.3f}",
                  delta=f"{d['recall']-dp['recall']:+.3f}" if multi else None)
        c2.metric("Precision", f"{d['precision']:.3f}",
                  delta=f"{d['precision']-dp['precision']:+.3f}" if multi else None)
        c3.metric("FN / FP", f"{int(d['false_negatives'])} / {int(d['false_positives'])}")
        c4.metric("Benchmark (n)", n_now, delta=(n_now - n_prev) if multi and n_now != n_prev else None,
                  help="Grows as the admin relabels hard cases — that growth is what makes this a "
                       "true concept-drift signal rather than a regression check.")
        c5.metric("Meets target", "✅" if bool(d.get("meets_target")) else "❌", help=f"recall ≥ {target:.0%}")
        if len(dyn) > 1:
            st.line_chart(dyn.set_index("ts")[["recall", "precision"]])
        st.caption("This set **gets harder over time** by design, so a dip here is not automatically a "
                   "regression. **Read it against gold:** if gold holds and this falls, the model isn't "
                   "generalizing to the newly relabeled cases — *genuine concept drift*. If both fall "
                   "together, it's the model/environment, not the data.")

    st.info("🔍 **How to read these two:** gold moving = **model/env**. Only dynamic moving = "
            "**concept drift**. Both moving = model/env (gold proves it). Neither = stable. "
            "That separation is the entire reason the sets are kept apart — one merged number can "
            "never distinguish these cases.")

    fnfp = df.set_index("ts")[["false_negatives", "false_positives"]].rename(
        columns={"false_negatives": "FN (missed — costly)", "false_positives": "FP (surfaced — cheap)"})
    st.bar_chart(fnfp)

    with st.expander("Raw drift log"):
        st.caption("`model_digest` / `ollama_version` are the first thing to check when **gold** moves: "
                   "Colab re-installs Ollama and re-pulls the model every run, both unpinned, so a "
                   "changed digest means the model itself changed — not that your classifier drifted.")
        cols = ["ts", "eval_set", "model", "model_digest", "ollama_version", "n", "recall", "precision",
                "f1", "accuracy", "false_negatives", "false_positives", "meets_target"]
        st.dataframe(df[[c for c in cols if c in df.columns]].iloc[::-1],
                     use_container_width=True, hide_index=True)
    st.divider()


def _render_trust(store, post: dict, classify_first: bool):
    """Full trust panel for one Bluesky post: header + links → shared assessment body.
    Surfaces signals; never asserts truth."""
    text = post["text"]
    st.markdown(f"**Post:** {_linkify_post(text, post.get('external_url', ''))}")
    url = _bsky_url(post.get("post_id", ""))
    if url:
        st.markdown(f"🔗 [Open the original Bluesky post to verify]({url})  ·  @{post.get('author','')}")
    # The URL in the post text is Bluesky's TRUNCATED display link (…/2026…) and breaks when
    # clicked — surface the full embed URL as a working link.
    ext = (post.get("external_url") or "").strip()
    if ext.startswith("http"):
        shown = ext if len(ext) <= 72 else ext[:72] + "…"
        st.markdown(f"📎 **Linked article (full URL):** [{shown}]({ext})")

    # Classification (fresh for a pasted post; stored label for a picked one)
    if classify_first:
        with st.spinner("Classifying claim vs opinion…"):
            cl = classify(text, model=MODEL)
        is_claim, reason = cl["has_claim"], cl["reason"]
    else:
        is_claim, reason = post.get("has_claim", 1) == 1, post.get("reason", "")

    vision = None
    if post.get("vision_signal"):
        try:
            vision = json.loads(post["vision_signal"])
        except Exception:
            vision = None

    _render_assessment_body(
        store, text=text, engagement=int(post.get("engagement", 0)), source="bluesky",
        followers=post.get("author_followers", 0) or 0, author=post.get("author", "") or "",
        external_url=post.get("external_url", "") or "", vision=vision,
        is_claim=is_claim, reason=reason)


def _render_assessment_body(store, *, text: str, engagement: int, source: str, followers: int,
                            author: str, external_url: str, vision: dict | None,
                            is_claim: bool, reason: str):
    """Shared assessment render used by BOTH the Bluesky-link path and the uploaded-screenshot
    path: classification verdict → missed-claim nomination → reader signal → retrieved evidence.
    Source-agnostic — the only difference between the two paths is the header their callers draw
    and the `source` they pass (which drives the unverified-source wording and the red flag)."""
    (st.info if is_claim else st.warning)(
        f"**Classification: {'CLAIM' if is_claim else 'OPINION'}** — {reason}")
    if not is_claim:
        st.caption("Opinions have no specific factual event to corroborate on their own — but strong "
                   "evidence below can flag a claim the classifier missed (see the check below).")

    with st.spinner("Retrieving news + checking corroboration…"):
        a = assess_claim(store, text, engagement=int(engagement), source=source,
                         followers=followers or 0, author=author or "", vision=vision, cfg=cfg,
                         external_url=external_url or "")
    sig, corro = a["signal"], a["corroboration"]

    # Runtime relabel nomination: an OPINION whose evidence contradicts the label (news covers the
    # event/topic, or it's an official / credibly-cited source) is likely a claim the classifier
    # missed. Surface it to the reader — this NEVER auto-relabels; the same signal feeds the eval-set
    # relabel queue (see MONITORING.md "evidence-nominated relabel candidates").
    if not is_claim:
        why = []
        if sig.get("news_status") == "REPORTED":
            why.append("independent news reports this event")
        if sig.get("treated_official"):
            why.append("it comes from a verified official source")
        if sig.get("credible_cite"):
            why.append("it links its own credible source")
        if why:
            st.warning("🔎 **Possibly a missed claim.** The classifier labeled this an **OPINION**, but "
                       + ", and ".join(why) + " — you may want to judge it as a **claim**. "
                       "(The classifier is recall-first; forecasts and evidence-backed statements are its "
                       "known weak spot.)")

    # ── THE SIGNAL — the clear reading for the reader, up top ──
    st.markdown("### 🧭 What the scanner is telling you")
    if sig["red_flag"]:
        st.error("🚩 **RED FLAG** — spreading widely, no news corroboration, unverified source, no cited "
                 "evidence. This is the misinformation-amplification pattern — **verify before you trust "
                 "or share.**")
    # Bold only the leading label (Source:, Reach:, …); the answer stays normal weight. Sentences
    # with no short "Label:" prefix (the reframe/red-flag lines) render plain.
    for b in sig.get("bullets", []):
        label, sep, rest = b.partition(": ")
        if sep and len(label) <= 40:
            st.markdown(f"- **{label}:** {rest}")
        else:
            st.markdown(f"- {b}")
    _ns = sig.get("news_status", "NO MATCH")
    _STATUS_NOTE = {
        "REPORTED":    "a retrieved article appears to report this event — open it to confirm.",
        "TOPIC MATCH": "retrieved news covers the same topic/region — a **topic match is not claim support** for your specific claim.",
        "NO MATCH":    "no retrieved news is even topically close to this specific claim.",
    }
    st.caption(f"Independent news check: **{_ns}** — {_STATUS_NOTE.get(_ns, '')} "
               "(does *other* news independently report the same event — separate from any source the post itself links).")
    if vision:
        st.caption(f"🖼️ Image (edge-case vision): **{vision.get('image_type')}** · "
                   f"depicts_claim={vision.get('depicts_claim')} — {vision.get('description','')}")

    # ── The evidence the reading is based on. Header adapts to the verdict so a NONE verdict
    #    with a list below doesn't read as a contradiction: the list is the CLOSEST topical
    #    neighbours we searched (shown for transparency), not confirmations. ──
    matches = a["retrieval"]["matches"]
    cited = sig.get("cited")
    if not matches:
        st.markdown("#### 🔍 News search — nothing relevant found")
        st.caption("No news in the corpus was even topically close to this claim, so there's nothing to "
                   "show. The news set is limited, so absence here is not proof either way.")
    else:
        if _ns == "REPORTED":
            st.markdown("#### 📰 News the scanner retrieved (RAG) — open them to verify")
            st.caption("The **number on the left** is a topic-similarity score (0–1): cosine similarity "
                       "between your claim and the article headline. Higher = closer wording/topic — it is "
                       "**not** proof they describe the same event (that's what the verdict above judges).")
        elif _ns == "TOPIC MATCH":
            st.markdown("#### 🔍 Related news on this topic — open to judge for yourself")
            st.caption("Shown for transparency: these are the nearest headlines by wording/topic. They "
                       "cover the same **subject**, but none is confirmed to report **this specific claim**. "
                       "The **number on the left** is the topic-similarity score (0–1) — close in topic is "
                       "not the same as reporting your exact claim.")
        elif sig.get("region_mismatch"):
            others = ", ".join(r.split(",")[0] for r in sig.get("other_regions", [])[:3])
            st.markdown("#### 🔍 Same topic — but from **other regions**")
            st.caption(f"These cover the same subject, but they're from **{others}** — not your claim's "
                       "region — so they likely report a *different* event. Shown for transparency; the "
                       "**number on the left** is topic-similarity (0–1), which doesn't account for region.")
        else:  # NO MATCH — only weak/noise neighbours cleared the display floor
            st.markdown("#### 🔍 Nearest headlines the search found — likely **not** about your claim")
            st.caption("**No real match was found.** These are just the closest headlines by wording — the "
                       "similarity is weak, so they're probably about a *different* subject (shown only for "
                       "transparency). The **number on the left** is the topic-similarity score (0–1).")
        claim_loc = a["retrieval"].get("location")
        if claim_loc:
            st.caption(f"📍 Region-aware retrieval: read your claim as **{claim_loc}** and folded that into "
                       "the search, so same-region news ranks higher.")
        for m in matches:
            mark = "  ← **closest match**" if cited and m["url"] == cited.get("url") else ""
            u = m["url"] if str(m["url"]).startswith("http") else ""
            title = f"[{m['title'][:90]}]({u})" if u else m["title"][:90]
            loc = f" · 📍 {m['location']}" if m.get("location") else ""
            st.markdown(f"- `{m['similarity']:.3f}` · **{m['domain']}**{loc} · {m.get('date','')} · {title}{mark}")


def _render_image_trust(store, extracted: dict, image_bytes: bytes, extra_text: str = ""):
    """Week 7 image-INPUT path: show the uploaded screenshot, GATE it to real social posts (must
    show engagement), then run the SAME assessment body as the Bluesky path. There is no canonical
    post to open, so the image itself + any visible citation stand in for the link, and the whole
    reading is labelled degraded-fidelity (best-effort OCR of off-platform content)."""
    st.image(image_bytes, caption="Uploaded screenshot", use_container_width=True)

    # ── Gate: must be a social-media post (visible engagement). Reject anything else with guidance. ──
    if not looks_like_social_post(extracted):
        st.error("🚫 **This isn't a social-media post the scanner can assess.** It found **no visible "
                 "engagement** (likes / reposts / comments). The scanner surfaces the **reach-vs-support** "
                 "mismatch of posts, so it needs a screenshot of an actual post showing its engagement. "
                 "Bare photos, satellite/comparison images, infographics, memes and illustrations aren't "
                 "accepted — upload a screenshot of the post itself, including its like/repost/comment counts.")
        with st.expander("What the vision model saw (why it was rejected)"):
            st.markdown(f"**Image type:** {extracted.get('image_type')} · "
                        f"platform: {extracted.get('platform') or '—'}")
            if extracted.get("claim_text"):
                st.markdown(f"**Text read:** {extracted['claim_text']}")
            if extracted.get("description"):
                st.caption(f"Description: {extracted['description']}")
            st.caption("No like / repost / comment count was found, so this doesn't qualify as a post.")
        return

    st.info("🖼️ **Degraded-fidelity path.** This is an off-platform screenshot: the text was read by a "
            "vision model (OCR) and there's no original post to open and verify. Treat the transcription "
            "as best-effort and these signals as **lower-confidence** than the Bluesky-link path.")

    with st.expander("What the vision model read from the image", expanded=True):
        eng = extracted.get("engagement") or {}
        def _n(v):
            return v if v is not None else "—"
        c1, c2 = st.columns(2)
        c1.markdown(f"**Platform:** {extracted.get('platform') or '—'}")
        c1.markdown(f"**Author:** {extracted.get('author_handle') or '—'}")
        c2.markdown(f"**Image type:** {extracted.get('image_type')}  ·  depicts_claim={extracted.get('depicts_claim')}")
        c2.markdown(f"**Engagement (read):** ❤️ {_n(eng.get('likes'))} · 🔁 {_n(eng.get('reposts'))} · "
                    f"💬 {_n(eng.get('replies'))}")
        if extracted.get("visible_citation"):
            st.markdown(f"**Visible citation:** {extracted['visible_citation']}")
        if extracted.get("description"):
            st.caption(f"Description: {extracted['description']}")
        st.caption("Fields shown as **—** were not clearly legible — the model is told to return null "
                   "rather than guess (a fabricated handle or count would be the worst failure).")

    # The claim = what OCR read, plus any post text the user pasted (for a screenshot that truncates
    # a long post with a 'See more'). The engagement/source/vision still come from the image.
    claim_text = (extracted.get("claim_text") or "").strip()
    if extra_text:
        claim_text = (claim_text + " " + extra_text).strip()
    if not claim_text:
        st.warning("📄 **Couldn't read a claim from this post.** No legible claim text was found and no "
                   "text was added. If the post's text is cut off, paste it in the box above and re-run.")
        return

    if extra_text:
        st.caption("📝 Using the screenshot text **plus** the post text you added.")
    st.markdown(f"**Transcribed claim:** {claim_text}")
    inp = screenshot_signal_inputs(extracted)
    with st.spinner("Classifying claim vs opinion…"):
        cl = classify(claim_text, model=MODEL)
    _render_assessment_body(
        store, text=claim_text, engagement=inp["engagement"], source="uploaded_screenshot",
        followers=0, author=inp["author"], external_url=inp["external_url"], vision=inp["vision"],
        is_claim=cl["has_claim"], reason=cl["reason"])


def _admin_authed() -> bool:
    """Gate the Maintenance tab. Admin password comes from `st.secrets['ADMIN_PASSWORD']` or the
    `ADMIN_PASSWORD` env var (.env). Not full auth — enough to keep relabel out of users' hands."""
    if st.session_state.get("is_admin"):
        return True
    import os
    pw = None
    try:
        pw = st.secrets.get("ADMIN_PASSWORD")
    except Exception:
        pw = None
    pw = pw or os.environ.get("ADMIN_PASSWORD")
    if not pw:
        st.error("🔒 Admin password not configured. Set `ADMIN_PASSWORD` in `.env` (or "
                 "`.streamlit/secrets.toml`) to use the Maintenance tab.")
        return False
    entered = st.text_input("Admin password", type="password", key="admin_pw")
    if entered:
        if entered == pw:
            st.session_state["is_admin"] = True
            st.rerun()
        st.error("Incorrect password.")
    return False


_ADD_NEW = "➕ Add new thought…"


def _render_relabel_section(title: str, items: list, corrected_label: str, note: str = ""):
    st.subheader(title)
    if note:
        st.caption(note)
    if not items:
        st.caption("None in the scanned set. ✅")
        return
    done = st.session_state.setdefault("relabel_done", {})   # post_id -> inline "done" message
    types = eval_post_types(str(EVAL_CSV))                   # FRESH read every rerun — never a stale list
    for c in items:
        pid = c["post_id"]
        with st.container(border=True):
            st.markdown(f"**Post:** {_linkify_post(c['text'][:400], c.get('external_url', ''))}")
            links = []
            bsky = _bsky_url(pid)
            if bsky:
                links.append(f"🔗 [Open the Bluesky post]({bsky})")
            ext = (c.get("external_url") or "").strip()
            if ext.startswith("http"):
                links.append(f"📎 [Linked article]({ext})")
            if links:
                st.markdown(" · ".join(links))

            # Already actioned this session — show a small confirmation, keep it in place, move on.
            if pid in done:
                st.success(done[pid])
                continue

            cur = "CLAIM" if c["has_claim"] else "OPINION"
            st.caption(f"Currently **{cur}** · @{c['author']} · {c['keyword_category']} · ❤️ {c['engagement']} "
                       f"· news: **{c['news_status']}**" + (f" · why: {c['why']}" if c['why'] else ""))
            # The "thought" (post_type) written to the eval set — pick an existing one for consistency,
            # or add a new category. Notes = the specific reason for this label.
            pt = st.selectbox("Thought (post_type)", types + [_ADD_NEW], key=f"pt_{pid}",
                              help="The reasoning category this post exemplifies — keeps hand-labels consistent.")
            if pt == _ADD_NEW:
                pt = st.text_input("New thought name (snake_case)", key=f"ptn_{pid}").strip()
            notes = st.text_input("Notes (why this label)", key=f"nt_{pid}",
                                  placeholder="e.g. Official source + specific verifiable comparison")
            col1, col2 = st.columns(2)
            if col1.button(f"✅ Relabel as {corrected_label.upper()}", key=f"rl_{pid}", type="primary"):
                if not pt:
                    st.warning("Pick or enter a thought (post_type) before relabeling.")
                else:
                    apply_relabel(DB_PATH, str(EVAL_CSV), pid, c["text"], corrected_label,
                                  c["keyword_category"], post_type=pt, notes=notes)
                    done[pid] = (f"✅ Label shifted to **{corrected_label.upper()}**. Appended to the "
                                 f"**dynamic** eval set (`{EVAL_CSV.name}`) as a `{corrected_label}` example "
                                 f"· thought `{pt}` — users now see the corrected label, and it will shape "
                                 "the next evaluation run. (The **gold** control set is never touched, so it "
                                 "stays comparable.) Scroll down to review the next one.")
                    st.rerun()   # re-render THIS card as done; queue + position preserved (no rescan)
            if col2.button("↩ Keep as is (model was right)", key=f"keep_{pid}"):
                mark_reviewed_ok(DB_PATH, pid, c["has_claim"])
                done[pid] = "↩ Kept as is — marked reviewed, nothing changed. Scroll down to the next one."
                st.rerun()


def _render_static_eval():
    """Static point-in-time evaluation: run the classifier on ONE of the two labeled sets and show
    the confusion matrix, per-class metrics, and error breakdowns. Admin diagnostic — a local run is
    slow; the canonical eval is the Colab GPU maintenance pass, which is what feeds the drift chart."""
    st.subheader("📏 Static evaluation — hand-labeled set")
    choice = st.radio(
        "Which eval set?", ["🥇 Gold (frozen control)", "📚 Dynamic (grows with relabels)"],
        horizontal=True, key="static_eval_set",
        help="Gold never changes, so it isolates model/env regressions. Dynamic grows with the "
             "relabel loop, so it measures concept drift on the new hard cases.")
    is_gold = choice.startswith("🥇")
    target_csv = GOLD_EVAL_CSV if is_gold else EVAL_CSV
    st.caption(
        f"Point-in-time run of the classifier against `{target_csv}` (human ground truth). "
        + ("**Gold = the frozen control** — the data never changes, so any movement is the model or "
           "the environment, not the benchmark. "
           if is_gold else
           "**Dynamic = the growing set** — it gets harder as you relabel, so a dip here isn't "
           "automatically a regression; read it against gold. ")
        + f"Success criterion: **recall on CLAIM ≥ {CLAIM_RECALL_TARGET:.0%}** — a missed claim (false "
        "negative) is discarded forever (costly); a false positive just surfaces on the dashboard (cheap).")
    if not target_csv.exists():
        st.warning(f"`{target_csv}` not found.")
        return
    n_eval = len(load_eval_set(str(target_csv)))
    if st.button(f"Run evaluation ({n_eval} labeled posts)"):
        with st.spinner(f"Classifying {n_eval} posts with {MODEL}..."):
            eval_results = run_eval(str(target_csv), model=MODEL, llm_batch_size=LLM_BATCH_SIZE)
            metrics = compute_metrics(eval_results, claim_recall_target=CLAIM_RECALL_TARGET)
        st.caption("🔬 Local diagnostic run — not added to the drift log (the drift series is produced by "
                   "the Colab GPU maintenance pass, for backend-comparable numbers).")

        claim_m = metrics["per_class"]["claim"]
        asym    = metrics["error_asymmetry"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recall on CLAIM", f"{claim_m['recall']:.3f}", delta=f"target ≥ {CLAIM_RECALL_TARGET:.2f}",
                  delta_color="normal" if metrics["meets_target"] else "inverse")
        m2.metric("Precision on CLAIM", f"{claim_m['precision']:.3f}")
        m3.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m4.metric("FN / FP", f"{asym['false_negatives']} / {asym['false_positives']}",
                  help="False negatives (missed claims — costly) vs false positives (opinions surfaced — cheap)")

        if metrics["meets_target"]:
            st.success(f"✅ PASS — the gate lets through {claim_m['recall']:.1%} of real claims "
                       f"(misses {asym['fn_rate']:.1%}); {asym['fp_rate']:.1%} of opinions slip through.")
        else:
            st.error(f"❌ BELOW TARGET — {asym['fn_rate']:.1%} of real claims are being discarded at the "
                     "gate. See the post-type breakdown for where.")

        cm = metrics["confusion_matrix"]
        col_cm, col_pc = st.columns(2)
        with col_cm:
            st.markdown("**Confusion matrix** (positive class = claim)")
            st.dataframe(pd.DataFrame({
                "predicted claim":   [cm["true_claim_predicted_claim"], cm["true_opinion_predicted_claim"]],
                "predicted opinion": [cm["true_claim_predicted_opinion"], cm["true_opinion_predicted_opinion"]],
            }, index=["actual claim", "actual opinion"]), use_container_width=True)
        with col_pc:
            st.markdown("**Per-class metrics**")
            st.dataframe(pd.DataFrame(metrics["per_class"]).T.round(3), use_container_width=True)

        st.markdown("**Error breakdown** — where the misses are concentrated")
        col_pt, col_kc = st.columns(2)
        with col_pt:
            st.caption("By post type (linguistic shape)")
            st.dataframe(pd.DataFrame(metrics["by_post_type"]).T.style.format({"error_rate": "{:.1%}"}),
                         use_container_width=True)
        with col_kc:
            st.caption("By keyword category (ingestion taxonomy)")
            st.dataframe(pd.DataFrame(metrics["by_keyword_category"]).T.style.format({"error_rate": "{:.1%}"}),
                         use_container_width=True)

        with st.expander(f"Misclassified posts ({len(metrics['misclassified'])})"):
            if metrics["misclassified"]:
                for m in metrics["misclassified"]:
                    direction = ("🔴 FALSE NEGATIVE (missed claim)" if m["expected_label"] == "claim"
                                 else "🟡 FALSE POSITIVE (opinion surfaced)")
                    st.markdown(f"{direction} · `{m['post_type']}`")
                    st.markdown(f"> {m['post_text']}")
                    st.caption(f"Model reason: {m['reason']}")
                    st.divider()
            else:
                st.success("No misclassifications.")


def maintenance():
    st.title("🔧 Maintenance — Admin")
    if not _admin_authed():
        return
    st.success("Signed in as admin.")
    st.caption("**Relabel review queue.** Evidence-nominated label mismatches. Confirming a relabel writes "
               "an **admin override** (users see the corrected label on refresh) **and** appends the "
               "corrected `(post, label, thought)` to the eval benchmark (`claim_eval.csv`). Users can never relabel.")
    with st.expander("📖 Labeling guide — which *thought* (post_type) to pick"):
        st.markdown(
            "Pick the **reasoning category** the post exemplifies (not just claim/opinion) so hand-labels "
            "stay consistent. Choose an existing one when it fits; add a new one only for a genuinely new pattern.\n\n"
            "**CLAIM thoughts** — a specific, checkable assertion (true *or* false):\n"
            "- `news_event` — named place + specific measurement/event\n"
            "- `news_headline_verbatim` — the post text IS the headline of the credible article it "
            "links (an outlet posting its own story: reporting, not commentary). A known classifier "
            "blind spot — worth its own bucket so the error breakdown shows it.\n"
            "- `official_alert` — official source + verifiable warning/comparison\n"
            "- `scientific_finding` — specific superlative + named period/metric\n"
            "- `false_but_checkable` — specific mechanism/effect/location, structurally a claim *even if false*\n"
            "- `denial_with_stat` — a denial that cites a checkable stat/source\n"
            "- `mixed_emotion_fact` — emotion wrapped around a verifiable fact/warning\n\n"
            "**OPINION thoughts** — no specific checkable assertion:\n"
            "- `emotional_reaction` — feeling, no factual content\n"
            "- `political_viewpoint` — stance, no verifiable claim\n"
            "- `hyperbole_doom` — prediction without sourcing/evidence\n"
            "- `vague_conspiracy` — accusation with no specific evidence\n"
            "- `sarcasm_joke` / `rhetorical_question` — nothing asserted\n\n"
            "**`real_data`** — provenance tag for real-ingested posts; use it only if none of the reasoning "
            "types fit. **Notes** = the one-line *why* (e.g. *“official source + specific verifiable comparison”*).")
    scan = st.slider("Posts to scan (top by engagement)", 40, 300, 100, step=20)
    if st.button("🔍 Scan for relabel candidates", type="primary"):
        with st.spinner(f"Assessing the top {scan} posts…"):
            st.session_state["relabel_cands"] = get_relabel_candidates(_trust_store(), DB_PATH, cfg, scan_limit=scan)
        st.session_state["relabel_done"] = {}          # fresh scan → clear this session's done markers
    cands = st.session_state.get("relabel_cands")
    if cands is None:
        st.info("Click **Scan** to find candidates.")
        return
    all_ids = [x["post_id"] for x in cands["opinion_to_claim"] + cands["claim_to_opinion"]]
    done = st.session_state.get("relabel_done", {})
    reviewed = sum(1 for pid in all_ids if pid in done)
    remaining = len(all_ids) - reviewed
    st.markdown(f"**{remaining} awaiting review**"
                + (f" · {reviewed} reviewed this session ✅" if reviewed else "")
                + "  ·  re-scan any time to refresh the list.")
    st.divider()
    _render_relabel_section(
        "🟠 Opinions that look like CLAIMS", cands["opinion_to_claim"], "claim",
        "Classified OPINION but evidence contradicts it (news covers it / official / credibly cited) — "
        "likely missed claims (the costly false negatives).")
    st.divider()
    _render_relabel_section(
        "🔵 Claims with no supporting evidence", cands["claim_to_opinion"], "opinion",
        "Classified CLAIM but nothing corroborates it. **Lower confidence** — the headline-only corpus "
        "means real claims often lack a match, so review carefully.")

    # ── Lane 2: signal-ranked, any reach (benchmark growth) ──────────────────────────────────
    # The queue above ranks by ENGAGEMENT, which is right for the red-flag product but wrong for
    # growing the eval set: the classifier's blind spots don't correlate with likes, so a 0-like
    # wire headline it misreads is as valuable an example as a 295-like one. This lane sweeps the
    # WHOLE corpus with cheap text/metadata signals (no retrieval) and ignores reach entirely.
    st.divider()
    st.subheader("🔬 Signal-nominated — any reach")
    st.caption("Sweeps **every** post (not just the top-N by engagement) for label contradictions the "
               "classifier's blind spots hide at low reach. Ranked by **signal strength**; engagement "
               "is only a tie-break. This is the lane that grows the benchmark.")
    if st.button("🔬 Sweep the whole corpus by signal", type="primary"):
        with st.spinner("Sweeping all posts (cheap signals — no retrieval)…"):
            st.session_state["signal_cands"] = get_signal_candidates(DB_PATH, cfg, store=_trust_store())
        st.session_state.setdefault("relabel_done", {})
    sc = st.session_state.get("signal_cands")
    if sc is None:
        st.info("Click **Sweep** to nominate by signal instead of reach.")
    else:
        st.markdown(f"**{len(sc['strong'])} strong** · {len(sc['weak'])} weak (lower confidence).")
        _render_relabel_section(
            "🟢 Strong signal — an outlet posting its own story", sc["strong"], "claim",
            "Classified OPINION, but the post text IS the headline of the credible article it links, "
            "or it comes from an official account — that's reporting, not commentary. The highest-value "
            "missed claims, at ANY reach.")
        st.divider()
        _render_relabel_section(
            "🟡 Weak signal — links a credible source", sc["weak"], "claim",
            "Classified OPINION and it cites a credible source. **Noisy on purpose:** sharing an article "
            "with a vibes caption is legitimately an OPINION, so many of these are the model being RIGHT. "
            "A backlog to mine, not a queue to clear.")

    st.divider()
    # Relabeling is what grows the benchmark, so this is where the admin most needs to see that a
    # relabel hasn't counted yet (uncommitted, or no GPU eval since).
    _render_eval_freshness()
    st.divider()
    _render_static_eval()   # point-in-time eval lives here (admin); the drift trend is in Evaluation


# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Claim Scanner",
    page_icon="🌪️",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER — Trust Checker: the user-facing product (check a Bluesky post, get signals)
# ═══════════════════════════════════════════════════════════════════════════════
def trust_checker():
    st.subheader("🛡️ Climate Claim Scanner — Trust Checker")
    st.markdown(
        "**What this does**\n"
        "- Scans **Bluesky** posts about climate & extreme weather.\n"
        "- Tells you whether a post is a checkable **claim** or just an **opinion**.\n"
        "- Checks whether **published news** reports the same event — and shows you those sources.\n"
        "- Flags the **reach-vs-support mismatch**: a post spreading fast with no news backing.\n\n"
        "**What this does NOT do**\n"
        "- It does **not** decide whether a post is true or false — *you* judge, using the signals.\n"
        "- It works on **Bluesky posts only** (not Facebook, X, or other platforms).\n"
        "- It matches claims against news **headlines, not full article text** — so it surfaces *related "
        "coverage* and the *reach-vs-support* signal, **not** a line-by-line fact-check of the exact claim."
    )
    with st.expander("🚩 How to read a red-flagged post — before you trust or share", expanded=True):
        st.markdown(_MISINFO_TIPS)

    store = _trust_store()
    if store.count() == 0:
        st.warning("Evidence index is empty — build it on the **Evidence Matching (RAG)** page "
                   "(sidebar → Course demos).")
        return

    tab_paste, tab_list, tab_photo = st.tabs(
        ["📋 Paste a post", "🏆 Pick from top posts", "🖼️ Upload a screenshot"])

    # ── Mode 1 — paste the LINK of a Bluesky post; we fetch its metadata ──
    with tab_paste:
        link = st.text_input("Paste the **bsky.app link** of the post you want assessed",
                             key="tc_link",
                             placeholder="https://bsky.app/profile/…/post/…   (Bluesky only)")
        st.caption("Paste the link — the scanner fetches the post's text, engagement, author and image "
                   "straight from Bluesky. Nothing to type by hand.")
        if st.button("View results ▸", type="primary", key="tc_paste_go"):
            other = _non_bluesky_url(link)
            if other:
                st.error(f"That looks like a **{other.capitalize()}** link. This scanner works on "
                         "**Bluesky posts only** — paste a Bluesky post link (bsky.app/…).")
            elif "bsky.app" not in link.lower() and not link.strip().startswith("at://"):
                st.warning("Paste a Bluesky post link, e.g. `https://bsky.app/profile/…/post/…`")
            else:
                with st.spinner("Fetching the post from Bluesky…"):
                    post = fetch_post_by_url(link.strip())
                if not post:
                    st.error("Couldn't fetch that post — check it's a valid, public Bluesky post link.")
                else:
                    post["engagement"] = post["likes"] + post["reposts"] + post["replies"] + post["quotes"]
                    post["classify_first"] = True
                    st.session_state["tc_result"] = {"posts": [post]}
                    st.switch_page(_page_results)

    # ── Mode 2 — pick one, many, or all from the day's top posts ──
    with tab_list:
        fc1, fc2 = st.columns([1, 1])
        sort_by = fc1.selectbox("Rank the day's top posts by", list(_SORT_LABEL),
                                format_func=lambda s: _SORT_LABEL[s], key="tc_sort")
        n_top = fc2.slider("How many top posts to choose from", 5, 50, 10, step=5, key="tc_n")
        claims = _load_posts(DB_PATH, 1, n_top, sort_by)
        opinions = _load_posts(DB_PATH, 0, n_top, sort_by)

        c1, c2 = st.columns(2)
        csel = c1.multiselect(f"Top CLAIMS ({len(claims)}) — pick one or more",
                              options=list(range(len(claims))),
                              format_func=lambda i: f"[{claims[i]['engagement']}] {claims[i]['text'][:55]}",
                              key="tc_cms")
        osel = c2.multiselect(f"Top OPINIONS ({len(opinions)}) — pick one or more",
                              options=list(range(len(opinions))),
                              format_func=lambda i: f"[{opinions[i]['engagement']}] {opinions[i]['text'][:55]}",
                              key="tc_oms")
        assess_all = st.checkbox("…or assess ALL shown claims + opinions", key="tc_all")

        if st.button("View results ▸", type="primary", key="tc_list_go"):
            if assess_all:
                chosen = ([dict(c, has_claim=1, classify_first=False) for c in claims] +
                          [dict(o, has_claim=0, classify_first=False) for o in opinions])
            else:
                chosen = ([dict(claims[i], has_claim=1, classify_first=False) for i in csel] +
                          [dict(opinions[i], has_claim=0, classify_first=False) for i in osel])
            if not chosen:
                st.warning("Pick at least one post, or check 'assess ALL shown'.")
            else:
                st.session_state["tc_result"] = {"posts": chosen}
                st.switch_page(_page_results)

    # ── Mode 3 — upload a SCREENSHOT of a social-media post (Week 7 image-input path) ──
    with tab_photo:
        st.caption("Encountered a climate/weather claim **off Bluesky** — on X, Facebook, Instagram, "
                   "LinkedIn or WhatsApp? Upload a **screenshot of the post**. A vision model reads the "
                   "text out of the image and the **same** pipeline assesses it (a **degraded-fidelity** "
                   "path: best-effort OCR, no original post to open).")
        st.warning("📱 **It must be a social-media post that shows its engagement** (likes / reposts / "
                   "comments). The scanner works on the post's **reach-vs-support** signal, so bare photos, "
                   "satellite/comparison images, infographics, memes and illustrations **aren't accepted** — "
                   "capture the whole post including its like/repost/comment counts.")
        up = st.file_uploader("Screenshot of a post — must show its engagement (PNG / JPG / WebP)",
                              type=["png", "jpg", "jpeg", "webp"], key="tc_img")
        extra = st.text_area("Post text, if the screenshot cuts it off (optional)", key="tc_img_text",
                             height=80, placeholder="Paste the full/remaining post text when the "
                             "screenshot truncates a long post (a 'See more'). Engagement still comes "
                             "from the image.")
        if st.button("Read image & assess ▸", type="primary", key="tc_img_go"):
            if not up:
                st.warning("Upload a screenshot of a social-media post first.")
            else:
                with st.spinner(f"Reading the image with the vision model ({VISION_MODEL})…"):
                    extracted = extract_from_image(up.getvalue(), model=VISION_MODEL,
                                                   corrector=VISION_CORRECTOR)
                st.session_state["tc_img_result"] = (
                    {"extracted": extracted, "img": up.getvalue(), "extra": extra.strip()}
                    if extracted else {"error": True})
        res = st.session_state.get("tc_img_result")
        if res:
            if res.get("error"):
                st.error(f"Couldn't read the image. The vision model **`{VISION_MODEL}`** may be "
                         "unavailable here — the image path needs Ollama with that model (GPU). "
                         "Confirm it's pulled and the Ollama server is reachable, then try again.")
            else:
                _render_image_trust(store, res["extracted"], res["img"], res.get("extra", ""))


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER — Results page (shown after 'View results')
# ═══════════════════════════════════════════════════════════════════════════════
def trust_results():
    st.subheader("🛡️ Scanner Results")
    if st.button("← Check another post"):
        st.session_state.pop("tc_result", None)
        st.switch_page(_page_trust)

    data = st.session_state.get("tc_result")
    if not data or not data.get("posts"):
        st.info("No results yet — go to **Trust Checker** in the sidebar and run a check.")
        return

    store = _trust_store()
    posts = data["posts"]
    if len(posts) > 1:
        st.caption(f"Assessing **{len(posts)}** posts — one corroboration call each; this can take a moment.")
    st.divider()
    for i, p in enumerate(posts, 1):
        cf = p.get("classify_first", False)
        if len(posts) == 1:
            _render_trust(store, p, classify_first=cf)
        else:
            with st.expander(f"#{i} · {p['text'][:75]}", expanded=(i == 1)):
                _render_trust(store, p, classify_first=cf)
        st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION EVALUATION (course demo)
# ═══════════════════════════════════════════════════════════════════════════════
def classification_eval():

    # Ingestion status bar
    last_ingest = get_last_ingestion_time(DB_PATH)
    if last_ingest:
        elapsed  = hours_since_last_ingestion(DB_PATH)
        interval = cfg["ingestion"]["interval_hours"]
        next_run = interval - elapsed
        if elapsed < interval:
            st.info(
                f"📡 Last ingestion: **{last_ingest.strftime('%Y-%m-%d %H:%M UTC')}** "
                f"({elapsed:.1f}h ago) · Next due in ~{next_run:.1f}h"
            )
        else:
            st.warning(
                f"⚠️ Last ingestion: **{last_ingest.strftime('%Y-%m-%d %H:%M UTC')}** "
                f"({elapsed:.1f}h ago) · Overdue — run the scheduler."
            )
    else:
        st.warning("⚠️ No ingestion has run yet. Start the scheduler to populate the database.")

    st.divider()

    # Pipeline stats. Only Bluesky posts are classifiable; GDELT rows are the evidence
    # corpus (retrieved, never labeled), so "pending" is over the Bluesky pool, not all posts.
    stats   = get_stats(DB_PATH)
    pending = max(0, stats.get("total_classifiable", 0) - stats["total_classified"])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📥 Bluesky posts",     stats.get("total_classifiable", 0),
                help=f"{stats.get('total_evidence', 0)} GDELT news articles are held separately as the evidence corpus.")
    col2.metric("🔍 Classified",        stats["total_classified"])
    col3.metric("✅ Claims Found",      stats["total_claims"])
    col4.metric("💬 Opinions Rejected", stats["total_opinions"])
    col5.metric("⏳ Pending",           pending)

    st.divider()

    # Classification STATUS — classification runs on the Colab GPU maintenance pass, not here.
    # A local run on qwen2.5:3b (CPU) is slow, blocks the app, and can time out, so this page only
    # reports whether a Colab classification pass is needed or was already done (from health.json).
    st.subheader("🔧 Classification status")
    _cls = load_health().get("classification")
    _last = (f"Last Colab classification: **{_age_label(_cls.get('ts',''))}** "
             f"({'✅ ok' if _cls.get('ok') else '🔴 failed'})." if _cls else
             "No Colab classification run recorded yet.")

    if stats["total_ingested"] == 0:
        st.warning("No posts in the database yet — run an ingestion cycle first.")
        st.code("uv run python -m climate_verifier.ingestion.scheduler --once --force")
    elif pending > 0:
        st.warning(f"⏳ **{pending} Bluesky posts awaiting classification.**")
        st.markdown(
            "Classification runs on the **Colab GPU** maintenance pass — the local model is too slow "
            "to run here without blocking the app. Push the DB to Drive and run the maintenance "
            "notebook (`classify → vision → reindex → evaluate`), or on a GPU machine:")
        st.code("python -m climate_verifier.maintenance --classify")
        st.caption(_last)
    else:
        st.success("✅ **All Bluesky posts are classified** — no Colab pass needed right now.")
        st.caption(_last)
    if st.button("🔄 Re-check status"):
        st.rerun()

    st.divider()

    # Dynamic evaluation — model drift over the GROWING hand-labeled benchmark (true concept drift).
    st.subheader("📈 Model drift — dynamic evaluation over time")
    st.caption(
        "How the classifier holds up on the **growing** hand-labeled benchmark — the true "
        "**concept-drift** signal, not just a regression guard. Each point is a Colab GPU evaluation "
        "run; the benchmark grows as the admin relabels hard cases (🔧 Maintenance), so a sustained "
        f"move here reflects real drift. Target: **recall on CLAIM ≥ {CLAIM_RECALL_TARGET:.0%}**. "
        "The point-in-time run (confusion matrix, breakdowns) lives in **🔧 Maintenance**."
    )
    _render_eval_freshness()   # are these numbers even measured on the current benchmark?
    _render_drift()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EMBEDDING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def embedding_analysis():

    st.subheader("🧬 Tokenisation & Embedding Evaluation")
    st.caption(
        f"Model: **{EMBED_MODEL}** · sentence-transformers · 384 dimensions · "
        "chosen over nomic-embed-text for higher MTEB STS scores"
    )

    st.divider()

    # ── Section A: Interactive similarity checker ──────────────────────────────
    st.markdown("### A · Interactive Similarity Checker")
    st.caption("Enter two texts and see their cosine similarity. Similar climate posts should score > 0.6.")

    col_left, col_right = st.columns(2)
    text_a = col_left.text_area("Text A", placeholder="e.g. Heat dome breaks BC temperature record", height=100)
    text_b = col_right.text_area("Text B", placeholder="e.g. Record heatwave scorches Pacific Northwest", height=100)

    if st.button("Compare", type="primary"):
        if text_a.strip() and text_b.strip():
            with st.spinner("Embedding..."):
                score = similarity(text_a, text_b)
            col_l, col_m, col_r = st.columns([1, 2, 1])
            with col_m:
                if score >= 0.6:
                    st.success(f"Cosine similarity: **{score:.4f}** — semantically similar")
                elif score >= 0.4:
                    st.warning(f"Cosine similarity: **{score:.4f}** — borderline / ambiguous")
                else:
                    st.error(f"Cosine similarity: **{score:.4f}** — semantically dissimilar")
        else:
            st.warning("Enter text in both boxes.")

    st.divider()

    # ── Section B: Embedding pairs evaluation ─────────────────────────────────
    st.markdown("### B · Domain Pair Evaluation")
    st.caption(
        f"20 labeled pairs from `{PAIRS_CSV}` — 10 similar (expected > 0.6) and "
        "10 dissimilar (expected < 0.4). Good separation confirms the model captures "
        "meaningful distance in the climate domain."
    )

    if PAIRS_CSV.exists():
        if st.button("Run embedding evaluation"):
            with st.spinner(f"Embedding 40 texts with {EMBED_MODEL}..."):
                results = eval_pairs(str(PAIRS_CSV))

            # Summary metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Similar pairs mean", results["similar_mean"],
                      delta="target > 0.6" if results["similar_mean"] >= 0.6 else "below target")
            c2.metric("Dissimilar pairs mean", results["dissimilar_mean"],
                      delta="target < 0.4" if results["dissimilar_mean"] <= 0.4 else "above target")
            c3.metric("Separation", results["separation"],
                      delta="good" if results["separation"] >= 0.2 else "weak")

            # Pairs table
            df = pd.DataFrame(results["pairs"])
            df["expected"] = df["should_be_similar"].map({True: "similar", False: "dissimilar"})
            df["pass"] = df.apply(
                lambda r: (r["should_be_similar"] and r["score"] >= 0.6)
                          or (not r["should_be_similar"] and r["score"] <= 0.4),
                axis=1,
            )
            df["result"] = df["pass"].map({True: "✅ pass", False: "❌ fail"})
            st.dataframe(
                df[["text_a", "text_b", "pair_type", "expected", "score", "result"]],
                use_container_width=True,
            )

            n_pass = df["pass"].sum()
            st.info(f"{n_pass}/20 pairs passed the threshold ({n_pass/20*100:.0f}%)")
    else:
        st.warning(f"`{PAIRS_CSV}` not found.")

    st.divider()

    # ── Section C: Category cluster analysis ──────────────────────────────────
    st.markdown("### C · Category Cluster Analysis")
    st.caption(
        "Mean intra-category cosine similarity across posts in the database. "
        "Scientific posts should cluster tighter than conspiracy posts — "
        "confirming the keyword taxonomy captures semantically coherent groups."
    )

    if st.button("Analyse category clusters"):
        stats_db = get_stats(DB_PATH)
        if stats_db["total_ingested"] == 0:
            st.warning("No posts in the database yet. Run ingestion first.")
        else:
            with st.spinner("Embedding up to 20 posts per category..."):
                cat_stats = category_similarity_stats(DB_PATH, limit_per_category=20)

            if not cat_stats:
                st.info("Not enough posts per category (need at least 2 per category).")
            else:
                cat_df = pd.DataFrame(
                    [{"category": k, "mean_intra_similarity": v} for k, v in cat_stats.items()]
                ).sort_values("mean_intra_similarity", ascending=False)

                st.bar_chart(cat_df.set_index("category")["mean_intra_similarity"])
                st.dataframe(cat_df, use_container_width=True)
                st.caption(
                    "Higher = posts in that category are more semantically similar to each other. "
                    "Scientific posts are expected to cluster tighter than conspiracy posts."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BASE vs ADAPTER  (Week 5)
# ═══════════════════════════════════════════════════════════════════════════════
def base_vs_adapter():
    st.subheader("🧪 Base vs LoRA Adapter")
    st.caption(
        "A comparison harness, **not** a production switch. The shipped classifier is the "
        f"base `{MODEL}` (recall-first). The adapter is the Week-5 precision experiment — "
        "kept here only to make the precision/recall tradeoff visible on real inputs."
    )

    with st.expander("What differs between the two calls?"):
        st.markdown(
            f"""
| | Base (production) | Adapter (demo) |
|---|---|---|
| Model | `{MODEL}` | `{ADAPTER_MODEL}` (LoRA → GGUF in Ollama) |
| Prompt | 8 few-shot examples | lean, zero-shot (few-shot folded into the weights) |

**Both the model path *and* the prompt format change** — the adapter was trained on the lean
prompt, so it must be served on it. Verdict from the eval: the adapter raised precision but
could not hold recall ≥ 0.90 **and** precision ≥ 0.85 together, so it is not deployed.
"""
        )

    examples = {
        "Vague conspiracy (they should DISAGREE)":
            "Weather manipulation is real and they do not want you to know",
        "Denial + statistic (they should AGREE — claim)":
            "50,000 acres of ice sheets have melted, what a lie these governments are trying to convince people of Alaska",
        "Specific conspiracy (claim)":
            "Cloud seeding planes triggered the flash floods in Dubai last week",
        "Emotional reaction (both → opinion)":
            "I am so done with this weather, it's honestly making me miserable",
        "False but checkable (both → claim)":
            "Antarctic sea ice has actually been expanding for the past decade",
        "Hyperbole / doom (adapter stricter → opinion)":
            "Honestly the planet is finished, there's no point even trying anymore",
        "Official alert (both → claim)":
            "The National Hurricane Center upgraded the system to Category 4 overnight",
    }
    pick = st.selectbox("Try a revealing example, or type your own below:", ["—"] + list(examples))
    default_text = examples.get(pick, "")
    post = st.text_area("Post to classify", value=default_text, height=90,
                        placeholder="Paste a climate / weather social media post...")

    if st.button("Compare base vs adapter", type="primary"):
        if not post.strip():
            st.warning("Enter a post first.")
        else:
            col_base, col_adapter = st.columns(2)

            with col_base:
                st.markdown(f"#### Base · `{MODEL}`")
                st.caption("8 few-shot prompt · production")
                with st.spinner("Base classifying..."):
                    b = classify_batch([post], model=MODEL)[0]
                (st.success if b["has_claim"] else st.warning)(
                    "**CLAIM**" if b["has_claim"] else "**OPINION**")
                st.caption(f"Reason: {b['reason']}")

            with col_adapter:
                st.markdown(f"#### Adapter · `{ADAPTER_MODEL}`")
                st.caption("lean zero-shot prompt · adapter experiment")
                with st.spinner("Adapter classifying..."):
                    a = classify_lean(post, model=ADAPTER_MODEL)
                if a["has_claim"] is None:
                    st.error(
                        f"Adapter not available — {a['reason']}\n\n"
                        f"Register it in Ollama first: convert the LoRA to GGUF, then "
                        f"`ollama create {ADAPTER_MODEL} -f Modelfile`."
                    )
                else:
                    (st.success if a["has_claim"] else st.warning)(
                        "**CLAIM**" if a["has_claim"] else "**OPINION**")
                    if a["thought"]:
                        st.caption(f"Thought: {a['thought']}")
                    st.caption(f"Reason: {a['reason']}")

            if a["has_claim"] is not None:
                if a["has_claim"] == b["has_claim"]:
                    st.info("Both models agree on this post.")
                elif b["has_claim"] and not a["has_claim"]:
                    # base=CLAIM, adapter=OPINION — the precision case
                    st.success(
                        "⚡ The models **disagree**, and the adapter is the stricter one — it "
                        "demoted this to **OPINION**. For a vague accusation with nothing specific "
                        "to check, the adapter is **correct** and the base's CLAIM is a false "
                        "positive — that's the precision gain. The same strictness is what can cost "
                        "recall on borderline *real* claims (try the denial-with-stat example)."
                    )
                else:
                    # base=OPINION, adapter=CLAIM — the rarer direction
                    st.warning(
                        "⚡ The models **disagree** — here the adapter is the more *permissive* one, "
                        "calling **CLAIM** where the base said OPINION. The adapter is right only if "
                        "the post names something specific and checkable; on a vague rant this would "
                        "be the kind of false positive the adapter is usually trained to avoid."
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVIDENCE MATCHING  (Week 6)
# ═══════════════════════════════════════════════════════════════════════════════
def evidence_matching():
    st.subheader("🔎 Evidence Matching (RAG)")
    st.caption(
        "Stage 4: for a classified **claim**, retrieve the nearest **GDELT news** articles "
        "(ChromaDB · all-MiniLM-L6-v2), then an LLM re-rank judges whether any article describes "
        "the *same specific event* — a **corroboration** signal, not a truth verdict. Combined with "
        "reach (engagement) it surfaces the **reach-vs-support mismatch**: a claim spreading widely "
        "with no news backing from an unverified source is the misinformation red flag. The flag is "
        "**not** raised for **official sources** (legitimate warnings/forecasts) or posts that "
        "**self-cite a credible source** — those are real reasons a genuine post lacks news corroboration."
    )

    @st.cache_resource
    def _evidence_store():
        return get_store()

    store = _evidence_store()
    n_idx = store.count()

    c1, c2 = st.columns([2, 1])
    c1.info(f"Evidence index: **{n_idx}** GDELT news articles" if n_idx else "Evidence index is empty — build it first.")
    if c2.button("🔃 Build / refresh index"):
        with st.spinner("Embedding GDELT articles into ChromaDB..."):
            n = store.build_index(DB_PATH)
        st.success(f"Index built — {n} articles.")
        st.rerun()

    st.divider()
    st.markdown("### Assess a claim")
    claim_text = st.text_area("Claim text (include any link the post cites, e.g. a study URL)",
                              placeholder="e.g. HAARP technology is causing the Alberta floods", height=80)
    ec1, ec2 = st.columns(2)
    eng = ec1.number_input("Reach (engagement = likes + reposts + replies + quotes)", min_value=0, value=0, step=10)
    author = ec2.text_input("Author handle (optional — e.g. nws.noaa.gov to test official sources)", value="")
    if st.button("Assess against news evidence", type="primary"):
        if not claim_text.strip():
            st.warning("Enter a claim.")
        elif n_idx == 0:
            st.warning("Build the evidence index first.")
        else:
            with st.spinner("Retrieving news + checking corroboration..."):
                a = assess_claim(store, claim_text, engagement=int(eng), source="bluesky",
                                 author=author.strip(), cfg=cfg)
            corro, sig = a["corroboration"], a["signal"]
            {"corroborated": st.success, "partial": st.warning, "none": st.error}[corro["verdict"]](
                f"**{corro['verdict'].upper()}** — {corro['reason']}")
            if sig["red_flag"]:
                st.error("🚩 RED FLAG — high reach, no corroboration, unverified source, no cited evidence "
                         "(misinformation pattern).")
            elif a["official"]:
                st.info("✓ Official source — legitimate warnings/forecasts aren't flagged even without news corroboration.")
            elif sig.get("reshared_official"):
                st.info("✓ Reshares an official source — credibility credited to the origin, not the account.")
            elif sig.get("credible_cite"):
                st.info("✓ Post self-cites a credible source — it supplies its own evidence for you to review.")
            st.caption(f"**Reader signal:** {sig['summary']}")
            st.markdown("**Retrieved news — open the source to verify (don't take the model's word):**")
            cited = sig.get("cited")
            for m in a["retrieval"]["matches"]:
                mark = "  ⬅ **cited**" if cited and m["url"] == cited["url"] else ""
                url = m["url"] if str(m["url"]).startswith("http") else ""
                title = f"[{m['title'][:90]}]({url})" if url else m["title"][:90]
                st.markdown(f"- `{m['similarity']:.3f}` · **{m['domain']}** · {title}{mark}")

    st.divider()
    st.markdown("### Scan top claims for red flags")
    st.caption("Assesses the highest-engagement classified claims — where a reach-vs-support mismatch matters most (one LLM call each).")
    n_scan = st.slider("How many top claims to scan", 5, 25, 10)
    if st.button(f"Scan top {n_scan} claims"):
        if n_idx == 0:
            st.warning("Build the evidence index first.")
        else:
            with st.spinner(f"Assessing {n_scan} claims..."):
                results = assess_db_claims(store, DB_PATH, limit=n_scan, cfg=cfg)
            flags = [r for r in results if r["signal"]["red_flag"]]
            st.write(f"**{len(flags)} red-flag claims** of {len(results)} scanned "
                     "(high reach + no corroboration + unverified source + no cited evidence; "
                     "official sources and self-cited claims are excluded).")
            for r in results:
                sig = r["signal"]
                icon = "🚩" if sig["red_flag"] else ("✅" if sig["verdict"] == "corroborated" else "•")
                with st.expander(f"{icon}  [{r['engagement']}]  {r['text'][:70]}"):
                    st.markdown(f"**Post:** {r['text']}")
                    st.markdown(f"**@{r['author']}** · {r['source']} · {r['engagement']} engagements")
                    st.caption(f"Corroboration: **{sig['verdict']}** — {sig['reason']}")
                    if r.get("vision"):
                        v = r["vision"]
                        st.caption(f"🖼️ Image (edge-case vision): **{v.get('image_type')}** · "
                                   f"depicts_claim={v.get('depicts_claim')} — {v.get('description','')}")
                    st.caption(f"Reader signal: {sig['summary']}")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION — Scanner (the product) vs Course demos (method views)
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🌪️ Climate Claim Scanner")
st.sidebar.caption("Surfaces signals about Bluesky climate posts — *you* judge the post.")
_render_health_sidebar()

# Page objects — referenced by trust_checker / trust_results for st.switch_page navigation.
_page_trust = st.Page(trust_checker, title="Trust Checker", icon="🛡️", default=True)
_page_results = st.Page(trust_results, title="Results", icon="📋")

pg = st.navigation({
    "🛡️ Scanner": [_page_trust, _page_results],
    "📚 Course demos": [
        st.Page(classification_eval, title="Classification Evaluation", icon="🔍"),
        st.Page(embedding_analysis, title="Embedding Analysis", icon="🧬"),
        st.Page(base_vs_adapter, title="Base vs Adapter", icon="🧪"),
        st.Page(evidence_matching, title="Evidence Matching (RAG)", icon="🔎"),
    ],
    "🔧 Admin": [st.Page(maintenance, title="Maintenance", icon="🔧")],
})
pg.run()
