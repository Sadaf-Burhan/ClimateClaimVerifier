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

import json
import sqlite3
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
EVAL_CSV       = Path(cfg["evaluation"]["claim_eval_csv"])
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
    rows = con.execute(f"""
        SELECT p.post_id, p.text, p.author, p.author_followers, p.vision_signal,
               p.keyword_category, p.created_at, p.external_url, c.reason,
               (p.likes + p.reposts + p.replies + p.quotes) AS engagement
        FROM posts p JOIN classifications c ON p.post_id = c.post_id
        WHERE c.has_claim = ? AND p.source = 'bluesky'
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


def _render_drift():
    """Model-drift panel: recall/precision across successive evals + the FN:FP balance.
    Reads data/eval_history.jsonl (one line per maintenance eval — Colab GPU or local).
    This is the signal that tells you the classifier is degrading before users feel it."""
    hist = load_eval_history()
    if not hist:
        st.info("📈 **Drift log is empty.** Each evaluation run (the Colab GPU maintenance pass, "
                "or the button below) appends recall / precision / FN / FP here, so you can watch "
                "for classifier drift over time.")
        return
    df = pd.DataFrame(hist)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return
    latest, prev = df.iloc[-1], (df.iloc[-2] if len(df) > 1 else df.iloc[-1])
    target = float(latest.get("target", CLAIM_RECALL_TARGET) or CLAIM_RECALL_TARGET)
    multi = len(df) > 1

    st.markdown("#### 📈 Model drift — evaluation history")
    st.caption(f"Last evaluated **{_age_label(str(latest['ts']))}** on model `{latest.get('model','?')}` "
               f"· {len(df)} run(s) logged.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (CLAIM)", f"{latest['recall']:.3f}",
              delta=f"{latest['recall']-prev['recall']:+.3f}" if multi else None,
              delta_color="normal")
    c2.metric("Precision", f"{latest['precision']:.3f}",
              delta=f"{latest['precision']-prev['precision']:+.3f}" if multi else None)
    c3.metric("FN / FP", f"{int(latest['false_negatives'])} / {int(latest['false_positives'])}",
              help="False negatives (missed claims — costly) vs false positives (opinions surfaced — cheap)")
    c4.metric("Meets target", "✅" if bool(latest.get("meets_target")) else "❌",
              help=f"recall ≥ {target:.0%}")

    st.line_chart(df.set_index("ts")[["recall", "precision"]])
    st.caption(f"Recall target ≥ {target:.0%}. A sustained **downward recall** trend is the costly "
               "drift — real claims dropped at the gate. Precision recovers downstream, so watch recall.")
    fnfp = df.set_index("ts")[["false_negatives", "false_positives"]].rename(
        columns={"false_negatives": "FN (missed — costly)", "false_positives": "FP (surfaced — cheap)"})
    st.bar_chart(fnfp)

    with st.expander("Raw drift log"):
        cols = ["ts", "model", "recall", "precision", "f1", "accuracy",
                "false_negatives", "false_positives", "meets_target"]
        st.dataframe(df[[c for c in cols if c in df.columns]].iloc[::-1],
                     use_container_width=True, hide_index=True)
    st.divider()


def _render_trust(store, post: dict, classify_first: bool):
    """Full trust panel for one post: classification → corroboration + credible sources →
    reader signal → verification links. Surfaces signals; never asserts truth."""
    text = post["text"]
    st.markdown(f"**Post:** {text}")
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
    (st.info if is_claim else st.warning)(
        f"**Classification: {'CLAIM' if is_claim else 'OPINION'}** — {reason}")
    if not is_claim:
        st.caption("Opinions have no specific factual event to corroborate — a strong match below would "
                   "suggest this is actually a claim the classifier missed.")

    vision = None
    if post.get("vision_signal"):
        try:
            vision = json.loads(post["vision_signal"])
        except Exception:
            vision = None

    with st.spinner("Retrieving news + checking corroboration…"):
        a = assess_claim(store, text, engagement=int(post.get("engagement", 0)), source="bluesky",
                         followers=post.get("author_followers", 0) or 0,
                         author=post.get("author", "") or "", vision=vision, cfg=cfg,
                         external_url=post.get("external_url", "") or "")
    sig, corro = a["signal"], a["corroboration"]

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
    _VERDICT_LABEL = {"corroborated": "REPORTED", "partial": "TOPIC MATCH", "none": "NO MATCH"}
    _vlabel = _VERDICT_LABEL.get(corro["verdict"], corro["verdict"].upper())
    st.caption(f"Independent news check: **{_vlabel}** — {corro['reason']} "
               "(does *other* news independently report the same event — a **topic match is not claim support**; "
               "separate from any source the post itself links).")
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
        if corro["verdict"] in ("corroborated", "partial"):
            st.markdown("#### 📰 News the scanner retrieved (RAG) — open them to verify")
            st.caption("The **number on the left** is a topic-similarity score (0–1): cosine similarity "
                       "between your claim and the article headline. Higher = closer wording/topic — it is "
                       "**not** proof they describe the same event (that's what the verdict above judges).")
        else:
            st.markdown("#### 🔍 Closest news the search found — none report *this specific* event")
            st.caption("Shown for transparency, **not as corroboration**: these are the nearest headlines by "
                       "wording/topic, so you can check the **NONE** above yourself. The **number on the "
                       "left** is the topic-similarity score (0–1) — close in topic is not the same as "
                       "reporting your specific claim.")
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

    tab_paste, tab_list = st.tabs(["📋 Paste a post", "🏆 Pick from top posts"])

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

    # Classifier evaluation against the hand-labeled set
    st.subheader("📏 Classifier Evaluation — Hand-Labeled Set")
    st.caption(
        f"Runs the classifier against `{EVAL_CSV}` (human ground truth). "
        f"Success criterion: **recall on CLAIM ≥ {CLAIM_RECALL_TARGET:.0%}** — "
        "a missed claim (false negative) is discarded forever, the costly error; "
        "a false positive just surfaces on the dashboard, the cheap one. "
        "Accuracy alone hides this asymmetry."
    )

    _render_drift()

    if not EVAL_CSV.exists():
        st.warning(f"`{EVAL_CSV}` not found.")
    else:
        n_eval = len(load_eval_set(str(EVAL_CSV)))
        if st.button(f"Run evaluation ({n_eval} labeled posts)"):
            with st.spinner(f"Classifying {n_eval} posts with {MODEL}..."):
                eval_results = run_eval(str(EVAL_CSV), model=MODEL,
                                        llm_batch_size=LLM_BATCH_SIZE)
                metrics = compute_metrics(eval_results,
                                          claim_recall_target=CLAIM_RECALL_TARGET)
            # Diagnostic only — does NOT write to the drift log. The drift series is populated by
            # the Colab GPU maintenance pass so every point shares one backend (the classifier is
            # nondeterministic across GPU/CPU; mixing backends would make the trend meaningless).
            st.caption("🔬 Local diagnostic run — not added to the drift log (the drift series is "
                       "produced by the Colab GPU maintenance pass, for backend-comparable numbers).")

            claim_m = metrics["per_class"]["claim"]
            asym    = metrics["error_asymmetry"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Recall on CLAIM", f"{claim_m['recall']:.3f}",
                      delta=f"target ≥ {CLAIM_RECALL_TARGET:.2f}",
                      delta_color="normal" if metrics["meets_target"] else "inverse")
            m2.metric("Precision on CLAIM", f"{claim_m['precision']:.3f}")
            m3.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m4.metric("FN / FP", f"{asym['false_negatives']} / {asym['false_positives']}",
                      help="False negatives (missed claims — costly) vs "
                           "false positives (opinions surfaced — cheap)")

            if metrics["meets_target"]:
                st.success(
                    f"✅ PASS — the gate lets through {claim_m['recall']:.1%} of real claims "
                    f"(misses {asym['fn_rate']:.1%}); {asym['fp_rate']:.1%} of opinions slip through."
                )
            else:
                st.error(
                    f"❌ BELOW TARGET — {asym['fn_rate']:.1%} of real claims are being "
                    "discarded at the gate. See the post-type breakdown for where."
                )

            cm = metrics["confusion_matrix"]
            col_cm, col_pc = st.columns(2)
            with col_cm:
                st.markdown("**Confusion matrix** (positive class = claim)")
                st.dataframe(pd.DataFrame(
                    {
                        "predicted claim":   [cm["true_claim_predicted_claim"], cm["true_opinion_predicted_claim"]],
                        "predicted opinion": [cm["true_claim_predicted_opinion"], cm["true_opinion_predicted_opinion"]],
                    },
                    index=["actual claim", "actual opinion"],
                ), use_container_width=True)
            with col_pc:
                st.markdown("**Per-class metrics**")
                st.dataframe(pd.DataFrame(metrics["per_class"]).T.round(3),
                             use_container_width=True)

            st.markdown("**Error breakdown** — where the misses are concentrated")
            col_pt, col_kc = st.columns(2)
            with col_pt:
                st.caption("By post type (linguistic shape)")
                pt_df = pd.DataFrame(metrics["by_post_type"]).T
                st.dataframe(pt_df.style.format({"error_rate": "{:.1%}"}),
                             use_container_width=True)
            with col_kc:
                st.caption("By keyword category (ingestion taxonomy)")
                kc_df = pd.DataFrame(metrics["by_keyword_category"]).T
                st.dataframe(kc_df.style.format({"error_rate": "{:.1%}"}),
                             use_container_width=True)

            with st.expander(f"Misclassified posts ({len(metrics['misclassified'])})"):
                if metrics["misclassified"]:
                    for m in metrics["misclassified"]:
                        direction = ("🔴 FALSE NEGATIVE (missed claim)"
                                     if m["expected_label"] == "claim"
                                     else "🟡 FALSE POSITIVE (opinion surfaced)")
                        st.markdown(f"{direction} · `{m['post_type']}`")
                        st.markdown(f"> {m['post_text']}")
                        st.caption(f"Model reason: {m['reason']}")
                        st.divider()
                else:
                    st.success("No misclassifications.")


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
})
pg.run()
