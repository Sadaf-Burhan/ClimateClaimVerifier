# Week 7 Status — Multimodal + Evaluation/Monitoring

Pointer summary of what was completed this session (branch `multimodal-edge-gating`), and the two
pieces still to build. Cross-references files, commits, and design decisions.

---

## ✅ Completed — built, tested, committed, pushed

### Region-aware evidence retrieval (Week 6 RAG upgrade) — `f802c06`
- New `src/climate_verifier/pipeline/geo.py` — derives a coarse `"Region, Country"` from headline
  place-names (event location wins) else the news domain (e.g. `weatherbc.com` → British Columbia).
- `pipeline/evidence.py`: `build_index()` embeds location into each GDELT document + stores
  `location`/`date_int` metadata; `evidence_for_claim()` folds the claim's location into the query so
  same-region news ranks higher (soft nudge, no hard prune).
- LLM re-rank made optional (`use_llm_rerank`, default off) with an honest proximity-based verdict
  (`retrieval_only_verdict`).
- Verified: identical claim scored 0.830 (Britain) vs 0.615 (Alberta) purely from the location token.

### Demand-driven GDELT ingestion (Week 6) — `6fddfe7`, `daa4f50`
- `ingestion/scheduler.py`: `topup_evidence_for_claims()` — each claim's topic + region drives
  targeted GDELT queries; deduped, no article cap, `max_article_age_days` recency guard, fail-fast
  `fetch_articles` timeout.
- Root fix for the empty-corpus false positive (a real "B.C. wildfires" claim had zero news to match).

### Reader-signal refinement (Week 7 vision-adjacent) — `84cb096`
- `pipeline/evidence.py`: `looks_like_eyewitness()` — first-person, locally-anchored observations get
  reframed, not red-flagged (with a conspiracy negative-guard so "I saw them spraying chemtrails"
  still flags).

### Monitoring & evaluation layer (Week 8 — Evaluation and Beyond) — `06d03b8`, `9e43460`, `f59574e`, `15b3aaf`, `ca67270`
- New `climate_verifier/health.py` — per-stage heartbeats → `data/health.json`; app sidebar chips
  🟢/🟡/🔴.
- `pipeline/evaluate.py`: `snapshot_metrics()` + `load_eval_history()` → `data/eval_history.jsonl`;
  drift charts (recall/precision over time, FN:FP) in the app's Evaluation tab.
- New `climate_verifier/maintenance.py` — the GPU maintenance chain `classify → vision → reindex →
  evaluate`, each with a health heartbeat.
- New `notebooks/colab_daily_maintenance.ipynb` — thin Colab wrapper for the maintenance pass.
- Autonomous cycle: `run_ingestion_cycle` wires the top-up + index rebuild; `--once` runner (exits
  non-zero on failure for OS-scheduler alerting).
- Standardized eval on Colab GPU — app/CLI eval are local diagnostics that don't write the drift log
  (backend-comparable series).
- New `MONITORING.md` — records what each signal measures / shows / helps; README "Full End-to-End
  Run" runbook + two-layer ops docs.
- Established finding: the "precision drop" (0.78 → 0.68) is nondeterminism, not regression
  (0.687 → 0.697 back-to-back same-machine); documented.

### Bug fix — `138ded6`
- `claim_classifier.py` + `get_stats`: classify only Bluesky posts, never the GDELT evidence corpus
  (the top-up had made 786 news articles show as "pending").

### Week 7 design decisions locked (assignment Q1–Q6) — decided, not yet code
- **Visual input:** off-platform screenshots/images of climate claims — the coverage path for content
  a user encounters anywhere off Bluesky.
- **Extraction schema (`extract_from_image()`):** `claim_text`, `has_readable_text`, `image_type`,
  `depicts_claim`, `author_handle`, `platform`, `engagement` (likes/reposts/replies),
  `visible_citation`, `description`.
  - Rules: **transcribe** `handle`/`engagement`/`claim_text` literally (null if not clearly legible);
    **infer** `platform` from visual branding (null if unclear); **never guess**.
- **Pipeline connection:** `extract_from_image()` → thin adapter (sum engagement, pack vision fields,
  null-fill followers, tag `source="uploaded_screenshot"`) → `assess_claim()`. Spine unchanged; ONE
  small downstream change — generalize the `source == "bluesky"` guard in `build_reader_signal()` to
  treat `"uploaded_screenshot"` as an unverified social source (so the red flag + source wording apply).
- **Evaluation:** 10–15 hand-labeled images with expected JSON; transcription fields near-exact,
  classification fields accuracy-scored, `platform` not gated; end-to-end verdict agreement as the
  primary bar.
- **Problem statement:** expanded from "Bluesky climate scanner" to "scanner for climate claims a
  person encounters anywhere" (Answer A), with the image path explicitly labeled lower-confidence
  (degraded-fidelity signal).
- **Streamlit:** `st.tabs` — "Text input" (existing Bluesky link) and "Photo input" (new); photo tab
  shows the uploaded image + extracted fields + full pipeline result with the degraded label; no
  "Open original post" link — image + any visible citation instead. On no-text/extraction failure,
  don't run the pipeline; show the raw output / a clear message.
- **Vision model:** `qwen2.5vl:7b` via Ollama (same model as the gated edge-case vision — one model,
  two entry points).

### End-to-end run
- Full `--force` ingestion cycle run to grow the corpus; Bluesky grown +524 posts (814 → 1,338)
  mid-run; GDELT accumulating. Next: Colab GPU maintenance pass to classify the backlog + evaluate.

---

## 🔨 To build (the two new pieces — noted, not started)

1. **Week 7 image-screenshot input path.** `extract_from_image()`; the thin adapter; generalize the
   `source == "bluesky"` guard for `"uploaded_screenshot"`; the Streamlit "Photo input" `st.tabs`
   panel; `has_readable_text` / extraction-failure handling.

2. **Evaluation layer.**
   - **Dynamically growing eval set** — label a sample of recent *ingested* posts and append to
     `claim_eval.csv`, so the drift chart catches real concept drift, not just model/env regressions.
   - **Retrieval-quality evaluation** — measure how well the RAG evidence layer returns *relevant*
     GDELT news for a claim (labeled claim → expected-article pairs, precision@k / relevance).
   - Both are also recorded in `MONITORING.md` ("Future work").
