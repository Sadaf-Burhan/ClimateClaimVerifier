# Session Handoff — ClimateClaimVerifier (updated 2026-07-14)

Read this first to continue in a fresh session without re-deriving. This supersedes the earlier
handoff. Companion docs: `WEEK7_STATUS.md` (Week-7 design pointers), `MONITORING.md` (the
monitoring/eval layer), and the auto-memory index (`memory/MEMORY.md`).

**Project =** a *reader-signal scanner* for **Bluesky** climate/extreme-weather posts. It surfaces
signals (claim vs opinion, independent news corroboration, source, reach, image nature) and a
**reach-vs-support red flag** — it **never** asserts true/false. Recall-first classifier; precision
recovered downstream by evidence + source signals.

---

## 0. Immediate next step
**Build the Week-7 image-input path** (§7 below — fully specced). Before that, the user planned to
run the overnight ingestion from their own VS Code terminal:
`uv run python -m climate_verifier.ingestion.scheduler --once --force` (keep the machine awake), then
push `data/ingested.db` to Drive and run the Colab classification pass.

## 1. Git / branch / data state
- **Branch `multimodal-edge-gating`** — all work here, pushed to `github.com/Sadaf-Burhan/ClimateClaimVerifier` (PUBLIC). `main` is frozen; merge only when the user says so.
- **Uncommitted** (as of writing): `.env.example` (safe template, placeholders only — user hasn't decided to commit it), a `.gitignore` `!​.env.example` exception, minor `README.md` line-ending, and **`data/claim_eval.csv`** (may contain a real admin relabel row appended via the app — review before committing).
- **SECURITY:** `.env` holds REAL secrets — `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`, `ADMIN_PASSWORD`. It is gitignored and NOT tracked. **NEVER commit or echo it.** Every push: grep the staged diff for secret patterns first.
- **Data:** `data/ingested.db` = **2030 bluesky + 3195 gdelt**, 1339 classified (≈691 bluesky pending → next Colab classify). `data/chroma_evidence` = the evidence index. `data/eval_history.jsonl` (drift), `data/health.json` (stage heartbeats) — both gitignored runtime artifacts.
- **Models:** local Ollama has `qwen2.5:3b` (classifier). `qwen2.5vl:7b` (vision) runs on **Colab GPU only**.

## 2. Two-layer architecture (as built)
- **🔧 In-house maintenance:** (a) **local daily ingestion** — Bluesky+GDELT → topic-filter → save → **demand-driven evidence top-up** → index rebuild → **corpus refresher** (all network/CPU, no GPU); (b) **Colab GPU maintenance** (`climate_verifier.maintenance --all`) — classify → vision → reindex → evaluate → export results back to Drive.
- **🌐 External-facing scanner:** the Streamlit app — Trust Checker + Results (users), Course demos, and an **authenticated Maintenance tab** (admin).
- The app only *reads* `data/`; maintenance *writes* it. Classification/eval run on **Colab GPU** (local CPU too slow); the app never classifies in bulk.

## 3. Everything built THIS session (with commits, newest→oldest)
**Region-aware evidence retrieval & ingestion**
- `f802c06` region-aware retrieval — new `pipeline/geo.py` derives `"Region, Country"` (headline place-names, else domain/ccTLD); `build_index` embeds location INTO each GDELT doc; `evidence_for_claim` folds the claim's location into the query. LLM re-rank made optional (`use_llm_rerank`, default OFF) with `retrieval_only_verdict`.
- `6fddfe7` demand-driven GDELT top-up — `scheduler.topup_evidence_for_claims()`: each claim's (region, subject) drives targeted GDELT queries; no count cap; `max_article_age_days` recency guard; fail-fast timeout.
- `9af825d` region-aware fixes — `geo` now reads **state/province abbreviations** in location context (`[AZ]`, `City, TX`; ambiguous IN/OR/ME/HI/OK/ON/OH only when bracketed); **region-mismatch demotion**: claim region + all matches from OTHER regions → demote TOPIC MATCH→NO MATCH ("covers the topic but from Pennsylvania/Tennessee — not Arizona").

**Reader-signal / verdict UX (many nuance fixes — see §6)**
- `84cb096` eyewitness reframe (`looks_like_eyewitness`, conspiracy-guarded) + wired top-up into the autonomous cycle.
- `47de569` self-cited posts lead with their own credible source, not "None retrieved"; de-jargoned the verdict line.
- `b65f237` retrieved-news header adapts to verdict (no "NONE + a list" contradiction).
- `f995fda` full clickable article URL, **TOPIC MATCH** label (not "PARTIAL"), "← closest match" (not "cited").
- `e41a258` **decoupled the display verdict from the red flag** (`news_status` REPORTED/TOPIC MATCH/NO MATCH) + runtime **"🔎 possibly a missed claim"** callout on opinions (strong evidence/official/credible-cite only).
- `e746add` **TOPIC MATCH needs a real bar** (`evidence.topic_proximity: 0.50`) — a 0.42 neighbour reads as NO MATCH, not a topic match.
- `818dad9`/`f7366af` truncated-link fix (`_linkify_post` rewrites Bluesky's truncated in-text URL to the full embed URL) + `import re`.
- `e97c539` embed-citation credit (a post linking Guardian counts as a credible cite), relevance floor, geo US, bullet formatting (bold only the label).

**Monitoring / evaluation layer (Module 8)**
- `06d03b8` `snapshot_metrics`/`load_eval_history` → `data/eval_history.jsonl`; drift charts; new `climate_verifier/health.py` heartbeats → `data/health.json`; `--once` daily runner.
- `9e43460` new `climate_verifier/maintenance.py` (classify→vision→reindex→evaluate) + `notebooks/colab_daily_maintenance.ipynb`.
- `f59574e` eval standardized on **Colab GPU** (app/CLI eval are local diagnostics, don't write the drift log — classifier is nondeterministic across GPU/CPU).
- `15b3aaf` `MONITORING.md`; `ca67270` README **Full End-to-End Run** runbook.
- `6ae6d80` replaced the local "Run Classifier" button with a **Colab-classification status checker**.
- `138ded6` classify **only Bluesky** posts (never the GDELT evidence corpus).
- `5c435d9` **static eval → Maintenance tab; drift → Evaluation tab** (drift shows benchmark size `n` growing = true concept drift).

**Corpus refresher & admin relabel loop**
- `87c76fe` corpus refresher (end of each cycle): age-expire Bluesky posts > `retention_days` (14) + availability-sweep (remove posts deleted on Bluesky, fail-safe on API error). Never touches GDELT/eval-train. `--refresh [--dry-run]`.
- `d7ec5e5` **Maintenance tab** (auth via `ADMIN_PASSWORD`): evidence-nominated **relabel queue**. `pipeline/relabel.py` — nominate opinions-that-look-like-claims (official/credible-cite/REPORTED) + claims-with-no-evidence; a relabel writes TWO stores: `classifications.admin_label` override (users see it, via COALESCE in `_load_posts`) **and** appends to `claim_eval.csv`.
- `7d4a6d1` relabel `post_type` "thought" from a **fresh-loaded dropdown** + add-new + notes + inline taxonomy guide.
- `569efe1` relabel UX — actions keep the queue in place + inline "done" note (no rescan/reset).

## 4. Key design decisions & nuances — DO NOT relitigate
- **Recall-first classifier; precision recovered downstream.** ~0.70 precision is a design ceiling (Week-4 prompts + Week-5 LoRA couldn't beat it). Classifier is **nondeterministic even at temp 0** (0.687↔0.697 back-to-back) — precision jitters ±several pts, more across GPU/CPU; **watch recall + sustained trends**, not single points.
- **Evaluation uses a FROZEN labeled CSV, independent of ingested data** — ingesting more never changes eval numbers. The drift layer catches model/env regressions; **true concept drift** needs the *growing* benchmark (relabel loop).
- **Evidence is HEADLINE-ONLY** (GDELT titles, no bodies). So retrieval/verdict operate at *related coverage* + *reach-vs-support*, NOT claim support. This caps everything — a headline can't verify a proposition. Documented in README "Limitations & Future Extensions".
- **Verdict vocabulary (user-facing, `news_status`):** REPORTED (LLM-rerank corroborated) / **TOPIC MATCH** (proximity ≥ `topic_proximity` 0.50, same subject) / **NO MATCH** (below bar, or region-mismatch, or only weak neighbours). A **topic match is NOT claim support** — the UI says so everywhere.
- **Red flag is decoupled from the display verdict** — it stays on the strict `corroboration.verdict` (only strong/HIGH counts as support), so a conspiracy post can show TOPIC MATCH *and* a red flag. Red flag is NOT raised for: official sources, credible self-citations (incl. embed `external_url`), real-photo vision saves, eyewitness observations, forecasts.
- **Reframes (parallel heuristics in `build_reader_signal`):** `looks_like_forecast` (future warning), `looks_like_eyewitness` (first-person local observation, conspiracy-guarded). Both suppress the red flag and reframe.
- **LLM ranker deferred to an agentic extension** — full body-fetch/rerank is its own project (bot walls). Region/entity mismatches are handled deterministically (cheaper). The `use_llm_rerank` toggle remains for same-region-different-event cases (best on Colab GPU).

## 5. WEEK 7 DESIGN — image-input path (READY TO BUILD, the next task)
Goal: let a user **upload a screenshot** of a climate claim (off-Bluesky content — X/FB/IG/WhatsApp, infographics, memes), extract structured text, and run the **unchanged** pipeline. See `WEEK7_STATUS.md` §Design and the Q1–Q7 reasoning.

- **Problem-statement expansion (Answer A):** from "Bluesky scanner" → "scanner for climate claims a person encounters anywhere." Bluesky link stays the full-fidelity path; screenshot is a **degraded-fidelity, clearly-labelled** coverage path.
- **`extract_from_image(image)` → JSON schema:** `claim_text`, `has_readable_text`, `image_type` (real_photo/meme_or_cartoon/synthetic_ai/screenshot/chart_infographic), `depicts_claim` (yes/partial/no), `author_handle`, `platform`, `engagement {likes,reposts,replies}`, `visible_citation`, `description`.
  - **RULES:** transcribe `claim_text`/`author_handle`/`engagement` LITERALLY (null if not clearly legible, never guess — hallucinated engagement/handle is the worst failure); INFER `platform` from visual branding (X/Bluesky/FB/IG logos), null if unclear. `has_readable_text=false` → don't run the pipeline; show the raw output / "couldn't read a claim."
- **Model:** `qwen2.5vl:7b` (same as the gated edge-case vision — one model, two entry points). GPU.
- **Pipeline connection:** `extract_from_image()` → **thin adapter** → `assess_claim()` (the SAME function the link path uses). Adapter transforms: sum `engagement` → one int; pack `image_type`/`depicts_claim`/`description` into the existing `vision` dict; null-fill `followers`; tag `source="uploaded_screenshot"`. **ONE real downstream change:** generalize the `source == "bluesky"` guard in `build_reader_signal` (red flag + source wording) to treat `"uploaded_screenshot"` as an unverified social source. The spine (classifier → region-aware RAG → signal assembly) is otherwise untouched. No canonical post URL exists → UI shows the uploaded image + any `visible_citation` instead of "Open original post."
- **Streamlit:** `st.tabs` — "Text input" (existing link path) + "Photo input" (new). Photo tab: upload → show image → extracted fields → full pipeline result with the **degraded-fidelity** note. Failure handling per `has_readable_text`.
- **Evaluation:** 10–15 hand-labeled images with expected JSON; transcription fields near-exact, classification fields accuracy-scored, `platform` not gated; **end-to-end verdict agreement** as the primary bar (does the image path reach the same reader signal as feeding the text directly?).

**CRITICAL Week-7 nuance (memory: `image-carried-claim-eval-example`):** some posts carry the claim in the IMAGE while the TEXT is pure opinion (canonical example: "RESIST TYRANNY" caption over a "DOGE fired NOAA's climate scientists → launched climate.us" card). Such a post is **OPINION in the text eval (correct — don't relabel it CLAIM there, it would corrupt the text classifier) and CLAIM in the image eval.** Two eval sets, two correct labels.

## 6. Future extensions (deferred, in memory)
- **Agentic retrieval + reranking agent** — fetch article bodies past bot walls → cross-encoder rerank (BGE/Jina) → LLM/NLI support/contradict verdict + source-credibility boost. (`planned-dynamic-eval-and-retrieval-eval`.)
- **Dynamically growing eval set** — the relabel loop already appends; wire periodic re-eval so the drift chart tracks true concept drift. Evidence NOMINATES, human DISPOSES (never auto-relabel).
- **Judge/arbiter agent for the FINAL verdict** — decide whether the surfaced verdict comes from the text classifier or the image classifier, so the USER sees ONE coherent answer (the text/image split is confusing to them). (`image-carried-claim-eval-example`.)
- **Retrieval-quality eval** — labeled claim→expected-article pairs, precision@k.

## 7. Key files map
- `pipeline/geo.py` — location extraction (place-names, domain/ccTLD, `[AZ]`/`City, TX` abbreviations), `extract_location`, `with_location`.
- `pipeline/evidence.py` — `evidence_for_claim` (region+time-aware retrieval, relevance floor), `build_reader_signal` (news_status, red flag, reframes, region-mismatch), `assess_claim`, `retrieval_only_verdict`, `corroboration_check`, `looks_like_forecast/eyewitness`, `is_official`, `extract_citations`.
- `pipeline/relabel.py` — `get_relabel_candidates`, `apply_relabel`/`set_admin_label`/`append_to_eval_csv`, `eval_post_types`, `ensure_admin_columns`.
- `pipeline/vision.py` — `gate_edge_cases`, `analyze_image`, `_fetch_jpeg` (WebP→JPEG), `vision_reader_note`. **Week 7 adds `extract_from_image` here.**
- `pipeline/claim_classifier.py` — `classify`, `classify_pending` (bluesky-only), `get_stats` (total_classifiable/evidence).
- `pipeline/evaluate.py` — `run_eval`, `compute_metrics`, `snapshot_metrics`, `load_eval_history`.
- `ingestion/scheduler.py` — `run_ingestion_cycle` (+ top-up + refresher), `topup_evidence_for_claims`, `refresh_corpus`, CLI `--once/--force/--topup/--refresh/--dry-run`.
- `ingestion/bluesky.py` — `fetch_posts`, `fetch_post_by_url`, `_extract_embed`, `check_posts_exist`.
- `ingestion/gdelt.py` — `fetch_articles(..., timeout)`; `ingestion/store.py` — schema, `save`, `delete_posts`, `old_/oldest_bluesky_post_ids`, `_heldout_guard`.
- `climate_verifier/maintenance.py` — Colab GPU chain. `climate_verifier/health.py` — heartbeats.
- `app.py` — Trust Checker, Results, `_render_trust`, `_linkify_post`, Maintenance (`maintenance`, `_render_relabel_section`, `_render_static_eval`, `_admin_authed`), `classification_eval` (drift), health sidebar, `st.navigation`.
- `config.yaml` — `evidence` (high/topic/low proximity, use_llm_rerank, official/citation domains, topup block), `vision` (qwen2.5vl:7b), `storage` (refresher: retention_days/verify_availability), `ingestion` keywords.

## 8. Ops gotchas
- **Windows Streamlit:** kill by port — `Get-NetTCPConnection -LocalPort 8501 -State Listen | %{ Stop-Process -Id $_.OwningProcess -Force }`. Module edits need a full restart; `app.py`-only edits just need a browser Rerun. File watcher OFF.
- **Running from VS Code terminal doesn't need `git pull`** — edits land in the same local working tree. Pull only on another machine.
- **Ingestion is I/O-bound + GDELT rate-limited**, not compute — Colab won't speed it up and throttles GDELT harder (shared IP). Run ingestion locally (own terminal so the harness doesn't kill it); GPU only helps classify/vision/eval.
- **Admin tab:** set `ADMIN_PASSWORD` in `.env` to use it.
