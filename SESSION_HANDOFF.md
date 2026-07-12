# Session Handoff — ClimateClaimVerifier (Week 7–8 work)

Read this first to continue without re-deriving. Project = a **reader-signal scanner** for **Bluesky**
climate/extreme-weather posts: surfaces signals (claim/opinion, news corroboration, source, reach,
vision), **never a true/false verdict**. Recall-first classifier; precision recovered downstream.

---

## 1. Git / branch state (CRITICAL)
- **All work is on branch `multimodal-edge-gating`.** `main` is FROZEN at `6be4a62` (reshare-provenance) — do NOT touch main; merge only when the user says so.
- **The user has uncommitted app.py edits stashed** → `git stash@{0}` ("WIP on multimodal-edge-gating"). Don't blow it away; `git stash pop` if they ask.
- Remote: `github.com/Sadaf-Burhan/ClimateClaimVerifier` (PUBLIC). Branch is pushed.
- **Every push MUST run a secret scan first** (repo is public, `.env` has real creds): grep the diff for
  the Bluesky app-password fragment, GitHub token prefixes, and app-password assignment lines.
  Keep any literal secret pattern OUT of committed files — use it only in the throwaway grep command.
- **SECURITY:** `.env` holds real Bluesky creds (handle + app password) — gitignored, NEVER commit/echo it.
  User was advised to **rotate** the app password (it was pasted in an earlier chat).

## 2. Data / environment state
- **`data/ingested.db`** = the shakeout: **745 bluesky + 284 gdelt** posts, all classified, **114 vision
  signals**. New schema (media/reshare/profile/vision columns).
- **`data/ingested_backup_premultimodal.db`** = the ORIGINAL corpus (4598 posts) — preserved, gitignored.
- **`data/chroma_evidence`** = ChromaDB index, **284 GDELT articles** (clean, headline-only). Rebuild:
  `uv run python -m climate_verifier.pipeline.evidence --build`.
- Ollama LOCAL has `qwen2.5:3b` (classifier), NOT `qwen2.5vl` (vision runs on Colab GPU only).
- **Windows quirks:** Streamlit's `WinError 10054` traceback is harmless; Ctrl+C often won't stop it —
  kill by port: `Get-NetTCPConnection -LocalPort 8501 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }`.
  File watcher is OFF (`.streamlit/config.toml` `fileWatcherType="none"` — silences torchvision noise),
  so **module edits need a full Streamlit restart**; app.py-only edits need a manual Rerun/refresh.

## 3. Architecture (as-built)
Ingest (Bluesky + GDELT) → topic filter (NA climate/wx) → **classify single-post** (`qwen2.5:3b`,
`llm_batch_size:1`, recall 0.938 / precision ~0.70) → **evidence matching** (ChromaDB dense retrieve →
LLM corroboration re-rank) + source/reach/self-citation/official/reshare signals → **gated vision** on
edge cases → **reader signal + red flag** → **Trust Checker** dashboard.
- **Two RAG layers (don't confuse):** Week 3 few-shot retriever (`retrieval/vector_store.py`,
  metadata-narrowed via `where`) vs **Week 6 evidence retriever** (`evidence.py:evidence_for_claim`,
  a PLAIN dense query `collection.query(query_texts=[claim_text])` — NO metadata narrowing). The Week 6
  one is what we're improving.

## 4. Key findings (established — don't relitigate)
- **Config beats model:** single-post inference lifted recall 0.750→0.938; classifier precision ~0.70 is
  a design ceiling (Week 4 prompts + Week 5 LoRA couldn't beat it). Precision is recovered downstream.
- **LoRA (Week 5): negative result** — no ratio hit recall≥0.90 AND precision≥0.85; adapter is demo-only.
- **Vision (Week 7): validated but ~1% coverage** — the WebP→JPEG fix was the real bug (Bluesky serves
  WebP, Ollama can't decode it → blank images). On 745 posts it changed exactly 1 red flag. Corroboration
  is the precision workhorse; vision is a low-coverage supplement. NOT merged; accumulating data to re-measure.
- **Ingestion scope:** keep weather posts (they're the corroboration corpus); the forecast wording handles
  future events. (Decided: keep as-is.)

## 5. What shipped this session (on the branch)
- Corroboration guardrails (grounding/no-truth/citation/scoped-none); red-flag guards: **self-citation,
  official-source allowlist (incl. `nws-bot.us`, `weather.im`), reshare-of-official**.
- GDELT date fix (`_iso_date`); **single-post classifier default** (removed batch-16 landmine).
- **Multimodal branch:** `store.py` schema (media/reshare/profile/vision cols), `bluesky.py`
  `_extract_embed`+`_batch_profiles`+**`fetch_post_by_url`**, `vision.py` (gate all categories, WebP→JPEG,
  analyze), vision→reader-signal wiring, `config.yaml` vision block.
- **Source-selectable ingestion:** `run_ingestion_cycle(sources=[...], force=...)` — GDELT throttles on
  Colab's IP, so run **GDELT locally / Bluesky+classify+vision on Colab GPU**, merge via Drive.
- **Colab notebook** `notebooks/colab_shakeout_pipeline.ipynb` (hardened).
- **Dashboard overhaul (`app.py`):** `st.navigation` sidebar — **Scanner** (Trust Checker + Results) vs
  **Course demos** (Classification Evaluation, Embedding, Base-vs-Adapter, Evidence Matching). Dropped
  "Week N" labels. Trust Checker: bullet summary + red-flag guide up top; **two entry modes** (paste a
  bsky.app link → auto-fetch metadata via `fetch_post_by_url`; OR filter+multiselect one/many/all top
  posts) → **"View results"** → dedicated **Results page** (signal-first BOLD bullets → sources with the
  similarity number EXPLAINED). Bluesky-only guard on paste. Sort (engagement/followers/recent) + count
  slider. **`looks_like_forecast()`** adds future-event wording.
- **Docs on branch:** `MULTIMODAL_REBUILD_PLAN.md`, `FINAL_SYSTEM_DESIGN.md`, `course_journey.md` (v2,
  spec-grounded), `PROJECT_JOURNEY.md` (v1 — user comparing the two, hasn't finalized).

## 6. Assignment (Week 6 RAG) — answers compiled but NOT yet saved to a file
User wants `week6_assignment_answers.md` for their partner. Answers drafted in chat: **Q1** (what info
helps — Thwaites), **Q2** (pre-compute vs retrieve), **Q3** (what completed), **Q3.5** (UI: two entry
modes, signal-first, Bluesky-only), **Q4** (short structured vs long text; chunking), **Q5** (which 3 of
10; hard-filter recency generously, NOT category/domain), **Q6** (correct retrieval; Belle Union tornado
example; "none" is a valid success), **Q7** (source-reputation from corroboration track record — Layer 1
analog), **Q8** (behavior/provenance > self-report; casualness is a trap), **Q9** (24h autonomous ingest;
INSERT OR IGNORE + upsert, not wipe; GDELT is the source). → Offer to save these.

## 7. WHERE WE LEFT OFF — region-aware evidence retrieval (NOT yet built)
The BC post pulled UK wildfire articles (region mismatch). Week 6 retriever is a plain dense query (no
region/time awareness). **Agreed direction (user's, better than my first re-rank idea):** strengthen
retrieval at the source with metadata, not a post-hoc re-rank.

**GDELT capture today (`gdelt.py`):** url→post_id (sourcelink), domain→author, title→text, seendate→date.
NOT captured: `sourcecountry` (available from GDELT DOC API, just dropped).

**ChromaDB reality:** dense similarity on the embedded document + a HARD `where` metadata filter
(exact/range) — no soft weighting. So:
- **Location → embed it INTO the document** (index `headline + location`; embed claim + its location) so
  dense retrieval prefers same-region. SOFT — avoids over-pruning. (Hard `where` location filter would
  wrongly drop a US outlet covering BC fires; location is best-effort anyway.)
- **Time → hard `where` date-window filter** (reliable). Needs a **numeric `date_int` (YYYYMMDD)** field.
- **LLM re-rank → make OPTIONAL** (`config vision/evidence: use_llm_rerank`), since strong retrieval
  reduces its need; its only unique job was "same specific event vs same region+time+topic different event."

**PLANNED BUILD (awaiting final user confirm + re-rank default off/on):**
1. Enrich GDELT index: embed `headline + location`; store metadata `{sourcelink(url), domain,
   sourcecountry, location, date, date_int, category}`. Derive `location` from domain + headline place-names
   for existing articles (re-index, no re-fetch); capture `sourcecountry` going forward. Rebuild ChromaDB.
2. Metadata-aware query in `evidence_for_claim`: embed claim + extracted location; add `where` date-window.
3. `use_llm_rerank` config flag.

## 8. Key files & functions (quick map)
- `src/climate_verifier/config.yaml` — model, `llm_batch_size:1`, `evidence` (official_sources,
  citation_domains, thresholds, high_reach), `vision` (model `qwen2.5vl:7b`, gate_categories:[], max_images),
  storage db_path, ingestion keywords (5 categories).
- `pipeline/evidence.py` — `ClimateEvidenceStore.build_index/evidence_for_claim`, `corroboration_check`
  (`_CORROBORATION_PROMPT`), `build_reader_signal` (returns `bullets`), `assess_claim`, `assess_db_claims`,
  `looks_like_forecast`, `is_official`, `extract_citations`, `_iso_date`.
- `pipeline/vision.py` — `gate_edge_cases`, `analyze_image`, `_fetch_jpeg` (WebP→JPEG), `vision_reader_note`.
- `pipeline/claim_classifier.py` — `classify` (single), `classify_pending`, `get_top_claims/opinions`
  (post_id + sort_by), `LLM_BATCH_SIZE=1`.
- `ingestion/bluesky.py` — `fetch_posts`, `fetch_post_by_url` (link→post metadata), `_extract_embed`,
  `_batch_profiles`.
- `ingestion/gdelt.py` — `fetch_articles` (add sourcecountry here for region work).
- `ingestion/scheduler.py` — `run_ingestion_cycle(sources, force)`.
- `ingestion/store.py` — `POST_COLUMNS`, `save`, `_iso_date`, schema migration.
- `app.py` — sidebar `st.navigation`; `trust_checker`, `trust_results`, `_render_trust`, `_load_posts`,
  `_bsky_url`, `_non_bluesky_url`, `_SORT_LABEL`, course-demo page fns.

## 9. Immediate next step
Confirm the region-aware plan (§7) — **location-in-embedding + hard date-window filter + optional
LLM re-rank** — and the re-rank default (off = trust retrieval / on = keep same-event guard). Then build:
enrich GDELT index → re-index → metadata-aware query. Also offer to save the Week 6 assignment answers.
