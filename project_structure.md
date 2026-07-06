# Project Structure — Climate Claim Scanner

This document explains every folder and file in this project and the reasoning behind each
decision. It is an end-to-end project: North American climate/extreme-weather social-media
posts are ingested, filtered, classified (claim vs opinion), evaluated, embedded, and — in
Week 5 — used to train and compare a LoRA adapter. (`project_description.md` is the
authoritative narrative; this file is the file-by-file reference.)

```
ClimateClaimVerifier/
├── app.py                              ← Streamlit dashboard (3 tabs: Weeks 1, 2, 5)
├── pyproject.toml                      ← dependencies managed by uv
├── .env                                ← Bluesky credentials (gitignored — never commit)
├── .gitignore
├── .streamlit/config.toml              ← disables the source-watcher (transformers noise)
├── project_description.md              ← authoritative project description
├── project_structure.md               ← this file
│
├── data/
│   ├── ingested.db                     ← SQLite: posts (+ in_eval_set / in_train_set flags),
│   │                                       classifications, metadata   (gitignored)
│   ├── claim_eval.csv                   ← 100 labeled posts — the held-out benchmark (TRACKED)
│   ├── lora_seed.jsonl                  ← 26 hand-labeled Week-5 LoRA seed examples (TRACKED)
│   └── embedding_pairs.csv              ← Week 2: 20 pairs for embedding quality (gitignored)
│
├── scripts/                            ← eval-set + LoRA dataset tooling (with leakage guards)
│   ├── sample_eval_candidates.py        ← sample real posts as eval candidates (excludes in_train_set)
│   ├── flag_eval_posts.py               ← marks eval posts in_eval_set=1 in the DB
│   ├── build_lora_trainset.py           ← two-phase LoRA dataset builder (select / label), flags in_train_set
│   ├── export_for_colab.py              ← export unclassified posts -> tmp/colab/ for GPU classification
│   └── import_from_colab.py             ← import Colab classification results back into the DB
│
├── models/                             ← Week-5 demo adapter (GGUF is gitignored, ~3.3 GB)
│   ├── Modelfile                        ← `ollama create qwen2.5-3b-claim-lora -f models/Modelfile`
│   └── README.md                        ← how to register the adapter locally
│
├── notebooks/
│   ├── week5_lora_training_colab.ipynb           ← clean, re-runnable Colab training source
│   └── week5_lora_training_colab.executed.ipynb  ← the actual run, with cell outputs (record)
│
├── specs/
│   └── week5_implementation_specs.yaml ← Week-5 plan + executed outcome banner
│
├── Analysis/evaluation.md              ← analysis notes
├── tmp/colab/                          ← transient Colab staging (gitignored)
│
└── src/
    └── climate_verifier/
        ├── config.yaml                 ← single source of truth for all parameters
        ├── __init__.py
        ├── ingestion/                  ← Infrastructure: data collection (no LLM)
        │   ├── bluesky.py
        │   ├── gdelt.py
        │   ├── store.py
        │   └── scheduler.py
        └── pipeline/                   ← Processing (course content)
            ├── topic_filter.py         ← Pre-LLM keyword relevance gate
            ├── claim_classifier.py     ← Weeks 1/4/5: LLM classifier + adapter-serving helper
            ├── evaluate.py             ← Week 1: classifier quality metrics
            ├── embedder.py             ← Week 2: sentence-transformer embeddings
            └── evidence.py             ← Week 6: evidence matching (GDELT RAG + corroboration re-rank)
```

The `data/chroma_evidence/` ChromaDB store (gitignored) is rebuilt from the DB by `evidence.py`.

---

## File by File

### `app.py`
Streamlit dashboard with **four tabs**:
- **Tab 1 — Claim Classifier**: ingestion status, run-classifier buttons, top claims/opinions,
  classifier evaluation (recall-on-CLAIM criterion, confusion matrix, error breakdowns)
- **Tab 2 — Embedding Analysis**: interactive similarity checker, pair eval, category clustering
- **Tab 3 — Evidence Matching (Week 6)**: build the GDELT index, assess a single claim (retrieval +
  corroboration verdict + reader signal + red flag), and scan the top-engagement claims for
  reach-vs-support red flags.
- **Tab 4 — Base vs Adapter (Week 5)**: a *comparison harness* (not a production switch) — runs a
  post through the base classifier (8-shot prompt) and the LoRA adapter (lean prompt, served via
  Ollama as a GGUF) side by side, with a direction-aware agreement/disagreement message.

Run: `uv run streamlit run app.py` from the project root.

### `data/ingested.db`  (gitignored)
SQLite database:
- `posts` — all ingested content (Bluesky + GDELT), deduplicated by `post_id`. Two held-out flags:
  **`in_eval_set`** (held out for evaluation) and **`in_train_set`** (reserved for LoRA training).
  A post is never both — these are the DB-level leakage guards.
- `classifications` — LLM claim/opinion decisions, keyed by `post_id`
- `metadata` — last successful ingestion timestamp

### `data/claim_eval.csv`  (TRACKED — the benchmark)
**100** human-labeled posts: 44 real (Bluesky + GDELT) and 56 synthetic hard-boundary cases,
stratified by `keyword_category` and `post_type`. Columns: `post_text`, `expected_label`
(`claim`/`opinion`), `keyword_category`, `post_type`, `notes`. Labeling rule: a post with
hostile/opinion tone but at least one checkable assertion is a `claim`. Real eval posts are
flagged `in_eval_set=1` so they are excluded from ChromaDB and LoRA training. Input to `evaluate.py`.

### `data/lora_seed.jsonl`  (TRACKED — Week 5)
26 hand-labeled contrastive examples on the conspiracy-specificity boundary (vague accusation →
opinion, specific mechanism / denial-with-statistic → claim), verified disjoint from the eval set.
The human-labeled "hard core" of the LoRA training set; the rest is teacher-labeled real posts.

### `data/embedding_pairs.csv`  (Week 2)
20 text pairs (10 similar, 10 dissimilar) for embedding quality. Similar > 0.6, dissimilar < 0.4.

---

## Scripts — eval-set & LoRA dataset tooling

| Script | Role |
|--------|------|
| `sample_eval_candidates.py` | sample real DB posts as eval candidates; **excludes `in_train_set`** so eval never overlaps training |
| `flag_eval_posts.py` | mark approved eval posts `in_eval_set=1` (by `post_id`); excluded from ChromaDB |
| `build_lora_trainset.py` | two-phase: **`select`** (local — flags posts `in_train_set=1`, exports candidates) and **`label`** (Colab — teacher-labels with per-post caching, merges the seed, asserts leak-free) |
| `export_for_colab.py` / `import_from_colab.py` | round-trip unclassified posts to a GPU (Colab) for fast classification, then import results |

---

## Pipeline Layer

### `ingestion/` — Infrastructure (no LLM)
| File | Role |
|------|------|
| `bluesky.py` | atproto SDK; posts + batched author follower counts per keyword; raw-HTTP fallback for new embed types |
| `gdelt.py` | GDELT API; rate-limit retry with backoff |
| `store.py` | `INSERT OR IGNORE` dedup; ingestion timestamp in `metadata` |
| `scheduler.py` | APScheduler loop; 24h guard; aligns Bluesky `since` with GDELT window |

### `pipeline/topic_filter.py` — Pre-LLM gate
Keyword relevance check (`WEATHER_TERMS` + `NA_TERMS`); discards irrelevant posts before any LLM call.

### `pipeline/claim_classifier.py` — Weeks 1 / 4 / 5 (LLM classification)
Binary claim/opinion classifier using `qwen2.5:3b` via Ollama. Returns `{"has_claim": bool, "reason": str}`;
identifies the *presence* of a checkable claim, never its truth.

- **Prompt (Week 4)**: 8 leak-free few-shot examples (4 claim / 4 opinion), a chain-of-thought
  `thought` field (first in the schema), and a "checkability is not evidence" rule. Examples are
  verified disjoint from the eval set.
- **Inference (Week 5 finding)**: `llm_batch_size: 1` (single-post). Batching 16 posts per call cost
  ~0.19 recall on the 3B model; single-post reaches recall **0.938** (precision ~0.70). See
  `project_description.md` Week 4/5 for the leakage + batch-mode corrections.
- **`classify_lean(post, model)`**: serves the LoRA adapter with the lean zero-shot prompt it was
  trained on (used only by the Base-vs-Adapter tab; not in the production pipeline).
- `classify_batch` / `classify_pending` drive the dashboard and headless backlog classification.

### `pipeline/evaluate.py` — Week 1 (classifier quality)
Runs the classifier against `data/claim_eval.csv`. **Recall on CLAIM ≥ 0.90** is the success
criterion (false negatives are unrecoverable; false positives are cheap, discounted downstream).
Reports per-class P/R/F1, confusion matrix, FN-vs-FP asymmetry, and breakdowns by `keyword_category`
and `post_type`. Run: `uv run python -m climate_verifier.pipeline.evaluate`.

### `pipeline/embedder.py` — Week 2 (embeddings)
`sentence-transformers` with `all-MiniLM-L6-v2` (chosen over `nomic-embed-text` for higher MTEB STS).
`similarity`, `eval_pairs`, `category_similarity_stats`. Foundation for Week-6 RAG evidence retrieval.

### `pipeline/evidence.py` — Week 6 (Evidence Matching / RAG)
Stage 4. A persistent ChromaDB collection of **GDELT news articles** (`all-MiniLM-L6-v2`, cosine),
excluding any post flagged `in_eval_set`/`in_train_set`. For each classified claim:
1. **Dense retrieval** — top-k nearest news articles (`evidence_for_claim`).
2. **Corroboration re-rank** (`corroboration_check`) — an LLM re-reads the claim against the
   retrieved articles and judges whether any describes the *same specific event*
   (`corroborated` / `partial` / `none`). This separates topical overlap from real corroboration —
   dense similarity alone scores a conspiracy claim high just for sharing a topic (Module 6's
   re-ranking lesson). **A relevance judgment, never a truth verdict.**
3. **Reader signal** (`build_reader_signal`) — a plain-language, *suggestive* summary plus the
   **reach-vs-support red flag**: high engagement + no corroboration + unverified source +
   no cited evidence = misinformation amplification pattern. The system never says "false".

The red flag has **two guards** so legitimate posts aren't flagged just for lacking a GDELT match:
- **Self-citation** (`extract_citations`) — if the post links its own credible source (a
  study/journal/news domain in `evidence.citation_domains`), it supplies its own evidence and
  isn't flagged. The linked domain is surfaced for the reader; the system doesn't vouch for it.
- **Official source** (`is_official`) — an allowlisted agency handle/domain
  (`evidence.official_sources`) posting a warning/forecast isn't flagged (news can't corroborate a
  *future* event). Matched conservatively by exact-or-dotted-suffix, so an "altgov" lookalike
  (`altcdc.altgov.info`) does **not** pass as `nws.noaa.gov`.

`assess_claim` runs all three; `assess_db_claims` assesses the top-engagement claims. Build the
index: `uv run python -m climate_verifier.pipeline.evidence --build`.

---

## Week 5 — LoRA adapter (trained, evaluated, NOT deployed)

A precision-targeted QLoRA adapter was trained on `qwen2.5:3b` (rank 8, Unsloth, Colab) across three
claim/opinion balances. It shifted the precision/recall frontier but no balance reached recall ≥ 0.90
**and** precision ≥ 0.85 together, so **production ships the base classifier (recall 0.938) with no
adapter**; the adapter is demo-only (the GGUF in `models/`, the Base-vs-Adapter tab). The finding —
not the artifact — is the result. Full write-up: `project_description.md`, "Week 5".

---

## Configuration (`config.yaml`)

```yaml
model:
  name: "qwen2.5:3b"                      # Ollama model (classification)
  temperature: 0.0
  llm_batch_size: 1                       # single-post (recall 0.938; batch-16 was 0.750)
  adapter_name: "qwen2.5-3b-claim-lora"   # demo-only LoRA, registered in Ollama as a GGUF
embedding:
  model_name: "all-MiniLM-L6-v2"
evidence:                                 # Week 6 — GDELT evidence matching
  chroma_path: data/chroma_evidence       # persistent vector store (gitignored)
  top_k: 5                                # nearest news articles per claim
  high_proximity: 0.60                    # retrieval tier thresholds
  low_proximity: 0.40
  high_reach: 50                          # engagement >= this = "high reach" for the red flag
  official_sources: [noaa.gov, weather.gov, weather.gc.ca, ...]   # allowlist — not flagged
  citation_domains: [gov, edu, sciencedaily.com, nature.com, ...] # credible self-cite → not flagged
evaluation:
  claim_eval_csv: data/claim_eval.csv
  claim_recall_target: 0.90
storage:
  db_path: data/ingested.db
ingestion: { interval_hours: 24, bluesky_limit: 100, ...keywords by category... }
```

---

## Running the Project

```bash
uv sync                                              # install deps
ollama serve && ollama pull qwen2.5:3b               # classifier model
uv run python -m climate_verifier.ingestion.scheduler  # ingest (24h guard)
uv run python -m climate_verifier.pipeline.evaluate    # evaluate the classifier
uv run streamlit run app.py                          # dashboard (4 tabs)
```
The Base-vs-Adapter tab needs the LoRA registered in Ollama — see `models/README.md`.

---

## Week 7 — Next Steps (Multimodality)

Testing the top-claims scan surfaced a class of posts the text-only pipeline can't resolve: a
first-person field observation with a **photo** (e.g. *"this deep sinkhole appeared at our field
site — abrupt permafrost thaw"*). No news corroborates it and the account is unverified, so the
current red flag fires — but the attached image is exactly the evidence a reader would weigh.

**Planned (Week 7 — multimodal models):** add an image signal to Stage 4/5. A multimodal model
inspects any attached media and contributes a descriptive signal — e.g. *a genuine on-the-ground
photo* supports a first-hand observation, whereas *a cartoon / meme / obviously synthetic image*
is a fabrication cue. Consistent with the reader-signal design: the image is **another signal for
the reader**, never an automated truth verdict. This folds into the same red-flag guard logic
(self-citation and official source already added in Week 6) as a third legitimate-context signal.
