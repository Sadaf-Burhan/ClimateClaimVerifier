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
            └── embedder.py             ← Week 2: sentence-transformer embeddings
```

---

## File by File

### `app.py`
Streamlit dashboard with **three tabs**:
- **Tab 1 — Claim Classifier**: ingestion status, run-classifier buttons, top claims/opinions,
  classifier evaluation (recall-on-CLAIM criterion, confusion matrix, error breakdowns)
- **Tab 2 — Embedding Analysis**: interactive similarity checker, pair eval, category clustering
- **Tab 3 — Base vs Adapter (Week 5)**: a *comparison harness* (not a production switch) — runs a
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
uv run streamlit run app.py                          # dashboard (3 tabs)
```
The Base-vs-Adapter tab needs the LoRA registered in Ollama — see `models/README.md`.
