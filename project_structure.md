# Project Structure — Climate Claim Scanner

This document explains every folder and file in this project and the reasoning behind each decision.
The project applies Weeks 1 and 2 of the Foundations of Language Models course to a real-world domain:
North American climate and extreme weather social media posts.

```
ClimateClaimVerifier/
├── app.py                              ← Streamlit dashboard (Weeks 1 + 2)
├── pyproject.toml                      ← dependencies managed by uv
├── .env                                ← Bluesky credentials (gitignored — never commit)
├── .gitignore
├── project_description.md             ← authoritative project description
├── project_structure.md               ← this file
│
├── data/
│   ├── ingested.db                     ← SQLite: posts + classifications
│   ├── claim_eval.csv                  ← Week 1: 60 labeled posts for classifier evaluation
│   └── embedding_pairs.csv            ← Week 2: 20 pairs for embedding quality eval
│
└── src/
    └── climate_verifier/
        ├── config.yaml                 ← single source of truth for all parameters
        ├── __init__.py
        │
        ├── ingestion/                  ← Infrastructure: data collection
        │   ├── bluesky.py
        │   ├── gdelt.py
        │   ├── store.py
        │   └── scheduler.py
        │
        └── pipeline/                   ← Processing (course content)
            ├── topic_filter.py         ← Pre-LLM keyword relevance gate
            ├── claim_classifier.py     ← Week 1: LLM binary classifier
            ├── evaluate.py             ← Week 1: classifier quality metrics
            └── embedder.py             ← Week 2: sentence-transformer embeddings
```

---

## File by File

### `app.py`
Streamlit dashboard with two tabs:
- **Tab 1 — Claim Classifier**: ingestion status, run classifier buttons, top claims/opinions,
  classifier evaluation (recall-on-CLAIM criterion, confusion matrix, error breakdowns)
- **Tab 2 — Embedding Analysis**: interactive similarity checker, pair eval, category clustering

Run: `uv run streamlit run app.py` from the project root.

### `data/ingested.db`
SQLite database with three tables:
- `posts` — all ingested content (Bluesky + GDELT), deduplicated by `post_id`
- `classifications` — LLM claim/opinion decisions, keyed by `post_id`
- `metadata` — stores the last successful ingestion timestamp

### `data/claim_eval.csv` ← Week 1
60 manually-labeled posts covering all keyword categories (`scientific`, `extreme_events`,
`sensationalist`, `conspiracy`, `combinations`) and twelve post types, with the hard
boundary cases (sarcasm, denial rants with embedded statistics, emotion wrapped around
official warnings) deliberately oversampled. Columns:

| Column | Purpose |
|--------|---------|
| `post_text` | the raw text as it would appear in the database |
| `expected_label` | `claim` or `opinion` (human ground truth) |
| `keyword_category` | which ingestion category this post represents |
| `post_type` | linguistic shape (e.g. `official_alert`, `sarcasm_joke`, `denial_with_stat`) — drives the error breakdown |
| `notes` | why this label is correct — useful for auditing failures |

Labeling rule: a post that expresses opinion or hostility but contains at least one
checkable assertion is a `claim`. Purpose: input to `pipeline/evaluate.py`.

### `data/embedding_pairs.csv` ← Week 2
20 text pairs for embedding quality evaluation — 10 similar and 10 dissimilar. Columns:

| Column | Purpose |
|--------|---------|
| `text_a`, `text_b` | the two texts to compare |
| `should_be_similar` | `True` if they describe the same type of event |
| `pair_type` | describes the pairing (e.g. `similar-extreme-events`) |

Similar pairs should score > 0.6; dissimilar pairs < 0.4.

---

## Pipeline Layer

### `ingestion/` — Infrastructure
Data collection. No LLM concepts apply here — these files handle HTTP requests, API auth,
SQLite writes, and scheduling.

| File | Role |
|------|------|
| `bluesky.py` | atproto SDK; fetches posts + author follower counts per keyword |
| `gdelt.py` | HTTP requests to GDELT API; rate-limit retry with backoff |
| `store.py` | `INSERT OR IGNORE` deduplication; ingestion timestamp in `metadata` table |
| `scheduler.py` | APScheduler blocking loop; 24-hour guard against re-ingestion |

### `pipeline/topic_filter.py` — Pre-LLM gate
Keyword-based relevance check using two lists:
- `WEATHER_TERMS` — confirms the post is about a climate/weather topic
- `NA_TERMS` — confirms geographic relevance to North America

No LLM call. This gate runs first and discards irrelevant posts before any model is invoked,
keeping the LLM workload focused on posts that have already passed a basic relevance check.

### `pipeline/claim_classifier.py` — Week 1 (LLM: Classification)
Binary claim/opinion classifier using `qwen2.5:3b` via Ollama — selected over `gemma2:2b`
and `llama3.2:3b` in a bake-off on the labeled eval set (recall-on-CLAIM 0.81 / precision
0.96 vs 0.75 / 0.86 for gemma2:2b).

**What the LLM does**: reads a post and decides whether it contains a verifiable factual
statement. Returns `{"has_claim": bool, "reason": str}`.

**What the LLM does NOT do**: verify whether the claim is true. It identifies the *presence*
of a claim, not its accuracy. This is the core justification for appropriate LLM use —
binary text classification matches Week 1's "decision" use case exactly.

Design decisions aligned with course content:
- Few-shot prompting: 6 labeled input/output examples in the prompt, including the hard
  boundary cases (false-but-checkable assertion, emotion wrapped around a fact, personal
  experience) and an explicit "never judge truth" instruction
- Structured JSON output with a fixed schema enforced by regex extraction
- `temperature: 0.0` — deterministic, reproducible outputs
- Results saved permanently to `classifications` table — never re-classified

### `pipeline/evaluate.py` — Week 1 (Classifier quality metrics)
Runs the classifier against `data/claim_eval.csv` and computes quality metrics with
CLAIM as the positive class.

**Why not accuracy alone**: the two error directions have asymmetric cost. A false
negative (claim labeled opinion) is discarded at the gate and never reaches the
dashboard — unrecoverable. A false positive (opinion labeled claim) merely surfaces
on the dashboard where context lets the reader discount it. The success criterion is
therefore **recall on CLAIM ≥ 0.90** (`claim_recall_target` in `config.yaml`), with
precision reported alongside.

Functions:
- `run_eval(csv_path, model)` → classifies every eval post, returns per-post results
- `compute_metrics(results)` → accuracy, per-class P/R/F1, confusion matrix, FN-vs-FP
  asymmetry stats, error breakdowns by `keyword_category` and `post_type`, misclassified posts
- `format_report(metrics)` → plain-text CLI report

Run: `uv run python -m climate_verifier.pipeline.evaluate` (also surfaced in dashboard Tab 1).

### `pipeline/embedder.py` — Week 2 (Tokenisation and Embeddings)
Embedding module using `sentence-transformers` with `all-MiniLM-L6-v2`.

**Model selection decision**: course baseline is `nomic-embed-text` (Ollama). Switched to
`all-MiniLM-L6-v2` because:
1. Higher MTEB STS scores — the primary task here is Semantic Textual Similarity
2. Runs offline without Ollama — the scheduler operates independently
3. 384-dimensional vectors sufficient for both evaluation and future RAG

Functions:
- `embed(texts)` → normalised embedding matrix (n × 384)
- `similarity(text_a, text_b)` → cosine similarity as float
- `eval_pairs(csv_path)` → runs against `embedding_pairs.csv`, returns accuracy stats
- `category_similarity_stats(db_path)` → mean intra-category similarity from the live DB

---

## Configuration (`config.yaml`)

Single source of truth. All parameters live here — no hardcoded values in Python files.

```yaml
model:
  name: "qwen2.5:3b"         # Ollama model for LLM classification (Week 1)

embedding:
  model_name: "all-MiniLM-L6-v2"   # sentence-transformer model (Week 2)

evaluation:
  claim_eval_csv: data/claim_eval.csv
  claim_recall_target: 0.90        # success criterion for the classifier gate

ingestion:
  interval_hours: 24
  bluesky_limit: 100
  ...keywords by category...

storage:
  db_path: data/ingested.db
```

---

## Running the Project

```bash
# Install dependencies
uv sync

# Start Ollama (required for claim classifier tab)
ollama serve
ollama pull qwen2.5:3b

# Run ingestion (first time populates DB; repeats every 24h automatically)
uv run python -m climate_verifier.ingestion.scheduler

# Launch dashboard (separate terminal)
uv run streamlit run app.py
```
