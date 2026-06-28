# Climate Claim Scanner
### Domain-Specific Social Media Claim Detection using LLM-Appropriate Tasks

A credibility signal aggregator for North American climate and extreme weather posts on social media. The system ingests live data, identifies posts that contain verifiable factual claims, and surfaces structural credibility signals — without asking an LLM to judge whether any claim is true.

**Course:** Foundations of Language Models — 8-week project  
**Scope:** Weeks 1 (LLM classification) and 2 (embeddings and evaluation)  
**Stack:** Python · Ollama (qwen2.5:3b) · sentence-transformers · Bluesky API · GDELT · SQLite · Streamlit  
**Authoritative description:** [project_description.md](project_description.md) — this README is the overview and quick-start view

---

## Why This Exists

Climate change content on social media attracts significant sensationalism and misinformation — claims like "CO₂ is just a trace gas (0.04%), so it can't affect the planet", "Record cold winters prove that global warming isn't happening", or "Extreme weather events (like hurricanes or wildfires) are engineered by the government or deliberately started to push 'green' agendas". Unlike daily weather, climate change topics unfold over long timescales and are harder for most people to evaluate, making them particularly susceptible to exaggeration and fabricated statistics. The goal of this project is to help build public awareness of what constitutes a verifiable claim versus noise.

This project ingests live posts from Bluesky in real time and classifies them as claims or opinions using an LLM. Tools like Climate Feedback and Google Fact Check Explorer already exist for verification; this project is not a verification system — it is a detection pipeline that surfaces which posts contain verifiable factual assertions, leaving evaluation to the reader.

### The Metal Detector Principle

A metal detector's job is to find metal, not to assess its value. The scanner narrows the search space; the human does the evaluating.

---

## The Central Design Decision: What This System Does NOT Do

This is the most important thing to understand about the project, and the one most likely to be misunderstood.

**This system does not use an LLM to verify whether a claim is true or false.**

That is not a gap in ambition. It is the correct design decision, grounded in what language models actually are.

### Why LLMs Cannot Verify Facts

LLMs are next-token predictors. They generate text by statistically predicting which word should come next, based on patterns in their training data. This mechanism is powerful for text generation, classification, and summarisation — but it is fundamentally incompatible with factual verification for three reasons:

**Knowledge cutoff.** An LLM trained on data through a fixed date has no knowledge of events after that date. A post about a wildfire that happened yesterday has no representation in the model's training data. Asking "is this claim true?" produces an answer based on pattern associations — not a comparison against actual ground truth.

**Hallucination.** When a model encounters a question whose answer falls outside its training distribution, it confabulates — generating confident-sounding, plausible text that may be entirely false. The model cannot reliably flag its own uncertainty, because it was trained to complete text, not to evaluate truth.

**Text generation is not fact checking.** The statistical mechanism that allows a model to write a poem, summarise a document, or classify sentiment cannot be repurposed to evaluate a claim against external physical reality. These are categorically different tasks. Fact checking requires access to authoritative external data — weather service records, scientific measurements, published reporting. A language model has none of these at inference time. It has text patterns.

The practical implication: asking `qwen2.5:3b` whether "HAARP controls the weather and caused the 2024 floods" is true would produce a statistically plausible answer — not a factually grounded one. Building a system around that would be misleading by design.

---

## What This System DOES Do: Appropriate LLM Use

The LLM is used for exactly one task it is genuinely well-suited for: **binary text classification**.

### The Question the LLM Answers

> "Does this post contain a verifiable factual statement — a named event, a measurement, an official warning, a documented occurrence — or is it an opinion, emotional reaction, or political commentary?"

This is a linguistic pattern recognition task, not a truth evaluation. A well-read generalist can reliably distinguish:

| Post | Classification | Why |
|------|---------------|-----|
| "NOAA confirms strongest Atlantic hurricane season since 2005" | **Claim** | Official source + specific comparative fact + named time period |
| "This storm is absolutely terrifying, I can't believe it" | **Opinion** | Emotional reaction, no verifiable factual content |
| "The government is using HAARP to control the weather" | **Claim** | Specific entity + mechanism + effect — structurally a factual assertion (even if false) |
| "Higher CO₂ makes the planet greener and benefits agriculture" | **Claim** | Specific gas + mechanism + effect — structurally a claim (even if contested) |
| "Climate change is just a government excuse to control us" | **Opinion** | Political viewpoint, no specific verifiable factual claim |
| "Environment Canada issues heat warning for BC interior through Thursday" | **Claim** | Official source + named warning type + named location + timeframe |

The LLM identifies the *presence* of a claim — the structural and linguistic features of a verifiable factual statement. It does not know whether that claim is accurate. That distinction is the entire basis for appropriate LLM use in this project.

### The Question the LLM Does NOT Answer

The LLM is never asked:
- "Is this claim true?"
- "Is this information accurate?"
- "Should I trust this post?"

Asking those questions would be the category error this project is specifically designed to avoid.

---

## System Design: How the Pipeline Works

```
  Social Media Posts
  Bluesky (social) + GDELT (news)
           │
           ▼
  ┌─────────────────────┐
  │  INGESTION LAYER    │  ← no LLM
  │  Runs every 24h     │
  │  Bluesky API        │
  │  GDELT API          │
  │  Stored in SQLite   │
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  TOPIC FILTER       │  ← no LLM
  │  Keyword-based gate │
  │  NA climate/weather │
  │  relevance check    │
  └─────────┬───────────┘
            │ passes / discards
            ▼
  ┌─────────────────────┐
  │  CLAIM CLASSIFIER   │  ← LLM (appropriate use)
  │  qwen2.5:3b         │
  │  Binary: claim /    │
  │  opinion            │
  │  Few-shot prompting │
  │  Structured output  │
  └──────┬──────┬───────┘
         │      │
    opinion   claim
    rejected  proceeds
              │
              ▼
  ┌─────────────────────┐
  │  EMBEDDING LAYER    │  ← no LLM (Week 2)
  │  all-MiniLM-L6-v2   │
  │  Semantic similarity│
  │  Category clustering│
  │  Future: RAG base   │
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  DASHBOARD          │
  │  Streamlit          │
  │  Top claims by      │
  │  engagement         │
  │  Embedding tools    │
  └─────────────────────┘
```

### What Each Component Decides — and What It Does Not

| Component | Decides | Does NOT decide |
|-----------|---------|----------------|
| Topic filter | Is this about NA climate/weather? | Is this claim accurate? |
| Claim classifier (LLM) | Does this text contain a verifiable factual statement? | Is that statement true? |
| Embedding layer | How semantically similar are two posts? | Which post is correct? |
| Dashboard | Surfaces signals for the reader | Makes any truth judgment |

**The final truth judgment is never made by the system.** The system surfaces structured signals — claim detected, engagement level, source domain, semantic proximity to published news. The reader draws the conclusion.

---

## How Success Is Measured

### Week 1 — Classifier Quality

The classifier gate has two failure directions with very different costs:

- **False negative** (claim labeled opinion) — the post is discarded at the gate and never reaches the dashboard. A missed claim is unrecoverable. This is the costly error.
- **False positive** (opinion labeled claim) — the post surfaces on the dashboard, where low evidence proximity and source context let the reader discount it. Cheap and self-correcting.

Accuracy alone treats these errors as equal, so it is reported but is **not** the success criterion.

**Evaluation set:** `data/claim_eval.csv` — 100 hand-labeled posts (44 real, 56 synthetic) with human-verified labels, stratified by all five keyword categories (`scientific`, `extreme_events`, `sensationalist`, `conspiracy`, `combinations`) **and** by post type (official alerts, news events, scientific findings, false-but-checkable conspiracy claims, denial rants with embedded statistics, emotion wrapped around official warnings, sarcasm/jokes, hyperbole, political viewpoints, vague conspiracy, emotional reactions, rhetorical questions). Hard boundary cases are deliberately oversampled — that is where a small model fails first. Labeling rule: a post that expresses opinion or hostility but contains at least one checkable assertion is a **claim**.

**Metrics** (`src/climate_verifier/pipeline/evaluate.py` — CLI report and dashboard section):

| Metric | Why it matters |
|--------|---------------|
| **Recall on CLAIM** | **Success criterion: ≥ 0.90** — share of real claims the gate lets through |
| Precision on CLAIM | Reported alongside — how many surfaced claims are genuine |
| Per-class precision / recall / F1 | Full picture for both classes |
| Confusion matrix | Shows exactly where the errors sit |
| False negatives vs false positives | Quantifies the cost asymmetry directly |
| Error breakdown by category and post type | "Misses are concentrated in sarcasm; zero misses on official alerts" is a stronger result than one aggregate number |

```bash
uv run python -m climate_verifier.pipeline.evaluate
```

The `reason` field on each classification must be coherent plain language — a legible audit log explaining the decision, not a model-generated placeholder.

### Week 2 — Embedding Quality

A 20-pair evaluation set (`data/embedding_pairs.csv`) contains 10 similar and 10 dissimilar text pairs. Quality is measured by cosine similarity scores.

| Metric | Target | Achieved |
|--------|--------|----------|
| Mean similar-pair similarity | > 0.60 | **0.64** ✅ |
| Mean dissimilar-pair similarity | < 0.40 | **0.21** ✅ |
| Separation (similar − dissimilar) | > 0.20 | **0.43** ✅ |
| `scientific` intra-similarity > `conspiracy` | confirmed | **confirmed** ✅ |

---

## Use Case Examples

### Case 1 — Credible factual claim surfaces to top of dashboard
**Post (Bluesky):** "Environment Canada issues extreme heat warning for BC interior through Thursday — temperatures expected to reach 42°C in Kamloops."

Pipeline:
- Topic filter: **passes** (heat warning, named province, named location)
- Claim classifier: `has_claim: true` — official source, specific warning type, named location, specific measurement
- Embedding: high cosine similarity to GDELT news articles covering the same heat event
- Dashboard: appears in Top Claims, ranked high by engagement

**Reader sees:** "Post contains a verifiable factual claim from an official source. Published news covers a similar heat event."

---

### Case 2 — Opinion discarded before reaching dashboard
**Post (Bluesky):** "This summer is absolutely terrifying, my backyard has been covered in wildfire smoke for two weeks straight."

Pipeline:
- Topic filter: **passes** (wildfire smoke)
- Claim classifier: `has_claim: false` — personal experience, emotional language, no specific verifiable factual content
- **Discarded here.** Does not proceed to embedding or dashboard.

**Reader sees:** nothing — no verifiable claim was detected.

---

### Case 3 — Conspiracy claim correctly routed without the system calling it "false"
**Post (Bluesky):** "HAARP technology is being used to cause the flooding in Alberta — wake up, this is weather warfare."

Pipeline:
- Topic filter: **passes** (flooding, Alberta)
- Claim classifier: `has_claim: true` — specific mechanism (HAARP), specific effect (flooding), specific location (Alberta). The LLM correctly identifies this as a structurally verifiable factual assertion.
- Embedding similarity to GDELT news: very low — no published news corroborates HAARP weather control
- Dashboard: appears in claims list with very low evidence proximity

**Reader sees:** "Post contains a specific claim. No published news sources describe a similar event. Source is unverified social media account."

**Important:** the system does **not** say "this is false." It says "this claim has no news corroboration and comes from an unverified source." The reader draws the conclusion. This is the correct design.

---

### Case 4 — Embedding similarity reveals same event, different framing
**User in Tab 2 (Embedding Analysis):**
- Text A: "Wildfire smoke from BC fires causes air quality alert across Alberta"
- Text B: "Forest fire smoke triggers health advisory for Edmonton residents"
- Cosine similarity: **0.71** — high similarity confirms these describe the same event despite different wording
- Interpretation: reliable evidence these posts are about the same weather event

---

### Case 5 — Category cluster analysis validates ingestion taxonomy
The Category Cluster Analysis button in Tab 2 shows:
- `scientific` posts: higher mean intra-category similarity — scientific vocabulary is consistent and specific
- `conspiracy` posts: lower mean intra-category similarity — conspiracy narratives vary widely in framing and vocabulary
- **Implication:** the keyword-based ingestion taxonomy captures semantically coherent groups, confirming the ingestion design is sound

---

## Project Structure

```
ClimateClaimVerifier/
├── README.md                           ← this file — overview and quick start
├── project_description.md             ← authoritative project description
├── project_structure.md               ← detailed technical reference
├── app.py                              ← Streamlit dashboard (Weeks 1 + 2)
├── pyproject.toml                      ← dependencies managed by uv
├── .env                                ← Bluesky credentials (gitignored — never commit)
├── .gitignore
│
├── data/
│   ├── ingested.db                     ← SQLite: posts + classifications (generated at runtime)
│   ├── claim_eval.csv                  ← 100 hand-labeled posts (44 real + 56 synthetic) — benchmark
│   └── embedding_pairs.csv            ← Week 2: 20 pairs for embedding quality eval
│
└── src/
    └── climate_verifier/
        ├── config.yaml                 ← single source of truth for all parameters
        ├── __init__.py
        │
        ├── ingestion/                  ← Infrastructure layer (no LLM)
        │   ├── bluesky.py              ← atproto SDK: fetches posts + author follower counts
        │   ├── gdelt.py                ← GDELT API: news articles with rate-limit retry
        │   ├── store.py                ← SQLite: INSERT OR IGNORE deduplication + timestamps
        │   └── scheduler.py           ← APScheduler: 24h guard, blocking loop
        │
        └── pipeline/                   ← Processing layer (course concepts)
            ├── topic_filter.py         ← Pre-LLM gate: NA climate/weather keyword check
            ├── claim_classifier.py     ← Week 1: LLM binary claim/opinion detector
            ├── evaluate.py             ← Week 1: eval metrics (P/R/F1, confusion matrix, breakdowns)
            └── embedder.py             ← Week 2: sentence-transformer semantic analysis
```

### Infrastructure Layer (`ingestion/`)

No course concepts apply here. These files handle data collection: HTTP requests, API authentication, SQLite writes, and scheduling. They run before any model is invoked.

| File | Purpose |
|------|---------|
| `bluesky.py` | Fetches posts per keyword via atproto SDK. Includes author follower counts via a separate profile API call — high follower accounts spreading false claims are a signal worth surfacing. |
| `gdelt.py` | Queries GDELT for news articles per keyword. Free, no auth. Implements exponential backoff for 429 rate-limit responses. |
| `store.py` | `INSERT OR IGNORE` on `post_id` ensures deduplication across runs. Tracks last successful ingestion time in `metadata` table — used by the 24h guard in the scheduler. |
| `scheduler.py` | Runs one full ingestion cycle at startup, then schedules repeats at the configured interval. Guard: checks elapsed time since last run and skips if interval hasn't passed. |

### Pre-LLM Gate (`pipeline/topic_filter.py`)

Keyword-based relevance filter using two term lists:
- `WEATHER_TERMS` — confirms the post is about a climate or weather topic
- `NA_TERMS` — confirms geographic relevance to North America, filtered against `FOREIGN_TERMS`

Runs before any model call. Discarding irrelevant posts here saves LLM inference time and keeps the classification workload focused on relevant content.

### Week 1: Claim Classifier (`pipeline/claim_classifier.py`)

LLM binary classifier using `qwen2.5:3b` via Ollama — selected over `gemma2:2b` and `llama3.2:3b` in a bake-off on the labeled eval set. On the 100-row held-out benchmark it reaches **recall-on-CLAIM 0.938** (single-post, leak-free; precision ~0.70). See `project_description.md` (Weeks 4–5) for the eval-leakage and batch-mode corrections behind that number.

**Prompt design** (Week 4):
- 8 leak-free few-shot examples (4 claim / 4 opinion) on the hard boundary cases (false-but-checkable assertions, emotion wrapped around a fact, vague conspiracy, personal experience) — verified disjoint from the eval set
- A chain-of-thought `thought` field (first in the schema) so the model reasons before committing, plus a "checkability is not evidence" rule
- Explicit instruction to never judge truth — only claim presence
- Structured JSON schema enforced via regex extraction; `temperature: 0.0` for deterministic results
- Inference runs single-post (`llm_batch_size: 1`) — batching 16/call cost ~0.19 recall on the 3B model

**Persistence**: results saved to `classifications` table keyed by `post_id`. Posts are never re-classified — the classifier is idempotent. The dashboard only runs classification on unclassified posts.

**Evaluation** (`pipeline/evaluate.py`): runs the classifier against `data/claim_eval.csv` and reports per-class precision/recall/F1, the confusion matrix, false-negative vs false-positive counts, and error breakdowns by keyword category and post type. Success criterion: recall on CLAIM ≥ 0.90 (set in `config.yaml`).

### Week 2: Embedding Layer (`pipeline/embedder.py`)

Sentence-transformer embeddings using `all-MiniLM-L6-v2`.

**Why this model over the course baseline (`nomic-embed-text`):**
- Higher MTEB Semantic Textual Similarity scores — the dominant task here is measuring whether two posts describe the same event
- Runs offline without an Ollama server — the ingestion scheduler operates independently
- 384-dimensional vectors are sufficient for both evaluation and the future RAG layer

**Functions:**
- `embed(texts)` — normalised 384-dim embedding matrix
- `similarity(text_a, text_b)` — cosine similarity as float
- `eval_pairs(csv_path)` — runs evaluation against labeled pairs CSV
- `category_similarity_stats(db_path)` — mean intra-category similarity from live database

### Dashboard (`app.py`)

Two-tab Streamlit application:

**Tab 1 — Claim Classifier (Week 1):**
- Ingestion status and freshness indicator
- Pipeline statistics (ingested, classified, claims found, opinions rejected, pending)
- Classify next N / Classify ALL buttons with real-time progress
- Top 10 Claims and Top 10 Opinions ranked by engagement
- Classifier evaluation — recall-on-CLAIM criterion, confusion matrix, per-category error breakdown, misclassified-post audit

**Tab 2 — Embedding Analysis (Week 2):**
- Interactive similarity checker (enter two texts, see cosine similarity)
- Embedding pairs evaluation (runs against `data/embedding_pairs.csv`, shows pass/fail per pair)
- Category cluster analysis (bar chart of mean intra-category similarity from live DB)

### Configuration (`config.yaml`)

Single source of truth. All parameters — model names, ingestion intervals, keyword lists, embedding model — are set here. No hardcoded values appear in Python files.

---

## Running the Project

```bash
# 1. Install dependencies
uv sync

# 2. Start Ollama (required for Tab 1 — claim classifier)
ollama serve
ollama pull qwen2.5:3b   # first time only

# 3. Start ingestion (runs once immediately, then every 24h automatically)
#    Run in a separate terminal and leave it running
uv run python -m climate_verifier.ingestion.scheduler

# 4. Launch the dashboard (separate terminal)
uv run streamlit run app.py
```

Tab 2 (Embedding Analysis) works without Ollama running — it uses sentence-transformers only.

---

## Week Scope

| Week | Concept | Implementation in this project |
|------|---------|-------------------------------|
| 1 | LLM use cases, classification | `claim_classifier.py` — binary claim/opinion detector · `evaluate.py` — quality metrics |
| 2 | Tokenisation, embeddings, cosine similarity | `embedder.py`, `claim_eval.csv`, `embedding_pairs.csv`, Tab 2 |
| 4 | Prompt engineering & evaluation | leak-free 8-shot + CoT prompt; found + fixed eval leakage and the batch-mode artifact (recall 0.750 → 0.938 single-post); precision is the open problem |
| 5 | Fine-tuning with LoRA adapters | trained a precision-targeted QLoRA adapter; measured the recall/precision tradeoff is unbreakable at the target corner → ship base, no adapter; Base-vs-Adapter demo tab (`models/`, Tab 3) |
| 6–8 | Future work | RAG evidence retrieval (ChromaDB), source credibility lookup, multimodal |

Your workflow from now on will be:

Run ingestion locally as normal (Bluesky/GDELT fetch)
When you have a backlog: export → Colab → import
Continue with dashboard/evaluation locally