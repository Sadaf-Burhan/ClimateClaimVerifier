# Project Description
## Climate Claim Scanner — Domain-Specific Social Media Claim Detection using LLM-Appropriate Tasks

> This is the authoritative project description. `README.md` is the overview and quick-start view; `project_structure.md` is the technical file-by-file reference.

---

## Problem Context

Every time a major weather event strikes North America — a wildfire, a heat dome, a flood — social media fills within hours. Some posts are accurate and sourced from official agencies. Many are not: exaggerated headlines, anonymous accounts amplifying fear, conspiracy narratives dressed up as fact. When a person cannot distinguish a verified Environment Canada tornado warning from a sensationalist blog post or a HAARP conspiracy claim, trust in actual climate science erodes — and the misinformation spreads further precisely because it looks like the real thing.

This project builds a credibility signal aggregator: a system that ingests climate and extreme weather posts from social media, identifies which ones contain verifiable factual claims, and surfaces the structural signals — source quality, evidence proximity, language patterns — that help a reader evaluate credibility for themselves.

---

## Vision behind this project

Climate change content on social media attracts significant sensationalism and misinformation — claims like "CO₂ is just a trace gas (0.04%), so it can't affect the planet", "Record cold winters prove that global warming isn't happening", or "Extreme weather events (like hurricanes or wildfires) are engineered by the government or deliberately started to push 'green' agendas". Unlike daily weather, climate change topics unfold over long timescales and are harder for most people to evaluate, making them particularly susceptible to exaggerated public opinion and fabricated statistics. The goal of this project is to help build public awareness of what constitutes a verifiable claim versus noise.

This project ingests live posts from Bluesky in real time and classifies them as claims or opinions using an LLM. Tools like Climate Feedback and Google Fact Check Explorer already exist for verification; this project is not a verification system — it is a detection pipeline that surfaces which posts contain verifiable factual assertions, leaving evaluation to the reader. An analogy: a metal detector's job is to find metal, not to assess its value. The scanner narrows the search space and surfaces key credibility signals, making it easier for humans to evaluate.

---

## The Central Design Decision: What This System Does NOT Do

This is the most important architectural decision in the project and the one most easily misunderstood.

**This system does not use an LLM to verify whether a claim is true or false.**

This is not a gap in ambition — it is a deliberate design constraint grounded in how language models actually work.

### Why LLMs Cannot Verify Facts

LLMs are next-token predictors. When a language model generates text, it is statistically predicting which word should come next based on patterns learned during training. This mechanism is powerful for generating, classifying, summarising, and analysing text — but it is fundamentally incompatible with factual verification for three reasons:

**1. Knowledge cutoff.** LLMs are trained on data up to a fixed date. A social media post about a wildfire that occurred yesterday has no representation in the model's training data. Asking "is this claim true?" produces an answer based on statistical associations with similar-sounding text — not a comparison against actual ground truth.

**2. Hallucination.** When a language model encounters a question whose answer falls outside its training distribution, it confabulates — generating confident-sounding, plausible text that may be entirely false. The model cannot flag its own uncertainty reliably, because uncertainty awareness is not what it was trained to produce. It was trained to complete text, not to assess truth.

**3. Text generation is not fact checking.** The same statistical mechanism that allows a model to write a poem, summarise a document, or classify sentiment cannot be repurposed to evaluate a claim against external physical reality. These are categorically different tasks. Fact checking requires access to authoritative external data — weather service records, scientific measurements, published reporting. A language model has none of these available at inference time. It has text patterns.

The implication is direct: asking `qwen2.5:3b` whether "HAARP controls the weather and caused the 2024 floods" is true would produce a statistically plausible answer — not a factually grounded one. Building a system around that answer would be misleading.

---

## What This System DOES Do: Appropriate LLM Use

The system uses the LLM for exactly one task it is genuinely well-suited for: **binary text classification**.

### Week 1 — Claim Presence Detection

The question the LLM answers is:

> "Does this text contain a verifiable factual statement — a named event, a measurement, an official warning, a documented occurrence — or is it an opinion, emotional reaction, or political commentary?"

This is a linguistic pattern recognition task, not a truth evaluation. A well-read generalist can reliably distinguish:

- "NOAA confirms strongest Atlantic hurricane season since 2005" → **factual claim** (official source, specific comparative fact)
- "This storm is absolutely terrifying, I can't believe how bad it is" → **opinion** (emotional reaction, no verifiable content)
- "The government is using HAARP to control the weather" → **claim** (specific entity, mechanism, effect — structurally a claim, even if false)
- "Higher CO₂ makes the planet greener and benefits agriculture" → **claim** (specific gas, mechanism, effect — structurally a claim, even if contested)

The LLM sees the structure and language of the post. It does not need to know whether the claim is accurate to identify that one exists. This distinction — claim presence vs. claim truth — is what makes the classification task appropriate for an LLM.

**Labeling rule for the boundary cases:** a post that expresses opinion, sarcasm, or hostility but contains at least one checkable assertion is a **claim**. A post with tone but nothing checkable — a joke, a vague prophecy, a political viewpoint — is an **opinion**. This rule is encoded in the evaluation set and is the standard the classifier is measured against.

Implementation: `qwen2.5:3b` (selected over `gemma2:2b` and `llama3.2:3b` in a measured bake-off on the eval set) with 6 few-shot examples, structured JSON output, temperature 0 for deterministic results. The few-shot examples deliberately include the hard boundary cases — a false-but-checkable assertion, emotion wrapped around a checkable fact, and a personal experience that is not externally checkable — and the prompt instructs the model to never judge truth, only claim presence. The model never sees a question like "is this true?" It sees: "does this post contain a verifiable factual statement?"

### Week 2 — Embedding-Based Semantic Analysis

The second layer uses sentence-transformer embeddings (`all-MiniLM-L6-v2`) to measure semantic similarity between posts. This is entirely non-LLM inference — it is mathematics on vector representations of text.

What this enables:
- Measuring whether two posts describe the same event (high cosine similarity = same topic)
- Identifying semantic coherence within keyword categories (scientific posts should cluster tighter than conspiracy posts)
- Laying the groundwork for future evidence retrieval: matching a classified claim against a database of GDELT news articles

The LLM plays no role in this layer.

---

## System Design: How the Pipeline Works

```
Social Media Posts (Bluesky + GDELT)
           │
           ▼
  [1] INGESTION LAYER          — no LLM
      Bluesky API + GDELT API
      Deduplicated into SQLite
      Runs every 24 hours
           │
           ▼
  [2] TOPIC FILTER             — no LLM
      Keyword-based gate:
      North American weather/climate relevance check
      Irrelevant posts discarded here
           │
           ▼
  [3] CLAIM CLASSIFIER         — LLM (appropriate use)
      qwen2.5:3b via Ollama
      Binary: has_claim / no_claim
      Opinions rejected with explanation
      Claims proceed to next stage
           │
           ▼
  [4] EVIDENCE MATCHING        — no LLM RAG (Week 6)
      all-MiniLM-L6-v2 embeddings + ChromaDB
      Retrieve top-k similar GDELT news articles per claim
      -> "evidence proximity" signal (does published news cover it?)
           │
           ▼
  [5] SIGNAL ASSEMBLY          — no LLM
      Per claim: claim + reason, evidence proximity + matched articles,
      source context (domain / account type / followers),
      engagement / reach (likes, reposts, replies, quotes)
           │
           ▼
  [6] READER SUMMARY           — suggestive, never a verdict
      Plain-language description of the signals; flags the red flag:
      high reach + low support (no corroboration, unverified source)
           │
           ▼
  [7] DASHBOARD                — Streamlit
      Top claims by engagement, the per-claim signal summary,
      classifier evaluation metrics, Base-vs-Adapter demo
```

### What Each Component Decides — and What It Does Not

| Component | What it decides | What it does NOT decide |
|-----------|----------------|------------------------|
| Topic filter | Is this post about NA climate/weather? | Is this claim accurate? |
| Claim classifier (LLM) | Does this text contain a verifiable factual statement? | Is that statement true? |
| Evidence matching (RAG) | How closely does published news cover this claim? | Whether the claim is true or false |
| Signal assembly | Which signals to surface (claim, proximity, source, reach) | A credibility verdict or score |
| Reader summary | A plain-language, suggestive description of the signals | Any yes/no or numeric truth judgment |
| Dashboard | Surfaces the signals + summary for the reader | The reader's conclusion |

**The final truth judgment is never made by the system.** It equips a reader with structured signals — claim detected, semantic proximity to published news, source context (domain / account type / followers), and engagement/reach (likes, reposts, replies, quotes) — plus a plain-language summary. The reader draws the conclusion. There is **no numeric credibility score and no HIGH/MEDIUM/LOW verdict**; an early design had one and it was deliberately removed, because an LLM cannot verify facts (see "What This System Does NOT Do").

**The headline signal is reach vs. support.** A claim spreading widely (high engagement) with no news corroboration and from an unverified source is the misinformation red flag — **high reach, low support.** The reader summary names that combination explicitly, making the mismatch between how far a claim travels and how well it is supported legible — without the system ever asserting the post is false.

---

## How Success Is Measured

### Week 1 — Classifier Quality

**Why accuracy alone is not the criterion.** The classifier gate has two failure directions with very different costs:

- **False negative** — a post containing a real verifiable claim is labeled opinion. It is discarded at the gate and never reaches the embedding layer or the dashboard. A missed claim is unrecoverable. This is the costly error.
- **False positive** — an opinion is labeled claim. It surfaces on the dashboard, where low evidence proximity and source context let the reader discount it. This error is cheap and self-correcting downstream.

A single accuracy number treats those two errors as equal. They are not — so accuracy is reported for continuity, but it is not the pass/fail criterion.

**Evaluation set.** `data/claim_eval.csv` contains 100 hand-labeled posts (44 real, 56 synthetic) with human-verified claim/opinion labels, stratified two ways:

1. **By `keyword_category`** — all five ingestion categories (`scientific`, `extreme_events`, `sensationalist`, `conspiracy`, `combinations`), so the eval mirrors what the pipeline actually ingests.
2. **By `post_type`** — the linguistic shape of the post: `official_alert`, `news_event`, `scientific_finding`, `false_but_checkable` (conspiracy assertions that are structurally claims), `denial_with_stat` (hostile rants with an embedded checkable fact), `mixed_emotion_fact` (emotional posts wrapped around an official warning), `sarcasm_joke`, `hyperbole_doom`, `political_viewpoint`, `vague_conspiracy`, `emotional_reaction`, `rhetorical_question`.

Hard boundary cases — sarcasm, denial rants with embedded statistics, emotion wrapped around official warnings — are deliberately oversampled, because that is where a small model fails first and where a single aggregate number would hide it.

**Metrics reported** (implemented in `src/climate_verifier/pipeline/evaluate.py`, surfaced both as a CLI report and in the dashboard):

| Metric | Why it matters |
|--------|---------------|
| **Recall on CLAIM** | **Success criterion: ≥ 0.90** — the share of real claims the gate lets through. A missed claim is the unrecoverable error. |
| Precision on CLAIM | Reported alongside — how many surfaced claims are genuine. |
| Per-class precision / recall / F1 | Full picture for both classes, not just the positive one. |
| Confusion matrix | Shows exactly where the errors sit. |
| False negatives vs false positives | Quantifies the cost asymmetry directly. |
| Error breakdown by `keyword_category` and `post_type` | "Misses are concentrated in sarcasm; zero misses on official alerts" is a stronger and more honest result than one aggregate number. |

Run it:

```bash
uv run python -m climate_verifier.pipeline.evaluate
```

The `reason` field on each classification must be coherent plain-language text explaining the decision — not a model-generated excuse, but a legible audit log entry.

### Week 2 — Embedding Quality

A 20-pair evaluation set (`data/embedding_pairs.csv`) contains 10 similar and 10 dissimilar text pairs. The embedding quality is evaluated by cosine similarity scores.

| Metric | Target |
|--------|--------|
| Mean similar-pair similarity | > 0.60 |
| Mean dissimilar-pair similarity | < 0.40 |
| Separation (similar − dissimilar) | > 0.20 |
| `scientific` intra-category > `conspiracy` intra-category | confirmed |

**Measured results (all-MiniLM-L6-v2 on domain pairs):**

| Metric | Achieved |
|--------|----------|
| Similar mean | 0.64 ✅ |
| Dissimilar mean | 0.21 ✅ |
| Separation | 0.43 ✅ |

---

## Use Case Examples

### Use Case 1 — High-credibility factual claim
**Post:** "Environment Canada issues extreme heat warning for BC interior through Thursday — temperatures expected to reach 42°C in Kamloops."

- Topic filter: passes (contains heat warning, named province, named location)
- Claim classifier: `has_claim: true` — official source, specific warning type, named location, temperature measurement
- Embedding similarity to GDELT news: high (news articles covering same heat event will cluster near this post)
- Dashboard: appears in Top Claims, ranked by engagement
- Reader signal: "This post contains a verifiable factual claim from an official source. Published news covers a similar heat event."

### Use Case 2 — Opinion discarded at classifier gate
**Post:** "This summer is absolutely terrifying, my backyard has been covered in wildfire smoke for two weeks."

- Topic filter: passes (contains wildfire smoke)
- Claim classifier: `has_claim: false` — personal experience, emotional language, no specific verifiable factual content
- Discarded here. Does not proceed to embedding or dashboard.
- Reader signal: not shown — no verifiable claim to surface

### Use Case 3 — Conspiracy claim correctly identified as claim
**Post:** "HAARP technology is being used to cause the flooding in Alberta — wake up, this is weather warfare."

- Topic filter: passes (contains flooding, Alberta)
- Claim classifier: `has_claim: true` — specific mechanism (HAARP), specific effect (flooding), specific location (Alberta). Structurally a factual assertion, even if the assertion is false.
- Embedding similarity to GDELT news: very low — no published news corroborates HAARP causing Alberta flooding
- Dashboard: appears in claims list with low evidence proximity
- Reader signal: "This post contains a specific claim. No published news sources describe a similar event. Source is unverified social media account."
- Note: the system does NOT say "this is false." It says "this has no news corroboration and comes from a low-credibility source." The difference matters.

### Use Case 4 — Embedding similarity analysis
**User checks two posts in the dashboard:**
- Text A: "Wildfire smoke from BC fires causes air quality alert across Alberta"
- Text B: "Forest fire smoke triggers health advisory for Edmonton residents"
- Cosine similarity: 0.71 — high similarity confirms these posts describe the same event despite different wording

### Use Case 5 — Category cluster analysis reveals semantic structure
The dashboard Category Cluster Analysis shows:
- `scientific` posts: higher mean intra-category similarity — scientific vocabulary is consistent
- `conspiracy` posts: lower mean intra-category similarity — conspiracy narratives vary widely in framing
- This confirms the keyword-based ingestion taxonomy captures semantically coherent groups, validating the ingestion design

---

## Scope Decision — Canada vs North America

A data-driven feasibility study was conducted comparing Canada and North America as candidate scopes.

Canada offered simpler source validation (fewer authoritative bodies to whitelist). However, Canada's most severe weather is seasonal and geographically concentrated, resulting in limited event activity during certain periods.

North America was selected:
- Continuous flow of natural hazard events year-round, eliminating seasonal data gaps
- Captures the cross-border nature of many disasters (wildfire smoke corridors, atmospheric rivers, drought zones)
- Much larger volume of social media content and misinformation to test the pipeline against
- Greater diversity of sources — official agencies, local news, social media, conspiracy accounts

---

## Data Stack Decision

A feasibility study compared candidate data sources for free, automated, text-based ingestion of current climate-related content.

**Twitter (X):** Rejected — the API costs $100–$5,000/month with severely limited free access.

**Kaggle pre-collected datasets:** Available climate Twitter datasets cover only up to 2019. This misses the post-2020 misinformation landscape — AI-generated content, state-sponsored disinformation, and algorithmic engagement farming represent a fundamentally different environment.

**Final stack — GDELT + Bluesky:**
- Both free and actively maintained
- Provides live, current text-based data — not historical snapshots
- Bluesky's API is open, well-documented, and returns rich metadata (engagement, follower counts, timestamps)
- GDELT provides structured global news data, enabling cross-referencing claims against published reporting
- Text-based only: image, audio, and video content are out of scope for Weeks 1–2

---

## Week 2 — Tokenisation and Embedding Evaluation

### Embedding Model Decision

The course baseline uses `nomic-embed-text` (via Ollama). For this project, `all-MiniLM-L6-v2` (via `sentence-transformers`) was selected.

**Reasoning:**
- The primary task is Semantic Textual Similarity (STS): measuring how closely two climate posts describe the same event. `all-MiniLM-L6-v2` achieves higher MTEB STS benchmark scores than `nomic-embed-text`.
- Runs fully offline — no Ollama server required. The ingestion scheduler and Streamlit dashboard do not need a second inference server.
- 384-dimensional vectors are sufficient for pairwise comparison and as the foundation for the Week 6 RAG evidence retrieval layer.

### Domain Pair Evaluation

Twenty embedding pairs in `data/embedding_pairs.csv` — 10 similar and 10 dissimilar — evaluated whether the model captures meaningful semantic distance in the climate domain.

**Similar pairs:** same event type, different wording (e.g., "Heat dome breaks BC temperature record" / "Record heatwave scorches Pacific Northwest"). Expected > 0.6.

**Dissimilar pairs:** posts from semantically different categories (e.g., factual scientific claim paired with conspiracy framing). Expected < 0.4.

Results: similar mean 0.64, dissimilar mean 0.21, separation 0.43. All targets met.

### What Embeddings Enable

1. **Category separation signal**: Confirms the keyword ingestion taxonomy is semantically coherent — `scientific` posts cluster together, `conspiracy` posts scatter.
2. **Foundation for Week 6 RAG**: GDELT news articles will be embedded into ChromaDB; each classified claim will be matched against the collection using the same cosine similarity metric.
3. **Classifier evaluation baseline**: The 100-post labeled eval set in `data/claim_eval.csv` converts "the classifier kind of works" into measured per-class precision, recall, F1, and a recall-on-CLAIM pass/fail criterion.

---

## Week 4 — Prompt Engineering and Evaluation

### The task being tuned

Week 4 tunes a **pure binary classification** prompt (claim vs opinion) with a fixed
output set, evaluated by exact match — not an open-ended instruction-following task. The
chain-of-thought `thought` field and the `reason` field are internal scaffolding and
explainability, never scored. The success criterion is **recall-on-CLAIM ≥ 0.90** on the
100-row `data/claim_eval.csv`, with precision, F1, the confusion matrix, and per-category
breakdowns (by `keyword_category` and by `post_type`) reported as guardrails.

### Prompt iteration

The prompt evolved from a single zero-shot instruction to a structured few-shot prompt:
8 examples (4 claim, 4 opinion) chosen as minimal pairs on the hardest boundaries
(tone-vs-checkability, false-but-checkable, vague-conspiracy, personal-experience); a
chain-of-thought `thought` field placed first in the structured output so the model
reasons before committing; and a "checkability is not evidence" rule to stop the model
demoting unverified-but-specific claims. The task definition lives in the system role; the
few-shot examples and the JSON format example stay in the user message (moving them to the
system role dropped recall to 0.708 — a 3B model needs the output format co-located with
the input).

### Two corrections that overturned the headline number

The benchmark did its job by exposing two problems that had been inflating every result:

1. **Eval-set leakage.** Several few-shot examples were themselves eval-set posts —
   including the exact `false_but_checkable` and `denial_with_stat` posts they were meant to
   test. The model was being shown the test answers, inflating recall to ~0.85.
   *Correction:* every few-shot example was replaced with a synthetic post verified disjoint
   from `claim_eval.csv`; honest held-out recall fell to **0.750**. (A parallel course-folder
   implementation had 6/8 examples leaked, inflating it to 0.938.) Leakage prevention is now
   enforced at the database level via an `in_eval_set` flag (and a symmetric `in_train_set`
   flag for Week 5), not by ad-hoc text matching.

2. **Batch-mode artifact.** The remaining shortfall was not a model limitation. Classifying
   16 posts per LLM call made the 3B model drift and under-call claims. Holding the leak-free
   prompt constant and switching to **single-post inference** (`llm_batch_size: 1`) lifted
   recall **0.750 → 0.938** and cleared the categories previously blamed on a "weights bias":
   `false_but_checkable` 5/6, `denial_with_stat` 2/2. Recall was a *configuration* problem,
   not a prompting or fine-tuning one. (Batch-16 had been chosen for CPU throughput; that
   choice was silently costing ~0.19 recall — single-post is ~16× more calls, run on GPU/Colab.)

### Result: where the prompt succeeds and where it fails

At leak-free single-post, **recall-on-CLAIM is 0.938** (PASS) with only 3 missed claims out
of 48. Claim recall is strong across every claim type — `news_event`, `official_alert`,
`scientific_finding`, `false_but_checkable`, and `denial_with_stat` are all near-perfect.
The remaining failure is **precision (~0.70)**: false positives concentrated in
`vague_conspiracy` posts ("weather manipulation is real, they don't want you to know")
over-called as claims.

A targeted prompt fix to tighten the OPINION boundary **overshot**: precision rose to 0.94
but recall collapsed to 0.69, because a single instruction cannot teach the model to
separate a *specific* conspiracy claim ("cloud seeding caused the Dubai floods last week")
from a *vague* one ("weather manipulation is real"). Prompting reaches high recall **or**
high precision on this boundary, not both — the prompt-engineering ceiling, and the
evidence-backed motivation for the Week 5 model-level experiment.

### Lessons / corrections of record

- **Never measure a prompt against a benchmark whose few-shot examples overlap it.** Verify
  disjointness before trusting any number; enforce it with a DB flag.
- **Isolate one variable at a time.** The "false_but_checkable is unfixable" conclusion was a
  confound of leakage *and* batch mode; separating them dissolved it.
- **Recall-first is deliberate**, given the gateway architecture (a missed claim is
  unrecoverable; a false positive is discounted downstream). F1 would reward trading recall
  for precision, which is backwards here.

---

## Week 5 — Fine-Tuning with LoRA Adapters

### The corrected journey (recall was a config problem, not a model problem)

Earlier analysis concluded the classifier needed LoRA to fix missed claims in the
`false_but_checkable` and `denial_with_stat` categories. That conclusion was wrong,
and the correction is itself an important project lesson:

1. **Eval-set leakage** inflated every number — several few-shot examples were
   themselves eval-set posts. After replacing them with synthetic examples disjoint
   from the eval set, the honest held-out recall fell from an inflated 0.854 to 0.750.
2. **The remaining recall shortfall was a batch-mode artifact.** Holding the
   leak-free prompt constant and changing only the batch size, recall went from
   0.750 (16 posts per LLM call) to **0.938 (one post per call)** — the 3B model
   drifts when classifying many posts at once. Recall is therefore solved by a
   configuration change (`llm_batch_size: 1`), not fine-tuning.

The eval set is now 100 hand-labeled posts (44 real, 56 synthetic), and the
inference default is single-post.

### What LoRA actually targets here: the precision/recall tradeoff

With recall met (0.938), the open problem is **precision (~0.70)**: vague conspiracy
posts ("weather manipulation is real and they don't want you to know") get over-called
as claims. A prompt fix tightening the OPINION boundary was tested and **overshot** —
it lifted precision to 0.943 but crashed recall to 0.688, because a single instruction
cannot teach the model to tell a *specific* conspiracy claim ("cloud seeding caused the
Dubai floods last week" → claim) from a *vague* one ("weather manipulation is real" →
opinion). Prompting can reach high recall **or** high precision on this boundary, but
not both. Breaking that tradeoff is the legitimate, evidence-backed case for LoRA.

This is an **Option B (accuracy-gap)** fine-tune in the Week 5 framework: train on the
specific failure slice — contrastive vague-vs-specific conspiracy pairs — to raise
precision without sacrificing recall. Training data: a hand-labeled hard core on the
conspiracy-specificity boundary plus teacher-model-labeled real posts for coverage,
~100–150 examples, all disjoint from the 100-row eval set (the leakage rule applies to
training data too).

### Generalisation and drift monitoring

**The adapter learns checkability structure, which generalizes to unseen events; drift
monitoring targets framing shift and eval staleness, not new phenomena.** The adapter is
not taught facts about specific events — it is taught a topic-independent behavior
("a specific, named, checkable assertion is a CLAIM; a vague accusation is an OPINION").
So a future phenomenon the model never saw in training (e.g. a new rifting event with a
named location and a measurement) is classified correctly by its *shape*, without the
model knowing the event exists. Drift monitoring is therefore aimed not at new science
but at (1) **new opinion/conspiracy framings** that did not exist at training time, and
(2) **eval-set staleness** — the 100-row benchmark reflects today's posts. The plan:
periodically re-sample recent posts into a fresh eval slice, re-run recall and precision,
and retrain the adapter only when the metrics degrade.

### Result: the adapter shifted the tradeoff but did not break it

A precision-targeted QLoRA adapter (qwen2.5:3b base, rank 8, Unsloth, ~120 leak-free
examples = hand-labeled conspiracy-boundary seed + teacher-labeled real posts) was
trained and evaluated on the unchanged 100-row eval set across three training balances,
measured against the base single-post classifier:

| Configuration | recall-on-CLAIM | precision-on-CLAIM |
|---|---|---|
| Base, single-post (no adapter) | **0.938** | 0.703 |
| Adapter — 26% claim training mix | 0.812 | **0.951** |
| Adapter — 40% claim training mix | 0.854 | 0.788 |
| Adapter — 50% claim training mix | 0.958 | 0.742 |

The adapter's training balance is a dial: more opinion examples raise precision and
lower recall; more claim examples do the reverse. LoRA genuinely *shifted* the frontier
— it reached 0.95 precision, which prompt engineering never could. But every point with
recall ≥ 0.90 capped precision at ~0.74, and every high-precision point (≥ 0.85) had
recall below 0.82. **The target corner — recall ≥ 0.90 *and* precision ≥ 0.85 together —
is not on the adapter's frontier.** Fine-tuning moved the curve; it did not break the
tradeoff at the operating point that matters.

### Final verdict: no adapter in production — ship recall-first

**The classifier ships as the base `qwen2.5:3b` single-post configuration
(`llm_batch_size: 1`), recall 0.938, with no LoRA adapter.** The reasoning:

1. **Recall is the success criterion (≥ 0.90), and the base already meets it (0.938).**
   The classifier is a gateway where false negatives are unrecoverable and false
   positives are cheap and self-correcting — surfaced opinions are discounted downstream
   by evidence matching (no news corroboration → low evidence proximity). Precision is
   therefore genuinely secondary by design.
2. **In the recall-passing zone, the best adapter (precision 0.742) beats the base
   (0.703) by only ~0.04.** That marginal gain does not justify the cost of merging,
   GGUF conversion, and maintaining a second model in an Ollama pipeline.
3. **The adapter cannot deliver what would have justified it** — high recall *and*
   meaningfully higher precision together — because that corner is off its frontier.

The LoRA work was not wasted: its value is the **finding, not the artifact.** It proved
empirically that the precision/recall tradeoff on the conspiracy-specificity boundary is
fundamental for a 3B model — not reachable by prompting *or* fine-tuning at the desired
corner — which confirms the original architectural decision to keep the binary gate
recall-first and push precision to the downstream evidence-matching stage. (A trained
50/50 adapter, 0.958/0.742, exists and could be deployed purely to demonstrate the
technique, but it is not part of the production pipeline.)

**Was the fine-tuning a failure? No.** It was the experiment that converted "precision is
secondary" from an assertion into a *measured* fact. Choosing not to deploy the adapter is
a result, not a dead end: it is the difference between defending the recall-first
architecture with evidence and merely hoping it is right. The negative result — that no
fine-tuning balance reaches the recall ≥ 0.90 / precision ≥ 0.85 corner — is what makes the
decision to push precision downstream to evidence matching a justified design choice rather
than a convenient assumption.
