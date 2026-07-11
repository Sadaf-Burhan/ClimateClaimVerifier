# ClimateClaimVerifier — Week 1–8 Project Journey

**Mission.** A *scanner* for **Bluesky** climate & extreme-weather posts that surfaces **signals** to
help a reader judge a post — is it a checkable claim or an opinion? does published news corroborate it?
who posted it, with what reach and sources? It **never renders a true/false verdict** ("metal detector,
not judge"). The reader concludes.

**Two ideas run through every week:**
- **Recall-first classification, precision recovered downstream.** Catch every claim (a missed claim is
  unrecoverable); tolerate false positives because later stages (corroboration, source context, vision)
  filter them.
- **Signals, not verdicts.** Nothing asserts truth; everything is descriptive and points the reader to
  primary sources.

---

## Week 1 — Introduction to LLMs → Claim/Opinion Classifier
**Tried.** An LLM binary classifier (`has_claim` / opinion) via Ollama; a model bake-off across
`gemma2:2b`, `llama3.2:3b`, `qwen2.5:3b` on a hand-labeled eval set; structured JSON output for
parseable results; a metrics harness (precision / recall / F1, confusion matrix, per-category breakdowns).
**Worked.** `qwen2.5:3b` selected. Established the **recall-first success criterion** (recall-on-CLAIM
≥ 0.90 is the pass/fail; a missed claim is the costly error). Structured outputs eliminated parse errors.
**Didn't / learned.** Small models are the binding constraint. Precision was set aside as *secondary* —
a decision that shaped the whole downstream architecture.

## Week 2 — Tokenization & Embedding → Semantic Analysis
**Tried.** `all-MiniLM-L6-v2` (sentence-transformers) embeddings; evaluated on labelled domain pairs and
on how cleanly DB categories cluster; considered `nomic-embed-text`.
**Worked.** `all-MiniLM-L6-v2` (384-dim, strong STS scores) validated — and reused later as the **RAG
retrieval engine** in Week 6 (same cosine similarity, new job).
**Didn't / learned.** 384 dimensions are plenty for pairwise comparison; no need for a heavier model.

## Week 3 — Transformer Architecture (conceptual) + Data Foundation
**Tried.** A conceptual week (attention, how the classifier model works under the hood). For the project
it consolidated the **data backbone**: Bluesky (atproto) + GDELT ingestion, SQLite storage, the NA
climate/weather **topic filter**, and the categorized keyword sweep (scientific / extreme-events /
sensationalist / conspiracy / combinations).
**Worked.** A reliable ingestion → topic-filter → store pipeline the later weeks all build on.
**Note.** Lighter on a distinct ML deliverable; understanding applied + infrastructure hardened.

## Week 4 — Prompt Tuning & Evaluation
**Tried.** Few-shot prompting, chain-of-thought structured output, the `classify_v3` pattern (task +
persona in the system role), a frozen 100-row eval set stratified by category *and* post-type.
**Worked.** 8 **leak-free** few-shot examples (4 claim / 4 opinion) + a "checkability is not evidence"
rule. Honest, reproducible metrics.
**Didn't / the two critical findings.**
- **Eval-set leakage** — the few-shot examples were themselves eval posts, inflating metrics (the
  0.854 / 0.938 numbers were fake). Fixed with synthetic leak-free examples → honest recall **0.750**.
- **Batch-mode artifact** — classifying 16 posts per prompt drifted the small model, costing ~**0.19
  recall**. Single-post (`llm_batch_size: 1`) recovered recall to **0.938**. *Recall was fixed by config
  (single-post), not by a better prompt or model.* **Precision ~0.70 emerged as the real problem.**

## Week 5 — Fine-Tuning with LoRA Adapters
**Tried.** QLoRA (Unsloth, Colab T4, rank 8) on `qwen2.5:3b`, targeting **precision**. A
class-balance-as-hyperparameter sweep (26 / 40 / 50 % claim). Teacher-labeling with `qwen2.5:7b`.
GGUF quantization for a demo (`q4_k_m` drifted on borderline cases; `q8_0` near-lossless).
**Worked.** The training pipeline, a Base-vs-Adapter **comparison harness**, and — most importantly —
a clean **negative result**.
**Didn't.** **No balance reached recall ≥ 0.90 AND precision ≥ 0.85 together** (a genuine tradeoff
frontier). Decision: **no adapter in production** (demo-only). Precision is *deliberately* pushed
**downstream** to evidence matching. The negative result *validated* the recall-first architecture —
fine-tuning wasn't a failure, it was the experiment that proved precision isn't a model problem.

## Week 6 — Semantic Search & RAG → Evidence Matching
**Tried.** A ChromaDB index of **GDELT news** (`all-MiniLM-L6-v2`, cosine); dense top-k retrieval; then
**retrieve-then-rerank** — an LLM **corroboration re-rank** that asks *"does any article report the same
specific event?"* (corroborated / partial / none).
**Worked.** The re-rank separates **topical overlap from real corroboration** — dense similarity alone
scores a HAARP conspiracy post 0.62 just for sharing the "weather" topic; the re-rank correctly returns
*none*. Built the **reader signal** + the **reach-vs-support red flag**, with three guards against false
flags: **self-citation**, **official-source allowlist**, and **reshare provenance** (credibility travels
with the origin). Tightened the corroboration prompt (grounding-only, never-judge-truth, mandatory
citation, scoped "none").
**Didn't / learned.** Naive dense similarity conflates topic with corroboration (fixed by the re-rank).
The corpus is **headline-only** (GDELT returns titles, not article bodies) — a documented limitation;
a full news-parser was ruled **out of scope** (bot walls, paywalls, JS rendering). "No match" is
**ambiguous, not negative** — scoped to the retrieved set, never a claim that no evidence exists.

## Week 7 — Multimodal Models → Gated Vision
**Tried.** A **gated** image signal (`qwen2.5vl:7b`) that fires *only* on edge-case posts (image-claims
not already resolved by metadata) — a real on-the-ground photo *rescues* a genuine post from a false
red flag; a cartoon/meme *reinforces* suspicion.
**Worked.** The rescue is correct (`@yourcier`: real event photo, uncorroborated → flag cleared). Gate
by **uncertainty, not category** (FPs occur everywhere — the eval showed combinations at 47%). The
official allowlist was expanded when widening exposed a flood of unrecognized NWS relay bots.
**Didn't / the two findings.**
- **The WebP bug** — Bluesky serves images as **WebP**, which Ollama's image loader can't decode; it
  silently fed the model blank images (identical canned descriptions). The real unlock was re-encoding
  to **JPEG** before inference.
- **~1 % coverage** — measured on 745 posts / 119 image-claims (96 % scanned), vision changed **exactly
  1 red-flag**. Its trigger combination (real photo **+** depicts the event **+** uncorroborated **+**
  reach) is rare, and it *must* stay strict (rescuing on any photo would hurt precision). Honest verdict:
  a **sound but low-coverage supplement**; **corroboration is the precision workhorse.** Not merged on a
  shakeout — accumulating data to re-measure. *(The twin of the Week-5 LoRA finding.)*

## Week 8 — Evaluation & Beyond → Finalization
**Tried.** The autonomous daily-pipeline design; the **Trust Checker** — a user-facing "check a Bluesky
post" product (classify → corroborate → credible sources → red flag → verification links + a
"how to spot misinformation wording" guide); the finalization / merge strategy.
**Worked.** Trust Checker shipped on the branch; the full system design is documented
(`FINAL_SYSTEM_DESIGN.md`).
**In progress.** Dashboard overhaul (sidebar nav, drop week labels, human-readable bullet summaries,
clickable Bluesky links everywhere, sort/filter criteria); the `run_daily()` autonomous loop; this
journey document.

---

## Cross-cutting lessons
- **Config beats model.** The biggest recall win (0.750 → 0.938) was single-post inference, not a
  fancier model — and fine-tuning couldn't beat the config-level baseline.
- **Negative results are results.** LoRA (can't beat ~0.70 precision) and vision (~1 % coverage) both
  *validated* the architecture by ruling out easy wins and locating precision downstream.
- **Discipline around evaluation.** A frozen, leak-free eval set caught inflated metrics twice; never
  relabel it, only extend it.
- **Signals over verdicts.** Every stage stays descriptive and points to primary sources — the design
  constraint that keeps the tool honest and defensible.

## Final state
Recall-first `qwen2.5:3b` classifier (single-post) → GDELT corroboration (RAG + LLM re-rank) → source /
reach / self-citation / official-source signals → gated vision on edge cases → **reader signal + red
flag**, surfaced through the **Trust Checker** dashboard. No adapter in production; vision a documented
low-coverage supplement; corroboration the precision workhorse.
