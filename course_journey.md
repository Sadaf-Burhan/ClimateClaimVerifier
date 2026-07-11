# ClimateClaimVerifier — Course Journey (Weeks 1–8)

Grounded in the weekly implementation specs (`LLM_Project/specs/week1–5`) plus the Week 6–8 build.
Per week: **what we tried / decided → what worked → what didn't (and how the design evolved).**

**Mission.** A *scanner* for **Bluesky** climate & extreme-weather posts that surfaces **signals** to
help a reader judge a post — never a true/false verdict. Two invariants: **recall-first** (catch every
claim; a missed one is unrecoverable) and **signals over verdicts** (precision recovered downstream).

---

## Week 1 — Introduction to LLMs → Claim/Opinion Classifier
- **Tried / decided.** A binary "does this post contain a verifiable factual claim?" gate via
  `ollama.generate(format="json")` on **`gemma2:2b`**, a fact-checker/climate-analyst persona, few-shot
  examples (claim / opinion / commentary), output `{has_claim, reason}`. Raw text only — metadata not
  needed at the presence stage. Standalone `main.py`.
- **Worked.** JSON output (auditable, parseable); the `reason` field for explainability; the recall-first
  framing was seeded here (the gate routes: discard vs pass forward).
- **Evolved.** `gemma2:2b` was the Week-1 pick; later bake-offs moved production to **`qwen2.5:3b`**.

## Week 2 — Tokenization & Embedding → Semantic Analysis
- **Tried / decided.** Compared 5 MTEB models on ~10 domain pairs; **chose `all-mpnet-base-v2`**
  (separation gap 0.369 vs `all-MiniLM-L6-v2` 0.326). Rejected `gte-small` (high close *and* far means =
  distribution mismatch). Built the minimal Streamlit v1. Locked the **data stack** — Google Trends +
  GDELT + Bluesky — over Twitter (API $100–5000/mo) and Kaggle (only ≤2019, misses post-2020 AI
  misinformation). **Scope: North America** (continuous events, cross-border coverage, volume).
- **Worked.** The embedding eval surfaced real failure modes: `hurricane→cyclone` 0.83 (synonyms) but
  `hurricane→wind` 0.52 (containment); "record high" read as a generic superlative, not a climate term.
- **Evolved.** Two reversals. (1) The system **reverted to `all-MiniLM-L6-v2`** (reused Week 3 onward —
  offline, 384-dim sufficient, and it became the RAG engine). (2) Week 2 planned a numeric **source
  credibility prior** (NOAA ≈0.95, unknown ≈0.40, suspicious ≈0.05) — **later removed entirely** (see
  the reader-signal arc below).

## Week 3 — Transformer Architecture → Dynamic Few-Shot Retrieval (RAG #1)
- **Tried / decided.** Replace fixed few-shot with **dynamically retrieved** examples: a ChromaDB store
  of labeled posts (`all-MiniLM-L6-v2`), **category-stratified, label-balanced** retrieval (k=20:
  5 categories × 2 labels × 2), with a **two-pass** fill (bucket → label backfill) so label balance
  (10 claim / 10 opinion) is guaranteed and topic diversity degrades gracefully. Topic filter used the
  post's ingestion `keyword_category` (no extra LLM call). Eval-set posts excluded from the store.
- **Worked.** The stratification rationale is sound — pure similarity retrieval risks a "wildfire =
  claim" topic shortcut instead of the structural pattern (mechanism + location + measurable effect).
- **Didn't / evolved.** Baseline to beat: fixed 8-shot **recall 0.792 / acc 0.820**, with an explicit
  **null hypothesis** — 8 well-chosen fixed examples may already span a binary decision space, so a flat
  result is a *valid finding*. Production ultimately uses **fixed leak-free few-shot** (Week 4), so the
  dynamic-retrieval approach did not displace the simpler baseline — exactly the null hypothesis holding.
  *(This is RAG for few-shot examples — distinct from Week 6's GDELT corroboration RAG.)*

## Week 4 — Prompt Tuning & Evaluation
- **Tried / decided.** Formalized **prompt v3**: persona + CLAIM/OPINION definitions + "checkability is
  not evidence" in the **system** role; **8 leak-free few-shot** (4+4) + full JSON example + input in the
  **user** role; `thought` field first (reason before committing). Froze the **100-row golden eval**
  (44 real + 56 synthetic hard-boundary), stratified by category and post-type.
- **Worked.** Empirical role placement: keeping the format example in the *user* message matters —
  moving it to *system* dropped recall **0.833 → 0.708**. Recall-on-CLAIM chosen as the headline metric
  (FN unrecoverable; FP self-corrects downstream) over F1 (which would reward trading recall for precision).
- **Didn't / the two findings.** (1) **Eval-set leakage** — 6 of 8 few-shot examples were eval posts,
  inflating recall to 0.854/0.938; fixed with synthetic disjoint examples. (2) **Batch-mode artifact** —
  batch-16 drifted the small model; single-post lifted recall **0.750 → 0.938** and cleared the hard
  categories. **Recall was fixed by config, not model.** LoRA declared **not** justified for recall;
  **precision ~0.70** (vague-conspiracy over-called) became the real target.

## Week 5 — Fine-Tuning with Adapters
- **Tried / decided.** A **precision-targeted QLoRA** on `qwen2.5:3b` (Option B — accuracy gap on
  vague-conspiracy FPs). Hybrid labels: **~26 hand-labeled** hard-boundary seed + **`qwen2.5:7b` teacher**
  for the bulk (rejected `gemma2:2b` — 87% claim rate, self-contradicting; `gemma4` — OOM on T4). Class
  ratio **swept as a hyperparameter** (26/40/50% claim). Three-gate validation (trust ≥8/10; weighted
  audit; near-zero tolerance for claim→opinion mislabels). Byte-identical train/inference format; fresh
  base each run. Base-vs-adapter Streamlit toggle.
- **Worked.** The pipeline, the comparison harness, and a clean **negative result**.
- **Didn't.** **No ratio cleared both gates** — every recall ≥ 0.90 point capped precision at **~0.74**;
  every high-precision point dropped recall **< 0.82**. A fundamental tradeoff at this data volume and
  model size, *not* label noise. Verdict: **do not deploy** — ship the recall-first base; keep the adapter
  as a demo toggle only. The negative result validated pushing precision **downstream**.

## Week 6 — Semantic Search & RAG → Evidence Matching (RAG #2)
- **Tried / decided.** A ChromaDB index of **GDELT news** (`all-MiniLM-L6-v2`, cosine) + **retrieve-then-
  rerank**: dense top-k, then an LLM **corroboration re-rank** asking *"does any article report the same
  specific event?"* (corroborated / partial / none). Built the reader signal + reach-vs-support **red flag**.
- **Worked.** The re-rank separates topic from corroboration — a HAARP post scores **0.62** on pure
  similarity (topical) but the re-rank correctly returns **none**. Three false-flag guards added:
  **self-citation**, **official-source allowlist**, **reshare provenance**. Corroboration prompt tightened
  (grounding-only, never-judge-truth, mandatory citation, scoped "none").
- **Didn't / learned.** Corpus is **headline-only** (GDELT returns titles, not bodies); a full news-parser
  was ruled out of scope. "No match" is **ambiguous, not negative** — scoped to the retrieved set.

## Week 7 — Multimodal Models → Gated Vision
- **Tried / decided.** A **gated** image signal (`qwen2.5vl:7b`) firing only on edge-case posts
  (image-claims not resolved by metadata) — a real event-photo *rescues* a genuine post from a false flag;
  a cartoon/meme *reinforces* suspicion. Gate by **uncertainty, not category**.
- **Worked.** The rescue is correct (`@yourcier`). Official allowlist expanded when widening exposed
  unrecognized NWS relay bots (`nws-bot.us`, `weather.im`).
- **Didn't / the two findings.** **The WebP bug** — Bluesky serves WebP, which Ollama can't decode; it
  silently fed blank images (identical canned captions). Fix: re-encode to **JPEG**. **~1% coverage** —
  on 745 posts / 119 image-claims (96% scanned) vision changed **exactly 1 red flag**; its trigger
  combination is rare and *must* stay strict. Verdict: sound but **low-coverage supplement**;
  **corroboration is the precision workhorse.** (The twin of the Week-5 LoRA result.) Not merged on a
  shakeout — accumulating data to re-measure.

## Week 8 — Evaluation & Beyond → Finalization
- **Tried / decided.** The autonomous daily-pipeline design and the **Trust Checker** — a user-facing
  "check a Bluesky post" tool (classify → corroborate → credible GDELT sources → red flag → verification
  links + a "how to spot misinformation wording" guide).
- **Worked.** Trust Checker shipped on the branch; the system design and this journey are documented.
- **In progress.** Dashboard overhaul (sidebar nav, human bullet summaries, Bluesky links everywhere,
  sort/filter); the `run_daily()` autonomous loop.

---

## Cross-cutting design evolutions (the honest arcs)
- **Credibility score → reader signal.** Weeks 2–3 designed a numeric **source-credibility prior**
  (NOAA 0.95 … RT 0.05). It was **deliberately removed** — an LLM can't verify facts, and a hardcoded
  credibility score is a curation/bias burden. The system surfaces **descriptive signals** (corroboration,
  source context, reach) and lets the reader conclude. No numeric score, no HIGH/MED/LOW.
- **`all-mpnet` → `all-MiniLM-L6-v2`.** The heavier model won the Week-2 gap test but the lighter one
  shipped — offline, 384-dim sufficient, and it doubles as the RAG retrieval engine.
- **Config beats model.** The biggest recall win (0.750 → 0.938) was single-post inference; fine-tuning
  couldn't beat the config-level baseline.
- **Negative results are results.** Dynamic few-shot RAG (flat vs fixed), LoRA (can't beat ~0.70/0.74
  precision), and vision (~1% coverage) each *validated* the architecture by locating precision downstream.
- **Signals over verdicts.** Every stage stays descriptive and points to primary sources — the constraint
  that keeps the tool honest.

## Final state
Recall-first `qwen2.5:3b` (single-post, leak-free fixed few-shot) → GDELT corroboration (RAG + LLM
re-rank) → source / reach / self-citation / official-source signals → gated vision on edge cases →
**reader signal + red flag**, surfaced through the **Trust Checker**. No adapter in production; vision a
documented low-coverage supplement; corroboration the precision workhorse.
