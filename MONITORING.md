# Monitoring & Evaluation Layer

How the in-house maintenance layer watches its own health. This records, for each signal, **what it
measures**, **what it shows**, and **how it helps** — so the design intent is not lost.

The system is two decoupled layers: an **external-facing scanner** (the Streamlit app — only *reads*
`data/`) and an **in-house maintenance** layer (local daily ingestion + a Colab GPU pass — *writes*
`data/`). This document covers the two signals the maintenance layer emits so the scanner never
silently serves stale or degraded data. See [README](README.md) "Full End-to-End Run" for how to run it.

---

## Signal 1 — Health heartbeats (`data/health.json`)

**What it measures.** The last outcome of each maintenance **stage** — `ingestion` (local, daily),
and `classification` / `evaluation` / `vision` (Colab GPU pass). Each record is
`{ok: bool, ts: ISO-8601, ...counts | error}`, written by `climate_verifier.health.update_health(...)`.

**What it shows.** The app sidebar renders a chip per stage:
- 🟢 **ok** — ran successfully, recently
- 🟡 **stale** — last good run is older than the staleness window (default 48h)
- 🔴 **fail** — the last run errored (the error string is shown)

**How it helps.** It surfaces *operational* failures and staleness — Ollama down, a model missing, a
network timeout, or a stage that simply hasn't run — so a half-updated corpus is visible instead of
silent. The one-shot runners (`scheduler --once`, the Colab cell) also exit **non-zero** on failure, so
the OS scheduler / Colab surfaces the failure through its own alerting. This is the project-scale form
of Module 8's *monitoring signals / instrumentation*.

---

## Signal 2 — Classifier drift chart, over TWO benchmarks (`data/eval_history.jsonl`)

**What it measures.** Every maintenance **evaluation** scores the classifier on **both** labeled sets
and writes one JSONL snapshot each, tagged `eval_set`, via
`climate_verifier.pipeline.evaluate.snapshot_metrics(...)`:

| Set | File | Grows? | A move here means |
|---|---|---|---|
| **🥇 Gold** | `data/claim_eval_gold.csv` (100: 48 claim / 52 opinion) | **never** | the **MODEL or ENVIRONMENT** changed — the data is constant, so nothing else *can* explain it |
| **📚 Dynamic** | `data/claim_eval.csv` (109 and growing) | via the relabel loop | model change **+** distribution change = **CONCEPT DRIFT** |

**They are kept apart deliberately.** Until 2026-07-15 both roles shared one CSV, so the relabel loop
silently mutated the set we called "static" and every recall move conflated *the model changed* with
*the test got harder*. One merged number can never separate those. Gold is the control that makes
dynamic readable.

**How to read it.**
- **Gold moves** → model/env. **Only dynamic moves** → concept drift (the new hard cases aren't being
  generalized to). **Both move** → model/env (gold proves it). **Neither** → stable.
- **Check `model_digest` before calling anything drift.** Colab re-installs Ollama and re-pulls the
  model every run, so a silent model swap can masquerade as a regression. Each snapshot records
  `model_digest` + `ollama_version` (`model_fingerprint()`), and the notebook pins `OLLAMA_VERSION`.
- **The classifier is deterministic on Colab GPU** at `llm_batch_size: 1`. Two runs 2h apart on the
  frozen gold set reproduced *byte-identically* (recall .875, precision .70, FN 6, FP 18). An earlier
  note here claimed nondeterminism at `temperature 0` (precision 0.687↔0.697) — that was a **local CPU**
  observation and does **not** hold for the canonical GPU backend. **Consequence: a gold move is real
  signal, not wobble — don't explain it away as noise.**
- Recall is still the metric that matters: a missed claim (FN) is dropped at the gate and unrecoverable.

**Policy.** Evaluation runs on **Colab GPU only**, so every drift point shares one backend and is
comparable. The app's *Run evaluation* button is a **local diagnostic that does not write the drift log.**
The drift log is an **append-only log**: the notebook must pull it from Drive *before* the run, or the
export overwrites your history (same for `health.json`, whose heartbeats merge).

---

## Signal 3 — Red-flag / signal eval (`scripts/eval_signal.py`, `data/signal_eval.csv`)

**What it measures.** Whether the **reader gets the right answer** — not whether the classifier does.
Gold and dynamic both score the bare classifier: recall (correctly — a dropped claim never reaches
downstream) and precision (the metric the design says to *ignore*, since it's "recovered downstream").
Nothing measured the scanner's one assertive output: **does the red flag fire when it should?**

**How it works.** Real posts + metadata, hand-labeled with the expected red flag. Retrieval is **pinned
to NO CORROBORATION** — the only state in which the flag can fire — because the flag depends on what
GDELT holds *today*, so `expected_red_flag` is only stable ground truth with the retrieval side held
constant. That isolates exactly the claim under test: the **source suppressors** (official / credible
cite / real-photo vision save / eyewitness / forecast). Runs locally; no GPU, no model, no corpus.

**What it found on its first run.** 6/7 — four suppressors demonstrably work (*that* is the downstream
recovery, finally measured), and one false alarm: `forecast` was computed and passed into
`build_reader_signal` but **never used in the red_flag condition**, despite the docs claiming forecasts
aren't flagged. Now fixed (conspiracy-guarded, so *"they're warning us about chemtrails"* still flags)
→ 7/7, red-flag precision 1.0.

**Caveat.** "Precision is recovered downstream" is only **partly** true: downstream never relabels, it
only decides whether the flag fires, and it suppresses solely for those source signals. A plain
opinion-misread-as-claim with reach and no citation still flags. (Though note the flag never consults
the claim/opinion label at all.)

---

## What these signals are NOT

- The drift chart is **not** a live-accuracy measure of production posts — those are unlabeled, so
  recall/precision can't be computed on them. Ingesting more data does **not** move the drift chart.
- A 🟢 health chip is **not** a quality guarantee — a green `evaluation` stage with a falling recall
  trend is still a problem. Read both signals together.

---

## Mapping to Module 8 (Evaluation and Beyond)

| Module 8 concept | Implemented here |
|---|---|
| Offline eval on a labeled set, tracked over time | The **drift chart** — 🥇 gold, the frozen control (`eval_history.jsonl`) |
| Monitoring signals / instrumentation (OpenTelemetry) | The **health heartbeats** (`health.json`) |
| Dynamic eval set that grows as the distribution shifts | **Built** — 📚 dynamic (`claim_eval.csv`), grown by the evidence-nominated relabel loop; read against gold |
| Evaluating the *system*, not just the model | **Built** — the **red-flag signal eval** (Signal 3) |
| Guarding against a silent model/runtime swap | **Built** — `model_digest` + `ollama_version` per snapshot; pinned `OLLAMA_VERSION` |
| "How would you know retrieval degraded before users notice?" | *Future work* — retrieval-quality eval |

---

## Built — the growing eval set (was "future work")

**Dynamically growing benchmark.** `data/claim_eval.csv` grows via the evidence-nominated relabel loop,
so the drift chart is a true **concept-drift** detector (framing shifts, new phenomena) *alongside* the
frozen gold set, which remains the regression guard. Both roles are wanted — they're separate files
precisely so each stays readable (see Signal 2).

**Evidence-nominated relabel candidates (active learning, not random sampling).** Grow the benchmark
from the boundary cases the pipeline already flags. A post classified **OPINION** is nominated on:
  - a **strong GDELT match** (REPORTED), an **official source**, a **credible self-cited source**, or
  - **`verbatim_headline_domain()`** — the post text IS the headline of the credible article it links
    (an outlet posting its own story is reporting, not commentary).

Guardrail: **evidence NOMINATES, a human DISPOSES — never auto-relabel.** A headline match is topic
similarity, not proof; op-ed headlines are verbatim headlines too; and we can't yet read the *cited
article's body* to confirm it quotes the claim (the deferred agentic-retrieval extension).

**Two lanes, because ranking by reach hides the classifier's blind spots:**
  - **Engagement-ranked** (`get_relabel_candidates`) — the **red-flag product** lane. Reach is the
    point: a reach-vs-support mismatch only matters at reach.
  - **Signal-ranked** (`get_signal_candidates`) — the **benchmark-growth** lane. Reach is irrelevant;
    a 0-like wire headline the classifier misreads is as valuable an example as a 295-like one. It
    sweeps the *whole* corpus (~0.03s — every signal is pure text/metadata, no retrieval, no LLM) and
    tiers results, because they differ in trust: *strong* (verbatim headline / official) vs *weak*
    (credible cite alone — "here's a great Guardian piece, so depressing" cites a credible source and is
    **correctly** an opinion). Measured: 65 contradicted opinions exist; only 3 are high-reach enough
    for the engagement lane to ever surface.

**Known workflow gaps** (surfaced in-app by `_render_eval_freshness()`): a relabel appends instantly, but
Colab **clones the benchmark from git** — uncommitted relabels are never evaluated; and evaluation is
**GPU-only**, so a grown benchmark doesn't refresh the numbers until the next Colab pass.

---

## Future work (planned, not yet built)

1. **Retrieval-quality evaluation.** Measure how well the Week-6 RAG evidence layer returns *relevant*
   GDELT news for a claim (labeled claim → expected-article pairs, precision@k / relevance), so a
   degrading retrieval layer is caught before users notice — currently unmeasured. This is the last
   stage with no eval of its own: Signal 3 deliberately **pins** retrieval to isolate the source
   suppressors, so nothing yet watches retrieval itself.

2. **Grow the signal eval (Signal 3) beyond its 7 seed rows** — one per suppressor path is enough to
   catch a missing suppressor (it found the forecast bug immediately) but too few to trust a rate.

3. **Name → domain citation resolution.** Credit is domain-based, so a screenshot showing the *name*
   "The New York Times" (rather than a URL) gets no credible-cite pass even though `nytimes.com` is
   allowlisted. Affects the image path most, where outlets appear as names.

4. **Resolve URL shorteners at ingestion.** ~67 corpus posts cite through `bit.ly` / `share.google` /
   `dlvr.it` / `ow.ly`, which hide the real domain — so a post citing the NYT via `bit.ly` gets **no**
   credible-cite credit and can be **red-flagged as "no cited evidence"** while genuinely citing a
   credible source. That is a false alarm the signal eval (Signal 3) would catch if such a post were
   in it. Fix = follow the redirect during ingestion and store the resolved domain in `external_url`;
   costs one network call per shortened link, and ingestion is already I/O-bound and GDELT-rate-limited,
   so it needs to be batched/cached rather than naive. Measured 2026-07-15 by auditing what the corpus
   actually cites (that same audit added `insideclimatenews.org` / `washingtonpost.com` / `grist.org`,
   which were high-frequency cites getting no credit).

4. **Judge/arbiter agent for the final verdict** — when the text classifier and the image path disagree
   (caption is opinion, the card carries the claim), decide the ONE label the user sees. Deferred.

5. **Agentic retrieval + reranking** — fetch article bodies past bot walls → cross-encoder rerank →
   support/contradict verdict. Would let evidence confirm a *cited article actually backs the claim*,
   which is the guardrail currently keeping nomination human-in-the-loop.
