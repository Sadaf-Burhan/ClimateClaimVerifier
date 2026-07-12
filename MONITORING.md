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

## Signal 2 — Classifier drift chart (`data/eval_history.jsonl`)

**What it measures.** The classifier's quality on the **frozen** hand-labeled benchmark
(`data/claim_eval.csv`, 100 posts) — recall, precision, F1, accuracy, FN, FP — one JSONL snapshot per
maintenance **evaluation**, written by `climate_verifier.pipeline.evaluate.snapshot_metrics(...)`.

**What it shows.** On the app's Evaluation tab: recall & precision **over time** (line chart), FN vs FP
over time (bar chart), a meets-target flag, and the raw log.

**How to read it — this is the important part.**
- It should stay **roughly constant, within a noise band.** The classifier is **nondeterministic even
  at `temperature 0`** — a borderline opinion can flip between two identical runs (observed: precision
  0.687 → 0.697 back-to-back on the same machine), and the swing is larger across backends (GPU vs CPU).
  Precision is computed over only ~52 opinions, so each flip moves it ~1.5 points. **Recall is far
  steadier.** Watch **recall** and **sustained** trends — not single-point wiggles.
- A move **beyond the noise band** is a **regression signal**: a model update, a re-pulled quantization,
  a prompt edit, a dependency bump, or a backend change. It tells you *"something in the system
  changed,"* — a software-engineering health check that the system still behaves as built.
- It does **NOT** capture concept/data drift. The benchmark is frozen, so it is blind to new real-world
  phenomena (new misinformation framings, new event types). Catching that requires **growing the eval
  set** — see Future work.

**Policy.** Evaluation runs on **Colab GPU only**, so every drift point shares one backend and is
comparable. The app's *Run evaluation* button and `evaluate.py` (without `--snapshot`) are **local
diagnostics that do not write the drift log.**

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
| Offline eval on a labeled set, tracked over time | The **drift chart** (`eval_history.jsonl`) |
| Monitoring signals / instrumentation (OpenTelemetry) | The **health heartbeats** (`health.json`) |
| Dynamic eval set that grows as the distribution shifts | *Future work* (below) |
| "How would you know retrieval degraded before users notice?" | *Future work* — retrieval-quality eval |

---

## Future work (planned, not yet built)

1. **Dynamically growing eval set.** Periodically hand-label a sample of recent *ingested* posts and
   append them to `claim_eval.csv`, so the drift chart becomes a true **concept-drift** detector
   (framing shifts, new phenomena) — not only a regression guard against model/env changes.

2. **Retrieval-quality evaluation.** Measure how well the Week-6 RAG evidence layer returns *relevant*
   GDELT news for a claim (e.g. labeled claim → expected-article pairs, precision@k / relevance), so a
   degrading retrieval layer is caught before users notice — currently unmeasured.
