# Session Handoff — ClimateClaimVerifier (updated 2026-07-16)

Read this first to continue in a fresh session without re-deriving. Supersedes all earlier handoffs.
Companions: `MONITORING.md` (the eval/monitoring layer — **rewritten this session, trust it**),
`WEEK7_STATUS.md`, and the auto-memory index (`memory/MEMORY.md`).

**Project =** a *reader-signal scanner* for **Bluesky** climate/extreme-weather posts. It surfaces
signals (claim vs opinion, independent news corroboration, source, reach, image nature) and a
**reach-vs-support red flag** — it **never** asserts true/false. Recall-first classifier; precision
recovered downstream by evidence + source signals.

---

## 0. Where things stand / next steps

**Everything below is committed + pushed on `multimodal-edge-gating` (HEAD `88d2afe`).** Working tree
is clean except `README.md` (yours, pre-existing) and `.env.example` (untracked, undecided).

**The single highest-value next task: identify the 6 stubborn false negatives.** See §3 — the
classifier is now *provably deterministic*, so the same 6 claims are missed in **every** run. They are
a fixed, reproducible target and they ARE your recall ceiling. Nothing else will move recall until you
know what they are.

Other open items, roughly by value:
1. **`news_status` agreement has never been measured** (image eval). Both runs had an empty ChromaDB.
   Needs cell 12 (`--reindex`) then cells 16-17 **in the same Colab session**. Half the image path's
   primary bar is unverified.
2. **The image-eval scorecard on Drive is stale** — cells 16-17 didn't run on 2026-07-16. Label fixes
   + the scorer whitespace fix have never been scored.
3. **12 signal-lane candidates await your judgment** (🔧 Maintenance → 🔬 Sweep). ⚠️ At least 2 are
   *opinion* content behind verbatim headlines (a Monbiot column, a Guardian Letters page) — do NOT
   bulk-accept. Also decide a policy: half are RSS bots reposting headlines (`@climate.skyfleet.blue`,
   `@bigearthdata.ai`) — is a bot reposting a NASA headline a claim worth benchmarking, or noise?
4. **Dynamic ≈ gold** (108 vs 100, ~99 rows shared) so concept drift can barely register. The dynamic
   set must GROW before it signals anything. That's what the signal lane is for.
5. **URL-shortener blind spot** — logged as future work, see §6.

---

## 1. Git / branch / data state
- **Branch `multimodal-edge-gating`** — all work here, pushed to `github.com/Sadaf-Burhan/ClimateClaimVerifier` (PUBLIC). `main` frozen; merge only when you say so.
- **SECURITY:** `.env` holds REAL secrets (`BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`, `ADMIN_PASSWORD`). Gitignored, NOT tracked. **NEVER commit or echo it.** Grep every staged diff for secret patterns first.
- **Data:** `ingested.db` = 2842 bluesky + 3606 gdelt, all 2842 classified (0 pending). 38 admin overrides.
- **Models:** local Ollama has `qwen2.5:3b` + `qwen2.5vl:7b` (6GB, pulled). Vision/eval run on **Colab GPU**.

### The eval sets — THREE benchmarks, deliberately separate (see §2)
| File | Rows | Tracked | Role |
|---|---|---|---|
| `data/claim_eval_gold.csv` | **100** (48 claim / 52 opinion) | yes | **FROZEN control.** Never append. Movement = model/env. |
| `data/claim_eval.csv` | **108** (50 / 58) | yes | **DYNAMIC**, grows via relabel loop. Movement = concept drift. |
| `data/signal_eval.csv` | **7** | yes | **Red-flag / signal eval** (Signal 3) — the reader's answer, not the model's. |
| `data/image_eval/labels.jsonl` | **16** | yes | Image-extraction benchmark. Images in `images/` are **gitignored**. |

Runtime artifacts (all **gitignored**, round-trip via Drive): `eval_history.jsonl` (6 entries),
`health.json`, `chroma_evidence/`, `image_eval/eval_results.json`, `signal_eval_results.json`.

---

## 2. THE BIG ARCHITECTURAL CHANGE — gold vs dynamic split (`8ba040e`)

**The bug:** until 2026-07-15 both evals read ONE csv (`claim_eval.csv`), which the relabel loop
appends to. So the "static" eval stopped being static on the first relabel, and every recall move
conflated *the model changed* with *the test got harder*. The separation existed only in UI labels.

**The fix:** two files, two questions. Each maintenance pass evaluates **both** and logs one drift
line each, tagged `eval_set`.

**Reading rule (memorize this):**
- **Gold moves** → model/env changed (data is constant, nothing else *can* explain it)
- **Only dynamic moves** → concept drift (not generalizing to newly relabeled cases)
- **Both move** → model/env (gold proves it)
- **Neither** → stable

Gold is the control that makes dynamic readable. `maintenance --evaluate` runs both; health/target
reports GOLD (a dynamic miss can just mean the new cases are hard). Pre-split log entries backfill
to `gold`. **Never let the relabel loop touch gold.**

---

## 3. THE RECALL STORY — corrected twice, and the real answer

**The drift log (`data/eval_history.jsonl`, 6 entries):**
```
2026-07-13 07:25  gold*    n=100  recall=0.9583  FN=2  FP=17   (no digest — pre-fingerprint)
2026-07-15 19:41  gold*    n=100  recall=0.8750  FN=6  FP=18   (no digest)
2026-07-15 21:54  gold     n=100  recall=0.8750  FN=6  FP=18   digest=357c53fb659c
2026-07-15 21:56  dynamic  n=109  recall=0.8846  FN=6  FP=21   digest=357c53fb659c
2026-07-16 03:02  gold     n=100  recall=0.8750  FN=6  FP=18   digest=357c53fb659c
2026-07-16 03:04  dynamic  n=108  recall=0.8800  FN=6  FP=22   digest=357c53fb659c
```

### ✅ FINDING 1 — the classifier is DETERMINISTIC on Colab GPU
**Three gold runs, byte-identical on every field** (recall, precision, f1, accuracy, FN, FP), across
separate sessions hours apart. **This overturns the old handoff/MONITORING note** claiming
nondeterminism at temperature 0 (`0.687↔0.697`) — that was a **local CPU** observation and does NOT
hold on the canonical GPU backend at `llm_batch_size: 1`.
**Consequence: any gold movement is REAL SIGNAL. Never explain a gold move away as noise again.**

### ❌ TWO WRONG HYPOTHESES (recorded so they aren't re-run)
1. *"The benchmark grew 60→100 between Jul 13 and Jul 15, so it's not like-for-like."* **WRONG** —
   the growth predated Jul 13; both ran the identical committed 100 (reconstructed TP reproduces the
   logged precision exactly: 46/63=0.7302, 42/60=0.7000). It **was** like-for-like: 4 claims flipped.
2. *"The out-of-scope (UK/Siberian) rows are the misses; removing them will lift recall to ~0.94."*
   **WRONG** — `FN stayed at exactly 6` across n=109 → n=108. All 6 removed out-of-scope claims were
   **TPs**, classified correctly. The scope leak was never the cause.

### ⭐ FINDING 2 — the model IGNORES the "North America" scope
The prompt says *"a verifiable factual claim about a climate or extreme weather event **in North
America**"*, but it classified Siberian and UK claims as CLAIM without hesitation. **The scope line is
decorative.** Geography is really enforced upstream by `topic_filter` at ingestion — arguably the
right architecture; the prompt just doesn't know it. **Decision (yours): keep the NA scope in the
prompt (it's the original design) and keep non-NA posts OUT of the eval sets.**

### ⭐ FINDING 3 — the real recall ceiling: 6 stubborn FNs
The same **6 claims are missed in every single run**. Not noise (determinism proved that), not scope
(finding 2), not the benchmark. Six specific posts this model cannot see. **Since the classifier is
deterministic, these are a fixed reproducible target — go read them.** ← the highest-value next task.

### The model/runtime fingerprint (`8ba040e`)
`model_fingerprint()` in `evaluate.py` records `model_digest` + `ollama_version` on every snapshot,
because Colab re-installs Ollama and re-pulls the model every run, **both unpinned** — a silent model
swap would masquerade as drift on the frozen set. **Check the digest FIRST when gold moves.**
- **Colab digest `357c53fb659c5076de1`** ≠ **desktop `845dbda0ea48ed749ca`** — genuinely different
  model builds. (Can't be proven retroactively; Jul 13 predates the fingerprint.)
- Notebook pins `OLLAMA_VERSION = "0.32.0"` to match desktop. Pinning changed nothing (0.32.0 *was*
  latest) — it's a guard for later. **Pins the RUNTIME only**: `ollama pull` has no digest-pin, which
  is exactly why the digest is recorded instead.
- **Untested idea:** run gold on the DESKTOP (still has the older build). ~0.958 would confirm the
  build caused the drop. (CPU-vs-GPU would confound it, but with the digest diff it'd be strong.)

---

## 4. Week-7 image-input path — BUILT, RUN, PASSING (`c0388f2`, `5aa970a`)

Screenshot upload → `extract_from_image` (qwen2.5vl:7b, OCR) → thin adapter → the **unchanged**
classify → region-aware RAG → reader-signal spine. Degraded-fidelity path, clearly labelled.

**Last run (16 images):** gate **100% (16/16)** · **classify agreement 100% (9/9)** ← the primary bar ·
zero hallucinated engagement on 8 non-posts · `news_status` agreement **never measured** (empty index).

- **HARD SCOPE RULE (yours):** the upload path only accepts **social-media-post screenshots that show
  engagement**. `looks_like_social_post()` gates it; no engagement → rejected with guidance. Bare
  photos, satellite/comparison images, infographics, memes, illustrations are OUT (they become
  *rejection* test cases in the eval). Photo tab has an optional "paste the post text if truncated" box.
- **BUG FIXED — worth remembering the shape:** `_correct_extraction` handed the 3B corrector a template
  that was ITSELF valid JSON with every value null and `has_readable_text: true`. The corrector
  **echoed the template**, `_parse_json` accepted it, and we produced a confident all-null extraction
  that looked like "no text in this image" — REJECTING two real high-engagement posts. Fixed:
  placeholders are now deliberately NOT valid JSON (`<...>`) so an echo fails to parse → honest
  failure. **Lesson: never show a weak corrector a fillable template that parses.**
- **`claim_text` labelling policy (settled, in `data/image_eval/README.md`):** label the primary
  claim-bearing text that is **VISIBLE** — usually the post body; the CARD for image-carried claims
  (DOGE); **never text behind a "See more" fold**. The first pass got this wrong and it read as model
  failure when the model was right.
- **Known weak fields:** `image_type` (~33%) and `depicts_claim` (~22%) — mixed model weakness and
  debatable labels. The prompt rule meant to stop "everything is a screenshot" did NOT work.
- **DOGE case:** the model reads it perfectly (11 likes / 1 repost, full card text) but the classifier
  calls the claim OPINION — a genuine false negative, not an image-path bug. The citation-credit path
  (§6) is what rescues it in the UI.

---

## 5. The signal eval (Signal 3) — measuring the READER's answer (`49e3afa`)

`scripts/eval_signal.py` + `data/signal_eval.csv` (7 real posts, one per suppressor path).
**Runs locally — no GPU, no model, no corpus.** Currently **7/7, red-flag precision 1.0, recall 1.0**.

**Why:** gold/dynamic score the bare *classifier*. Nothing measured the scanner's one assertive
output — the red flag. Retrieval is **pinned to NO CORROBORATION** (`assess_claim(retrieval=...)`),
the only state where the flag can fire, because the flag depends on what GDELT holds *today* and
`expected_red_flag` is only stable ground truth with retrieval held constant. That isolates the
**source suppressors** — i.e. exactly the "precision is recovered downstream" claim.

**It found a real bug on run 1 (`1b054d8`):** `forecast` was computed and passed into
`build_reader_signal` but **never used in the `red_flag` condition**, despite the docs claiming
forecasts aren't flagged. Fixed via `forecast_defused`, and `looks_like_forecast` now carries the
shared `_CONSPIRACY_RE` guard (so *"they're warning us about chemtrails"* still flags). Blast radius
measured before the change: 3 of 44 flagged posts, 0 conspiratorial.

**Nuance discovered:** "precision recovered downstream" is only **partly** true. Downstream never
relabels — it only decides whether the flag fires, suppressing solely for official / credible-cite /
vision-save / eyewitness / forecast. **And `build_reader_signal` never receives `has_claim`** — the
red flag doesn't depend on the claim/opinion label at all, so a classifier FP can't cause a false flag.

---

## 6. Nomination, relabel, and the labeling guide

### Two lanes (`164bc72`) — ranking by reach hides the classifier's blind spots
- **Engagement-ranked** (`get_relabel_candidates`) — the **red-flag product** lane. Reach is the point.
- **Signal-ranked** (`get_signal_candidates`) — the **benchmark-growth** lane. Reach irrelevant.
  Sweeps the WHOLE corpus in ~0.03s (every signal is pure text/metadata — no retrieval, no LLM); only
  the small strong shortlist is enriched with `news_status`. Engagement is a tie-break, never a filter.
  **Measured: 65 contradicted opinions exist; only 3 are high-reach enough for the engagement lane.**
  Tiered: **strong** (verbatim headline / official) vs **weak** (credible cite alone — "here's a great
  Guardian piece, so depressing" cites a credible source and is CORRECTLY an opinion).

### `verbatim_headline_domain()` (`relabel.py`)
Post text == `external_title` (captured at ingestion, already in the schema) + the linked domain is
credible → a credible outlet posting its own headline = reporting, not commentary. **Stays a
NOMINATION, never an auto-override** — op-ed headlines are verbatim headlines too. **Key insight: it
can NOT fix the classifier**, because the headline and the post text are the *same string* — the model
already read those exact words and still said OPINION. It's provenance, not text.

### The labeling guide (`fa37217`) — sidebar, renders the prompt VERBATIM
`_render_labeling_guide()` puts the criteria in the **sidebar** (the only surface that stays visible
while scrolling the queue) and renders `claim_classifier._SYSTEM_PROMPT` **imported, never
transcribed** — because the admin's label becomes the ground truth the classifier is SCORED against.
Judge by a different definition than the model was handed and the eval measures a definition mismatch
instead of model error. Copy-pasting the criteria would let guide and prompt drift — the same failure
one level up.

**The three rules that catch almost every bad relabel (all straight from the prompt):**
1. **A claim need NOT be true** — a false/conspiratorial assertion is still a CLAIM if specific and checkable.
2. **Tone is irrelevant** — sarcasm/fury/jokes don't make it an opinion if *one* checkable assertion exists.
3. **"No source" is NEVER a reason for OPINION** — checkability ≠ evidence.

**Label and `post_type` must AGREE.** An OPINION carrying a CLAIM thought is a mechanical
contradiction — it's how two bad relabels were caught (`2a53a96`): *"Over 2,700 people have died from
heat related causes…"* (`mixed_emotion_fact`) and *"Trump doesnt believe in climate change"*
(`false_but_checkable`), both labelled OPINION. Both flipped to claim, **in BOTH stores** —
`apply_relabel` writes the eval CSV *and* the DB `admin_label`; changing only the CSV leaves users
seeing the old label. Taxonomy is now fully snake_case (`news_headline_verbatim` replaced
`Ref_source_contain_same_wording`).

### Citation domains (`8fdd325`) — audited against the corpus
Added this session: `nytimes.com`, `climate.us` (deliberate editorial choice, documented in config),
`insideclimatenews.org` (19 cites), `washingtonpost.com` (9), `grist.org` (7).
- A **bare-domain** `visible_citation` (e.g. `climate.us`) is normalized to `https://climate.us` in the
  image adapter, because `extract_citations` only matches http/www URLs.
- **Known gaps:** source *names* ("The New York Times", not a domain) aren't credited — name→domain
  resolution is future work. **URL shorteners hide the real source on ~67 posts** (`bit.ly` 23,
  `share.google` 21, `dlvr.it` 14, `ow.ly` 9) — a post citing the NYT via bit.ly gets NO credit and may
  be red-flagged for "no cited evidence". **Decision: log it, don't fix** (needs a network call per URL
  at ingestion).
- Content scrapers (`byteseu.com`, `europesays.com`) correctly excluded — they're not the source.
- **UK outlets stay** (`theguardian.com` is your #1 cited at 85): credibility ≠ geography.

---

## 7. Ops / the artifacts round-trip

**THE RULE: logs accumulate, state gets versioned.** (`84758ac`)
- `eval_history.jsonl` = an append-only **LOG**. It IS the history. **Never version, never prune**
  (~250 bytes/run). The notebook **pulls it from Drive BEFORE** the chain so evaluate appends —
  otherwise the git-ignored (empty) clone writes one entry and the export **destroys your history**.
  This already happened once; it was recovered from the **recycle bin**, not git (never committed).
- `health.json` = per-stage heartbeats, **merged** by `update_health`. Must also be pulled, or the
  Colab export wipes the local **ingestion** heartbeat. ⚠️ **Check this survived the last run.**
- `ingested.db` = **STATE**, replaced each run → dated backup in `<DRIVE>/backups/` with 30-day
  retention. The corpus is rolling anyway (refresher expires Bluesky posts >14 days).

**Colab notebook (`notebooks/colab_daily_maintenance.ipynb`):**
- Cell 8: **`zstd` MUST install before Ollama** (Colab lacks it; the installer fails without it).
  `OLLAMA_VERSION = "0.32.0"` pinned. Prints the fingerprint — compare the digest to the desktop.
- Cell 10: pulls `ingested.db` + `eval_history.jsonl` + `health.json`. `DRIVE_DIR = /content/drive/MyDrive/ClimateScanner`.
- Cell 12: `maintenance --all` = classify → vision → reindex → evaluate (**both** sets).
- Cells 15-17 (§7b): image eval — pulls images from `<DRIVE_DIR>/image_eval_images/`, runs the eval,
  exports the scorecard. **Needs cell 12's reindex in the same session for `news_status`.**
- **Drive naming trap:** the scorecard exports FLAT as `image_eval_results.json` but the code writes
  `data/image_eval/eval_results.json`. Both names are now gitignored (`88d2afe`).

**Two silent workflow gaps** (now surfaced in-app by `_render_eval_freshness()`):
1. **Commit gap** — a relabel appends to `claim_eval.csv` instantly, but **Colab clones from git**, so
   uncommitted relabels are never evaluated.
2. **Staleness gap** — evaluation is **GPU-only**, so a grown benchmark doesn't refresh the numbers
   until the next Colab pass. Hence a *pending flag*, not an auto-run.

**Ingestion (`e44c580`):** the banner now separates **last COMPLETED cycle** from **newest post
ingested** — they diverged badly (72h vs 19h) because ingestion **commits posts as it goes but only
stamps completion at the very end**, so a mid-cycle failure looks like "never ran" while the data is
fresh. **Your last two cycles failed partway** (Jul 14 05:59→07:15, Jul 15 05:08→07:28). The error is
in `health.json` → `ingestion` — *which the Colab pass overwrites unless you upload health.json first*.
Runs start ~05:00-06:00 UTC, take 1-2h. Interval 24h. **Run it from your own VS Code terminal** —
`uv run python -m climate_verifier.ingestion.scheduler --once --force` — it's I/O-bound, Colab won't
help and throttles GDELT harder (shared IP).

**Windows:** kill Streamlit by port — `Get-NetTCPConnection -LocalPort 8501 -State Listen | %{ Stop-Process -Id $_.OwningProcess -Force }`. Module edits need a full restart. Admin tab needs `ADMIN_PASSWORD` in `.env`.

---

## 8. Design decisions — DO NOT relitigate
- **Recall-first classifier; precision recovered downstream** — but see §5: only *partly* true, and now
  measured rather than assumed.
- **Evaluation uses FROZEN labeled CSVs, independent of ingested data.** Ingesting more NEVER changes
  eval numbers. Re-classifying the corpus does NOT affect recall. Different populations.
- **Evidence is HEADLINE-ONLY** (GDELT titles, no bodies) — caps everything at *related coverage* +
  *reach-vs-support*, NOT claim support. A headline can't verify a proposition.
- **Verdict vocabulary:** REPORTED / TOPIC MATCH (≥ `topic_proximity` 0.50) / NO MATCH. **A topic match
  is NOT claim support.**
- **Red flag is decoupled from the display verdict** (stays on the strict `corroboration.verdict`).
  NOT raised for: official sources, credible self-citations, real-photo vision saves, eyewitness
  observations, **forecasts** (as of `1b054d8` — this was documented but not implemented).
- **Evidence NOMINATES, a human DISPOSES — never auto-relabel.** A wrong relabel is permanent,
  committed, and silently teaches the wrong boundary. (Claude relabeling on your behalf violates this
  — it introduced a scope violation when it tried.)
- **Eval standardized on Colab GPU** — app/CLI eval are local diagnostics that don't write the drift log.
- **LLM ranker deferred to an agentic extension** (`use_llm_rerank` default OFF).

## 9. Key files
- `pipeline/evidence.py` — `assess_claim(retrieval=...)`, `build_reader_signal` (news_status, red flag, `forecast_defused`/`eyewitness_defused`, region-mismatch), `_CONSPIRACY_RE` (shared guard), `looks_like_forecast/eyewitness`, `is_official`, `extract_citations`, `retrieval_only_verdict`.
- `pipeline/relabel.py` — `get_relabel_candidates` (engagement lane), **`get_signal_candidates`** (signal lane), `verbatim_headline_domain`, `apply_relabel`/`set_admin_label`/`append_to_eval_csv`, `eval_post_types`.
- `pipeline/vision.py` — `extract_from_image`, `normalize_extraction`, `looks_like_social_post`, `screenshot_signal_inputs`, `_citation_as_url`, `_correct_extraction`, `gate_edge_cases`, `analyze_image`, `vision_reader_note`.
- `pipeline/evaluate.py` — `run_eval`, `compute_metrics`, **`snapshot_metrics(eval_set=...)`**, **`model_fingerprint`**, `load_eval_history`.
- `pipeline/claim_classifier.py` — **`_SYSTEM_PROMPT`** (the guide renders this), `build_prompt` (5 few-shot examples — verified scope-clean and **zero leakage** into either eval set), `classify`, `classify_pending`.
- `pipeline/topic_filter.py` — where geography is ACTUALLY enforced (`382105c` added UK/Europe terms).
- `app.py` — `_render_drift` (two series), `_render_eval_freshness`, `_render_static_eval` (gold/dynamic picker), `_render_labeling_guide` (sidebar), `_render_image_trust`, `_render_assessment_body` (shared by both paths), `maintenance`.
- `scripts/eval_image_extraction.py` · `scripts/eval_signal.py` · `climate_verifier/maintenance.py` · `climate_verifier/health.py`.

## 10. This session's commits (`84758ac` → `88d2afe`)
```
84758ac Drift log accumulates; eval staleness surfaced; verbatim-headline nomination
8ba040e Split gold vs dynamic eval sets; record model/runtime fingerprint
85bd800 Commit 9 admin relabels to the dynamic set; pin Colab Ollama to 0.32.0
5aa970a Fix corrector echoing its own null template; correct image-eval labels
49e3afa Add the signal eval: measure the red flag, not the classifier
1b054d8 Suppress the red flag on forecasts (conspiracy-guarded); fix scorer whitespace
164bc72 Add the signal-ranked nomination lane (benchmark growth, any reach)
e44c580 Ingestion banner: separate "completed cycle" from "newest post"
f6de68c Fix stale pre-split wording in the app and MONITORING.md
2a53a96 Commit 5 relabels; flip 2 whose label contradicted their own post_type
fa37217 Pin the labeling criteria in the sidebar, rendered from the classifier's own prompt
382105c Fix the NA scope leak: topic_filter had no UK/Europe terms
8fdd325 Audit citation_domains against the corpus; add 3 newsrooms; snake_case the last post_type
88d2afe Ignore the image-eval scorecard under either name
```
