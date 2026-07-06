# Multimodal Rebuild Plan — edge-case vision gating + fresh pipeline run

**Branch:** `multimodal-edge-gating` (do NOT work on `main`).
**Goal:** add a *gated* image-modality signal that fires only on precision-hurting edge cases,
on a fresh, current dataset — without disturbing the working architecture until it's proven.
**Promotion rule:** merge to `main` only if edge-case **precision improves with no recall regression**.

Baseline to beat (frozen, from `main`): classifier **recall 0.938 / precision ~0.70** (single-post),
the ~0.70 precision — concentrated in **conspiracy / ambiguous** posts — is exactly what this change targets.

---

## 0. Safety / preservation (DONE)
- Test branch `multimodal-edge-gating` created off `main`.
- Current DB snapshotted → `data/ingested_backup_premultimodal.db` (gitignored, local-only).
- `data/claim_eval.csv` (frozen ground truth) and `data/lora_seed.jsonl` are untouched files.
- A *new* `data/ingested.db` is produced by the fresh run; the backup is the rollback.

---

## Part A — Data Requirements (all angles)

Legend: **HAVE** = already captured · **NEW** = must add · **DERIVED** = produced by a pipeline stage.

### A1. Post core — HAVE
`post_id`, `source`, `keyword`, `keyword_category`, `author`, `text`, `created_at`, `ingested_at`.

### A2. Engagement / reach — HAVE
`likes`, `reposts`, `replies`, `quotes`  → engagement = sum (drives the reach-vs-support flag).

### A3. Source / author signals
- `author_followers` — HAVE.
- **NEW (near-free):** the existing batched `get_profiles` call also returns, per author:
  - `author_bio` (self-reported "what I post" — a *self-report*, to be verified, not trusted)
  - `author_post_count`, `author_created_at` (account age) — behavioral reliability signals.
  These come from the same call we already make for followers → no extra API cost.

### A4. Media / images — NEW (metadata universal, inference selective)
Captured at ingest for **all** posts from `post.embed` (cheap, no download):
- `has_image` (bool), `image_url` (fullsize), `image_alt` (author's alt-text — a free text signal).
Downloaded + analyzed **only for gated edge cases** (see Part B): `vision_signal` (DERIVED).

### A5. Provenance — NEW
From `post.embed` (same parse as images):
- **Quote/reshare:** `reshare_of_author`, `reshare_of_uri` (`app.bsky.embed.record`).
- **External link:** `external_url`, `external_title` (`app.bsky.embed.external`) — the *structured*
  form of self-citation (today we regex links out of `text`; this is cleaner and more reliable).

### A6. Corroboration corpus (GDELT) — HAVE (+ date fix already on main)
`title` (headline), `domain`, `url`, `date` (now normalized), `keyword_category`. Body text remains
out of scope (news-parser problem). Retrieval + LLM corroboration re-rank unchanged.

### A7. Derived signals
- `has_claim` + `reason` (classifier) — HAVE.
- `vision_signal` (edge-case vision) — **NEW / DERIVED** (schema in Part B).
- Reader signal fields (official / self_cited / reshared_official / red_flag) — HAVE.

### A8. Governance
- `in_eval_set`, `in_train_set` — HAVE (re-derived on the new DB by matching `claim_eval.csv`).
- `claim_eval.csv` — **frozen** ground truth. Extend only by *adding* rows (never relabel existing).

---

## Part B — Vision layer: gated escalation (the core change)

Vision is a **cascade / precision intervention**, NOT a per-post feature. Cheap text+metadata signals
decide first; the expensive vision model fires only on the uncertain subset that has an image.

### B1. The gate (confirmed)
| Resolved by text/metadata → **skip vision** | Edge case → **escalate to vision** (only if `has_image`) |
|---|---|
| Official source (`is_official`) | Claim **phrased as an opinion**, no source link |
| Links a trusted/credible source (`credible_cite`) | **Confusing / ambiguous** language |
| Reshares an official post (`reshared_official`) | **Uncorroborated personal event** ("fishy" — not in news, unverified, no metadata support) |
| Corroborated by news | Conspiracy / ambiguous category with no supporting signal |
| Clear-cut classification (no opinion-wording muddle) | |

### B2. Vision model
Candidates (both run on Ollama, same `image + text → text` contract):
- **`qwen2.5-vl`** *(recommended)* — strong on natural photos **and** OCR/screenshots. Edge-case
  images are mostly natural (real-photo vs cartoon), with some screenshots → best fit.
- `gemma4:e4b` — the course/module default; document-tuned (weaker on natural photos).
- `llava` — natural-photo generalist, weaker OCR.
Decision to confirm at implementation (verify the exact Ollama tag is pullable).

### B3. Vision signal schema (`vision_signal`, JSON per edge-case post)
- `image_type`: `real_photo | meme_or_cartoon | screenshot | infographic | ai_suspected | other`
- `depicts_claim`: `yes | partial | no | unrelated`
- `description`: short caption (grounded in the image only)
- `signal`: derived reader phrase, e.g.
  - real_photo + depicts_claim → *"genuine on-the-ground image consistent with the claim — supports a real observation"* (precision **save**)
  - meme_or_cartoon → *"cartoon/meme imagery — leans satire/opinion, not evidence"*
  - mocking cartoon of a real news frame → *"fabrication cue — mocking imagery over a news framing"*
  - screenshot → *"screenshot of a source — treat as a citation, open and verify"*

**Guardrail (unchanged philosophy):** the vision output is another *signal for the reader*, never an
automated truth verdict. It only disambiguates the red-flag decision on edge cases.

### B4. Malformed-output recovery (Week 7 pattern)
Vision model → JSON; on parse failure, strip fences then have a cheap **text** model (`qwen2.5:3b` /
`gemma2:2b`) reformat to the schema. The corrector never sees the image.

### B5. Vision eval (extends validation, per "add to it if needed")
Add a small **image-edge eval set** (~15–20 edge-case posts with images, hand-labeled
`image_type` + should-the-red-flag-stand). Measure whether the vision signal improves the edge-case
decision. This *extends* the frozen `claim_eval.csv` discipline; it does not alter existing rows.

---

## Part C — Compute split (why each stage runs where)

| Stage | Where | Why |
|---|---|---|
| Ingestion (Bluesky + GDELT) | **Colab** | Co-locate so the produced DB has all fields before export; API-bound (no GPU need, but keeps data-gathering in one place) |
| Topic filter + classification (`qwen2.5:3b`, batch=1, ~thousands of posts) | **Colab GPU** | The CPU bottleneck (multi-hour locally) → minutes on GPU |
| Edge-gating + vision (`qwen2.5-vl`) | **Colab GPU** | Vision model needs GPU; runs on the gated subset only |
| Export DB → Drive | **Colab** | `ingested.db` is the single artifact that crosses over |
| Re-flag eval/train | **Local CPU** | Light SQL matching against frozen CSV |
| Evaluation (100 eval posts) | **Local CPU** | DB-independent (CSV-based), few minutes on CPU; you hold the ground truth locally |
| Evidence index build (embeddings) | **Local CPU** | `all-MiniLM-L6-v2` is light |
| Dashboard (Streamlit) | **Local CPU** | Interactive, local |

**Data movement:** only `ingested.db` (SQLite) crosses Colab↔local, via **Google Drive** (it's
gitignored — never commit it). **Code** crosses via **git** (this branch). Images are downloaded and
analyzed on Colab and never leave it — only the derived `vision_signal` text is stored in the DB.

---

## Part D — Execution sequence (back-and-forth runbook)

### Phase 1 — LOCAL (VS Code), on `multimodal-edge-gating`: implement + push code
**Status: code complete & tested locally.** Capture is **live-validated** (real Bluesky fetch:
images/external-links/bio/post-count all captured). Gating + reader-signal consumption unit-tested
(real photo → precision save; cartoon/AI → flag stands). **Vision *inference* is tested on Colab**
(its GPU home), not locally, per decision.
1. `store.py` — add columns: `has_image, image_url, image_alt, reshare_of_author, reshare_of_uri,`
   `external_url, external_title, author_bio, author_post_count, author_created_at, vision_signal`.
2. `bluesky.py` — parse `post.embed` (images / external / record) in BOTH the SDK and raw-HTTP paths;
   extend `get_profiles` to also return `description`, `posts_count`, `created_at`.
3. `pipeline/vision.py` — NEW: `gate_edge_cases(db)` (Part B1) → `analyze_image(url)` (vision + parse
   + correct) → write `vision_signal`. CLI: `--gate-and-analyze`.
4. `evidence.py` / reader signal — let `vision_signal` adjust the red flag on edge cases only.
5. `config.yaml` — add `vision:` block (model name, gate thresholds).
6. Commit + push the branch (code only; DB never committed).

### Phase 2 — COLAB (GPU): fresh data + heavy compute  [`notebooks/week7_colab_pipeline.ipynb`]
- **C1** Clone repo w/ PAT → `git checkout multimodal-edge-gating`.
- **C2** Install deps; start Ollama; `ollama pull qwen2.5:3b` + the vision model.
- **C3** Provide creds: upload `.env` (Bluesky) or Colab secrets. *(Use the rotated app password.)*
- **C4** Ingest: `python -m climate_verifier.ingestion.scheduler` → new `data/ingested.db` (all fields).
- **C5** Classify: `python -m climate_verifier.pipeline.claim_classifier` (batch=1).
- **C6** Gate + vision: `python -m climate_verifier.pipeline.vision --gate-and-analyze`.
- **C7** Export: copy `data/ingested.db` → Google Drive.

### Phase 3 — LOCAL (VS Code, CPU): validate + decide
- **L1** Pull `ingested.db` from Drive into `data/` (backup already preserves the old one).
- **L2** Re-flag: match `claim_eval.csv` texts → set `in_eval_set` (in_train_set optional).
- **L3** Evaluate: `python -m climate_verifier.pipeline.evaluate` → recall/precision. Compare to baseline.
- **L4** Build index: `python -m climate_verifier.pipeline.evidence --build`.
- **L5** Dashboard: `streamlit run app.py` — spot-check edge cases with the vision signal.
- **L6** Vision eval: score the image-edge eval set (Part B5).

### Phase 4 — Promote or iterate
- If **edge-case precision ↑ with recall ≥ 0.90 preserved** → merge `multimodal-edge-gating` → `main`.
- Else iterate on the gate / vision prompt on the branch. `main` stays untouched throughout.

---

## Part E — Validation / ground truth
`claim_eval.csv` stays the frozen yardstick — never relabel existing rows. If coverage of edge cases
is thin, **add** rows (and the separate image-edge eval set in B5) to strengthen the signal. All
metric comparisons are branch-vs-baseline on this same frozen set.

---

## Decisions (locked)
1. **Vision model:** `qwen2.5-vl` (best natural-photo + OCR mix for real-vs-cartoon).
2. **Colab data transfer:** Google Drive mount.
3. **Ingest volume:** **capped shakeout first** — a small run (e.g. `bluesky_limit: 15`) to prove the
   whole Colab↔local pipeline end-to-end fast, then the full sweep once it's clean. The cap is a
   config value, not a code change, so the shakeout and full run use identical code.
