# ClimateClaimVerifier — Final System Design

The end-state: an **autonomous daily pipeline** that ingests climate/extreme-weather social posts,
classifies claim-vs-opinion, corroborates claims against news (RAG), applies a gated image signal to
edge cases, and surfaces everything through a **reader-facing dashboard** — as *signals that guide the
reader*, never an automated truth verdict (the "metal detector," not "judge").

---

## 1. Finalization — what merges to `main`

Everything on `multimodal-edge-gating` is validated, tested, and documented. **Recommendation: merge
the whole branch to `main`.** It's all sound, isolated code; vision is included because it's *correct*
and cleanly gated (fires on ~1% of posts, harms nothing), with its low coverage honestly documented.

| Change | Status | Merge? |
|---|---|---|
| Data capture (images / reshare / profile) — `store.py`, `bluesky.py` | live-validated | ✅ |
| Source-selectable ingestion (`run_ingestion_cycle(sources=…)`) | tested | ✅ |
| WebP→JPEG vision fix + `vision.py` (gated) | validated | ✅ |
| Widened gate (all categories) + official allowlist (weather bots) | tested | ✅ |
| Single-post classifier default (batch-16 landmine removed) | pure fix | ✅ |
| Reader-signal: reshare + vision consumption | tested | ✅ |

Merge as a **merge commit** (preserve the branch history) after a final secret scan. Tag it `v1.0`.

---

## 2. Autonomous system architecture (the daily loop)

```
              ┌──────────────────────── DAILY (scheduler / cron, 24h guard) ────────────────────────┐
              │                                                                                       │
  [1] INGEST ─┼─► Bluesky (keywords, images/reshare/profile)   ┐                                      │
              │   GDELT news (home IP — not rate-limited)       ├─► [2] TOPIC FILTER (NA climate/wx)  │
              │                                                 ┘         │                            │
              │                                                          ▼                            │
              │                                    [3] CLASSIFY  (qwen2.5:3b, SINGLE-POST, recall-first)
              │                                                          │  has_claim / opinion + reason
              │                                             ┌────────────┴────────────┐               │
              │                                             ▼                         ▼               │
              │                              [4] EVIDENCE INDEX refresh        (opinions: stored,      │
              │                                  (embed new GDELT → ChromaDB)   shown, not corroborated)│
              │                                             │                                          │
              │                              [5] VISION (gated) — image-claims not resolved by         │
              │                                  metadata → qwen2.5vl (real-photo vs cartoon)          │
              └──────────────────────────────────────────────┬──────────────────────────────────────┘
                                                              ▼
                                         [6] READER SIGNALS (on-demand / nightly cache)
                                    corroboration re-rank + reach + source + vision → red flag
                                                              ▼
                                         [7] DASHBOARD (Streamlit) — reader-facing product
```

**Scheduling:** `apscheduler` (already in `scheduler.py`) or Windows Task Scheduler / cron calling a
single `run_daily()` entrypoint that chains ingest → classify → index → vision. The 24h guard prevents
redundant runs.

**Vision in the daily loop:** it only touches ~1% of posts (image-edge-cases), so even on local CPU
`qwen2.5vl` handles a day's ~10–15 edge cases in a few minutes. (Alternative: a weekly GPU batch.)

---

## 3. Data model (SQLite `ingested.db` + ChromaDB)

- **`posts`** — text, author, engagement, followers, timestamp, category; **captured signals**:
  `has_image/image_url/image_alt`, `reshare_of_author/uri`, `external_url/title`, `author_bio/
  post_count/created_at`, `vision_signal` (JSON). Governance: `in_eval_set`, `in_train_set`.
- **`classifications`** — `has_claim` + `reason`.
- **ChromaDB `gdelt_evidence`** — embedded GDELT headlines (the RAG corpus).
- **`claim_eval.csv`** — frozen 100-row benchmark (never regenerated).

---

## 4. Dashboard design (the reader-facing product)

Five sections, mapping directly to what you described:

### § A — Daily Summary & Detector Health
- **Ingestion:** N new posts today, broken down by category.
- **Classifier health:** recall / precision on the frozen `claim_eval.csv` (the "how good is our
  detector" number), framed recall-first (a missed claim is the costly error). Reminds the reader the
  classifier *detects claims*, it does not judge truth.
- **Split:** N classified CLAIM vs N OPINION (today / window).

### § B — Claims vs Opinions (top lists)
Two ranked lists (by engagement, then recency):
- **Top CLAIMS** — snippet · author · engagement · category · classifier reason.
- **Top OPINIONS** — same, so the reader sees what was filtered out and why.
Click a row → detail view (§C or §D).

### § C — Red-Flag Deep-Dive (misinformation-amplification cases)
Among top claims, the **red-flagged** ones (high reach + no corroboration + unverified source + no
cited evidence + not rescued by a real event-photo). For each:
- **Full post content** (text + image if any).
- **"Why this is flagged"** — plain-language reasons, *suggestive not accusatory*: *"spreading widely
  (N engagements); no published news in our set reports this specific event; unverified account; no
  source linked."* Plus the vision note if present (*"cartoon imagery — leans satire/fabrication"*).
- **How to spot misinformation wording** (reusable educational panel) — the patterns the eval surfaced:
  vague conspiracy (*"they're hiding the truth"*), unfalsifiable claims, urgency/ALL-CAPS, *"PROOF
  inside"*, no source, denial-with-cherry-picked-stat. Teaches the reader to judge — the metal-detector
  principle applied to language.
- **Verification links** — the **Bluesky post URL** + the retrieved **GDELT article URLs**, so the
  reader checks the primary sources themselves.

### § D — Corroboration Panel (credible sources from the RAG layer)  ← Module 6
For a selected top **claim**, show the evidence the RAG layer pulled:
- The top-k retrieved **GDELT articles** (domain · date · headline · clickable URL).
- The **corroboration verdict** (corroborated / partial / none) and which article the re-rank cited.
- Framed as *"published news our system found on this topic — open them to verify,"* never *"this is
  true."* This is the "list of credible sources for top claim posts" you asked for.

### § E — Analysis tabs (keep existing)
Embedding Analysis (Week 2) and Base-vs-Adapter demo (Week 5) stay as method/demonstration tabs.

---

## 5. Guardrail (unchanged philosophy)

Every surface **shows signals and teaches the reader**; none renders a truth verdict. No numeric
credibility score, no HIGH/MED/LOW. Corroboration = "did published news cover this specific event,"
not "is it true." Vision = a descriptive image signal. The reader concludes.

---

## 6. Module 6 & 7 alignment (course deliverables, in the final product)

- **Module 6 (RAG):** § D *is* the deliverable — retrieve GDELT (dense) → LLM re-rank for
  same-specific-event → surface credible sources. **Optional enhancement:** hybrid search (dense +
  BM25) for rare terms (HAARP, place names), straight from Module 6's "Beyond" section.
- **Module 7 (multimodal):** § C's image signal *is* the deliverable — vision as a second entry point,
  gated to edge cases, feeding the reader signal (validated; low-coverage, documented).

---

## 7. Build plan (what's done vs new)

**Already built:** classifier (single-post), evidence/corroboration (RAG), vision (gated), dashboard
tabs (Claim Classifier, Embedding, Evidence Matching, Base-vs-Adapter).

**New for finalization (proposed order):**
1. **Merge branch → `main`** (secret scan, merge commit, tag `v1.0`).
2. **`run_daily()` entrypoint** — chain ingest → classify → index → vision; wire to the scheduler.
3. **Dashboard §A–§D** — the daily summary, claims-vs-opinions lists, red-flag deep-dive (with the
   misinformation-wording panel + verification links), and the corroboration source panel. Most of the
   data plumbing exists (`assess_db_claims`, `evidence_for_claim`); this is mostly UI assembly.
4. **(Optional) hybrid search** in the retrieval layer (Module 6 extension).
5. **Docs:** fold this design into `project_description.md`; update `README.md` run instructions.

**Deferred / honest scope:** vision stays a low-coverage supplement until the accumulate-and-re-measure
plan shows otherwise; corroboration is the precision workhorse.
