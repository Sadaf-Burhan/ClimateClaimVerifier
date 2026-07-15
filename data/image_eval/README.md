# Image-extraction eval set (Week 7)

Benchmark for the **image-input path** (`extract_from_image` → thin adapter → `assess_claim`).
It answers: *does reading a claim out of a screenshot reach the same reader signal as feeding the
same claim text in directly?*

## Layout
- `labels.jsonl` — **tracked** benchmark, one JSON object per line (the ground truth).
- `images/` — **git-ignored** (third-party screenshots; keep binaries/copyright out of the public
  repo). Put the actual `.png`/`.jpg`/`.webp` files here, named to match each row's `"image"`.

## A label row
```json
{
  "image": "doge_noaa_card.png",
  "note": "why this case matters",
  "expected": {
    "claim_text": "<the exact claim text visible in the image — transcribe literally>",
    "has_readable_text": true,
    "image_type": "real_photo|meme_or_cartoon|screenshot|infographic|ai_suspected|other",
    "depicts_claim": "yes|partial|no|unrelated",
    "author_handle": "@handle or null",
    "platform": "x|twitter|bluesky|facebook|instagram|tiktok|reddit|whatsapp|other or null",
    "engagement": {"likes": <int|null>, "reposts": <int|null>, "replies": <int|null>},
    "visible_citation": "<source/URL shown in image, or null>"
  },
  "expected_label": "claim|opinion"
}
```
**Rules for labelling** (mirror the model's rules): transcribe `claim_text` / `author_handle` /
`engagement` **literally**; use `null` for anything not clearly legible — never guess. `platform`
is scored as *info only* (not gated), because it's an inference, not a transcription.

Target: **10–15 rows**, spread across `image_type`s and across claim vs opinion, and including at
least one *image-carried-claim* case (caption is opinion, the claim is in the image — the case only
this path can catch; see `doge_noaa_card.png`).

## Run
Needs the vision model (`qwen2.5vl:7b`) reachable via Ollama — so run it on the **Colab GPU** (or a
machine that has pulled the model), the same place classification/vision run.

```bash
uv run python scripts/eval_image_extraction.py
uv run python scripts/eval_image_extraction.py --limit 5      # first N rows
uv run python scripts/eval_image_extraction.py --no-assess    # skip the RAG/news-status check (no ChromaDB)
```

## `claim_text` labelling policy (settled 2026-07-15 — read before labelling)
Label **the primary claim-bearing text that is actually VISIBLE**, wherever it lives:
- Usually that's the **post's body/caption** (Yale, Melghat, the NOAA report) — *not* a short slogan
  overlaid on the picture.
- For an **image-carried claim** it's the **card/infographic text** and *not* the caption, because the
  caption is only vibes (the DOGE case: caption "RESIST TYRANNY", claim on the card).
- **Never include text hidden behind a "See more"/"Show more" fold** — the model can't see it, so
  scoring against it measures nothing.

The first pass got this wrong (labelled overlay slogans, truncated bodies, and text past the fold),
which read as model failure when the model was right: mean sim 0.40 with **classify agreement at
100%**. If the transcription drifts but the claim still classifies identically, the difference is
cosmetic — the primary bar is what counts.

## Accept vs reject rows
The upload path only assesses **social-media posts**, whose reach (engagement) is the core signal.
So a row is one of two kinds, decided by its `expected.engagement`:
- **Post (accept case)** — has at least one engagement count. The gate should ACCEPT it and the
  pipeline runs; it's scored on transcription / classification / end-to-end.
- **Non-post (reject case)** — all engagement is `null` (a bare photo, infographic, meme, satellite
  image, illustration). The gate should REJECT it; there's nothing downstream to score. These rows
  test that the gate refuses non-posts (and that the model didn't *hallucinate* engagement).

## What it scores
0. **Social-post gate:** did the gate correctly ACCEPT posts and REJECT non-posts.
1. **Transcription (near-exact):** `claim_text` similarity, `author_handle` exact match, per-field
   `engagement` exact match. These must be tight — a fabricated handle/count is the worst failure.
2. **Classification (accuracy):** `image_type`, `depicts_claim`, `has_readable_text`.
3. **End-to-end verdict agreement (the primary bar):** classify the *extracted* claim vs the
   *expected* claim → label agreement; and (with the evidence index built) the retrieved
   `news_status` agreement. This is the real question — did the image path land on the same reader
   signal as feeding the text straight in?

> **Filename gotcha:** save the file as exactly the `"image"` value (e.g. `doge_noaa_card.png`). If
> your browser adds its own extension you'll get `doge_noaa_card.png.png` / `.png.webp`, which won't
> match. The content format doesn't matter (PIL reads WebP/JPG even if named `.png`) — only the name.
