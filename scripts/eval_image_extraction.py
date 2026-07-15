"""
Week-7 image-extraction eval — how well the image-INPUT path reads a climate claim out of a
screenshot, and whether it reaches the SAME reader signal as feeding the claim text in directly.

Pipeline under test:  extract_from_image()  →  screenshot_signal_inputs()  →  classify() + assess_claim()

Scores three things (see data/image_eval/README.md):
  1. Transcription (near-exact)  — claim_text similarity, author_handle exact, engagement per-field exact.
  2. Classification (accuracy)   — image_type, depicts_claim, has_readable_text.
  3. End-to-end verdict agreement (the PRIMARY bar) — classify(extracted) vs classify(expected) label
     agreement, and (with the evidence index built) retrieved news_status agreement.

Needs the vision model (qwen2.5vl:7b) reachable via Ollama — run on the Colab GPU (same place as
classification/vision), not on the CPU app machine.

  uv run python scripts/eval_image_extraction.py
  uv run python scripts/eval_image_extraction.py --limit 5
  uv run python scripts/eval_image_extraction.py --no-assess     # skip the RAG/news-status check
"""

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

try:                                    # Windows console is cp1252 — the ✅/▸ glyphs need UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from climate_verifier.pipeline.vision import (
    extract_from_image, screenshot_signal_inputs, looks_like_social_post, _load_cfg,
)
from climate_verifier.pipeline.claim_classifier import classify

EVAL_DIR    = Path("data/image_eval")
LABELS_PATH = EVAL_DIR / "labels.jsonl"
IMAGES_DIR  = EVAL_DIR / "images"
NEAR_EXACT  = 0.85          # claim_text similarity at/above this counts as a near-exact transcription


def _norm(s: str | None) -> str:
    """Lowercase, drop punctuation, and COLLAPSE whitespace runs — for a forgiving comparison.
    The collapse matters: the model preserves the post's real newlines while a CSV/JSON label
    flattens them, so without it an identical transcription scores as drift purely on how many
    spaces a paragraph break became."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())


def _text_sim(a: str | None, b: str | None) -> float:
    """Similarity of two transcriptions in [0,1]; both-empty = 1.0 (agreeing on 'no text')."""
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _load_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        raise SystemExit(f"No labels at {LABELS_PATH}. See data/image_eval/README.md.")
    rows = []
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def _score_engagement(exp: dict, got: dict) -> tuple[int, int]:
    """Per-field exact match on the three engagement counts (null must match null). Returns
    (correct, total) — hallucinating a count where the truth is null is a miss, the worst kind."""
    correct = total = 0
    for k in ("likes", "reposts", "replies"):
        total += 1
        if (exp or {}).get(k) == (got or {}).get(k):
            correct += 1
    return correct, total


def main():
    ap = argparse.ArgumentParser(description="Evaluate the Week-7 image-extraction path.")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N labeled rows")
    ap.add_argument("--no-assess", action="store_true",
                    help="skip the RAG/news-status agreement check (no ChromaDB needed)")
    ap.add_argument("--out", default=str(EVAL_DIR / "eval_results.json"),
                    help="write the per-image results + summary here (JSON)")
    args = ap.parse_args()
    out_path = Path(args.out)

    cfg = _load_cfg()
    vz = cfg.get("vision", {})
    model, corrector = vz.get("model", "qwen2.5vl:7b"), vz.get("corrector_model", "qwen2.5:3b")
    mdl = cfg["model"]["name"]

    labels = _load_labels()
    if args.limit:
        labels = labels[:args.limit]

    # Optional end-to-end retrieval agreement (needs the built evidence index).
    store = None
    if not args.no_assess:
        try:
            from climate_verifier.pipeline.evidence import get_store, assess_claim
            store = get_store()
            if store.count() == 0:
                print("⚠️  Evidence index empty — skipping news_status agreement (build it with "
                      "`python -m climate_verifier.pipeline.evidence --build`).")
                store = None
        except Exception as e:
            print(f"⚠️  Could not open evidence store ({e}) — skipping news_status agreement.")
            store = None

    agg = {"claim_sim": [], "claim_near": [], "handle_ok": [], "handle_n": 0,
           "eng_correct": 0, "eng_total": 0, "type_ok": [], "depicts_ok": [], "hrt_ok": [],
           "label_agree": [], "news_agree": [], "gate_ok": [], "missing": 0, "read_fail": 0}

    records = []
    print(f"\nImage-extraction eval — {len(labels)} labeled image(s), model {model}\n" + "=" * 78)

    for row in labels:
        img = IMAGES_DIR / row["image"]
        exp = row.get("expected", {})
        rec = {"image": row["image"], "expected_label": row.get("expected_label")}
        print(f"\n▸ {row['image']}")
        if not img.exists():
            print(f"  ⏭  image file missing ({img}) — drop it in {IMAGES_DIR}/ to score this row.")
            agg["missing"] += 1
            rec["status"] = "missing"; records.append(rec); continue

        got = extract_from_image(img.read_bytes(), model=model, corrector=corrector)
        if not got:
            print("  ❌ extraction failed (vision model unreachable, or unreadable image).")
            agg["read_fail"] += 1
            rec["status"] = "extraction_failed"; records.append(rec); continue
        rec["got"] = got

        # 0) Social-post gate. A row with NO expected engagement is a NON-post -> the correct system
        #    behavior is REJECTION. Score whether the gate accepts posts and rejects non-posts.
        exp_eng = exp.get("engagement") or {}
        expected_reject = not any(exp_eng.get(k) is not None for k in ("likes", "reposts", "replies"))
        accepted = looks_like_social_post(got)
        gate_ok = accepted != expected_reject
        agg["gate_ok"].append(gate_ok)
        rec.update(expected_reject=expected_reject, accepted=accepted, gate_ok=gate_ok)
        print(f"  gate: expected {'REJECT (not a post)' if expected_reject else 'ACCEPT (post)'}, "
              f"system {'accepted' if accepted else 'rejected'} {'✅' if gate_ok else '❌'}")
        if expected_reject:
            # a non-post: correct behavior is rejection, so the pipeline never runs — nothing else to score.
            if not gate_ok:
                print(f"     ⚠️ gate let a non-post through — model read engagement {got.get('engagement')}")
            rec["status"] = "reject_case"; records.append(rec); continue

        # 1) Transcription
        csim = _text_sim(exp.get("claim_text"), got.get("claim_text"))
        near = csim >= NEAR_EXACT
        agg["claim_sim"].append(csim); agg["claim_near"].append(near)
        rec.update(claim_sim=round(csim, 3), claim_near=near)
        print(f"  claim_text sim={csim:.2f} {'✅near-exact' if near else '⚠️drift'}")
        print(f"     expected: {exp.get('claim_text')!r}")
        print(f"     got:      {got.get('claim_text')!r}")
        if exp.get("author_handle"):                       # only score when a handle is expected
            agg["handle_n"] += 1
            ok = _norm(exp["author_handle"]) == _norm(got.get("author_handle"))
            agg["handle_ok"].append(ok)
            rec["author_handle_ok"] = ok
            print(f"  author_handle {'✅' if ok else '❌'} exp={exp['author_handle']!r} got={got.get('author_handle')!r}")
        ec, et = _score_engagement(exp.get("engagement"), got.get("engagement"))
        agg["eng_correct"] += ec; agg["eng_total"] += et
        rec["engagement_exact"] = f"{ec}/{et}"
        print(f"  engagement {ec}/{et} fields exact  exp={exp.get('engagement')} got={got.get('engagement')}")

        # 2) Classification
        for field, bucket in (("image_type", "type_ok"), ("depicts_claim", "depicts_ok"),
                              ("has_readable_text", "hrt_ok")):
            ok = exp.get(field) == got.get(field)
            agg[bucket].append(ok)
            rec[f"{field}_ok"] = ok
            print(f"  {field} {'✅' if ok else '❌'} exp={exp.get(field)} got={got.get(field)}")
        if got.get("platform") != exp.get("platform"):     # info only — inferred, not gated
            print(f"  platform (info only) exp={exp.get('platform')} got={got.get('platform')}")

        # 3) End-to-end verdict agreement — the primary bar
        exp_claim, got_claim = exp.get("claim_text"), got.get("claim_text")
        if exp_claim and got_claim:
            lt = classify(exp_claim, model=mdl)["has_claim"]
            lg = classify(got_claim, model=mdl)["has_claim"]
            agree = lt == lg
            agg["label_agree"].append(agree)
            rec["classify_agree"] = agree
            rec["image_label"] = "claim" if lg else "opinion"
            exp_lbl = row.get("expected_label")
            extra = f" (label≈expected: {'✅' if (('claim' if lg else 'opinion') == exp_lbl) else '❌'})" if exp_lbl else ""
            print(f"  ↳ classify agreement {'✅' if agree else '❌'}  text={'CLAIM' if lt else 'OPINION'} "
                  f"vs image={'CLAIM' if lg else 'OPINION'}{extra}")
            if store is not None:
                inp = screenshot_signal_inputs(got)
                st_img = assess_claim(store, got_claim, engagement=inp["engagement"],
                                      source="uploaded_screenshot", author=inp["author"],
                                      external_url=inp["external_url"], vision=inp["vision"], cfg=cfg
                                      )["signal"]["news_status"]
                st_txt = assess_claim(store, exp_claim, engagement=inp["engagement"],
                                      source="uploaded_screenshot", cfg=cfg)["signal"]["news_status"]
                na = st_img == st_txt
                agg["news_agree"].append(na)
                rec.update(news_status_agree=na, news_status_image=st_img, news_status_text=st_txt)
                print(f"  ↳ news_status agreement {'✅' if na else '❌'}  image={st_img} vs text={st_txt}")
        rec["status"] = "assessed"
        records.append(rec)

    # ── Summary ──
    def pct(xs):
        return f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "—"
    def rate(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    print("\n" + "=" * 78 + "\nSUMMARY")
    scored = len(agg["claim_sim"])
    n_gated = len(agg["gate_ok"])
    n_reject = n_gated - scored
    print(f"  ran {n_gated} image(s): {scored} post(s) assessed, {n_reject} non-post(s) (reject cases)"
          + (f", {agg['missing']} missing file(s)" if agg["missing"] else "")
          + (f", {agg['read_fail']} extraction failure(s)" if agg["read_fail"] else ""))
    if n_gated:
        print(f"  Social-post gate · correct accept/reject: {pct(agg['gate_ok'])}")
    if scored:
        print(f"  Transcription  · claim_text near-exact: {pct(agg['claim_near'])}  "
              f"· mean sim: {sum(agg['claim_sim'])/scored:.2f}")
        print(f"                 · author_handle exact: {pct(agg['handle_ok']) if agg['handle_n'] else '— (none expected)'}  "
              f"· engagement fields exact: {agg['eng_correct']}/{agg['eng_total']}")
        print(f"  Classification · image_type: {pct(agg['type_ok'])}  "
              f"· depicts_claim: {pct(agg['depicts_ok'])}  · has_readable_text: {pct(agg['hrt_ok'])}")
        print(f"  End-to-end     · classify agreement: {pct(agg['label_agree'])}  "
              f"· news_status agreement: {pct(agg['news_agree'])}   ← PRIMARY BAR")

    summary = {
        "rows": len(labels), "gated": n_gated, "posts_assessed": scored, "reject_cases": n_reject,
        "missing_files": agg["missing"], "extraction_failures": agg["read_fail"],
        "gate_accuracy": rate(agg["gate_ok"]),
        "claim_near_exact_rate": rate(agg["claim_near"]),
        "claim_mean_sim": round(sum(agg["claim_sim"]) / scored, 3) if scored else None,
        "author_handle_exact_rate": rate(agg["handle_ok"]),
        "engagement_fields_exact": f"{agg['eng_correct']}/{agg['eng_total']}",
        "image_type_acc": rate(agg["type_ok"]), "depicts_claim_acc": rate(agg["depicts_ok"]),
        "has_readable_text_acc": rate(agg["hrt_ok"]),
        "classify_agreement": rate(agg["label_agree"]),
        "news_status_agreement": rate(agg["news_agree"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "assessed_news_status": store is not None,
                                    "summary": summary, "records": records},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  📄 wrote results → {out_path}")
    print()


if __name__ == "__main__":
    main()
