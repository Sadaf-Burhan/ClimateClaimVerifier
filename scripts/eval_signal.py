"""
SIGNAL eval — does the READER get the right answer? The third benchmark, and the only one that
tests the system rather than the classifier.

Why this exists: the project's core design decision is "recall-first classifier; PRECISION IS
RECOVERED DOWNSTREAM by evidence + source signals." That claim had never been measured. The gold and
dynamic evals both score the bare classifier — recall (which must be measured there, because a claim
the classifier drops never reaches downstream and is unrecoverable) and precision (which the design
explicitly says to ignore). Nothing measured the thing users actually see: **does the red flag fire
when it should, and stay quiet when it shouldn't?**

What it measures: RED-FLAG precision/recall against hand-labeled real posts. The red flag is the
scanner's one assertive output — a false one accuses an innocent post, a missed one is the
misinformation pattern going unflagged.

Why the retrieval is held constant: the red flag depends on `verdict == "none"`, which depends on
whatever the GDELT corpus happens to hold today. That moves daily, so "expected_red_flag" is only
stable ground truth if the retrieval side is pinned. We pin it to NO CORROBORATION — the case where
the flag can fire — which isolates exactly what the design claims recovers precision: the SOURCE
suppressors (official / credible cite / real-photo vision / eyewitness).

  uv run python scripts/eval_signal.py
  uv run python scripts/eval_signal.py --out data/signal_eval_results.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from climate_verifier.pipeline.evidence import assess_claim, _load_cfg

EVAL_CSV = Path("data/signal_eval.csv")

# The pinned retrieval: no corroboration found. This is the ONLY state in which the red flag can
# fire, so it's the state worth testing — anything the system suppresses here, it suppresses because
# of the post's own source signals, which is the "precision recovered downstream" claim under test.
NO_CORROBORATION = {"proximity": 0.0, "tier": "NONE", "location": "", "matches": []}


def _b(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def _int(v) -> int:
    try:
        return int(str(v).strip() or 0)
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description="Evaluate the reader SIGNAL (red flag), not the classifier.")
    ap.add_argument("--out", default="data/signal_eval_results.json", help="write results here (JSON)")
    args = ap.parse_args()

    if not EVAL_CSV.exists():
        raise SystemExit(f"No signal eval set at {EVAL_CSV}.")
    rows = list(csv.DictReader(open(EVAL_CSV, newline="", encoding="utf-8")))
    cfg = _load_cfg()

    tp = fp = tn = fn = 0
    records = []
    print(f"\nSignal eval — {len(rows)} labeled post(s) · retrieval pinned to NO CORROBORATION\n" + "=" * 78)

    for r in rows:
        expected = _b(r["expected_red_flag"])
        a = assess_claim(None, r["post_text"], engagement=_int(r["engagement"]), source="bluesky",
                         followers=_int(r["followers"]), author=r["author"], cfg=cfg,
                         external_url=r["external_url"], retrieval=NO_CORROBORATION)
        s = a["signal"]
        got = bool(s["red_flag"])
        ok = got == expected
        # why did it (not) fire — the suppressor that decided it
        why = []
        if s.get("treated_official"):
            why.append("official")
        if s.get("credible_cite"):
            why.append("credible-cite")
        if s.get("eyewitness"):
            why.append("eyewitness")
        if s.get("vision_supports"):
            why.append("vision-save")
        if s.get("forecast"):
            why.append("forecast(detected, NOT a suppressor in code)")
        if _int(r["engagement"]) < cfg["evidence"].get("high_reach", 50):
            why.append("below high_reach")

        if got and expected:
            tp += 1
        elif got and not expected:
            fp += 1
        elif not got and not expected:
            tn += 1
        else:
            fn += 1

        icon = "✅" if ok else "❌"
        print(f"\n{icon} @{r['author']}  eng={r['engagement']}")
        print(f"   {r['post_text'][:88]!r}")
        print(f"   expected red_flag={expected}  got={got}"
              + (f"   [{', '.join(why)}]" if why else "   [no suppressor — flag fires]"))
        if not ok:
            print(f"   ⚠️  {'FALSE ALARM' if got else 'MISSED FLAG'} — {r['notes'][:96]}")
        records.append({"author": r["author"], "engagement": _int(r["engagement"]),
                        "expected": expected, "got": got, "correct": ok, "why": why,
                        "notes": r["notes"]})

    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    acc = (tp + tn) / len(rows) if rows else None
    print("\n" + "=" * 78 + "\nSUMMARY — the RED FLAG (what the reader actually sees)")
    print(f"  correct: {tp + tn}/{len(rows)}" + (f"  ({acc:.0%})" if acc is not None else ""))
    print(f"  flagged correctly (TP): {tp}   correctly quiet (TN): {tn}")
    print(f"  FALSE ALARMS (FP): {fp}   <- accuses an innocent post; the cost of bad precision")
    print(f"  MISSED FLAGS (FN): {fn}   <- misinformation pattern goes unflagged")
    print(f"  red-flag precision: {prec if prec is None else round(prec, 3)}   "
          f"recall: {rec if rec is None else round(rec, 3)}")
    print("\n  Reads on the design claim 'precision is recovered downstream': every TN above with a"
          "\n  suppressor listed IS that recovery happening. Every FP is a case where it did not.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(rows), "true_positives": tp, "false_positives": fp,
                               "true_negatives": tn, "false_negatives": fn,
                               "red_flag_precision": prec, "red_flag_recall": rec,
                               "accuracy": acc, "records": records}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n  📄 wrote results → {out}\n")


if __name__ == "__main__":
    main()
