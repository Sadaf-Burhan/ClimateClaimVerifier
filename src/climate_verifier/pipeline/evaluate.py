"""
Week 1 evaluation: classifier quality metrics on the hand-labeled eval set.

Accuracy alone hides the asymmetry between the two error directions:

  False negative (claim labeled opinion) — the post is discarded at the
  classifier gate and never reaches the dashboard. A missed claim is
  unrecoverable, so this is the costly error.

  False positive (opinion labeled claim) — the post surfaces on the
  dashboard, where low evidence proximity and source context let the
  reader discount it. Cheap and self-correcting.

The success criterion is therefore recall on CLAIM (target set in
config.yaml), reported alongside per-class precision/recall/F1, the
confusion matrix, false-negative vs false-positive counts, and error
breakdowns by keyword category and post type.

Run from the project root:
  uv run python -m climate_verifier.pipeline.evaluate
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .claim_classifier import classify_batch, LLM_BATCH_SIZE

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
EVAL_HISTORY_PATH = Path("data/eval_history.jsonl")   # drift log — one JSON line per eval run


def load_eval_set(csv_path: str) -> list[dict]:
    """Loads the hand-labeled eval CSV. Raises if missing or empty."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Eval set is empty: {csv_path}")
    return rows


def run_eval(csv_path: str, model: str,
             llm_batch_size: int = LLM_BATCH_SIZE) -> list[dict]:
    """
    Classifies every post in the eval set and attaches the prediction.
    Returns one dict per post: the CSV row plus predicted_label, reason, correct.
    """
    rows = load_eval_set(csv_path)
    results = []
    for start in range(0, len(rows), llm_batch_size):
        chunk = rows[start:start + llm_batch_size]
        predictions = classify_batch([r["post_text"] for r in chunk], model=model)
        for row, pred in zip(chunk, predictions):
            predicted = "claim" if pred["has_claim"] else "opinion"
            results.append({
                **row,
                "predicted_label": predicted,
                "reason": pred["reason"],
                "correct": predicted == row["expected_label"],
            })
    return results


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _breakdown(results: list[dict], key: str) -> dict:
    """Per-group error breakdown: n, errors, FN/FP split, error rate."""
    groups: dict[str, dict] = {}
    for r in results:
        g = groups.setdefault(r[key], {
            "n": 0, "errors": 0, "false_negatives": 0, "false_positives": 0,
        })
        g["n"] += 1
        if not r["correct"]:
            g["errors"] += 1
            if r["expected_label"] == "claim":
                g["false_negatives"] += 1
            else:
                g["false_positives"] += 1
    for g in groups.values():
        g["error_rate"] = g["errors"] / g["n"]
    return dict(sorted(groups.items(),
                       key=lambda kv: (kv[1]["error_rate"], kv[1]["n"]),
                       reverse=True))


def compute_metrics(results: list[dict],
                    claim_recall_target: float = 0.90) -> dict:
    """
    Classification quality metrics with CLAIM as the positive class.

    Returns accuracy, per-class precision/recall/F1, confusion matrix,
    FN-vs-FP asymmetry stats, per-category breakdowns, the misclassified
    posts, and whether the recall-on-CLAIM success criterion is met.
    """
    tp = sum(1 for r in results if r["expected_label"] == "claim"   and r["predicted_label"] == "claim")
    fn = sum(1 for r in results if r["expected_label"] == "claim"   and r["predicted_label"] == "opinion")
    fp = sum(1 for r in results if r["expected_label"] == "opinion" and r["predicted_label"] == "claim")
    tn = sum(1 for r in results if r["expected_label"] == "opinion" and r["predicted_label"] == "opinion")

    claim_recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(results),
        "accuracy": (tp + tn) / len(results) if results else 0.0,
        "confusion_matrix": {
            "true_claim_predicted_claim": tp,
            "true_claim_predicted_opinion": fn,    # missed claim — costly
            "true_opinion_predicted_claim": fp,    # extra dashboard entry — cheap
            "true_opinion_predicted_opinion": tn,
        },
        "per_class": {
            "claim":   {**_prf(tp, fp, fn), "support": tp + fn},
            "opinion": {**_prf(tn, fn, fp), "support": tn + fp},
        },
        "error_asymmetry": {
            "false_negatives": fn,
            "false_positives": fp,
            "fn_rate": fn / (tp + fn) if tp + fn else 0.0,  # share of real claims missed
            "fp_rate": fp / (tn + fp) if tn + fp else 0.0,  # share of opinions surfaced
        },
        "claim_recall_target": claim_recall_target,
        "meets_target": claim_recall >= claim_recall_target,
        "by_keyword_category": _breakdown(results, "keyword_category"),
        "by_post_type": _breakdown(results, "post_type"),
        "misclassified": [
            {k: r[k] for k in ("post_text", "expected_label", "predicted_label",
                               "keyword_category", "post_type", "reason")}
            for r in results if not r["correct"]
        ],
    }


def snapshot_metrics(metrics: dict, model: str, path: Path | str = EVAL_HISTORY_PATH,
                     ts: str | None = None) -> dict:
    """Append a compact, timestamped eval snapshot to the drift log (JSONL).

    Each maintenance eval (Colab GPU, or a local run) records one line here; the app's
    Evaluation tab reads them to plot recall/precision over time and the FN:FP balance —
    the signal that tells you if the classifier is drifting. Pass `ts` to stamp a specific
    time (else now, UTC)."""
    claim = metrics["per_class"]["claim"]
    asym = metrics["error_asymmetry"]
    rec = {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "model": model,
        "n": metrics["n"],
        "recall": round(claim["recall"], 4),
        "precision": round(claim["precision"], 4),
        "f1": round(claim["f1"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "false_negatives": asym["false_negatives"],
        "false_positives": asym["false_positives"],
        "fn_rate": round(asym["fn_rate"], 4),
        "fp_rate": round(asym["fp_rate"], 4),
        "meets_target": metrics["meets_target"],
        "target": metrics["claim_recall_target"],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load_eval_history(path: Path | str = EVAL_HISTORY_PATH) -> list[dict]:
    """Read the drift log (oldest-first). Returns [] if it doesn't exist yet."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def format_report(metrics: dict) -> str:
    """Plain-text report (ASCII only, safe for any console)."""
    cm = metrics["confusion_matrix"]
    asym = metrics["error_asymmetry"]
    claim = metrics["per_class"]["claim"]
    opinion = metrics["per_class"]["opinion"]
    target = metrics["claim_recall_target"]
    verdict = "PASS" if metrics["meets_target"] else "FAIL"

    lines = [
        "=" * 68,
        f"CLAIM CLASSIFIER EVALUATION  (n={metrics['n']})",
        "=" * 68,
        "",
        f"HEADLINE  Recall on CLAIM: {claim['recall']:.3f}  "
        f"(target >= {target:.2f})  ->  {verdict}",
        f"          Precision on CLAIM: {claim['precision']:.3f}",
        f"          Accuracy: {metrics['accuracy']:.3f}",
        "",
        "Confusion matrix (positive class = claim):",
        "                     pred claim   pred opinion",
        f"  actual claim     {cm['true_claim_predicted_claim']:>10}   "
        f"{cm['true_claim_predicted_opinion']:>12}   <- false negatives (missed claims, costly)",
        f"  actual opinion   {cm['true_opinion_predicted_claim']:>10}   "
        f"{cm['true_opinion_predicted_opinion']:>12}      false positives are the cheap error",
        "",
        "Per-class metrics:",
        f"  {'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
        f"  {'claim':<10}{claim['precision']:>10.3f}{claim['recall']:>10.3f}"
        f"{claim['f1']:>10.3f}{claim['support']:>10}",
        f"  {'opinion':<10}{opinion['precision']:>10.3f}{opinion['recall']:>10.3f}"
        f"{opinion['f1']:>10.3f}{opinion['support']:>10}",
        "",
        "Error asymmetry:",
        f"  False negatives (claim -> opinion): {asym['false_negatives']}  "
        f"({asym['fn_rate']:.1%} of real claims missed)",
        f"  False positives (opinion -> claim): {asym['false_positives']}  "
        f"({asym['fp_rate']:.1%} of opinions surfaced)",
        "",
    ]

    for title, key in (("By post type:", "by_post_type"),
                       ("By keyword category:", "by_keyword_category")):
        lines.append(title)
        lines.append(f"  {'group':<22}{'n':>4}{'errors':>8}{'FN':>5}{'FP':>5}{'error rate':>12}")
        for name, g in metrics[key].items():
            lines.append(
                f"  {name:<22}{g['n']:>4}{g['errors']:>8}"
                f"{g['false_negatives']:>5}{g['false_positives']:>5}"
                f"{g['error_rate']:>11.1%}"
            )
        lines.append("")

    if metrics["misclassified"]:
        lines.append("Misclassified posts:")
        for m in metrics["misclassified"]:
            lines.append(
                f"  [{m['expected_label']} -> {m['predicted_label']}] "
                f"({m['post_type']}) \"{m['post_text'][:70]}\""
            )
            lines.append(f"      model reason: {m['reason'][:90]}")
    else:
        lines.append("Misclassified posts: none")

    return "\n".join(lines)


def main():
    import argparse, sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    eval_cfg = cfg.get("evaluation", {})
    csv_path = eval_cfg.get("claim_eval_csv", "data/claim_eval.csv")
    target = float(eval_cfg.get("claim_recall_target", 0.90))
    llm_batch_size = cfg["model"].get("llm_batch_size", LLM_BATCH_SIZE)

    parser = argparse.ArgumentParser(description="Evaluate the claim classifier on the labeled eval set.")
    parser.add_argument("--model", default=cfg["model"]["name"],
                        help="Ollama model to evaluate (default: model.name from config.yaml)")
    parser.add_argument("--snapshot", action="store_true",
                        help="append this run to the drift log (data/eval_history.jsonl). Off by "
                             "default — the drift series is produced by the Colab GPU maintenance "
                             "pass so every point shares one backend; a local CPU run would not compare.")
    args = parser.parse_args()
    model = args.model

    rows = load_eval_set(csv_path)
    print(f"Evaluating {model} on {len(rows)} labeled posts from {csv_path} ...")
    results = run_eval(csv_path, model=model, llm_batch_size=llm_batch_size)
    metrics = compute_metrics(results, claim_recall_target=target)
    print(format_report(metrics))
    if args.snapshot:
        rec = snapshot_metrics(metrics, model=model)
        print(f"\nDrift log updated: {EVAL_HISTORY_PATH} (recall {rec['recall']}, "
              f"precision {rec['precision']}, FN/FP {rec['false_negatives']}/{rec['false_positives']}).")
    else:
        print("\n(Diagnostic run — not written to the drift log. Use --snapshot to record it, "
              "or run the Colab GPU maintenance pass for a backend-comparable point.)")


if __name__ == "__main__":
    main()
