"""
In-house maintenance chain — the GPU pass (run from Colab, 2–3×/week).

Local daily ingestion (climate_verifier.ingestion.scheduler) accumulates raw posts on CPU.
This module runs the compute-heavy stages that want a GPU, in order:

    classify → vision → reindex → evaluate

- classify : label pending posts claim/opinion (single-post, recall-first)
- vision   : gated image analysis on edge cases → posts.vision_signal
- reindex  : rebuild the ChromaDB evidence index so new claims are retrievable
- evaluate : score the classifier on the labeled set AND append a drift snapshot
             (data/eval_history.jsonl) — the signal the app plots for model drift

Every stage writes a health heartbeat (data/health.json) so the external-facing app shows
freshness/failure instead of silently serving stale data. Export data/ back to the project
(ingested.db, eval_history.jsonl, health.json, chroma_evidence/) for the website to use.

    uv run python -m climate_verifier.maintenance --all
    uv run python -m climate_verifier.maintenance --classify --evaluate
"""

import argparse
from pathlib import Path

import yaml

from climate_verifier.pipeline.claim_classifier import classify_pending, LLM_BATCH_SIZE
from climate_verifier.pipeline.evaluate import (
    run_eval, compute_metrics, snapshot_metrics, format_report,
)
from climate_verifier.pipeline import vision
from climate_verifier.pipeline.relabel import apply_overrides, ADMIN_OVERRIDES_PATH
from climate_verifier.health import update_health

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_classification(cfg: dict) -> int:
    """Classify every pending post. Records health; returns the number newly classified."""
    db = cfg["storage"]["db_path"]
    model = cfg["model"]["name"]
    lbs = cfg["model"].get("llm_batch_size", LLM_BATCH_SIZE)
    print(f"[classify] model={model} llm_batch_size={lbs} …")
    try:
        done = claims = 0
        for u in classify_pending(db, model=model, batch_size=10**9, llm_batch_size=lbs):
            done = u["done"]
            claims += 1 if u["has_claim"] else 0
        update_health("classification", True, counts={"classified": done, "claims": claims})
        print(f"[classify] {done} newly classified ({claims} claims).")
        return done
    except Exception as e:
        update_health("classification", False, error=f"{type(e).__name__}: {e}")
        raise


def run_vision(cfg: dict) -> dict:
    """Gated vision on edge cases → posts.vision_signal. Records health; returns {gated, analyzed}."""
    db = cfg["storage"]["db_path"]
    print("[vision] gating edge cases + analyzing images …")
    try:
        res = vision.gate_and_analyze(db, cfg)
        update_health("vision", True, counts=res)
        print(f"[vision] gated {res['gated']}, analyzed {res['analyzed']}.")
        return res
    except Exception as e:
        update_health("vision", False, error=f"{type(e).__name__}: {e}")
        raise


def run_reindex(cfg: dict) -> int:
    """Rebuild the evidence index so newly classified claims can be matched. Returns article count."""
    from climate_verifier.pipeline.evidence import get_store  # lazy: heavy import
    print("[reindex] rebuilding the evidence index …")
    n = get_store().build_index(cfg["storage"]["db_path"])
    print(f"[reindex] {n} GDELT articles indexed.")
    return n


def run_evaluation(cfg: dict) -> dict:
    """Evaluate the classifier on BOTH labeled sets + append one drift snapshot each. Records health.

    Two sets, two questions — never collapse them into one number:
      gold    (frozen)  — data constant, so any move = MODEL/ENV change (the +/- jitter band)
      dynamic (growing) — relabel loop appends, so a move = model change + distribution change
                          (concept drift: does it generalize to the new hard cases?)
    Gold is the control that makes dynamic readable. Health/target status reports GOLD, because a
    dynamic miss can just mean "the new cases are hard", while a gold miss is a real regression.
    A missing gold file degrades to dynamic-only rather than failing the pass.
    """
    ev = cfg.get("evaluation", {})
    target = float(ev.get("claim_recall_target", 0.90))
    model = cfg["model"]["name"]
    lbs = cfg["model"].get("llm_batch_size", LLM_BATCH_SIZE)
    sets = [("gold", ev.get("gold_eval_csv", "data/claim_eval_gold.csv")),
            ("dynamic", ev.get("claim_eval_csv", "data/claim_eval.csv"))]
    try:
        recs = {}
        for name, csv_path in sets:
            if not Path(csv_path).exists():
                print(f"[evaluate] {name}: {csv_path} not found — skipping.")
                continue
            print(f"[evaluate] {name} set — {model} on {csv_path} …")
            results = run_eval(csv_path, model=model, llm_batch_size=lbs)
            metrics = compute_metrics(results, claim_recall_target=target)
            print(format_report(metrics))
            rec = snapshot_metrics(metrics, model=model, eval_set=name)
            recs[name] = rec
            print(f"[evaluate] {name}: n={rec['n']} recall={rec['recall']} "
                  f"precision={rec['precision']} meets_target={rec['meets_target']} "
                  f"(digest={rec.get('model_digest')}, ollama={rec.get('ollama_version')})")
        if not recs:
            raise FileNotFoundError("no eval set found (checked gold + dynamic)")
        primary = recs.get("gold") or recs.get("dynamic")     # gold is the regression signal
        update_health("evaluation", True, recall=primary["recall"], precision=primary["precision"],
                      meets_target=primary["meets_target"],
                      sets={k: {"n": v["n"], "recall": v["recall"]} for k, v in recs.items()})
        if "gold" in recs and "dynamic" in recs:
            g, d = recs["gold"], recs["dynamic"]
            print(f"\n[evaluate] gold recall {g['recall']} (n={g['n']})  vs  dynamic "
                  f"{d['recall']} (n={d['n']}) — gold moving = model/env; only dynamic moving = "
                  "concept drift on the newly relabeled cases.")
        return primary
    except Exception as e:
        update_health("evaluation", False, error=f"{type(e).__name__}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="GPU maintenance pass: classify / vision / reindex / evaluate.")
    parser.add_argument("--classify", action="store_true", help="classify pending posts")
    parser.add_argument("--vision", action="store_true", help="gated vision on edge cases")
    parser.add_argument("--reindex", action="store_true", help="rebuild the evidence index")
    parser.add_argument("--evaluate", action="store_true", help="evaluate + record a drift snapshot")
    parser.add_argument("--all", action="store_true", help="classify → vision → reindex → evaluate")
    args = parser.parse_args()

    cfg = load_cfg()

    # ALWAYS restore admin overrides first, whatever else was asked for.
    #
    # Colab clones the repo and pulls ingested.db from Drive, so the DB it works on is whatever
    # snapshot Drive last held — and its export then replaces the local DB. Any relabel made after
    # that snapshot has its admin_label wiped, and since the nomination queries filter
    # `admin_label IS NULL`, every wiped post floods back into the review queue as if it had never
    # been judged. That already cost 5 of the 14 relabels committed on 2026-07-15.
    #
    # The overrides log is git-tracked, so the clone always carries it: replaying it here makes the
    # DB's cache column self-healing rather than depending on anyone remembering a manual step.
    # Unconditional and outside the failure loop — it is a cheap UPDATE per override, and if it
    # cannot run, every downstream count is quietly wrong.
    try:
        res = apply_overrides(cfg["storage"]["db_path"])
        if res["applied"] or res["missing"]:
            print(f"[overrides] restored {res['applied']} admin label(s) from "
                  f"{ADMIN_OVERRIDES_PATH}" +
                  (f"; {res['missing']} post(s) have aged out of the corpus" if res["missing"] else ""))
    except Exception as e:
        print(f"[overrides] FAILED to restore admin labels: {type(e).__name__}: {e}")

    do_all = args.all or not any([args.classify, args.vision, args.reindex, args.evaluate])
    # keep the causal order: classify produces claims, vision annotates them, reindex makes
    # evidence retrievable, evaluate scores the classifier (independent, runs last).
    steps = [
        ("classify", args.classify or do_all, run_classification),
        ("vision",   args.vision   or do_all, run_vision),
        ("reindex",  args.reindex  or do_all, run_reindex),
        ("evaluate", args.evaluate or do_all, run_evaluation),
    ]
    failures = []
    for name, enabled, fn in steps:
        if not enabled:
            continue
        try:
            fn(cfg)
        except Exception as e:
            # keep going so one broken stage (e.g. vision model missing) doesn't block the rest;
            # the health file already recorded the failure for the app banner.
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            failures.append(name)

    if failures:
        print(f"\nMaintenance finished with failures in: {', '.join(failures)}")
        raise SystemExit(1)      # non-zero → the notebook cell / caller shows the run failed
    print("\nMaintenance complete — export data/ (ingested.db, eval_history.jsonl, "
          "health.json, chroma_evidence/) back to the project for the website.")


if __name__ == "__main__":
    main()
