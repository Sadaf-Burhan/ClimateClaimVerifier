"""
Build the LoRA training set (Week 5, precision-targeted, Option B).

Two phases so DB flagging stays on the canonical local DB and the slow
teacher-labeling can run on Colab/GPU:

  select  (LOCAL, fast)  — pick real posts for training, flag them in_train_set=1
                           in the DB (by post_id), and export their text, UNLABELED,
                           to tmp/colab/train_candidates.jsonl. Idempotent: reuses
                           already-flagged posts and only tops up to the target.

  label   (COLAB/GPU)    — teacher-label train_candidates.jsonl, merge with the
                           hand-labeled seed (data/lora_seed.jsonl), and write
                           tmp/colab/lora_trainset.jsonl. Asserts leak-free vs eval.

  all     — run both locally (only if a teacher model is available on this machine).

LEAKAGE MODEL — two DB flags are the single source of truth, by post_id:
  in_eval_set = 1  -> held out for evaluation
  in_train_set = 1 -> used to train the adapter
A post can never be both. The eval sampler excludes in_train_set; this builder
excludes in_eval_set; ChromaDB/RAG population (Week 6) must exclude BOTH so neither
held-out set leaks into retrieval. The synthetic seed is not in the DB, so it needs
no flag — it is guaranteed disjoint.

Run:
  uv run python scripts/build_lora_trainset.py select --target-size 120
  # then on Colab (teacher pulled):
  uv run python scripts/build_lora_trainset.py label --teacher gemma4:latest
"""

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "src" / "climate_verifier" / "config.yaml"
SEED = ROOT / "data" / "lora_seed.jsonl"
EVAL = ROOT / "data" / "claim_eval.csv"
CANDIDATES = ROOT / "tmp" / "colab" / "train_candidates.jsonl"
OUT = ROOT / "tmp" / "colab" / "lora_trainset.jsonl"

OVERSAMPLE_CATS = ("conspiracy", "combinations", "sensationalist")  # where precision fails

TEACHER_PROMPT = """You are labeling training data for a climate claim detector.
A CLAIM contains a specific, checkable assertion — a named event, mechanism, place,
date, measurement, official source, or attributed study — checkable even if false.
A vague conspiracy accusation with NO named event/mechanism/place/date/measurement is
an OPINION, as are emotional reactions, jokes, and general viewpoints. Never judge truth.

Return JSON only: {"thought": "<what is or isn't checkable, <8 words>", "has_claim": true|false, "reason": "<under 8 words>"}

Post: "%s"
Output:"""


def norm(t: str) -> str:
    return " ".join((t or "").split()).strip().lower()


def _db_path() -> Path:
    p = Path(yaml.safe_load(open(CONFIG))["storage"]["db_path"])
    return p if p.is_absolute() else ROOT / p


def _ensure_flag(con: sqlite3.Connection) -> None:
    cols = [r[1] for r in con.execute("PRAGMA table_info(posts)").fetchall()]
    if "in_train_set" not in cols:
        con.execute("ALTER TABLE posts ADD COLUMN in_train_set INTEGER NOT NULL DEFAULT 0")
        con.commit()


def phase_select(target_size: int, oversample_frac: float) -> None:
    seed = [json.loads(l) for l in open(SEED, encoding="utf-8")]
    need = max(0, target_size - len(seed))

    con = sqlite3.connect(_db_path())
    con.row_factory = sqlite3.Row
    _ensure_flag(con)
    cols = [r[1] for r in con.execute("PRAGMA table_info(posts)").fetchall()]
    has_eval = "in_eval_set" in cols

    eval_norm = {norm(r["post_text"]) for r in csv.DictReader(open(EVAL, encoding="utf-8"))}

    # reuse already-flagged training posts (idempotent), then top up to target
    flagged = con.execute(
        "SELECT post_id, text FROM posts WHERE in_train_set = 1"
    ).fetchall()
    picked = [{"post_id": r["post_id"], "text": r["text"]} for r in flagged
              if norm(r["text"]) not in eval_norm]

    if len(picked) < need:
        exclude = "in_train_set = 0" + (" AND (in_eval_set IS NULL OR in_eval_set = 0)" if has_eval else "")
        rows = con.execute(f"SELECT post_id, text, keyword_category FROM posts WHERE {exclude}").fetchall()
        over = [r for r in rows if r["keyword_category"] in OVERSAMPLE_CATS]
        rest = [r for r in rows if r["keyword_category"] not in OVERSAMPLE_CATS]
        ordered = over + rest  # oversample the failing categories first

        want_new = need - len(picked)
        n_over_target = int(want_new * oversample_frac)
        chosen, seen = [], {norm(p["text"]) for p in picked} | eval_norm
        n_over = 0
        for r in ordered:
            if len(chosen) >= want_new:
                break
            n = norm(r["text"])
            if not n or n in seen:
                continue
            if r["keyword_category"] not in OVERSAMPLE_CATS and n_over < n_over_target and \
               sum(1 for x in ordered if x["keyword_category"] in OVERSAMPLE_CATS and norm(x["text"]) not in seen):
                # still have oversample candidates left — prefer them
                pass
            seen.add(n)
            if r["keyword_category"] in OVERSAMPLE_CATS:
                n_over += 1
            chosen.append({"post_id": r["post_id"], "text": r["text"]})

        # flag the newly chosen posts in the DB by post_id
        con.executemany("UPDATE posts SET in_train_set = 1 WHERE post_id = ?",
                        [(c["post_id"],) for c in chosen])
        con.commit()
        picked += chosen

    con.close()

    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES, "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Selected {len(picked)} real posts, flagged in_train_set=1 in the DB (by post_id).")
    print(f"Exported UNLABELED -> {CANDIDATES}")
    print(f"Seed adds {len(seed)} hand-labeled examples at label time "
          f"(target total ~{len(picked) + len(seed)}).")
    print("Next: run the 'label' phase (on Colab with the teacher model pulled).")


def phase_label(teacher: str) -> None:
    import ollama
    if not CANDIDATES.exists():
        raise SystemExit(f"{CANDIDATES} not found — run the 'select' phase first.")

    seed = [json.loads(l) for l in open(SEED, encoding="utf-8")]
    cands = [json.loads(l) for l in open(CANDIDATES, encoding="utf-8")]
    eval_norm = {norm(r["post_text"]) for r in csv.DictReader(open(EVAL, encoding="utf-8"))}

    labeled = []
    for i, c in enumerate(cands, 1):
        t = c["text"]
        try:
            resp = ollama.chat(
                model=teacher,
                messages=[{"role": "user", "content": TEACHER_PROMPT % t[:500]}],
                format={"type": "object",
                        "properties": {"thought": {"type": "string"},
                                       "has_claim": {"type": "boolean"},
                                       "reason": {"type": "string"}},
                        "required": ["thought", "has_claim", "reason"]},
                options={"temperature": 0.0, "num_predict": 200},
            )
            m = re.search(r"\{.*\}", resp["message"]["content"], re.DOTALL)
            if not m:
                continue
            d = json.loads(m.group())
            labeled.append({"text": t, "has_claim": bool(d.get("has_claim", False)),
                            "thought": str(d.get("thought", "")), "reason": str(d.get("reason", ""))})
            if i % 10 == 0:
                print(f"  labeled {i}/{len(cands)}")
        except Exception as e:
            print(f"  teacher error on #{i}: {e}")

    combined = seed + labeled
    bad = [e for e in combined if norm(e["text"]) in eval_norm]
    assert not bad, f"LEAK: {len(bad)} training rows are in the eval set"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for e in combined:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    claims = sum(1 for e in combined if e["has_claim"])
    print(f"\nWrote {OUT}: {len(combined)} examples "
          f"({claims} claim / {len(combined) - claims} opinion). "
          f"Seed {len(seed)} + teacher {len(labeled)}. Leak-free.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "label", "all"])
    ap.add_argument("--teacher", default="gemma4:latest")
    ap.add_argument("--target-size", type=int, default=120)
    ap.add_argument("--oversample-frac", type=float, default=0.6)
    args = ap.parse_args()

    if args.phase in ("select", "all"):
        phase_select(args.target_size, args.oversample_frac)
    if args.phase in ("label", "all"):
        phase_label(args.teacher)


if __name__ == "__main__":
    main()
