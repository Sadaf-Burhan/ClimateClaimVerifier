"""
Stage 4: Evidence Matching  (Week 6 — Semantic Search / RAG)

For each classified CLAIM, retrieve the most similar GDELT news articles from a
ChromaDB vector store and produce an *evidence-proximity* signal: does published
news cover a similar event?

This is NOT a truth verdict. It is a reader signal. Combined with reach
(engagement) and source context it surfaces the **reach-vs-support** mismatch —
a claim spreading widely with no news backing from an unverified source is the
misinformation red flag the reader should evaluate. The system never says "false".

Design:
  - Evidence corpus = GDELT news articles (`source='gdelt'`), excluding any post
    flagged `in_eval_set`/`in_train_set` (held out from the classifier).
  - Query = a classified claim (usually a Bluesky post).
  - Embedding model = `all-MiniLM-L6-v2` (same as Week 2), cosine space.
  - similarity = 1 - cosine_distance; proximity tier by config thresholds.

Build the index, then query:
  uv run python -m climate_verifier.pipeline.evidence --build
  uv run python -m climate_verifier.pipeline.evidence --claim "HAARP is causing the Alberta floods"
"""

import json
import re
import sqlite3
from pathlib import Path

import chromadb
import ollama
import yaml
from chromadb.utils import embedding_functions

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
COLLECTION = "gdelt_evidence"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class ClimateEvidenceStore:
    """Persistent ChromaDB collection of GDELT news articles, queried by claim text."""

    def __init__(self, chroma_path: str, embed_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embed_model
            ),
            metadata={"hnsw:space": "cosine"},  # distances are cosine → similarity = 1 - distance
        )

    def count(self) -> int:
        return self.collection.count()

    def build_index(self, db_path: str) -> int:
        """(Re)index leak-free GDELT articles. Idempotent — upsert by post_id."""
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cols = [r[1] for r in con.execute("PRAGMA table_info(posts)").fetchall()]
        # exclude held-out posts from the retrieval pool (leakage guard)
        guard = ""
        if "in_eval_set" in cols:
            guard += " AND (in_eval_set IS NULL OR in_eval_set = 0)"
        if "in_train_set" in cols:
            guard += " AND (in_train_set IS NULL OR in_train_set = 0)"
        rows = con.execute(
            f"SELECT post_id, text, author, keyword_category, created_at "
            f"FROM posts WHERE source = 'gdelt'{guard}"
        ).fetchall()
        con.close()

        rows = [r for r in rows if (r["text"] or "").strip()]
        if rows:
            self.collection.upsert(
                ids=[r["post_id"] for r in rows],
                documents=[r["text"] for r in rows],
                metadatas=[{
                    "domain":   r["author"] or "",
                    "url":      r["post_id"],
                    "category": r["keyword_category"] or "",
                    "date":     (r["created_at"] or "")[:10],
                } for r in rows],
            )
        return self.collection.count()

    def evidence_for_claim(self, claim_text: str, k: int = 5,
                           high: float = 0.60, low: float = 0.40) -> dict:
        """
        Returns {proximity, tier, matches}:
          proximity — top cosine similarity to any GDELT article (0-1)
          tier      — HIGH (news covers a similar event) / LOW / NONE (no corroboration)
          matches   — top-k articles [{title, domain, url, date, similarity}]
        """
        n = self.collection.count()
        if n == 0 or not (claim_text or "").strip():
            return {"proximity": 0.0, "tier": "NONE", "matches": []}
        res = self.collection.query(query_texts=[claim_text], n_results=min(k, n))
        sims = [round(1 - d, 3) for d in res["distances"][0]]
        matches = [{
            "title":      doc,
            "domain":     m.get("domain", ""),
            "url":        m.get("url", ""),
            "date":       m.get("date", ""),
            "similarity": s,
        } for doc, m, s in zip(res["documents"][0], res["metadatas"][0], sims)]
        top = sims[0] if sims else 0.0
        tier = "HIGH" if top >= high else ("LOW" if top >= low else "NONE")
        return {"proximity": top, "tier": tier, "matches": matches}


# Re-ranking pass (Module 6 "Advanced RAG"): dense retrieval finds topically-similar
# news, but topical overlap is NOT corroboration. The LLM re-reads the claim against the
# retrieved articles and judges whether any describes the SAME specific event — a
# relevance judgment, NOT a truth verdict.
_CORROBORATION_PROMPT = """You check whether published NEWS corroborates a claim — whether any article describes the SAME specific event, mechanism, place, or measurement the claim asserts. You are NOT judging whether the claim is true. Topical overlap (same general subject or region) is NOT corroboration.

Claim: "{claim}"

Candidate news articles:
{articles}

Return JSON only: {{"verdict": "corroborated" | "partial" | "none", "article": <article number, or 0>, "reason": "<under 12 words>"}}
- corroborated: an article describes the same specific event/mechanism as the claim.
- partial: an article covers the same topic or region but not the specific claim.
- none: no article describes the specific event — only loose topical overlap, or nothing."""


def corroboration_check(claim_text: str, matches: list[dict], model: str) -> dict:
    """LLM re-rank: does any retrieved article corroborate the SPECIFIC claim?
    Returns {verdict: corroborated|partial|none, article: int, reason: str}."""
    if not matches:
        return {"verdict": "none", "article": 0, "reason": "no candidate articles"}
    listing = "\n".join(f'{i}. [{m["domain"]}] {m["title"][:120]}' for i, m in enumerate(matches, 1))
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user",
                       "content": _CORROBORATION_PROMPT.format(claim=claim_text[:400], articles=listing)}],
            format={"type": "object",
                    "properties": {"verdict": {"type": "string"},
                                   "article": {"type": "integer"},
                                   "reason": {"type": "string"}},
                    "required": ["verdict", "article", "reason"]},
            options={"temperature": 0.0, "num_predict": 100},
        )
        m = re.search(r"\{.*\}", resp["message"]["content"], re.DOTALL)
        d = json.loads(m.group()) if m else {}
        verdict = str(d.get("verdict", "none")).lower()
        if verdict not in ("corroborated", "partial", "none"):
            verdict = "none"
        return {"verdict": verdict, "article": int(d.get("article", 0) or 0),
                "reason": str(d.get("reason", ""))}
    except Exception as e:
        return {"verdict": "none", "article": 0, "reason": f"check error: {e}"}


def build_reader_signal(retrieval: dict, corro: dict, engagement: int, source: str,
                        followers: int = 0, domain: str = "", high_reach: int = 50) -> dict:
    """
    Plain-language, *suggestive* reader signal from the corroboration verdict, reach,
    and source context. Flags the reach-vs-support mismatch. Never asserts truth.
    """
    verdict = corro["verdict"]
    evidence_phrase = {
        "corroborated": "Published news corroborates this claim.",
        "partial":      "Published news covers the topic but not this specific claim.",
        "none":         "No published news describes this specific event.",
    }[verdict]
    art = corro.get("article", 0)
    if verdict != "none" and 1 <= art <= len(retrieval["matches"]):
        evidence_phrase += f" ({retrieval['matches'][art - 1]['domain']})"

    if source == "bluesky":
        src_phrase = f"Source: an unverified social account ({followers:,} followers)."
    else:
        src_phrase = f"Source: news domain {domain}." if domain else "Source: a news article."
    reach_phrase = f"Reach: {engagement:,} engagements." if engagement else "Reach: low engagement."

    red_flag = engagement >= high_reach and verdict == "none" and source == "bluesky"
    parts = [evidence_phrase, src_phrase, reach_phrase]
    if red_flag:
        parts.append("High reach with no news corroboration from an unverified source — "
                     "evaluate carefully; this is the pattern of misinformation amplification.")
    return {"summary": " ".join(parts), "red_flag": red_flag, "verdict": verdict,
            "proximity": retrieval["proximity"], "reason": corro.get("reason", "")}


def assess_claim(store: "ClimateEvidenceStore", claim_text: str, engagement: int = 0,
                 source: str = "bluesky", followers: int = 0, domain: str = "",
                 cfg: dict | None = None) -> dict:
    """Full Stage-4 assessment: retrieve → corroborate (LLM re-rank) → reader signal."""
    cfg = cfg or _load_cfg()
    ev = cfg.get("evidence", {})
    retrieval = store.evidence_for_claim(claim_text, k=ev.get("top_k", 5),
                                         high=ev.get("high_proximity", 0.60),
                                         low=ev.get("low_proximity", 0.40))
    corro = corroboration_check(claim_text, retrieval["matches"], model=cfg["model"]["name"])
    signal = build_reader_signal(retrieval, corro, engagement, source,
                                 followers=followers, domain=domain,
                                 high_reach=ev.get("high_reach", 50))
    return {"retrieval": retrieval, "corroboration": corro, "signal": signal}


def get_store() -> ClimateEvidenceStore:
    cfg = _load_cfg()
    ev = cfg.get("evidence", {})
    return ClimateEvidenceStore(
        chroma_path=ev.get("chroma_path", "data/chroma_evidence"),
        embed_model=cfg["embedding"]["model_name"],
    )


def main():
    import argparse
    cfg = _load_cfg()
    ev = cfg.get("evidence", {})
    parser = argparse.ArgumentParser(description="Evidence matching — GDELT news corroboration for claims.")
    parser.add_argument("--build", action="store_true", help="(re)build the GDELT evidence index")
    parser.add_argument("--claim", type=str, help="a claim to match against news evidence")
    parser.add_argument("--engagement", type=int, default=0, help="engagement count (for the reach-vs-support red flag)")
    args = parser.parse_args()

    store = get_store()
    if args.build:
        n = store.build_index(cfg["storage"]["db_path"])
        print(f"Evidence index built: {n} GDELT articles in ChromaDB ({ev.get('chroma_path')}).")
    if args.claim:
        a = assess_claim(store, args.claim, engagement=args.engagement, cfg=cfg)
        print(f"\nClaim: {args.claim}")
        print(f"Retrieved (top proximity {a['retrieval']['proximity']:.3f}):")
        for m in a["retrieval"]["matches"]:
            print(f"  {m['similarity']:.3f}  [{m['domain']}]  {m['title'][:75]}")
        print(f"Corroboration: {a['corroboration']['verdict'].upper()} — {a['corroboration']['reason']}")
        print(f"Red flag: {a['signal']['red_flag']}")
        print(f"READER SIGNAL: {a['signal']['summary']}")
    if not args.build and not args.claim:
        parser.print_help()


if __name__ == "__main__":
    main()
