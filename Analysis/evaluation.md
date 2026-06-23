Back to clean baseline — and actually better than the messy 105-row run:

| Run | Recall | Accuracy | Notes |
|---|---|---|---|
| 60 synthetic rows (original) | ~0.88 | ~0.88 | Synthetic only, easy |
| 105 rows (messy real_data) | 0.745 | 0.762 | Bad labels inflated errors |
| **100 rows (clean)** | **0.792** | **0.820** | Fixed 3 mislabeled rows |

The 3 label fixes moved 3 FNs to TNs (they were posts the model correctly called opinion — we just had them mislabeled as claims). That's where the jump from 0.745 → 0.792 came from.

**What's left (18 errors — structural model limits):**

The 10 remaining FNs split into two honest explanations:

1. **Model reasoning contradicts its own output** — `denial_with_stat` posts: model says *"Satellite data is checkable"* / *"Arctic ice claim is checkable"* in the reason, yet outputs `has_claim=false`. This is a small-model structured-output failure, not fixable with few-shot examples as we saw.

2. **Model judges truth, not checkability** — `false_but_checkable` posts ("Higher CO2 makes the planet greener", chemtrails in Texas). Even with example 7 (HAARP) in the prompt, the model still filters by plausibility.

The 8 FPs are the cheaper error (opinions surfacing on dashboard), and they're all coherent: hyperbole doom sounds like predictions, vague conspiracy posts have confident language, the Greenland "buffoon" post genuinely contains a climate claim buried in opinion framing.

**The honest conclusion:** `qwen2.5:3b` has a ceiling of ~0.79 recall on this task without a better model or a two-pass approach. For your assignment write-up, 0.792 on a clean 100-post stratified eval set with this analysis is strong work — you can document exactly *why* each failure category is hard rather than just missing the number.

---

**Baseline:** My fixed few-shot uses 8 hardcoded examples in the prompt — same examples for every post regardless of what it's about. I've already measured this: recall on CLAIM = 0.792, accuracy = 0.820 on my 100-post eval set.

**Test set:** I have a held-out eval set of 100 labeled posts stratified across 5 keyword categories (scientific, extreme_events, sensationalist, conspiracy, combinations), roughly 10 claims and 10 opinions per category. These posts are flagged in the database so they're excluded from ChromaDB — the retrieval pool never contains posts I'm evaluating on, so there's no data leakage.

**Metric:** Primary metric is recall on CLAIM. I care about this more than accuracy because a missed claim is unrecoverable — it never reaches the dashboard and can't be corrected downstream. False positives are cheap because a human reviewer can discard a surfaced opinion. So my headline comparison is: does RAG recall beat 0.792?

**How I'd run it:** Run the same evaluation on the same 100 posts, but this time each post retrieves its own few-shot examples from ChromaDB using category-stratified retrieval before classification. Everything else stays the same — same model, same temperature, same eval script.

**What I'd look for beyond the headline number:** My current per-category breakdown shows conspiracy and sensationalist have ~30% error rates while scientific has ~10%. The hypothesis is RAG helps most in the hard categories because it retrieves posts with similar rhetorical framing — a conspiracy post gets conspiracy examples rather than a generic one. If RAG improves recall in conspiracy and sensationalist specifically, that confirms the retrieval is doing something meaningful. If improvement is uniform or absent, it suggests 8 well-chosen fixed examples already cover the decision space for a binary task, which is also a valid finding.

What result would make you confident retrieval is helping? And what result would make you doubt it — or even reconsider the approach entirely?

---

**What would make me confident retrieval is helping:**

If RAG recall on CLAIM is meaningfully higher than 0.792 — I'd say at least +0.05 improvement to rule out noise given the small eval set size. More importantly, I'd want to see the improvement concentrated in conspiracy and sensationalist, which are currently my hardest categories at ~30% error rate. If those two improve while scientific stays flat, that's the mechanism working as intended: retrieval is surfacing rhetorically similar examples that the fixed prompt can't cover with 8 generic posts.

**What would make me doubt it:**

If recall improves overall but the per-category breakdown shows the gains are in scientific or extreme_events — the categories that were already easy. That would suggest the model just got lucky on a few posts, not that retrieval is doing anything structurally useful.

Also if recall on CLAIM improves but false positives increase significantly — more opinions being called claims — that's a tradeoff worth flagging, not a clean win.

**What would make me reconsider the approach entirely:**

Two things. First, if RAG recall is the same or worse than fixed. For a binary task, 8 well-chosen fixed examples may already span the decision space well enough that dynamic retrieval adds noise rather than signal — especially with a 3B model that has limited context capacity.

Second, if I look at the misclassified posts and the retrieved examples have nothing semantically in common with what's being misclassified. My current failures aren't vocabulary problems — denial_with_stat posts fail because the model contradicts its own reasoning, and false_but_checkable posts fail because the model judges truth instead of checkability. Neither of those is a retrieval problem. If that pattern holds, the right fix is a stronger model or a different prompt structure, not better retrieval.

