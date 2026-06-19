# Evaluated alternative: NLI contradiction model as an L2 precision stage

**Question:** can a natural-language-inference (contradiction) model close the
residual precision gap the deterministic polarity guard leaves (~90% held-out)?

**Setup:** `cross-encoder/nli-distilroberta-base` (CPU), run on candidates that
pass cosine ≥ 0.90, scored in both directions (a→b, b→a). Paraphrase = mutual
entailment; reject signal = max contradiction. Held-out sets v3+v4 (the guard was
never tuned on these). Measured 2026-06-19.

## Result

| Pipeline (cosine ≥ 0.90 candidates) | Precision | Recall | False hits |
|---|---|---|---|
| cosine only | 63% | 68% | 15 |
| cosine + deterministic guard | 90% | 68% | 3 |
| cosine + NLI alone (contradiction < 0.5) | 76% | 66% | 8 |
| **cosine + guard + NLI (contradiction < 0.5)** | **96%** | **66%** | **1** |

## Findings

1. **NLI alone is worse than the guard** (76% vs 90%). NLI is trained on
   declarative sentence pairs; on imperative/question prompts ("how to encode…")
   it is unreliable — requiring high entailment collapses recall to 24%.
2. **NLI is complementary, not a replacement.** It catches exactly the cases the
   guard structurally cannot — numeric/quantity contradictions
   (`round to 2 decimals` vs `4 decimals`, `limit 10 rows` vs `100 rows` both score
   `contradiction = 1.00`). The guard catches morphological/antonym/direction
   flips NLI misses. Stacked (guard → NLI) they reach 96% held-out precision.
3. **It still does not reach 100%.** `login/logout` survives both: single tokens
   (so the guard's in/out rule never fires) and NLI rates them as entailment
   (`con = 0.04`). Open-domain polarity is not fully solvable this cheaply.
4. **Cost:** ~10 ms/candidate on CPU (two directions), only on L2 candidates — within
   the <50ms hit budget — plus a ~330 MB model and its startup load.

## Recommendation

Keep the deterministic guard as the **zero-cost default** (90–92% held-out, no
model, <1ms). Offer NLI as an **opt-in third stage** (`guard → NLI`) for
deployments that can absorb the model — e.g. on Databricks via model serving /
Foundation Model API, where the verify call is a governed endpoint rather than
local RAM. Expected lift: ~90% → ~96% held-out precision. Not wired into the
default pipeline: the cost/benefit is a deployment decision, and it does not
reach perfect precision.
