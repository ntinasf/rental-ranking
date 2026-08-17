"""Review sentiment — a demonstration of the Azure AI Language path, **not a model feature**.

**Decided 2026-08-17, before any spend, on measurement.** Piloted locally at zero cost on an
ephemeral environment: 80 *whole* query groups, 1,643 listings, 14,525 reviews scored with a
multilingual XLM-R sentiment model. Measured **within query groups**, which is the only
comparison a pairwise ranker ever makes, mean Spearman against the label:

* ``review_scores_value`` **+0.128**, ``review_scores_rating`` +0.082
* sentiment **+0.049**, and **+0.015 once the rating is partialled out** — indistinguishable from
  zero across 80 groups at SE ~ 0.025.

**The ceiling test settles it.** Among the 463 listings at rating >= 4.95, where the rating
separates nothing, sentiment does vary (sd 0.117) — it genuinely de-compresses the ceiling — but
its correlation with the label among them is **-0.026**. The mechanism works; the signal is
absent.

**Aspect mining fails on coverage rather than on signal.** The aspects that are well covered are
exactly the ones Airbnb already ships as numeric sub-scores (clean 85.7 % of listings, location
80.7 %). The unrated ones that could add something have a **median of zero mentions per listing**
— noise 45.4 %, bed 23.7 %, parking 17.8 %, wifi 17.5 %, air conditioning 13.1 %.

So no sentiment column reaches the feature matrix, and **no ``torch``/``transformers`` dependency
is added** — the cloud image is pinned from ``uv.lock`` and would have carried ~2-3 GB for a
feature worth +0.015.

**What this module is for.** The one sanctioned run is a stratified demonstration inside the
Azure AI Language **F0 free tier** (5,000 text records per month), showing sentiment and opinion
mining on a handful of whole query groups. Two rules from BUILD_GUIDE gotcha #5 apply to it
exactly as they would to a production run: **cache the raw responses before any aggregation**, so
re-aggregating never re-bills, and **never call inside a loop over listings** — batch the
documents. Aggregation from cached responses is a pure transform and belongs here; the call
itself belongs to the Azure scripts, run once.

Rejected alongside the full run, and worth not re-deriving: scoring only the **top-N
most-reviewed listings within a group** makes "has sentiment" a proxy for review count, which is
the label's dominant driver — the model would rank on the *missingness* without reading the
value, which is the banned "has price" flag pattern. Scoring only the **smallest groups** covers
about 30 listings of 44,684 and yields a two-regime model.
"""
