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

from collections.abc import Sequence

import pandas as pd

from rental_ranking.data.validate import require_columns

#: Azure AI Language bills a **text record per 1,000 characters, rounded up, minimum one per
#: document**. The minimum is the part that decides the budget here and it is easy to get wrong:
#: this corpus has a median review of 195 characters (mean 263, p90 574), so almost every review
#: costs **one whole record** rather than the 0.2 its length suggests. The F0 tier's 5,000
#: records/month is therefore worth about **5,000 short reviews**, not 19,000 — measured
#: 2026-08-18 against the real selection, which priced 393 reviews at exactly 393 records.
CHARS_PER_RECORD = 1_000

#: Documents per sentiment request — the service's own cap, and the mechanism behind gotcha #5.
#: Batching is not an optimisation here; it is the difference between one call and five hundred.
MAX_DOCUMENTS_PER_REQUEST = 10

#: Characters accepted in one document. Longer reviews are **truncated, never split**: a review
#: is one opinion, and splitting it would let a long review vote twice in its listing's mean.
MAX_CHARS_PER_DOCUMENT = 5_120

#: Free-tier allowance. Exceeding it fails the call rather than charging for the overage, which
#: is the right failure mode for a demonstration.
FREE_TIER_RECORDS_PER_MONTH = 5_000


def text_records(documents: Sequence[str]) -> int:
    """Billable text records for ``documents`` — the number to check *before* calling.

    Args:
        documents: Review texts, already truncated to :data:`MAX_CHARS_PER_DOCUMENT`.

    Returns:
        Total records. Each document costs ``ceil(len / 1000)``, and never fewer than one.
    """
    return sum(max(1, -(-len(text) // CHARS_PER_RECORD)) for text in documents)


def batches(documents: Sequence[str], size: int = MAX_DOCUMENTS_PER_REQUEST) -> list[list[str]]:
    """Split ``documents`` into request-sized batches.

    Exists as a named function rather than an inline slice because **gotcha #5 is about this
    line**: the banned version is a loop that calls the API once per listing.
    """
    if size < 1:
        raise ValueError(f"size={size}: a batch needs at least one document")
    return [list(documents[i : i + size]) for i in range(0, len(documents), size)]


def aggregate_sentiment(responses: pd.DataFrame) -> pd.DataFrame:
    """Per-listing sentiment from the **cached** per-review responses.

    A pure transform over what the API already returned, which is the half of this demonstration
    that may run repeatedly. The call itself lives in ``cloud/sentiment.py`` and runs once.

    The listing score is the **mean of positive minus negative confidence**, not a majority vote
    over the returned labels: the label is a three-way argmax that throws away how confident the
    service was, and a listing whose reviews are all faintly positive is not the same as one
    whose reviews are all emphatically positive.

    Args:
        responses: One row per review, carrying ``listing_id``, the ``positive``/``neutral``/
            ``negative`` confidences, and the argmax ``sentiment``.

    Returns:
        One row per listing, indexed by ``listing_id``: ``reviews_scored``, ``sentiment_score``
        in ``[-1, 1]``, ``positive_share``, and the mean positive and negative confidences.

    Raises:
        KeyError: If a required column is absent.
    """
    require_columns(
        responses, ("listing_id", "positive", "negative", "sentiment"), "language responses"
    )
    grouped = responses.groupby("listing_id", observed=True)
    frame = pd.DataFrame(
        {
            "reviews_scored": grouped.size(),
            "sentiment_score": grouped["positive"].mean() - grouped["negative"].mean(),
            "positive_share": grouped["sentiment"].apply(lambda s: float(s.eq("positive").mean())),
            "mean_positive": grouped["positive"].mean(),
            "mean_negative": grouped["negative"].mean(),
        }
    )
    frame.index.name = "listing_id"
    return frame
