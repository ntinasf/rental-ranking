"""Review sentiment — a demonstration of the Azure AI Language path, **not a model feature**.

**Measured before any spend**, on a local pilot over whole query groups. Within query groups —
the only comparison a pairwise ranker ever makes — sentiment adds almost nothing to the label
correlation once the rating is partialled out, and Airbnb's own ``review_scores_value``, which is
free, beats it several times over.

**The ceiling test settles it.** Among listings at the rating ceiling, where the rating separates
nothing, sentiment does vary — it genuinely de-compresses the ceiling — but its correlation with
the label among them is *negative*. The mechanism works; the signal is absent.

**Aspect mining fails on coverage rather than on signal.** The well-covered aspects are exactly
the ones Airbnb already ships as numeric sub-scores (cleanliness, location). The unrated ones
that could add something have a median of zero mentions per listing.

So no sentiment column reaches the feature matrix, and **no ``torch``/``transformers`` dependency
is added** — the cloud image is pinned from ``uv.lock`` and would have carried ~2-3 GB for it.

**What this module is for.** The one sanctioned run is a demonstration inside the Azure AI
Language **F0 free tier** (5,000 text records per month) over a handful of whole query groups.
Two rules apply to it exactly as they would to a production run: **cache the raw responses before
any aggregation**, so re-aggregating never re-bills, and **never call inside a loop over
listings** — batch the documents. Aggregation from cached responses is a pure transform and
belongs here; the call itself belongs to the Azure scripts.

Rejected and worth not re-deriving: scoring only the **top-N most-reviewed listings within a
group** makes "has sentiment" a proxy for review count, the label's dominant driver, so the model
would rank on the *missingness* without reading the value — the banned "has price" flag pattern.
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
