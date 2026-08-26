"""Review-history windows anchored at each listing's own T.

Review counts serve two roles that must not drift apart: the independent instrument the label is
validated against, and a model feature. Both come from this module, so a window used in
validation is computed the same way as the one used in training.

**The anchor is passed in, never re-derived.** ``min(calendar.date)`` is T and belongs to
``features.label``; ``min(reviews.date)`` is ``first_review``, a median of 1,106 days earlier. A
module that recomputed its own anchor from whichever frame it happens to hold would silently
answer a different question.

The primary instrument is the **same season, one year earlier** — reviews in
``[T - 365, T - 365 + 90)``. It is entirely pre-T, so leakage-free, and seasonally matched to the
July-September label window, unlike a trailing window from T which lands in shoulder season. It
correlates better with the label than the trailing window in two cities of three; the exception
is Crete, the most seasonal market, where recent activity tracks the coming summer better than
last summer does.
"""

import pandas as pd

from rental_ranking.data.validate import require_columns, warn_violations

#: Length of the counting window, matching the label window.
WINDOW_DAYS = 90

#: How far before T the window opens. 365 puts it in the same season one year earlier.
SEASON_LAG_DAYS = 365

#: Prior strength for :func:`rating_shrunk`, in pseudo-reviews: the number of reviews at which a
#: listing's own rating and the city prior carry equal weight.
#:
#: **It is a prior, not a fitted quantity.** The empirical-Bayes optimum is ``within-listing
#: variance / between-listing variance``, and the numerator is unobservable here: the reviews file
#: carries no per-review score. So 20 is a stated convention. If it is ever selected rather than
#: asserted it belongs in a validation comparison — tuning it against label correlation on the
#: full population is target encoding performed through the author's eyes.
SHRINKAGE_K = 20

#: Raw review columns carried through unchanged. ``number_of_reviews`` is also what the
#: reviews baseline ranks by, so it must be present in the matrix that baseline reads.
#:
#: **The six sub-scores are Airbnb's own aspect ratings**, kept on measurement rather than on
#: completeness. They are *not* one factor with the overall rating — location in particular tracks
#: it loosely — every one of them discriminates inside a query group, and ``review_scores_value``
#: is the strongest single label correlate in the whole review family, beating the overall rating
#: in two cities of three.
#:
#: **Their null pattern is not quite the overall rating's.** ``review_scores_rating`` is null for
#: exactly the never-reviewed listings; the sub-scores add a handful more, where a single reviewer
#: gave an overall score and left the sub-categories blank.
#:
#: They are passed through raw rather than shrunk: the small-n noise that motivates
#: :func:`rating_shrunk` applies to them too, but six more shrunk columns doubles the block for an
#: unmeasured gain.
PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
)

_REQUIRED_COLUMNS = ("listing_id", "date")

_LISTING_COLUMNS = (
    "city",
    "T",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "review_scores_rating",
    "first_review",
    "last_review",
    "reviews_per_month",
)


def reviews_in_window(
    reviews: pd.DataFrame,
    anchors: pd.Series,
    window_days: int = WINDOW_DAYS,
    starts_days_before: int = SEASON_LAG_DAYS,
    name: str | None = None,
) -> pd.Series:
    """Count each listing's reviews inside a window defined relative to its own anchor.

    The window is half-open: ``[T - starts_days_before, T - starts_days_before + window_days)``.
    Defaults give the same-season-last-year instrument; ``starts_days_before=window_days``
    gives a trailing window ending at T.

    Args:
        reviews: Processed reviews frame with ``listing_id`` and ``date``.
        anchors: T per listing, indexed by ``listing_id`` — pass ``occupancy_label(...)["T"]``.
            Listings absent from this index are ignored, which is how calendar orphans drop out.
        window_days: Width of the window.
        starts_days_before: Days before the anchor at which the window opens.
        name: Name for the returned Series; defaults to a description of the window.

    Returns:
        One row per listing **in** ``anchors``, in that index's order. A listing with no
        reviews in the window counts **0**, never NaN — 25-30 % of listings sit in exactly
        that state, and NaN would be read as missingness by LightGBM and dropped by scipy.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If ``anchors`` is not uniquely indexed, or if no review resolves against
            it — see the note on the index below.
    """
    require_columns(reviews, _REQUIRED_COLUMNS, "reviews")

    if not anchors.index.is_unique:
        raise ValueError(
            f"`anchors` must be uniquely indexed by listing_id; found "
            f"{int(anchors.index.duplicated().sum())} duplicated entries."
        )

    # Broadcast each review's own anchor onto its row. Reviews whose listing is absent from
    # `anchors` map to NaT, and every comparison against NaT is False, so they fall out.
    anchor = reviews["listing_id"].map(anchors)

    # `.map` matches on the *index* of `anchors`. A listings-shaped frame carries a RangeIndex
    # after a merge, so `frame["T"]` resolves nothing and every count comes back 0 — a wrong
    # answer that looks like a real one. Refuse it instead.
    if len(reviews) and len(anchors) and not anchor.notna().any():
        raise ValueError(
            "no review resolved against `anchors`, so every count would be 0. `anchors` must "
            "be indexed by listing_id: pass occupancy_label(...)['T'], or "
            "frame.set_index('id')['T'] — not frame['T'], whose index is positional. "
            f"Got an index of {anchors.index.dtype} starting {list(anchors.index[:3])}."
        )
    opens = anchor - pd.Timedelta(days=starts_days_before)
    closes = opens + pd.Timedelta(days=window_days)
    inside = reviews["date"].ge(opens) & reviews["date"].lt(closes)

    counts = reviews.loc[inside].groupby("listing_id").size()

    # reindex, not join: a listing with an empty window is absent from the groupby result, and
    # the honest value there is 0. Leaving it missing would understate every downstream mean.
    return (
        counts.reindex(anchors.index, fill_value=0)
        .astype("int64")
        .rename(name or f"reviews_w{window_days}_lag{starts_days_before}")
    )


def same_season_last_year(
    reviews: pd.DataFrame, anchors: pd.Series, window_days: int = WINDOW_DAYS
) -> pd.Series:
    """Reviews in the same calendar window one year before T — the validation instrument.

    Thin wrapper so the intent is readable at the call site and the lag is not a magic number
    passed positionally.
    """
    return reviews_in_window(
        reviews,
        anchors,
        window_days=window_days,
        starts_days_before=SEASON_LAG_DAYS,
        name="reviews_same_season_ly",
    )


def has_reviews(listings: pd.DataFrame) -> pd.Series:
    """The cold-start flag: has this listing ever been reviewed.

    **Plain, and never conjoined with a label condition.** "Zero reviews *and* a fully blocked
    window" implies ``label == 0`` exactly for every row it marks — a label wearing a feature's
    name. The flag earns its place alone: never-reviewed listings carry a markedly lower mean
    label than the rest, and a tree learns the interaction from the ingredient.

    Args:
        listings: Frame carrying ``number_of_reviews``.

    Returns:
        A boolean Series aligned to ``listings``, named ``has_reviews``.

    Raises:
        KeyError: If ``number_of_reviews`` is absent.
    """
    require_columns(listings, ("number_of_reviews",), "listings")
    return listings["number_of_reviews"].gt(0).rename("has_reviews")


def rating_shrunk(listings: pd.DataFrame, k: int = SHRINKAGE_K) -> pd.Series:
    """Shrink each listing's rating toward its city's mean in proportion to its evidence.

    ``(n * rating + k * city_mean) / (n + k)`` — the posterior mean of listing quality under a
    Normal-Normal model, where ``k`` is the prior's weight in pseudo-reviews. It exists because
    the raw rating cannot rank:

    * **It is ceiling-compressed.** 32.7 % of reviewed listings sit at exactly 5.0 and 71.4 % at
      4.8 or above, so a third of the population is tied.
    * **The ties are mostly unproven.** The raw score correlates *negatively* with the number of
      reviews behind it, so ranking on it promotes the untested. Shrinking flips that sign while
      staying strongly rank-correlated with the raw score.

    **Cold start is handled by construction.** With ``n = 0`` the expression returns the city mean
    exactly, so a listing with a null ``review_scores_rating`` receives a neutral prior rather
    than an imputed value, and this module needs no missing-value rule.

    The city mean is **leave-one-out**. At city scale the difference from the include-self mean is
    negligible — this is correctness, not a rescue.

    Args:
        listings: Frame carrying ``city``, ``number_of_reviews`` and ``review_scores_rating``.
        k: Prior strength in pseudo-reviews. Defaults to :data:`SHRINKAGE_K`; see its note on
            why this is asserted rather than fitted.

    Returns:
        A float Series aligned to ``listings``, named ``rating_shrunk``, never null.

    Warns:
        UserWarning: Once, with a count, if any listing has reviews but no rating — so the
            arithmetic never silently treats a real rating as absent.

    Raises:
        KeyError: If a required column is missing.
    """
    require_columns(listings, ("city", "number_of_reviews", "review_scores_rating"), "listings")

    rating = listings["review_scores_rating"]
    rated = rating.notna()
    warn_violations(
        listings["number_of_reviews"].gt(0) & ~rated,
        "listing(s) carry reviews but no review_scores_rating, so they are shrunk to the "
        "city prior as if unrated",
    )

    # The listing's own contribution is removed from its city's total, so the prior it is
    # pulled toward never contains itself. Unrated listings contribute nothing either way, so
    # for them this is simply the city mean — which is exactly the prior they should get.
    by_city = rating.groupby(listings["city"])
    city_total = by_city.transform("sum")
    city_count = by_city.transform("count")
    prior = (city_total - rating.fillna(0.0)) / (city_count - rated.astype("int64"))

    # Evidence is counted only where there is a rating to weight, so a hypothetical
    # reviews-without-rating row falls back to the prior rather than dragging it toward zero.
    evidence = listings["number_of_reviews"].where(rated, 0).astype("float64")
    shrunk = (evidence * rating.fillna(0.0) + k * prior) / (evidence + k)
    return shrunk.astype("float64").rename("rating_shrunk")


def listing_age_days(listings: pd.DataFrame) -> pd.Series:
    """Days from a listing's first review to its own T — how long it has been established.

    Null for the never-reviewed cohort, and **left null**: an age of 0 for a listing that has
    never been reviewed is a value that looks like data. ``has_reviews`` names that cohort
    explicitly and LightGBM learns a direction for the missing branch.
    """
    require_columns(listings, ("T", "first_review"), "listings")
    age = (listings["T"] - listings["first_review"]).dt.days
    warn_violations(age.lt(0), "listing(s) have a first review dated after their own T")
    return age.astype("float64").rename("listing_age_days")


def days_since_last_review(listings: pd.DataFrame) -> pd.Series:
    """Days from a listing's most recent review to its own T — recency of activity.

    Null for the same never-reviewed cohort, and left null for the same reason. Strictly pre-T:
    ``last_review <= T`` for every row.
    """
    require_columns(listings, ("T", "last_review"), "listings")
    gap = (listings["T"] - listings["last_review"]).dt.days
    warn_violations(gap.lt(0), "listing(s) have a last review dated after their own T")
    return gap.astype("float64").rename("days_since_last_review")


def review_features(
    listings: pd.DataFrame,
    reviews: pd.DataFrame | None = None,
    k: int = SHRINKAGE_K,
) -> pd.DataFrame:
    """Assemble the review feature block.

    Every column is strictly pre-T: ``first_review`` and ``last_review`` are ``<= T`` for every
    row, the counts are lifetime or trailing, and the same-season window closes a year before T.

    **These are model features and nothing else.** They must never reach the price-imputation
    cascade, which keys on what a property *is* rather than how it has performed.

    Args:
        listings: The ranked population, carrying ``id`` and ``_LISTING_COLUMNS``.
        reviews: Processed reviews frame. When given, ``reviews_same_season_ly`` is added;
            when omitted it is left out rather than filled, so a caller without the reviews
            parquet gets a smaller block rather than a wrong one.
        k: Passed to :func:`rating_shrunk`.

    Returns:
        One row per listing, indexed as ``listings``, carrying ``id`` and the review block.

    Raises:
        KeyError: If a required column is missing.
    """
    require_columns(listings, ("id", *_LISTING_COLUMNS), "ranked listings")

    features = pd.DataFrame({"id": listings["id"]}, index=listings.index)
    features[list(PASSTHROUGH_COLUMNS)] = listings[list(PASSTHROUGH_COLUMNS)]
    features["has_reviews"] = has_reviews(listings)
    features["rating_shrunk"] = rating_shrunk(listings, k)
    features["listing_age_days"] = listing_age_days(listings)
    features["days_since_last_review"] = days_since_last_review(listings)

    if reviews is not None:
        anchors = listings.set_index("id")["T"]
        same_season = same_season_last_year(reviews, anchors)
        features["reviews_same_season_ly"] = features["id"].map(same_season).astype("int64")
    return features
