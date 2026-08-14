"""Review-history windows anchored at each listing's own T.

Review counts serve two roles that must not drift apart: the independent instrument notebook
02 validates the label against, and a Phase 2 feature. Both come from this module, so a
window used in validation is computed the same way as the one used in training.

**The anchor is passed in, never re-derived.** ``min(calendar.date)`` is T and belongs to
``features.label``; ``min(reviews.date)`` is ``first_review``, a median of 1,106 days earlier
(verified 2026-08-04, identical for 0.02 % of listings). A module that recomputed its own
anchor from the frame it happens to hold would silently answer a different question.

The primary instrument is the **same season, one year earlier** — reviews in
``[T - 365, T - 365 + 90)``. It is entirely pre-T, so leakage-free, and seasonally matched to
the July-September label window, unlike a trailing window from T which lands in shoulder
season. Measured against the label (Thessaloniki / Athens / Crete): rho 0.124 / 0.225 / 0.194,
against 0.083 / 0.193 / 0.195 for a trailing 40-day window.

At **equal 90-day width**, where season is the only difference, the same-season window gives
0.124 / 0.225 / 0.194 against the trailing window's 0.061 / 0.194 / 0.215 — it doubles
Thessaloniki, gains in Athens, and loses slightly in Crete. It does this while having *more*
empty windows in two of the three cities (44.8 / 48.7 % against 38.2 / 42.5 %), so the gain is
seasonal alignment itself, not extra data.
"""

import pandas as pd

from rental_ranking.data.validate import require_columns

#: Length of the counting window, matching the label window.
WINDOW_DAYS = 90

#: How far before T the window opens. 365 puts it in the same season one year earlier.
SEASON_LAG_DAYS = 365

_REQUIRED_COLUMNS = ("listing_id", "date")


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
    passed positionally. Extending the window to absorb review-posting lag was tested
    (104 / 110 / 120 days) and does not help: Thessaloniki degrades monotonically.
    """
    return reviews_in_window(
        reviews,
        anchors,
        window_days=window_days,
        starts_days_before=SEASON_LAG_DAYS,
        name="reviews_same_season_ly",
    )
