"""Tests for rental_ranking.features.reviews.

The window is defined by two things that can each be wrong without raising: the anchor it is
measured from, and which end of the interval is closed. Both are pinned here, and the anchor
test is a regression guard — an earlier draft re-derived it from ``min(reviews.date)``, which
is ``first_review``, a median of 1,106 days earlier than T.
"""

import pandas as pd
import pytest

from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import reviews as rv
from rental_ranking.features.label import occupancy_label


def _reviews(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": [listing_id for listing_id, _ in rows],
            "date": pd.to_datetime([date for _, date in rows]),
        }
    )


def _anchors(mapping: dict[str, str]) -> pd.Series:
    return pd.Series(
        {key: pd.Timestamp(value) for key, value in mapping.items()},
        name="T",
        dtype="datetime64[ns]",
    ).rename_axis("listing_id")


# --- the anchor ---------------------------------------------------------------------------


def test_anchor_comes_from_the_caller_never_from_the_reviews_frame() -> None:
    """T is ``min(calendar.date)`` and belongs to ``features.label``.

    ``min(reviews.date)`` is ``first_review`` — here 2020, six years earlier. If the module
    re-derived its own anchor the window would sit in 2019 and count nothing.
    """
    counts = rv.reviews_in_window(
        _reviews([("a", "2020-01-15"), ("a", "2025-07-10")]),
        _anchors({"a": "2026-06-29"}),
    )
    assert counts["a"] == 1


def test_each_listing_uses_its_own_anchor() -> None:
    """Scrape dates spread over four days inside one city; a shared anchor would shift them."""
    counts = rv.reviews_in_window(
        _reviews([("early", "2025-07-10"), ("late", "2025-07-10")]),
        _anchors({"early": "2026-06-29", "late": "2026-11-01"}),
    )
    assert counts["early"] == 1  # window [2025-06-29, 2025-09-27) contains it
    assert counts["late"] == 0  # window [2025-11-01, 2026-01-30) does not


# --- the interval -------------------------------------------------------------------------


def test_the_window_is_half_open() -> None:
    """[opens, closes): the opening day counts, the closing day belongs to the next window."""
    counts = rv.reviews_in_window(
        _reviews(
            [
                ("a", "2025-06-28"),  # before it opens
                ("a", "2025-06-29"),  # opens exactly here
                ("a", "2025-09-26"),  # last day inside
                ("a", "2025-09-27"),  # closes exactly here
            ]
        ),
        _anchors({"a": "2026-06-29"}),
    )
    assert counts["a"] == 2


def test_trailing_window_ends_at_the_anchor() -> None:
    """``starts_days_before == window_days`` gives a trailing window closing at T."""
    counts = rv.reviews_in_window(
        _reviews([("a", "2026-06-28"), ("a", "2026-05-31"), ("a", "2026-05-29")]),
        _anchors({"a": "2026-06-29"}),
        window_days=30,
        starts_days_before=30,
    )
    assert counts["a"] == 2


def test_reviews_outside_the_window_on_either_side_are_excluded() -> None:
    counts = rv.reviews_in_window(
        _reviews([("a", "2024-07-10"), ("a", "2025-07-10"), ("a", "2026-06-01")]),
        _anchors({"a": "2026-06-29"}),
    )
    assert counts["a"] == 1


# --- coverage and alignment ---------------------------------------------------------------


def test_listing_with_an_empty_window_counts_zero_not_missing() -> None:
    """44-49 % of real listings have an empty same-season window; NaN would be read as
    missingness by LightGBM and dropped by scipy, and would understate every mean."""
    counts = rv.reviews_in_window(
        _reviews([("a", "2025-07-10")]),
        _anchors({"a": "2026-06-29", "b": "2026-06-29"}),
    )
    assert counts["b"] == 0
    assert counts.notna().all()


def test_listing_with_no_reviews_at_all_still_appears() -> None:
    counts = rv.reviews_in_window(_reviews([]), _anchors({"a": "2026-06-29"}))
    assert counts.to_dict() == {"a": 0}


def test_reviews_for_a_listing_without_an_anchor_are_ignored() -> None:
    """Athens ships five calendar orphans; a review pointing at one must not invent a row."""
    counts = rv.reviews_in_window(
        _reviews([("a", "2025-07-10"), ("orphan", "2025-07-10")]),
        _anchors({"a": "2026-06-29"}),
    )
    assert list(counts.index) == ["a"]
    assert counts["a"] == 1


def test_result_is_aligned_to_the_anchor_index_and_integer_typed() -> None:
    anchors = _anchors({"b": "2026-06-29", "a": "2026-06-29"})
    counts = rv.reviews_in_window(_reviews([("a", "2025-07-10")]), anchors)

    assert counts.index.equals(anchors.index)
    assert counts.dtype == "int64"


# --- naming and the contract --------------------------------------------------------------


def test_default_name_describes_the_window() -> None:
    counts = rv.reviews_in_window(
        _reviews([("a", "2025-07-10")]), _anchors({"a": "2026-06-29"}), 60, 120
    )
    assert counts.name == "reviews_w60_lag120"


def test_name_can_be_overridden() -> None:
    counts = rv.reviews_in_window(
        _reviews([("a", "2025-07-10")]), _anchors({"a": "2026-06-29"}), name="custom"
    )
    assert counts.name == "custom"


def test_same_season_last_year_is_the_365_day_lag() -> None:
    reviews = _reviews([("a", "2025-07-10")])
    anchors = _anchors({"a": "2026-06-29"})

    wrapper = rv.same_season_last_year(reviews, anchors)
    explicit = rv.reviews_in_window(reviews, anchors, window_days=90, starts_days_before=365)

    assert wrapper.name == "reviews_same_season_ly"
    assert wrapper.tolist() == explicit.tolist()


def test_missing_column_raises_a_readable_keyerror() -> None:
    frame = _reviews([("a", "2025-07-10")]).drop(columns=["date"])
    with pytest.raises(KeyError, match="reviews"):
        rv.reviews_in_window(frame, _anchors({"a": "2026-06-29"}))


# --- against the real snapshots -----------------------------------------------------------


@pytest.fixture(scope="module")
def real_frames() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    for name in ("listings", "calendar", "reviews"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    anchors = occupancy_label(pd.read_parquet(PROCESSED_DIR / "calendar.parquet"))["T"]
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet", columns=["listing_id", "date"])
    return listings, anchors, reviews


def test_real_window_covers_every_anchor_with_no_nulls(real_frames) -> None:
    _, anchors, reviews = real_frames
    counts = rv.same_season_last_year(reviews, anchors)

    assert counts.index.equals(anchors.index)
    assert counts.notna().all()
    assert (counts >= 0).all()


def test_real_trailing_window_reproduces_the_shipped_review_count(real_frames) -> None:
    """The cross-check that licenses every derived window: Inside Airbnb ships a trailing
    30-day count, and ours must reproduce it without ever having seen it."""
    listings, anchors, reviews = real_frames
    derived = rv.reviews_in_window(reviews, anchors, window_days=30, starts_days_before=30)

    merged = listings.assign(derived=listings["id"].map(derived))
    agreement = merged.groupby("city").apply(
        lambda part: part["derived"].eq(part["number_of_reviews_l30d"]).mean(),
        include_groups=False,
    )
    assert (agreement > 0.99).all(), agreement.to_dict()


def test_positionally_indexed_anchors_raise_instead_of_counting_zero() -> None:
    """`frame["T"]` after a merge carries a RangeIndex, so `.map` resolves nothing and every
    count silently comes back 0 — a wrong answer shaped exactly like a real one."""
    positional = pd.Series(
        [pd.Timestamp("2026-06-29")], index=[0], name="T", dtype="datetime64[ns]"
    )
    with pytest.raises(ValueError, match="indexed by listing_id"):
        rv.reviews_in_window(_reviews([("a", "2025-07-10")]), positional)


def test_duplicated_anchor_index_raises() -> None:
    duplicated = pd.Series(
        [pd.Timestamp("2026-06-29")] * 2, index=["a", "a"], name="T", dtype="datetime64[ns]"
    )
    with pytest.raises(ValueError, match="uniquely indexed"):
        rv.reviews_in_window(_reviews([("a", "2025-07-10")]), duplicated)


def test_an_anchor_set_with_no_matching_reviews_is_still_allowed() -> None:
    """Legitimately empty: a listing subset none of whose members were reviewed in-window.
    The guard must fire on an unusable *index*, not on a genuinely empty intersection."""
    counts = rv.reviews_in_window(_reviews([("a", "2019-01-01")]), _anchors({"a": "2026-06-29"}))
    assert counts.to_dict() == {"a": 0}
