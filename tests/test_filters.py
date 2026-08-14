"""Tests for rental_ranking.data.filters.

Both rules are conjunctions or comparisons that fail *quietly* when written wrong — a
precedence slip drops half the condition and still returns a plausible frame. So every rule
is tested in both directions: it removes what it should, and it keeps what it must not touch.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data import filters
from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features.label import occupancy_label

_DEFAULTS = {
    "city": "thessaloniki",
    "number_of_reviews": 5,
    "number_of_reviews_ly": 2,
    "blocked_fraction_90": 0.5,
    "minimum_nights": 2,
}


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Listings joined with the label; each row states only what it varies."""
    return pd.DataFrame([{**_DEFAULTS, **row} for row in rows])


# --- the inactive rule, both directions ---------------------------------------------------


def test_zero_reviews_and_fully_blocked_is_removed() -> None:
    kept, counts = filters.filter_listings(
        _frame([{"number_of_reviews": 0, "blocked_fraction_90": 1.0}])
    )
    assert len(kept) == 0
    assert counts.loc["thessaloniki", "inactive"] == 1


def test_zero_reviews_but_not_fully_blocked_is_kept() -> None:
    """Zero reviews alone is cold start, which the project keeps and later studies.

    This is the direction a precedence slip breaks: `a == 0 & (b == 1.0)` parses as
    `a == (0 & (b == 1.0))`, which is just `a == 0` — the blocked half vanishes silently.
    """
    kept, counts = filters.filter_listings(
        _frame([{"number_of_reviews": 0, "blocked_fraction_90": 0.4}])
    )
    assert len(kept) == 1
    assert counts.loc["thessaloniki", "inactive"] == 0


def test_fully_blocked_but_reviewed_is_kept() -> None:
    """The other half of the conjunction: a booked-out listing with a history stays."""
    kept, _ = filters.filter_listings(
        _frame([{"number_of_reviews": 40, "blocked_fraction_90": 1.0}])
    )
    assert len(kept) == 1


def test_lifetime_reviews_with_none_last_year_is_kept() -> None:
    """The rule reads `number_of_reviews`, never `number_of_reviews_ly`.

    `_ly` is the calendar-2025 count, so a listing reviewed steadily until 2024 reads as
    never-reviewed under it — 1,123 real listings sit in exactly that state.
    """
    kept, counts = filters.filter_listings(
        _frame(
            [
                {
                    "number_of_reviews": 30,
                    "number_of_reviews_ly": 0,
                    "blocked_fraction_90": 1.0,
                }
            ]
        )
    )
    assert len(kept) == 1
    assert counts.loc["thessaloniki", "inactive"] == 0


# --- the long-term rule, both directions --------------------------------------------------


def test_minimum_nights_above_threshold_is_removed() -> None:
    kept, counts = filters.filter_listings(_frame([{"minimum_nights": 31}]))
    assert len(kept) == 0
    assert counts.loc["thessaloniki", "long_term"] == 1


def test_minimum_nights_at_the_threshold_is_kept() -> None:
    """The comparison is strict: 30 nights is still in scope."""
    kept, _ = filters.filter_listings(_frame([{"minimum_nights": 30}]))
    assert len(kept) == 1


def test_threshold_is_configurable() -> None:
    rows = _frame([{"minimum_nights": 14}])
    assert len(filters.filter_listings(rows)[0]) == 1
    assert len(filters.filter_listings(rows, threshold=7)[0]) == 0


def test_plausible_short_stay_listing_passes_untouched() -> None:
    original = _frame([{}])
    kept, counts = filters.filter_listings(original)
    pd.testing.assert_frame_equal(kept, original)
    assert counts.loc["thessaloniki", "removed"] == 0


def test_listing_first_reviewed_shortly_before_the_anchor_is_kept() -> None:
    """Thin history is the cold-start cohort Phase 2 flags — a decision, not an oversight."""
    rows = _frame([{"number_of_reviews": 1, "blocked_fraction_90": 0.2}])
    rows["first_review"] = pd.Timestamp("2026-06-20")
    rows["T"] = pd.Timestamp("2026-06-29")

    kept, _ = filters.filter_listings(rows)
    assert len(kept) == 1


# --- counts -------------------------------------------------------------------------------


def test_counts_are_reported_per_city_and_per_rule() -> None:
    kept, counts = filters.filter_listings(
        _frame(
            [
                {"city": "athens", "number_of_reviews": 0, "blocked_fraction_90": 1.0},
                {"city": "athens", "minimum_nights": 90},
                {"city": "crete", "minimum_nights": 90},
                {"city": "crete"},
            ]
        )
    )
    assert counts.loc["athens", "inactive"] == 1
    assert counts.loc["athens", "long_term"] == 1
    assert counts.loc["crete", "inactive"] == 0
    assert counts.loc["crete", "long_term"] == 1
    assert len(kept) == 1


def test_overlap_is_reported_not_absorbed() -> None:
    """A row both rules catch is counted once in `removed` and named in `both_rules`."""
    _, counts = filters.filter_listings(
        _frame([{"number_of_reviews": 0, "blocked_fraction_90": 1.0, "minimum_nights": 90}])
    )
    assert counts.loc["thessaloniki", "inactive"] == 1
    assert counts.loc["thessaloniki", "long_term"] == 1
    assert counts.loc["thessaloniki", "both_rules"] == 1
    assert counts.loc["thessaloniki", "removed"] == 1


def test_counts_reconcile_with_the_returned_frame() -> None:
    rows = _frame(
        [
            {"number_of_reviews": 0, "blocked_fraction_90": 1.0},
            {"minimum_nights": 90},
            {"city": "athens"},
            {"city": "athens", "minimum_nights": 90},
        ]
    )
    kept, counts = filters.filter_listings(rows)

    assert counts["n"].sum() == len(rows)
    assert counts["kept"].sum() == len(kept)
    assert (counts["kept"] + counts["removed"] == counts["n"]).all()
    assert (
        counts["removed"] == counts["inactive"] + counts["long_term"] - counts["both_rules"]
    ).all()


def test_rules_are_independent_predicates_not_a_chain() -> None:
    """Chained rules would make the second count depend on the first and hide the overlap."""
    _, counts = filters.filter_listings(
        _frame(
            [
                {"number_of_reviews": 0, "blocked_fraction_90": 1.0, "minimum_nights": 90},
                {"minimum_nights": 90},
            ]
        )
    )
    # 2 long-term rows, even though one of them was already caught by the inactive rule.
    assert counts.loc["thessaloniki", "long_term"] == 2
    assert counts.loc["thessaloniki", "inactive"] == 1


# --- shape and contract -------------------------------------------------------------------


def test_unhashable_columns_survive_the_filter() -> None:
    """`amenities` round-trips from parquet as an ndarray; anything that dedups on all
    columns raises `unhashable type: 'numpy.ndarray'` on real data."""
    rows = _frame([{"minimum_nights": 90}, {}])
    rows["amenities"] = [np.array(["Wifi", "Kitchen"]), np.array(["Wifi"])]

    kept, _ = filters.filter_listings(rows)
    assert len(kept) == 1
    assert list(kept["amenities"].iloc[0]) == ["Wifi"]


def test_result_is_a_copy_not_a_view() -> None:
    """Callers add imputed price, tier and grade columns to the survivors."""
    original = _frame([{"minimum_nights": 2}])
    kept, _ = filters.filter_listings(original)
    kept.loc[kept.index[0], "minimum_nights"] = 999

    assert original.loc[0, "minimum_nights"] == 2


def test_predicates_return_row_aligned_boolean_series() -> None:
    rows = _frame([{"number_of_reviews": 0, "blocked_fraction_90": 1.0}, {}])
    for mask in (filters.is_inactive(rows), filters.is_long_term(rows)):
        assert mask.dtype == bool
        assert mask.index.equals(rows.index)


def test_missing_column_raises_a_readable_keyerror() -> None:
    rows = _frame([{}]).drop(columns=["blocked_fraction_90"])
    with pytest.raises(KeyError, match="listings joined with the label"):
        filters.filter_listings(rows)


# --- against the real snapshots -----------------------------------------------------------


def test_real_filter_counts_match_the_contract() -> None:
    """The counts docs/data_pipeline_design.md publishes; a drift here invalidates the doc."""
    for name in ("listings", "calendar"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    labels = occupancy_label(pd.read_parquet(PROCESSED_DIR / "calendar.parquet"))
    joined = listings.merge(labels, left_on="id", right_index=True, how="inner")

    kept, counts = filters.filter_listings(joined)

    expected = {
        "thessaloniki": {"inactive": 75, "long_term": 25, "removed": 99},
        "athens": {"inactive": 101, "long_term": 1, "removed": 102},
        "crete": {"inactive": 523, "long_term": 43, "removed": 565},
    }
    for city, values in expected.items():
        for column, want in values.items():
            assert counts.loc[city, column] == want, f"{city}.{column}"

    assert counts["both_rules"].sum() == 2
    assert len(kept) == 45_869
