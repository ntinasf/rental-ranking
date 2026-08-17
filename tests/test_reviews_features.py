"""Tests for the Phase 2 feature half of rental_ranking.features.reviews.

The window functions are covered by ``test_reviews.py``; this file covers the derived features.

What matters here is that the shrinkage arithmetic is *exactly* what it claims. Every wrong
variant still returns a plausible number in [1, 5] — an include-self prior, a prior taken over
the wrong population, a zero-review listing quietly imputed to the global mean. So each is
pinned against a hand-computed value rather than against itself.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import reviews as rv

_DEFAULTS = {
    "id": "a",
    "city": "athens",
    "T": pd.Timestamp("2026-07-01"),
    "number_of_reviews": 10,
    "number_of_reviews_ltm": 4,
    "review_scores_rating": 5.0,
    "review_scores_accuracy": 4.9,
    "review_scores_cleanliness": 4.8,
    "review_scores_checkin": 5.0,
    "review_scores_communication": 5.0,
    "review_scores_location": 4.7,
    "review_scores_value": 4.6,
    "first_review": pd.Timestamp("2024-07-01"),
    "last_review": pd.Timestamp("2026-06-01"),
    "reviews_per_month": 1.0,
}


def _listings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_DEFAULTS, **row} for row in rows])


@pytest.fixture
def _no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --- has_reviews ---------------------------------------------------------------------------


def test_has_reviews_is_plain_and_never_conjoined_with_the_label() -> None:
    frame = _listings([{"number_of_reviews": 0}, {"number_of_reviews": 1}])
    assert rv.has_reviews(frame).tolist() == [False, True]


# --- rating_shrunk -------------------------------------------------------------------------


def test_the_prior_leaves_the_listing_itself_out(_no_warning) -> None:
    """Include-self would give 3.667 here; leave-one-out gives 2.333. Both look plausible."""
    frame = _listings(
        [
            {"review_scores_rating": 5.0, "number_of_reviews": 10},
            {"review_scores_rating": 1.0, "number_of_reviews": 10},
        ]
    )
    shrunk = rv.rating_shrunk(frame, k=20)

    # (10*5 + 20*1) / 30 — the prior for row 0 is row 1's rating alone.
    assert shrunk.iloc[0] == pytest.approx(70 / 30)
    assert shrunk.iloc[1] == pytest.approx(110 / 30)


def test_a_never_reviewed_listing_collapses_to_its_city_prior(_no_warning) -> None:
    """n = 0 returns the city mean exactly, which is why this module needs no imputation rule."""
    frame = _listings(
        [
            {"number_of_reviews": 0, "review_scores_rating": float("nan")},
            {"review_scores_rating": 4.0},
            {"review_scores_rating": 3.0},
        ]
    )
    shrunk = rv.rating_shrunk(frame)

    assert shrunk.iloc[0] == pytest.approx(3.5)
    assert shrunk.notna().all()


def test_the_prior_is_per_city_not_global(_no_warning) -> None:
    frame = _listings(
        [
            {"city": "athens", "number_of_reviews": 0, "review_scores_rating": float("nan")},
            {"city": "athens", "review_scores_rating": 4.0},
            {"city": "crete", "review_scores_rating": 2.0},
        ]
    )
    assert rv.rating_shrunk(frame).iloc[0] == pytest.approx(4.0)


def test_evidence_moves_a_listing_off_the_prior(_no_warning) -> None:
    """The whole point: a 1-review 5.0 must not outrank a 200-review 4.9."""
    frame = _listings(
        [
            {"review_scores_rating": 5.0, "number_of_reviews": 1},
            {"review_scores_rating": 4.9, "number_of_reviews": 200},
            {"review_scores_rating": 3.0, "number_of_reviews": 50},
        ]
    )
    shrunk = rv.rating_shrunk(frame)

    assert shrunk.iloc[1] > shrunk.iloc[0]


def test_a_larger_k_shrinks_harder(_no_warning) -> None:
    frame = _listings(
        [
            {"review_scores_rating": 5.0, "number_of_reviews": 10},
            {"review_scores_rating": 1.0, "number_of_reviews": 10},
        ]
    )
    prior = rv.rating_shrunk(frame, k=1000).iloc[0]
    loose = rv.rating_shrunk(frame, k=1).iloc[0]

    assert loose > prior
    assert prior == pytest.approx(1.0, abs=0.05)  # driven almost entirely by the other listing


def test_reviews_without_a_rating_warn_and_fall_back_to_the_prior() -> None:
    """Measured zero today, so the guard is here to keep the arithmetic honest if it changes."""
    frame = _listings(
        [
            {"number_of_reviews": 30, "review_scores_rating": float("nan")},
            {"review_scores_rating": 4.0},
        ]
    )
    with pytest.warns(UserWarning, match="no review_scores_rating"):
        shrunk = rv.rating_shrunk(frame)

    assert shrunk.iloc[0] == pytest.approx(4.0)


# --- ages ----------------------------------------------------------------------------------


def test_ages_are_measured_from_the_listings_own_T(_no_warning) -> None:
    frame = _listings([{"first_review": pd.Timestamp("2026-06-01")}])

    assert rv.listing_age_days(frame).iloc[0] == 30
    assert rv.days_since_last_review(frame).iloc[0] == 30


def test_a_never_reviewed_listing_keeps_a_null_age(_no_warning) -> None:
    """Age 0 for a listing never reviewed is a value that looks like data."""
    frame = _listings([{"first_review": pd.NaT, "last_review": pd.NaT}])

    assert rv.listing_age_days(frame).isna().all()
    assert rv.days_since_last_review(frame).isna().all()


def test_a_review_dated_after_T_is_reported() -> None:
    """A negative age means a review inside the label window — leakage, not a quirk."""
    frame = _listings([{"first_review": pd.Timestamp("2026-08-01")}])

    with pytest.warns(UserWarning, match="first review dated after"):
        rv.listing_age_days(frame)


# --- the assembled block --------------------------------------------------------------------


def test_the_block_carries_id_index_and_every_feature(_no_warning) -> None:
    frame = _listings([{"id": "a"}, {"id": "b"}]).set_axis([3, 5])
    block = rv.review_features(frame)

    assert block["id"].tolist() == ["a", "b"]
    assert block.index.tolist() == [3, 5]
    for column in ("has_reviews", "rating_shrunk", "listing_age_days"):
        assert column in block.columns


def test_the_same_season_window_is_omitted_rather_than_faked(_no_warning) -> None:
    """A caller without the reviews parquet gets a smaller block, never a wrong one."""
    assert "reviews_same_season_ly" not in rv.review_features(_listings([{}])).columns


def test_the_same_season_window_joins_on_listing_id(_no_warning) -> None:
    frame = _listings([{"id": "a"}, {"id": "b"}])
    reviews = pd.DataFrame(
        {
            "listing_id": ["a", "a", "b"],
            "date": pd.to_datetime(["2025-07-05", "2025-07-20", "2020-01-01"]),
        }
    )
    block = rv.review_features(frame, reviews)

    assert block["reviews_same_season_ly"].tolist() == [2, 0]


def test_a_missing_column_raises_a_readable_keyerror() -> None:
    with pytest.raises(KeyError, match="ranked listings"):
        rv.review_features(_listings([{}]).drop(columns=["first_review"]))


# --- against the real snapshots --------------------------------------------------------------


@pytest.fixture(scope="module")
def real_block() -> tuple[pd.DataFrame, pd.DataFrame]:
    for name in ("listings", "calendar", "reviews"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    from rental_ranking.data.filters import filter_listings
    from rental_ranking.features.label import occupancy_label

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    labels = occupancy_label(pd.read_parquet(PROCESSED_DIR / "calendar.parquet"))
    ranked, _ = filter_listings(listings.merge(labels, left_on="id", right_index=True, how="inner"))
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet", columns=["listing_id", "date"])
    return ranked, rv.review_features(ranked, reviews)


def test_real_rating_shrunk_is_complete_and_bounded(
    real_block: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    ranked, block = real_block

    assert block["rating_shrunk"].notna().all()
    assert block["rating_shrunk"].between(1.0, 5.0).all()
    # Every never-reviewed listing in a city shares one value: its city's prior.
    never = ranked["number_of_reviews"].eq(0)
    assert (
        block.loc[never].groupby(ranked.loc[never, "city"])["rating_shrunk"].nunique().eq(1).all()
    )


def test_real_ages_are_null_for_exactly_the_never_reviewed(
    real_block: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """One clean cohort, not a data-quality problem — 16.3 % of listings, zero exceptions."""
    ranked, block = real_block
    never = ranked["number_of_reviews"].eq(0)

    assert block["listing_age_days"].isna().equals(never)
    assert block["days_since_last_review"].isna().equals(never)


def test_real_review_history_is_strictly_pre_T(
    real_block: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The leakage guard: a review dated after T would sit inside the label window."""
    _, block = real_block

    assert block["listing_age_days"].dropna().ge(0).all()
    assert block["days_since_last_review"].dropna().ge(0).all()


def test_real_correlations_reproduce_the_recorded_figures(
    real_block: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Within the established cohort as notebook 02 defines it: first review at least a year
    before T. The recorded triples are Thessaloniki / Athens / Crete."""
    from scipy.stats import spearmanr

    ranked, block = real_block
    age = ranked["T"] - ranked["first_review"]
    established = ranked["first_review"].notna() & age.ge(pd.Timedelta(days=365))

    expected = {
        "rating_shrunk": {"thessaloniki": 0.267, "athens": 0.153, "crete": 0.239},
        "listing_age_days": {"thessaloniki": 0.060, "athens": 0.096, "crete": 0.062},
        "days_since_last_review": {"thessaloniki": -0.090, "athens": -0.102, "crete": -0.193},
    }
    for feature, per_city in expected.items():
        for city, recorded in per_city.items():
            mask = established & ranked["city"].eq(city)
            got = spearmanr(block.loc[mask, feature], ranked.loc[mask, "blocked_fraction_90"])
            assert got.statistic == pytest.approx(recorded, abs=0.01), f"{feature} / {city}"


def test_real_shrinkage_reverses_the_sign_of_the_evidence_relationship(
    real_block: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The reason the feature exists: raw rating is *negatively* related to its own evidence."""
    from scipy.stats import spearmanr

    ranked, block = real_block
    reviewed = ranked["number_of_reviews"].gt(0)
    n = ranked.loc[reviewed, "number_of_reviews"]

    raw = spearmanr(ranked.loc[reviewed, "review_scores_rating"], n).statistic
    shrunk = spearmanr(block.loc[reviewed, "rating_shrunk"], n).statistic

    assert raw < -0.2
    assert shrunk > 0.1
    assert np.sign(raw) != np.sign(shrunk)


def test_the_six_aspect_sub_scores_are_carried(_no_warning) -> None:
    """Airbnb's own aspect ratings: not one factor, all discriminating inside a group."""
    block = rv.review_features(_listings([{}]))

    for aspect in ("accuracy", "cleanliness", "checkin", "communication", "location", "value"):
        assert f"review_scores_{aspect}" in block.columns
