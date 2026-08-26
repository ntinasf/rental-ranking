"""Tests for rental_ranking.train.baseline — and the freeze itself.

The point of a frozen baseline is that it cannot be adjusted once the model's score is known.
A note in a log is a promise; the real-snapshot test at the bottom is the mechanism. If either
baseline is "improved" later, that test fails and the change has to be argued for rather than
made quietly.

The recorded figures are **full-population reference numbers**, not held-out estimates: the
grouped split does not exist yet when they are taken, and a full-population baseline compared
against a test-set model would be a comparison of two different things. Training re-runs this
same code on the test split alongside the model, and that comparison is the headline.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.evaluate.metrics import evaluate_ranking
from rental_ranking.train import baseline

_DEFAULTS = {"number_of_reviews": 10, "price": 100.0, "rating_shrunk": 4.5}


def _listings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_DEFAULTS, **row} for row in rows])


# --- baseline A ------------------------------------------------------------------------------


def test_baseline_a_ranks_by_review_count() -> None:
    frame = _listings([{"number_of_reviews": 3}, {"number_of_reviews": 40}])
    assert baseline.rank_by_reviews(frame).tolist() == [3.0, 40.0]


def test_baseline_a_needs_only_its_own_column() -> None:
    with pytest.raises(KeyError, match="number_of_reviews"):
        baseline.rank_by_reviews(pd.DataFrame({"price": [1.0]}))


# --- baseline B ------------------------------------------------------------------------------


def test_baseline_b_prefers_a_better_rating_at_the_same_price() -> None:
    frame = _listings([{"rating_shrunk": 4.9}, {"rating_shrunk": 4.1}])
    score = baseline.rank_by_price_and_rating(frame, pd.Series(["g", "g"]))

    assert score.iloc[0] > score.iloc[1]


def test_baseline_b_prefers_a_lower_price_at_the_same_rating() -> None:
    frame = _listings([{"price": 60.0}, {"price": 300.0}])
    score = baseline.rank_by_price_and_rating(frame, pd.Series(["g", "g"]))

    assert score.iloc[0] > score.iloc[1]


def test_baseline_b_percentiles_are_taken_inside_the_group() -> None:
    """A cheap listing in an expensive group must not be judged against the other group."""
    frame = _listings(
        [
            {"price": 100.0},
            {"price": 200.0},
            {"price": 1_000.0},
            {"price": 2_000.0},
        ]
    )
    groups = pd.Series(["a", "a", "b", "b"])
    score = baseline.rank_by_price_and_rating(frame, groups)

    # The cheaper member of each group outranks the dearer one, at equal ratings.
    assert score.iloc[0] > score.iloc[1]
    assert score.iloc[2] > score.iloc[3]
    assert score.iloc[0] == pytest.approx(score.iloc[2])


def test_baseline_b_is_robust_to_a_price_outlier() -> None:
    """Percentiles rather than z-scores: price has a skew of 6.5 in this data."""
    frame = _listings([{"price": p} for p in (80.0, 90.0, 100.0, 9_000.0)])
    groups = pd.Series(["g"] * 4)
    score = baseline.rank_by_price_and_rating(frame, groups)

    assert score.is_monotonic_decreasing


def test_baseline_b_refuses_an_incomplete_frame() -> None:
    """`price` is imputed upstream and `rating_shrunk` is never null, so a null is a wrong frame."""
    frame = _listings([{"price": float("nan")}, {}])

    with pytest.raises(ValueError, match="complete inputs"):
        baseline.rank_by_price_and_rating(frame, pd.Series(["g", "g"]))


def test_the_weight_is_a_frozen_constant_not_a_knob() -> None:
    """Tuning it against NDCG would make the baseline a model, selected on the evaluation set."""
    assert baseline.PRICE_RATING_WEIGHT == 0.5


# --- the freeze, against the real snapshots ----------------------------------------------------


@pytest.fixture(scope="module")
def real_ranked() -> pd.DataFrame:
    for name in ("listings", "calendar"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    from rental_ranking.data.filters import filter_listings
    from rental_ranking.features.groups import query_group
    from rental_ranking.features.label import assign_grades, occupancy_label
    from rental_ranking.features.price import impute_price
    from rental_ranking.features.reviews import rating_shrunk

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    labels = occupancy_label(pd.read_parquet(PROCESSED_DIR / "calendar.parquet"))
    kept, _ = filter_listings(listings.merge(labels, left_on="id", right_index=True, how="inner"))
    ranked, _ = impute_price(kept)
    ranked["grade"], _ = assign_grades(ranked)
    ranked["rating_shrunk"] = rating_shrunk(ranked)
    ranked["query_group"], _ = query_group(ranked)
    return ranked


def test_the_frozen_baseline_numbers_have_not_moved(real_ranked: pd.DataFrame) -> None:
    """**The freeze.** Recorded before any model existed, over the full population.

    If this fails, either a baseline definition changed or the snapshots rotated — both of which
    must be a decision, not a surprise discovered while comparing against a model.
    """
    groups = real_ranked["query_group"]
    scores = {
        "reviews": baseline.rank_by_reviews(real_ranked),
        "price_rating": baseline.rank_by_price_and_rating(real_ranked, groups),
    }
    expected = {"reviews": 0.6424, "price_rating": 0.6218}

    for name, score in scores.items():
        got = evaluate_ranking(real_ranked["grade"], groups, score).loc["overall", "ndcg@10"]
        assert got == pytest.approx(expected[name], abs=0.005), name


def test_both_baselines_beat_a_random_ranking(real_ranked: pd.DataFrame) -> None:
    """NDCG has a high floor — a random order already scores ~0.540, so 0.64 is a modest lead.

    The floor is averaged over twenty draws rather than pinned from one: a single random
    ranking varies by sd 0.006 across seeds, so quoting it to four decimals would be false
    precision on the very number every model headline is read against. The usable range is
    ~0.540 to 1.0, and baseline A traverses under a quarter of it.
    """
    groups = real_ranked["query_group"]
    floors = [
        evaluate_ranking(
            real_ranked["grade"],
            groups,
            pd.Series(
                np.random.default_rng(seed).random(len(real_ranked)), index=real_ranked.index
            ),
        ).loc["overall", "ndcg@10"]
        for seed in range(20)
    ]
    random_ndcg = float(np.mean(floors))
    reviews_ndcg = evaluate_ranking(
        real_ranked["grade"], groups, baseline.rank_by_reviews(real_ranked)
    ).loc["overall", "ndcg@10"]

    assert random_ndcg == pytest.approx(0.540, abs=0.01)
    assert np.std(floors) < 0.02  # one draw is not a stable floor; twenty are
    assert reviews_ndcg > random_ndcg + 0.05


def test_the_establishment_baseline_leads_the_value_heuristic(real_ranked: pd.DataFrame) -> None:
    """As the label analysis predicted: the label is substantially establishment-driven."""
    groups = real_ranked["query_group"]
    a = evaluate_ranking(real_ranked["grade"], groups, baseline.rank_by_reviews(real_ranked)).loc[
        "overall", "ndcg@10"
    ]
    b = evaluate_ranking(
        real_ranked["grade"], groups, baseline.rank_by_price_and_rating(real_ranked, groups)
    ).loc["overall", "ndcg@10"]

    assert a > b


# --- the split comparators, frozen before any model exists --------------------------------------


@pytest.fixture(scope="module")
def real_features() -> pd.DataFrame:
    """The shipped feature table — what training actually fits and evaluates on.

    Read from disk rather than rebuilt from the processed layer, because ``query_group`` and
    ``cluster_id`` are positional ids: a rebuild in a different row order would produce a
    different, equally valid fold assignment, and the frozen numbers below would not be
    comparable to the ones training uses.
    """
    from rental_ranking.data.paths import FEATURE_TABLE_PATH

    if not FEATURE_TABLE_PATH.exists():
        pytest.skip("feature table not built")
    return pd.read_parquet(FEATURE_TABLE_PATH)


def _split_table(features: pd.DataFrame, sealed: bool) -> pd.DataFrame:
    from rental_ranking.evaluate.report import comparison_table
    from rental_ranking.train.split import assign_folds, sealed_mask

    fold, _ = assign_folds(features)
    mask = sealed_mask(fold) if sealed else ~sealed_mask(fold)
    held = features[mask]
    return comparison_table(
        held["grade"],
        held["query_group"],
        {
            "reviews": baseline.rank_by_reviews(held),
            "price_rating": baseline.rank_by_price_and_rating(held, held["query_group"]),
        },
        reference="reviews",
    )


@pytest.mark.parametrize(
    ("sealed", "slice_name", "expected"),
    [
        (True, "overall", {"reviews": 0.6390, "price_rating": 0.6429, "floor": 0.5519}),
        (True, "n>10", {"reviews": 0.5951, "price_rating": 0.5859, "floor": 0.4808}),
        (False, "overall", {"reviews": 0.6432, "price_rating": 0.6169, "floor": 0.5402}),
        (False, "n>10", {"reviews": 0.5903, "price_rating": 0.5590, "floor": 0.4620}),
    ],
)
def test_the_frozen_split_comparators_have_not_moved(
    real_features: pd.DataFrame, sealed: bool, slice_name: str, expected: dict[str, float]
) -> None:
    """**The split freeze.** Recorded before any model existed.

    A model scored on the sealed fold has to be compared against baselines scored on *the same
    groups*: the baselines move enough between folds that the full-population figures are the
    wrong comparator for a sealed-fold result. Freezing them
    here, before training, is what stops the comparator being computed after the model's number
    is known and framed around it.
    """
    table = _split_table(real_features, sealed)
    for ranker in ("reviews", "price_rating"):
        got = table.loc[(slice_name, ranker), "ndcg@10"]
        assert got == pytest.approx(expected[ranker], abs=0.005), ranker
    assert table.loc[(slice_name, "reviews"), "floor"] == pytest.approx(
        expected["floor"], abs=0.005
    )


def test_the_baseline_ordering_flips_on_the_sealed_fold(real_features: pd.DataFrame) -> None:
    """Recorded so it cannot be a surprise found mid-comparison.

    On the full population the establishment baseline leads the value heuristic by 0.0207. On
    the sealed fold it does not: price+rating scores 0.6429 against reviews at 0.6390. Nothing
    is wrong — 72 groups is a small sample and the paired interval spans zero — but a model
    report that assumes "A is the baseline to beat" would name the wrong comparator, and the
    honest headline compares against **both**.
    """
    sealed = _split_table(real_features, sealed=True)
    dev = _split_table(real_features, sealed=False)

    assert sealed.loc[("overall", "price_rating"), "vs_reviews"] > 0
    assert dev.loc[("overall", "price_rating"), "vs_reviews"] < 0
    # The flip is noise, not a finding: the sealed interval covers zero, the dev one does not.
    assert (
        sealed.loc[("overall", "price_rating"), "vs_low"]
        < 0
        < sealed.loc[("overall", "price_rating"), "vs_high"]
    )
    assert dev.loc[("overall", "price_rating"), "vs_high"] < 0


def test_the_cutoff_slice_is_where_the_baselines_can_be_told_apart(
    real_features: pd.DataFrame,
) -> None:
    """In groups of ten or fewer the metric cannot discriminate: the two frozen baselines tie
    to four decimals on the full population (0.8069 / 0.8070) against a random floor of 0.7891,
    while over the groups the cut-off actually cuts the floor is 0.4655. Reporting one number
    across both averages a quarter of the metric's weight in from where nothing can be shown.
    """
    dev = _split_table(real_features, sealed=False)
    assert dev.loc[("n<=10", "reviews"), "floor"] > 0.75
    assert dev.loc[("n>10", "reviews"), "floor"] < 0.50
    # The lead over the floor is roughly ten times larger where the cut-off cuts.
    small = dev.loc[("n<=10", "reviews"), "ndcg@10"] - dev.loc[("n<=10", "reviews"), "floor"]
    large = dev.loc[("n>10", "reviews"), "ndcg@10"] - dev.loc[("n>10", "reviews"), "floor"]
    assert large > 5 * small
