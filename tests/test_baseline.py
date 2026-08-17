"""Tests for rental_ranking.train.baseline — and the freeze itself.

The point of a frozen baseline is that it cannot be adjusted once the model's score is known.
A note in a log is a promise; the real-snapshot test at the bottom is the mechanism. If either
baseline is "improved" later, that test fails and the change has to be argued for rather than
made quietly.

The recorded figures are **full-population reference numbers**, not held-out estimates — the
grouped split does not exist until Phase 3, and a full-population baseline compared against a
test-set model would be a comparison of two different things. Phase 3 re-runs this same code on
the test split alongside the model; that comparison is the headline.
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
    """`price` is imputed in Phase 1 and `rating_shrunk` is never null, so a null is a wrong frame."""
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
    """**The freeze.** Recorded 2026-08-17, before any model existed. Full population.

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
    """Phase 1 predicted this: the label is substantially establishment-driven."""
    groups = real_ranked["query_group"]
    a = evaluate_ranking(real_ranked["grade"], groups, baseline.rank_by_reviews(real_ranked)).loc[
        "overall", "ndcg@10"
    ]
    b = evaluate_ranking(
        real_ranked["grade"], groups, baseline.rank_by_price_and_rating(real_ranked, groups)
    ).loc["overall", "ndcg@10"]

    assert a > b
