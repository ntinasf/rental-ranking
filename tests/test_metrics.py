"""Tests for rental_ranking.evaluate.metrics.

A ranking metric is the easiest thing in the project to get quietly wrong: every bug returns a
number in [0, 1] that looks like a score. The three that matter are pinned against hand
computation rather than against the implementation — a favourable tie-break, a linear gain where
LightGBM uses exponential, and degenerate groups scored 1.0 all inflate the headline without
failing anything.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.evaluate import metrics


def _frame(grades: list[int], scores: list[float], group: str = "g") -> dict:
    return {
        "grades": pd.Series(grades),
        "groups": pd.Series([group] * len(grades)),
        "scores": pd.Series(scores),
    }


# --- NDCG ------------------------------------------------------------------------------------


def test_a_perfect_ranking_scores_one() -> None:
    out = metrics.ndcg_at_k(**_frame([3, 2, 1], [3.0, 2.0, 1.0]))
    assert out.iloc[0] == pytest.approx(1.0)


def test_a_reversed_ranking_matches_the_hand_computation() -> None:
    """Gains 1, 3, 7 over discounts 1, log2(3), 2 against an ideal of 7, 3, 1."""
    out = metrics.ndcg_at_k(**_frame([1, 2, 3], [3.0, 2.0, 1.0]))
    expected = (1 + 3 / np.log2(3) + 7 / 2) / (7 + 3 / np.log2(3) + 1 / 2)
    assert out.iloc[0] == pytest.approx(expected)


def test_gain_is_exponential_by_default_to_match_lightgbm() -> None:
    exponential = metrics.ndcg_at_k(**_frame([4, 1, 0], [1.0, 2.0, 3.0]))
    linear = metrics.ndcg_at_k(**_frame([4, 1, 0], [1.0, 2.0, 3.0]), exponential_gain=False)

    assert exponential.iloc[0] != pytest.approx(linear.iloc[0])


def test_the_cut_off_is_respected() -> None:
    """A grade-4 listing ranked 11th must not count toward NDCG@10."""
    grades = [0] * 10 + [4]
    scores = list(range(11, 0, -1))
    assert metrics.ndcg_at_k(**_frame(grades, scores), k=10).iloc[0] == pytest.approx(0.0)


def test_ties_are_not_broken_favourably() -> None:
    """Every score identical: the metric must read row order, never sneak a look at the grade."""
    good_first = metrics.ndcg_at_k(**_frame([4, 0], [1.0, 1.0])).iloc[0]
    bad_first = metrics.ndcg_at_k(**_frame([0, 4], [1.0, 1.0])).iloc[0]

    assert good_first > bad_first  # order decides, and it is the row order


def test_a_group_with_one_grade_is_degenerate_not_perfect() -> None:
    """Any permutation scores 1.0 there; counting it as a win inflates the mean for free."""
    out = metrics.ndcg_at_k(**_frame([2, 2, 2], [3.0, 1.0, 2.0]))
    assert out.isna().all()

    included = metrics.ndcg_at_k(**_frame([2, 2, 2], [3.0, 1.0, 2.0]), include_degenerate=True)
    assert included.iloc[0] == pytest.approx(1.0)


def test_a_group_with_no_relevance_at_all_is_undefined() -> None:
    assert metrics.ndcg_at_k(**_frame([0, 0], [1.0, 2.0])).isna().all()


def test_each_group_is_scored_separately() -> None:
    grades = pd.Series([3, 0, 3, 0])
    groups = pd.Series(["a", "a", "b", "b"])
    scores = pd.Series([2.0, 1.0, 1.0, 2.0])
    out = metrics.ndcg_at_k(grades, groups, scores)

    assert out.loc["a"] == pytest.approx(1.0)
    assert out.loc["b"] < 1.0


# --- Recall ----------------------------------------------------------------------------------


def test_recall_counts_relevant_listings_in_the_top_k() -> None:
    grades = [4, 3, 0, 0]
    out = metrics.recall_at_k(**_frame(grades, [4.0, 1.0, 3.0, 2.0]), k=2)
    assert out.iloc[0] == pytest.approx(0.5)  # one of two relevant made the top 2


def test_recall_is_undefined_when_nothing_is_relevant() -> None:
    """Scoring it zero would punish the ranker for the grading, not for its ordering."""
    assert metrics.recall_at_k(**_frame([0, 1, 2], [1.0, 2.0, 3.0])).isna().all()


# --- bootstrap -------------------------------------------------------------------------------


def test_the_bootstrap_brackets_the_mean_and_is_deterministic() -> None:
    values = pd.Series(np.linspace(0.2, 0.9, 120))
    mean, low, high = metrics.bootstrap_ci(values, iterations=500, seed=3)

    assert low < mean < high
    assert metrics.bootstrap_ci(values, iterations=500, seed=3) == (mean, low, high)


def test_the_bootstrap_survives_an_all_degenerate_slice() -> None:
    assert all(np.isnan(v) for v in metrics.bootstrap_ci(pd.Series([np.nan, np.nan])))


# --- the report ------------------------------------------------------------------------------


def test_evaluate_ranking_reports_slices_and_counts_degenerates() -> None:
    grades = pd.Series([3, 0, 2, 2, 4, 1])
    groups = pd.Series(["a", "a", "b", "b", "c", "c"])
    scores = pd.Series([2.0, 1.0, 1.0, 2.0, 2.0, 1.0])
    cities = pd.Series(["athens", "athens", "crete", "crete", "crete", "crete"])

    out = metrics.evaluate_ranking(grades, groups, scores, breakdown=cities)

    assert list(out.index) == ["overall", "athens", "crete"]
    assert out.loc["overall", "degenerate"] == 1  # group b is single-grade
    assert out.loc["athens", "ndcg@10"] == pytest.approx(1.0)


def test_size_band_labels_each_row_by_its_groups_size() -> None:
    groups = pd.Series(["a"] * 3 + ["b"] * 12)
    bands = metrics.size_band(groups)

    assert bands.iloc[0] == "1-10"
    assert bands.iloc[-1] == "11-50"
