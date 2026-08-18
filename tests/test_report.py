"""Tests for rental_ranking.evaluate.report.

The module exists to stop three silent errors, so those are what is asserted: that every ranker
in a table is scored on **identical** groups, that the floor is recomputed **per slice** rather
than carried over as a constant, and that the ``n>k`` slice is produced whether or not anyone
asked for it. Each failure produces a table that looks entirely reasonable and misstates the
result by a fifth or more.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.evaluate import report


def _graded(sizes: list[int], seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Grades and group ids for groups of the given sizes, grades cycling 0..4."""
    rng = np.random.default_rng(seed)
    grades, groups = [], []
    for g, size in enumerate(sizes):
        grades.extend(rng.integers(0, 5, size).tolist())
        groups.extend([g] * size)
    return pd.Series(grades, name="grade"), pd.Series(groups, name="query_group")


# --- the floor ---------------------------------------------------------------------------------


def test_random_floor_is_per_group_and_seed_stable() -> None:
    grades, groups = _graded([20, 30, 40])
    first = report.random_floor(grades, groups, draws=5, seed=3)
    again = report.random_floor(grades, groups, draws=5, seed=3)

    assert first.index.tolist() == [0, 1, 2]
    assert first.equals(again)
    assert not first.equals(report.random_floor(grades, groups, draws=5, seed=4))


def test_random_floor_marks_degenerate_groups_nan() -> None:
    grades = pd.Series([2, 2, 2, 0, 1, 3])
    groups = pd.Series([0, 0, 0, 1, 1, 1])
    floor = report.random_floor(grades, groups, draws=3)

    assert np.isnan(floor.loc[0])
    assert np.isfinite(floor.loc[1])


def test_the_floor_is_much_higher_in_groups_the_cutoff_cannot_cut() -> None:
    """The measurement the module is built around: at n <= k a random order already scores high,
    because the whole group is inside the top k and NDCG@k is scoring the full list."""
    grades, groups = _graded([6] * 30 + [200] * 30, seed=1)
    floor = report.random_floor(grades, groups, k=10, draws=5)
    sizes = groups.value_counts()

    small = floor.loc[floor.index.intersection(sizes.index[sizes <= 10])].mean()
    large = floor.loc[floor.index.intersection(sizes.index[sizes > 10])].mean()
    assert small > large + 0.2


# --- pairing -----------------------------------------------------------------------------------


def test_paired_difference_is_tighter_than_the_unpaired_interval() -> None:
    rng = np.random.default_rng(0)
    shared = rng.normal(0.6, 0.18, 80)
    a = pd.Series(shared + rng.normal(0, 0.02, 80))
    b = pd.Series(shared + rng.normal(0, 0.02, 80) - 0.03)

    _, low, high = report.paired_difference(a, b)
    unpaired = 1.96 * np.sqrt(a.var() / 80 + b.var() / 80) * 2
    assert (high - low) < unpaired / 2


def test_paired_difference_on_a_constant_offset_has_a_zero_width_interval() -> None:
    a = pd.Series([0.4, 0.6, 0.8, 0.5])
    _, low, high = report.paired_difference(a, a - 0.1)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.1)


def test_paired_difference_drops_groups_either_side_is_missing() -> None:
    a = pd.Series([0.5, np.nan, 0.7])
    b = pd.Series([0.4, 0.3, np.nan])
    mean, _, _ = report.paired_difference(a, b)
    assert mean == pytest.approx(0.1)


def test_paired_difference_of_nothing_is_nan() -> None:
    empty = pd.Series(dtype="float64")
    assert all(np.isnan(v) for v in report.paired_difference(empty, empty))


# --- slices ------------------------------------------------------------------------------------


def test_cutoff_slices_are_always_produced_and_partition_the_whole() -> None:
    _, groups = _graded([5, 8, 20, 300])
    slices = report.metric_slices(groups, k=10)

    assert list(slices)[:3] == ["overall", "n>10", "n<=10"]
    assert sorted(slices["n>10"].tolist() + slices["n<=10"].tolist()) == sorted(
        slices["overall"].tolist()
    )
    assert set(slices["n<=10"]) == {0, 1}


def test_the_cutoff_follows_k() -> None:
    _, groups = _graded([5, 8, 20, 300])
    assert set(report.metric_slices(groups, k=25)["n<=25"]) == {0, 1, 2}


def test_breakdown_slices_are_appended() -> None:
    _, groups = _graded([10, 10])
    breakdown = pd.Series(["athens"] * 10 + ["crete"] * 10)
    slices = report.metric_slices(groups, breakdown=breakdown)

    assert set(slices["athens"]) == {0}
    assert set(slices["crete"]) == {1}


# --- the table ---------------------------------------------------------------------------------


def _table(**kwargs) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    grades, groups = _graded([4, 7, 15, 40, 120], seed=2)
    rng = np.random.default_rng(0)
    scores = {
        "model": pd.Series(grades + rng.normal(0, 1, len(grades))),
        "baseline": pd.Series(rng.normal(0, 1, len(grades))),
    }
    return (
        report.comparison_table(grades, groups, scores, reference="baseline", **kwargs),
        grades,
        groups,
    )


def test_every_ranker_is_scored_on_identical_groups() -> None:
    """The invariant the whole module exists for. If two rankers are sliced differently, the
    comparison is between populations rather than between rankers, and nothing says so."""
    table, _, _ = _table()
    counted = table[["groups", "degenerate"]].sum(axis=1).unstack("ranker")
    assert (counted["model"] == counted["baseline"]).all()


def test_the_reference_scores_zero_against_itself() -> None:
    table, _, _ = _table()
    assert (table.xs("baseline", level="ranker")["vs_baseline"] == 0).all()


def test_the_floor_differs_between_the_cutoff_slices() -> None:
    """A single floor quoted across slices misstates whichever slice it was not computed on."""
    table, _, _ = _table()
    floors = table["floor"].groupby("slice").first()
    assert floors["n<=10"] > floors["n>10"] + 0.1


def test_range_share_uses_the_slice_floor_not_zero() -> None:
    table, _, _ = _table()
    row = table.loc[("n>10", "model")]
    assert row["range_share"] == pytest.approx((row["ndcg@10"] - row["floor"]) / (1 - row["floor"]))


def test_an_unknown_reference_raises_rather_than_silently_omitting_the_column() -> None:
    grades, groups = _graded([10, 10])
    with pytest.raises(KeyError, match="not one of the rankers"):
        report.comparison_table(grades, groups, {"a": pd.Series(range(20))}, reference="b")


def test_the_table_needs_no_reference() -> None:
    grades, groups = _graded([10, 10])
    table = report.comparison_table(grades, groups, {"a": pd.Series(range(20), dtype="float64")})
    assert not any(c.startswith("vs_") for c in table.columns)


def test_a_precomputed_floor_is_used_unchanged() -> None:
    grades, groups = _graded([12, 12])
    fixed = pd.Series({0: 0.25, 1: 0.25})
    table = report.comparison_table(
        grades, groups, {"a": pd.Series(range(24), dtype="float64")}, floor=fixed
    )
    assert table.loc[("overall", "a"), "floor"] == pytest.approx(0.25)


def test_headline_quotes_the_floor_and_both_levels() -> None:
    table, _, _ = _table()
    sentence = report.headline(table, "model", "baseline", "n>10")

    assert "random floor" in sentence
    assert "paired" in sentence
    assert f"{table.loc[('n>10', 'model'), 'ndcg@10']:.4f}" in sentence
    assert f"{table.loc[('n>10', 'baseline'), 'ndcg@10']:.4f}" in sentence


def test_headline_against_a_baseline_the_table_was_not_paired_on() -> None:
    """The sealed fold is where the two frozen baselines swap order, so both deserve a sentence.
    Asking for the one the table was not built against must degrade to levels, not raise."""
    table, _, _ = _table()
    sentence = report.headline(table, "model", "baseline", "overall")
    assert "paired" in sentence

    unpaired = table.drop(columns=[c for c in table.columns if c.startswith("vs_")])
    fallback = report.headline(unpaired, "model", "baseline", "overall")
    assert "no paired interval" in fallback
    assert f"{table.loc[('overall', 'baseline'), 'ndcg@10']:.4f}" in fallback
