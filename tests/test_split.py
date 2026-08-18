"""Tests for rental_ranking.train.split.

**Every failure here is silent and irreversible.** A cluster that straddles the split leaks a
near-twin into the test half and inflates every metric computed afterwards; a query group that
straddles it turns test NDCG into a measurement over a partial candidate set, which is not the
measurement the frozen baselines made; a sealed fold that leaks into a training index produces
numbers that look exactly like honest ones. So the two structural invariants — components keep
clusters and groups whole, and the sealed fold appears nowhere in the development pool — are
asserted directly rather than inferred from a balance report.

The grade-blindness of ``assign_folds`` is also tested, because it is a property that decays:
the objective is one obvious edit away from balancing on the target.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data.paths import FEATURE_TABLE_PATH
from rental_ranking.train import split

_DEFAULTS = {"city": "athens"}


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_DEFAULTS, **row} for row in rows])


def _population(n_components: int = 40, seed: int = 0) -> pd.DataFrame:
    """A frame of disjoint components: one query group per component, clusters inside it.

    Deliberately conflict-free, so a test that fails is failing on the fold assignment rather
    than on the graph.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for component in range(n_components):
        size = int(rng.integers(4, 40))
        for i in range(size):
            rows.append(
                {
                    "query_group": component,
                    "cluster_id": f"c{component}_{i // 2}",
                    "city": ["athens", "crete", "thessaloniki"][component % 3],
                    "grade": int(rng.integers(0, 5)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- components


def test_shared_cluster_merges_two_query_groups() -> None:
    """The conflict this module exists for: one cluster spanning two groups fuses both."""
    frame = _frame(
        [
            {"query_group": 0, "cluster_id": "a"},
            {"query_group": 0, "cluster_id": "twin"},
            {"query_group": 1, "cluster_id": "twin"},
            {"query_group": 1, "cluster_id": "b"},
        ]
    )
    assert split.split_component(frame).nunique() == 1


def test_disjoint_groups_stay_separate() -> None:
    frame = _frame(
        [
            {"query_group": 0, "cluster_id": "a"},
            {"query_group": 0, "cluster_id": "b"},
            {"query_group": 1, "cluster_id": "c"},
            {"query_group": 2, "cluster_id": "d"},
        ]
    )
    assert split.split_component(frame).nunique() == 3


def test_components_chain_transitively() -> None:
    """Group 0 -- cluster t1 -- group 1 -- cluster t2 -- group 2 is one component, not two."""
    frame = _frame(
        [
            {"query_group": 0, "cluster_id": "t1"},
            {"query_group": 1, "cluster_id": "t1"},
            {"query_group": 1, "cluster_id": "t2"},
            {"query_group": 2, "cluster_id": "t2"},
        ]
    )
    assert split.split_component(frame).nunique() == 1


def test_component_ids_are_dense_and_aligned() -> None:
    frame = _population(n_components=6)
    frame.index = frame.index * 3 + 7  # a non-trivial index must survive
    components = split.split_component(frame)
    assert components.index.equals(frame.index)
    assert sorted(components.unique()) == list(range(6))


@pytest.mark.parametrize("column", ["query_group", "cluster_id"])
def test_null_id_raises_rather_than_dropping_the_row(column: str) -> None:
    frame = _population(n_components=4)
    frame.loc[0, column] = None
    with pytest.raises(ValueError, match="no fold"):
        split.split_component(frame)


def test_missing_column_raises() -> None:
    with pytest.raises(KeyError):
        split.split_component(pd.DataFrame({"query_group": [0, 1]}))


# --------------------------------------------------------------------------- size bands


@pytest.mark.parametrize(
    ("size", "band"),
    [
        (1, "<10"),
        (10, "<10"),
        (11, "10-30"),
        (30, "10-30"),
        (31, "30-100"),
        (400, "100-400"),
        (401, "400+"),
    ],
)
def test_group_size_band_edges(size: int, band: str) -> None:
    frame = pd.DataFrame({"query_group": [0] * size})
    assert split.group_size_band(frame).iloc[0] == band


# --------------------------------------------------------------------------- fold assignment


def test_no_cluster_and_no_group_spans_a_fold() -> None:
    """The invariant option A was chosen to guarantee, asserted on a frame full of conflicts."""
    frame = _frame(
        [
            {"query_group": g, "cluster_id": f"c{g}_{i}", "grade": i % 5}
            for g in range(30)
            for i in range(12)
        ]
    )
    # Wire ten spanning clusters through consecutive group pairs.
    for g in range(0, 20, 2):
        frame.loc[(frame.query_group == g + 1) & (frame.index % 12 == 0), "cluster_id"] = f"c{g}_0"

    fold, _ = split.assign_folds(frame, folds=3)
    assert fold.groupby(frame.cluster_id).nunique().max() == 1
    assert fold.groupby(frame.query_group).nunique().max() == 1


def test_every_row_gets_exactly_one_fold() -> None:
    frame = _population()
    fold, report = split.assign_folds(frame)
    assert fold.notna().all()
    assert fold.index.equals(frame.index)
    assert set(fold.unique()) == {0, 1, 2, 3, 4}
    assert report["rows"].sum() == len(frame)
    assert report["groups"].sum() == frame.query_group.nunique()


def test_rows_are_balanced_across_folds() -> None:
    frame = _population(n_components=120)
    _, report = split.assign_folds(frame)
    assert report["row_share"].max() - report["row_share"].min() < 0.02


def test_city_composition_is_reproduced_in_every_fold() -> None:
    frame = _population(n_components=150)
    fold, _ = split.assign_folds(frame)
    shares = split.fold_balance(fold, frame.city, frame.query_group)
    overall = frame.groupby("query_group").city.first().value_counts(normalize=True)
    assert (shares - overall).abs().to_numpy().max() < 0.10


def test_assignment_is_deterministic_and_seed_sensitive() -> None:
    frame = _population(n_components=60)
    first, _ = split.assign_folds(frame, seed=0)
    again, _ = split.assign_folds(frame, seed=0)
    other, _ = split.assign_folds(frame, seed=7)
    assert first.equals(again)
    assert not first.equals(other)


def test_grade_is_never_read() -> None:
    """The objective must stay grade-blind: balancing the split on the target is how a split
    starts choosing its own answer. A frame with no grade column must assign identically."""
    frame = _population(n_components=60)
    with_grade, _ = split.assign_folds(frame, seed=0)
    without_grade, _ = split.assign_folds(frame.drop(columns="grade"), seed=0)
    assert with_grade.equals(without_grade)


def test_too_few_folds_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        split.assign_folds(_population(n_components=10), folds=1)


def test_fewer_components_than_folds_raises() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        split.assign_folds(_population(n_components=3), folds=5)


# --------------------------------------------------------------------------- the sealed fold


def test_sealed_fold_appears_in_no_development_index() -> None:
    """A sealed fold leaking into a training index is invisible in every metric it produces."""
    frame = _population(n_components=60)
    fold, _ = split.assign_folds(frame)
    sealed = fold.index[fold.eq(split.SEALED_FOLD)]
    for train, valid in split.dev_cv_splits(fold):
        assert sealed.intersection(train).empty
        assert sealed.intersection(valid).empty


def test_dev_splits_partition_the_pool_exactly_once() -> None:
    frame = _population(n_components=60)
    fold, _ = split.assign_folds(frame)
    splits = split.dev_cv_splits(fold)
    assert len(splits) == split.DEFAULT_FOLDS - 1

    seen = pd.Index([])
    for train, valid in splits:
        assert train.intersection(valid).empty
        assert len(train) + len(valid) == int((~split.sealed_mask(fold)).sum())
        seen = seen.union(valid)
    assert seen.equals(fold.index[fold.ne(split.SEALED_FOLD)].sort_values())


def test_sealed_mask_selects_one_fold() -> None:
    fold = pd.Series([0, 1, 2, 0, 3])
    assert split.sealed_mask(fold).tolist() == [True, False, False, True, False]
    assert split.sealed_mask(fold, sealed=2).sum() == 1


# --------------------------------------------------------------------------- reporting


def test_constant_grade_groups_flags_only_unrankable_groups() -> None:
    grades = pd.Series([2, 2, 2, 0, 1, 3])
    groups = pd.Series([0, 0, 0, 1, 1, 1])
    flagged = split.constant_grade_groups(grades, groups)
    assert flagged.loc[0]
    assert not flagged.loc[1]


def test_fold_balance_rows_sum_to_one() -> None:
    frame = _population(n_components=40)
    fold, _ = split.assign_folds(frame)
    shares = split.fold_balance(fold, frame.grade)
    assert np.allclose(shares.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- the real table


def test_shipped_feature_table_splits_without_leakage() -> None:
    """The measurement the module docstring records, pinned against the shipped table."""
    table = FEATURE_TABLE_PATH
    if not table.exists():
        pytest.skip("feature table not built")

    frame = pd.read_parquet(table)
    assert split.split_component(frame).nunique() == 345

    fold, report = split.assign_folds(frame)
    assert report["row_share"].max() - report["row_share"].min() < 0.001
    assert fold.groupby(frame.cluster_id).nunique().max() == 1
    assert fold.groupby(frame.query_group).nunique().max() == 1
