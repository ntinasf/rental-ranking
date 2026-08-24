"""Tests for the cross-query comparability measurement.

The one that carries the most weight is
``test_a_score_with_a_per_group_offset_is_caught_as_incomparable``. That is the failure the
module exists to detect: a score whose *within-group* ordering is perfect but whose scale shifts
from group to group. Every within-group metric in the project — NDCG included — reports such a
score as flawless, which is precisely why a separate measurement was needed. A version of this
module that failed to catch it would agree with every other number and be worthless.

The rest pin the supporting rules: truth ties carry no ordering information and must be dropped
rather than counted as half-right, cells too thin to estimate must be absent rather than noisy,
and the cell key must carry the fold.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.evaluate import comparability

# --- the rule the module exists to keep ---------------------------------------------------------


def _two_groups_in_one_cell(offset: float, n: int = 60) -> dict[str, pd.Series]:
    """Two query groups inside one cell, truth running 0..4, score faithful within each group."""
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 5, 2 * n).astype("float64")
    group = np.repeat([0, 1], n)
    # Perfect within-group ordering; group 1's scale is shifted by `offset`.
    score = truth + group * offset
    return {
        "scores": pd.Series(score),
        "groups": pd.Series(group),
        "truth": pd.Series(truth),
        "cells": pd.Series(["c"] * (2 * n)),
    }


def test_a_score_with_a_per_group_offset_is_caught_as_incomparable() -> None:
    """Within-group ordering is perfect, so NDCG would call this model flawless. The cross-group
    ordering is dictated by the offset instead of the truth, and that is what must show up."""
    data = _two_groups_in_one_cell(offset=100.0)

    inside = comparability.pair_accuracy(**data, within=True)
    across = comparability.pair_accuracy(**data, within=False)

    assert inside["c"] == pytest.approx(1.0), "within-group ordering is exact by construction"
    assert across["c"] < 0.75, "a 100-point per-group shift must not read as comparable"


def test_a_score_that_is_the_truth_is_comparable_everywhere() -> None:
    data = _two_groups_in_one_cell(offset=0.0)

    table = comparability.comparability(**data)

    assert table.loc["within", "estimate"] == pytest.approx(1.0)
    assert table.loc["cross", "estimate"] == pytest.approx(1.0)
    assert table.loc["difference", "estimate"] == pytest.approx(0.0)


def test_a_score_unrelated_to_the_truth_sits_at_chance_both_ways() -> None:
    rng = np.random.default_rng(1)
    n = 4_000
    data = {
        "scores": pd.Series(rng.normal(size=n)),
        "groups": pd.Series(rng.integers(0, 4, n)),
        "truth": pd.Series(rng.integers(0, 5, n).astype("float64")),
        "cells": pd.Series(["c"] * n),
    }

    table = comparability.comparability(**data)

    assert table.loc["within", "estimate"] == pytest.approx(0.5, abs=0.05)
    assert table.loc["cross", "estimate"] == pytest.approx(0.5, abs=0.05)


# --- the supporting rules -----------------------------------------------------------------------


def test_the_cell_key_carries_the_fold_so_two_calibrations_never_share_a_candidate_set() -> None:
    """Out-of-fold scores come from four fitted models. A universe spanning folds must split."""
    universe = pd.Series([7, 7, 7, 7])
    fold = pd.Series([1, 1, 3, 3])

    cells = comparability.evaluation_cells(universe, fold)

    assert cells.nunique() == 2, "one universe over two folds is two cells, not one"
    assert list(cells) == ["7|f1", "7|f1", "7|f3", "7|f3"]


def test_truth_ties_are_dropped_rather_than_counted_as_half_right() -> None:
    """Counting a tie as half-right pulls every estimate toward 0.5 by the tie rate, which reads
    as 'less comparable' for no reason but the grading being coarse."""
    n = 200
    rng = np.random.default_rng(2)
    truth = np.where(np.arange(2 * n) % 2 == 0, 1.0, 1.0)  # every truth identical
    truth[:4] = [0.0, 4.0, 0.0, 4.0]  # four rows that do differ
    data = {
        "scores": pd.Series(truth + rng.normal(scale=0.01, size=2 * n)),
        "groups": pd.Series(np.repeat([0, 1], n)),
        "truth": pd.Series(truth),
        "cells": pd.Series(["c"] * (2 * n)),
    }

    # Only the handful of discordant pairs count, and the score orders them correctly, so the
    # answer is 1.0 rather than something near 0.5 diluted by the ties.
    across = comparability.pair_accuracy(**data, within=False, min_pairs=1)

    assert across["c"] == pytest.approx(1.0)


def test_a_cell_too_thin_to_estimate_is_absent_rather_than_noisy() -> None:
    data = _two_groups_in_one_cell(offset=0.0, n=3)

    across = comparability.pair_accuracy(**data, within=False, min_pairs=200)

    assert across.empty, "a handful of pairs must not enter the bootstrap at full weight"


def test_within_and_cross_draw_from_the_populations_they_claim() -> None:
    """A cell whose listings all sit in one query group can supply no cross-group pair."""
    n = 80
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 5, n).astype("float64")
    data = {
        "scores": pd.Series(truth),
        "groups": pd.Series(np.zeros(n, dtype="int64")),
        "truth": pd.Series(truth),
        "cells": pd.Series(["c"] * n),
    }

    assert comparability.pair_accuracy(**data, within=True)["c"] == pytest.approx(1.0)
    assert comparability.pair_accuracy(**data, within=False).empty


def test_the_paired_difference_covers_only_cells_reporting_both() -> None:
    """One cell holds a single query group, so it reports a within accuracy and no cross one; it
    must not enter the difference with a silently imputed value."""
    n = 80
    rng = np.random.default_rng(4)
    truth = rng.integers(0, 5, 2 * n).astype("float64")
    data = {
        "scores": pd.Series(truth),
        "groups": pd.Series(
            np.concatenate([np.repeat([0, 1], n // 2), np.zeros(n, dtype=int) + 2])
        ),
        "truth": pd.Series(truth),
        "cells": pd.Series(["both"] * n + ["single"] * n),
    }

    table = comparability.comparability(**data)

    assert table.loc["within", "cells"] == 2
    assert table.loc["cross", "cells"] == 1
    assert table.loc["difference", "cells"] == 1


def test_the_estimate_is_stable_across_sampling_seeds() -> None:
    data = _two_groups_in_one_cell(offset=0.0, n=400)

    estimates = [
        comparability.pair_accuracy(**data, within=False, seed=seed)["c"] for seed in range(4)
    ]

    assert max(estimates) - min(estimates) < 0.02
