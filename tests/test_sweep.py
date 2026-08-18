"""Tests for rental_ranking.train.sweep.

The search itself is uninteresting to test — random draws from stated ranges. What matters is
the **protocol around it**, because each of these failures produces a plausible winner:

* the defaults must be evaluated as configuration 0, by the same code on the same folds, or the
  reference is not comparable to its challengers;
* the acceptance rule must refuse a point-estimate win whose interval covers zero, because the
  maximum of ~35 noisy draws is positive by construction;
* pruned configurations must stay in the results table, or the "configurations that lost" record
  the roadmap asks for is quietly incomplete.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.train import sweep


def _table(n_groups: int = 15, size: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(n_groups):
        for i in range(size):
            grade = int(rng.integers(0, 5))
            rows.append(
                {
                    "id": f"{group}_{i}",
                    "query_group": group,
                    "cluster_id": group * 100 + i,
                    "grade": grade,
                    "blocked_fraction_90": grade / 4,
                    "strong": float(grade) + rng.normal(0, 0.6),
                    "noise": rng.normal(),
                }
            )
    frame = pd.DataFrame(rows)
    return frame, (frame["query_group"] % 5).rename("fold")


# --- sampling ------------------------------------------------------------------------------------


def test_configuration_zero_is_always_the_defaults() -> None:
    """The reference has to run through the same code path as its challengers."""
    assert sweep.sample_configs(5)[0] == {}


def test_sampling_is_seed_stable_and_seed_sensitive() -> None:
    assert sweep.sample_configs(6, seed=3) == sweep.sample_configs(6, seed=3)
    assert sweep.sample_configs(6, seed=3) != sweep.sample_configs(6, seed=4)


def test_every_drawn_configuration_covers_the_whole_space() -> None:
    for config in sweep.sample_configs(8, seed=1)[1:]:
        assert set(sweep.SEARCH_SPACE) <= set(config)


def test_row_subsampling_always_arrives_with_a_frequency() -> None:
    """``subsample`` without ``subsample_freq`` does nothing — the search would silently
    explore a parameter that cannot take effect."""
    for config in sweep.sample_configs(10, seed=2)[1:]:
        assert config["subsample_freq"] == 1


def test_drawn_values_stay_inside_their_ranges() -> None:
    for config in sweep.sample_configs(30, seed=5)[1:]:
        assert 15 <= config["num_leaves"] <= 127
        assert 5 <= config["min_child_samples"] <= 100
        assert 0.02 <= config["learning_rate"] <= 0.1
        assert 0.5 <= config["colsample_bytree"] <= 1.0
        assert config["lambdarank_truncation_level"] in (10, 20, 30, 50, 100)
        assert isinstance(config["lambdarank_norm"], bool)


def test_an_empty_search_raises() -> None:
    with pytest.raises(ValueError, match="at least the default"):
        sweep.sample_configs(0)


# --- the guard -----------------------------------------------------------------------------------


def test_the_guard_abandons_a_configuration_and_returns_no_vector() -> None:
    frame, fold = _table()
    oof, folds, iterations = sweep.evaluate_config(frame, fold, {}, guard=2.0)
    assert oof is None
    assert len(folds) == 1
    assert len(iterations) == 1


def test_no_guard_runs_every_fold() -> None:
    frame, fold = _table()
    oof, folds, _ = sweep.evaluate_config(frame, fold, {"n_estimators": 20}, guard=None)
    assert oof is not None
    assert len(folds) == 4


def test_the_default_configuration_is_never_pruned() -> None:
    """It is the reference; a pruned reference leaves nothing to compare against."""
    frame, fold = _table()
    results, vectors = sweep.run_search(
        frame, fold, n_configs=3, margin=0.0, verbose=False, fit_seed=0
    )
    assert results.loc[0, "status"] == "default"
    assert 0 in vectors


# --- the results table ---------------------------------------------------------------------------


def test_pruned_configurations_stay_in_the_table() -> None:
    """The losers record has to be complete, or 'here is what lost' is not falsifiable."""
    frame, fold = _table()
    results, _ = sweep.run_search(frame, fold, n_configs=4, margin=-1.0, verbose=False)
    assert len(results) == 4
    assert (results["status"] == "pruned").sum() >= 1
    assert results.loc[results["status"].eq("pruned"), "oof_ndcg"].isna().all()


def test_every_completed_configuration_is_scored_on_the_same_groups() -> None:
    frame, fold = _table()
    results, _ = sweep.run_search(frame, fold, n_configs=3, margin=None, verbose=False)
    scored = results[results["groups"].notna()]
    assert scored["groups"].nunique() == 1


def test_the_default_scores_zero_against_itself() -> None:
    frame, fold = _table()
    results, _ = sweep.run_search(frame, fold, n_configs=3, margin=None, verbose=False)
    assert results.loc[0, "vs_default"] == 0.0


# --- the acceptance rule -------------------------------------------------------------------------


def _results(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("config")


def test_a_win_whose_interval_covers_zero_is_refused() -> None:
    """The maximum of many noisy draws is positive by construction. A point estimate is not
    enough to replace the defaults."""
    table = _results(
        [
            {"config": 0, "status": "default", "vs_default": 0.0, "vs_low": 0.0, "vs_high": 0.0},
            {
                "config": 1,
                "status": "complete",
                "vs_default": 0.012,
                "vs_low": -0.004,
                "vs_high": 0.028,
            },
        ]
    )
    chosen, verdict = sweep.accept(table)
    assert chosen == 0
    assert "did not clear the noise floor" in verdict


def test_a_win_clear_of_zero_is_accepted() -> None:
    table = _results(
        [
            {"config": 0, "status": "default", "vs_default": 0.0, "vs_low": 0.0, "vs_high": 0.0},
            {
                "config": 1,
                "status": "complete",
                "vs_default": 0.021,
                "vs_low": 0.006,
                "vs_high": 0.037,
            },
        ]
    )
    chosen, verdict = sweep.accept(table)
    assert chosen == 1
    assert "accepted" in verdict


def test_the_largest_point_estimate_does_not_win_on_its_own() -> None:
    """A bigger but wilder gain must lose to nothing, not beat a smaller reliable one by size."""
    table = _results(
        [
            {"config": 0, "status": "default", "vs_default": 0.0, "vs_low": 0.0, "vs_high": 0.0},
            {
                "config": 1,
                "status": "complete",
                "vs_default": 0.040,
                "vs_low": -0.010,
                "vs_high": 0.090,
            },
            {
                "config": 2,
                "status": "complete",
                "vs_default": 0.015,
                "vs_low": 0.004,
                "vs_high": 0.026,
            },
        ]
    )
    assert sweep.accept(table)[0] == 0


def test_a_search_where_nothing_completed_keeps_the_defaults() -> None:
    table = _results(
        [{"config": 0, "status": "default", "vs_default": 0.0, "vs_low": np.nan, "vs_high": np.nan}]
    )
    chosen, verdict = sweep.accept(table)
    assert chosen == 0
    assert "no configuration completed" in verdict
