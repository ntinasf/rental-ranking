"""Tests for rental_ranking.train.train — the orchestrator.

The protocol is the thing under test, not the arithmetic. Two properties carry the whole
evaluation and both fail silently: **the sealed fold must appear in no training index at any
point**, and the out-of-fold scores must cover each development row exactly once, from the model
that held it out. A breach of either produces a table that looks entirely normal.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.train import train as trainer
from rental_ranking.train.split import SEALED_FOLD


def _table(n_groups: int = 20, size: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A feature-table-shaped frame with a hand-built fold column, five folds, groups whole."""
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(n_groups):
        for i in range(size):
            grade = int(rng.integers(0, 5))
            rows.append(
                {
                    "id": f"{group}_{i}",
                    "query_group": group,
                    "cluster_id": group * 1000 + i,
                    "grade": grade,
                    "blocked_fraction_90": grade / 4,
                    "strong": float(grade) + rng.normal(0, 0.5),
                    "noise": rng.normal(),
                }
            )
    frame = pd.DataFrame(rows)
    fold = (frame["query_group"] % 5).rename("fold")
    return frame, fold


# --- run tags ------------------------------------------------------------------------------------


def test_dataset_version_reads_the_registered_asset() -> None:
    version = trainer.dataset_version()
    assert version != "unregistered"
    assert version.startswith("2026.")


def test_a_missing_asset_file_reports_unregistered_rather_than_raising(tmp_path) -> None:
    """A training run must not fail because an asset file moved — but the tag must not claim a
    version it never read either."""
    assert trainer.dataset_version(tmp_path / "absent.yml") == "unregistered"


def test_dataset_digest_changes_with_the_bytes(tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert trainer.dataset_digest(first) != trainer.dataset_digest(second)
    assert len(trainer.dataset_digest(first)) == 12


# --- cross-validation ----------------------------------------------------------------------------


def test_out_of_fold_scores_cover_the_pool_exactly_once_and_omit_the_sealed_fold() -> None:
    """The invariant the per-city claim rests on: every development row scored by a model that
    never saw it, and no sealed row scored at all."""
    frame, fold = _table()
    oof, iterations, report, curves = trainer.cross_validate(
        frame, fold, params={"min_child_samples": 5, "n_estimators": 25}
    )

    pool = frame.index[fold.ne(SEALED_FOLD)]
    assert sorted(oof.index) == sorted(pool)
    assert len(oof) == len(oof.index.unique())
    assert oof.index.intersection(frame.index[fold.eq(SEALED_FOLD)]).empty
    assert len(iterations) == 4
    assert SEALED_FOLD not in report.index


def test_cross_validation_report_accounts_for_every_pool_row_per_fold() -> None:
    frame, fold = _table()
    _, _, report, _ = trainer.cross_validate(
        frame, fold, params={"min_child_samples": 5, "n_estimators": 25}
    )
    pool_size = int(fold.ne(SEALED_FOLD).sum())
    assert (report["train_rows"] + report["valid_rows"] == pool_size).all()
    assert report["valid_rows"].sum() == pool_size


# --- the refit -----------------------------------------------------------------------------------


def test_refit_warns_when_several_seeds_cannot_differ() -> None:
    """A spread of 0.000 from a deterministic parameter set reads as stability and is not."""
    frame, fold = _table(n_groups=10, size=30)
    pool = frame[fold.ne(SEALED_FOLD)]
    with pytest.warns(UserWarning, match="deterministic"):
        trainer.refit(pool, n_estimators=10, seeds=(0, 1, 2))


def test_refit_does_not_warn_for_one_seed_or_a_stochastic_set() -> None:
    frame, fold = _table(n_groups=10, size=30)
    pool = frame[fold.ne(SEALED_FOLD)]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trainer.refit(pool, n_estimators=10, seeds=(0,))
        trainer.refit(
            pool,
            n_estimators=10,
            seeds=(0, 1),
            params={"colsample_bytree": 0.8, "min_child_samples": 5},
        )


def test_refit_returns_one_model_per_seed_at_the_given_size() -> None:
    frame, fold = _table(n_groups=10, size=30)
    pool = frame[fold.ne(SEALED_FOLD)]
    models = trainer.refit(pool, n_estimators=12, seeds=(0, 1), params={"colsample_bytree": 0.8})
    assert sorted(models) == [0, 1]
    assert all(m.booster_.num_trees() == 12 for m in models.values())


# --- seed summary --------------------------------------------------------------------------------


def _seed_table() -> pd.DataFrame:
    rows = [
        {
            "slice": "overall",
            "ranker": "model_seed0",
            "groups": 70,
            "ndcg@10": 0.70,
            "floor": 0.55,
            "vs_reviews": 0.06,
        },
        {
            "slice": "overall",
            "ranker": "model_seed1",
            "groups": 70,
            "ndcg@10": 0.74,
            "floor": 0.55,
            "vs_reviews": 0.10,
        },
        {
            "slice": "overall",
            "ranker": "reviews",
            "groups": 70,
            "ndcg@10": 0.64,
            "floor": 0.55,
            "vs_reviews": 0.0,
        },
    ]
    return pd.DataFrame(rows).set_index(["slice", "ranker"])


def test_seed_summary_collapses_only_the_seed_rows() -> None:
    summary = trainer.summarise_seeds(_seed_table())
    assert summary.loc["overall", "seeds"] == 2
    assert summary.loc["overall", "ndcg_mean"] == pytest.approx(0.72)
    assert summary.loc["overall", "ndcg_min"] == pytest.approx(0.70)
    assert summary.loc["overall", "vs_reviews"] == pytest.approx(0.08)


def test_seed_summary_flags_a_zero_spread_as_determinism() -> None:
    table = _seed_table()
    table.loc[("overall", "model_seed1"), "ndcg@10"] = 0.70
    assert trainer.summarise_seeds(table).loc["overall", "deterministic"]
    assert not trainer.summarise_seeds(_seed_table()).loc["overall", "deterministic"]


# --- learning curves ------------------------------------------------------------------------------


def test_cross_validation_returns_one_curve_per_development_fold() -> None:
    """The curves are the evidence behind "the stopping point is barely identified". An assertion
    in prose is cheap; four lines that go flat and stay flat is not."""
    frame, fold = _table()
    _, iterations, _, curves = trainer.cross_validate(
        frame, fold, params={"min_child_samples": 5, "n_estimators": 25}
    )
    assert sorted(curves) == [1, 2, 3, 4]
    assert SEALED_FOLD not in curves
    assert all(len(c) == 25 for c in curves.values())
    assert len(iterations) == len(curves)


def test_plot_learning_curves_writes_a_figure(tmp_path) -> None:
    frame, fold = _table()
    _, iterations, _, curves = trainer.cross_validate(
        frame, fold, params={"min_child_samples": 5, "n_estimators": 15}
    )
    path = trainer.plot_learning_curves(curves, iterations, tmp_path / "curves.png")
    assert path.exists()
    assert path.stat().st_size > 5_000
