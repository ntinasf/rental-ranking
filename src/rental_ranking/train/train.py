"""The training run: cross-validate on the development pool, refit, score the sealed fold once.

The orchestrator for ``train/`` — the only module here that reads a file, writes one, or has a
``main()``, exactly as ``features/build.py`` is to ``features/``. Everything it composes is a
pure function in ``lambdamart.py``, ``split.py``, ``evaluate/metrics.py`` and
``evaluate/report.py``.

**The protocol, decided 2026-08-18 and irreversible.** ``assign_folds`` cuts the population into
five folds over the ``cluster_id`` x ``query_group`` connected components. **Fold 0 is sealed.**
Folds 1-4 are the development pool, and every decision — the stopping iteration here, the
hyperparameters of any later sweep — is made by cross-validating inside it. The sealed fold is
scored once, at the end, by a model refit on the whole pool.

Why not one validation split: baseline A minus baseline B is a *constant*, and read off five
candidate 20 % test halves it reports 0.0151 / 0.0116 / 0.0487 / 0.0249 / 0.0048. Per-group NDCG
has standard deviation 0.187, so a single ~77-group validation set would choose the stopping
iteration on noise of the same size as the effect being measured.

**Out-of-fold predictions are the second product, and they are free.** ``dev_cv_splits``
partitions the pool exactly once, so concatenating each fold model's predictions on its own
held-out fold gives one prediction per development row, each from a model that never trained on
it. That is where the **per-city** numbers come from: the sealed fold holds 4 Thessaloniki
groups (+/- 0.183, uninterpretable) against the pool's 24 (+/- 0.071).

**The two numbers describe different objects and the report must say which.** The out-of-fold
table estimates the *procedure* — a model trained this way on three quarters of the pool. The
sealed table estimates the *artifact* — the model that would ship. Neither substitutes for the
other, and "the model scores X in Thessaloniki" without naming which is an overstatement.

**Cross-fold score scales never meet.** Each fold model calibrates its own scores, but a query
group lies wholly inside one fold, and NDCG only ever compares scores *within* a group. So the
concatenated out-of-fold scores are safe to rank even though they are not on a common scale —
the usual trap in out-of-fold stacking does not arise here.

**Seeds are reported as a mean and a spread, never as the best run** — but read the spread
carefully. Measured 2026-08-18: at the starting parameters LightGBM is **fully deterministic**
(``subsample_freq=0``, ``colsample_bytree=1.0``), so five seeds give bit-identical predictions
and a spread of exactly 0.000. That is not a stable model; it is no randomness to average over,
and :func:`refit` says so rather than letting the zero read as stability. The variance that does
exist is over *data* — the group bootstrap and the fold-to-fold spread in the cross-validation
report are what measure it. A sweep that turns on ``subsample`` or ``colsample_bytree`` makes the
seed spread meaningful, and it must be re-read then.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from rental_ranking.data.paths import (
    FEATURE_TABLE_PATH,
    PROJECT_ROOT,
    SWEEP_RESULTS_PATH,
    TRAIN_DIR,
)
from rental_ranking.evaluate.report import comparison_table, headline, random_floor
from rental_ranking.features.assemble import check_feature_table, feature_columns
from rental_ranking.train import baseline as bl
from rental_ranking.train import lambdamart as lm
from rental_ranking.train.split import (
    DEFAULT_FOLDS,
    SEALED_FOLD,
    assign_folds,
    dev_cv_splits,
    group_size_band,
    sealed_mask,
)

#: One experiment for the whole project, local and on Azure alike. On Azure ML the workspace is
#: the tracking server, so the same call reaches a different store with no code change.
EXPERIMENT = "rental-ranking"

#: The cut-off the curves and the stopping rule both read, kept in step with lambdamart.EVAL_AT.
EVAL_K = lm.EVAL_AT[0]

#: Seeds the final refit is repeated over. Five is the roadmap's number and it is a floor, not a
#: target: one run that beats the baseline by 0.01 has told you nothing.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Prefix the per-seed sealed-fold rankers are named with, so :func:`summarise_seeds` can find
#: them without being told how many there were.
_SEED_PREFIX = "model_seed"

_VERSION_YAML = PROJECT_ROOT / "pipelines" / "data" / "feature_table.yml"


def dataset_version(path: Path = _VERSION_YAML) -> str:
    """The registered feature-table version, read from the asset YAML.

    **On Azure ML this file is not present** — the job snapshot is ``src/`` alone, so
    ``PROJECT_ROOT`` resolves to the working directory and the lookup misses. That is why
    :func:`run` takes an explicit ``version``: a cloud run passes the value from the job YAML
    rather than silently tagging itself ``unregistered``, which would break the one traceability
    claim the Azure step exists to demonstrate.

    Regex rather than a YAML parser: ``pyyaml`` is not a declared dependency of this project and
    one pinned line does not justify adding one. Returns ``"unregistered"`` rather than raising —
    a training run must not fail because an asset file moved, but the tag must not silently claim
    a version it did not read either.
    """
    if not path.is_file():
        return "unregistered"
    match = re.search(r'^version:\s*"?([^"\n]+)"?', path.read_text(), flags=re.MULTILINE)
    return match.group(1).strip() if match else "unregistered"


def dataset_digest(path: Path) -> str:
    """SHA-256 of the feature table, first 12 hex — the tag that survives a version reused."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12]


def git_commit() -> str:
    """Short HEAD, or ``"unknown"`` outside a repository. Never raises into a training run."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def cross_validate(
    table: pd.DataFrame,
    fold: pd.Series,
    params: dict[str, object] | None = None,
    seed: int = 0,
    features: list[str] | None = None,
) -> tuple[pd.Series, list[int], pd.DataFrame, dict[int, list[float]]]:
    """Fit one model per development fold; return out-of-fold scores, iterations and curves.

    Args:
        table: The full feature table.
        fold: Fold ids from :func:`~rental_ranking.train.split.assign_folds`.
        params: LightGBM parameters, defaulting to :data:`~rental_ranking.train.lambdamart.DEFAULT_PARAMS`.
        seed: Seed for every fold model. Held fixed across folds so the spread in the returned
            iteration counts is fold variance, not seed variance.
        features: Column list, defaulting to ``feature_columns``.

    Returns:
        ``(oof, iterations, per_fold, curves)``. ``oof`` is one score per development row, from
        the model that held that row out. ``iterations`` is each fold's ``best_iteration_``.
        ``per_fold`` reports rows, groups and the fold's own NDCG@10. ``curves`` maps fold id to
        LightGBM's validation NDCG@10 at every boosting iteration — the evidence behind the
        claim that the stopping point is barely identified, rather than the assertion of it.
    """
    from rental_ranking.evaluate.metrics import ndcg_at_k

    columns = features if features is not None else feature_columns(table)
    blocks, iterations, rows, curves = [], [], [], {}

    splits = dev_cv_splits(fold)
    for index, (train_index, valid_index) in enumerate(splits, start=1):
        train, valid = table.loc[train_index], table.loc[valid_index]
        model = lm.fit(train, valid, params=params, seed=seed, features=columns)
        scores = lm.predict(model, valid, features=columns)
        blocks.append(scores)
        iterations.append(int(model.best_iteration_ or model.n_estimators))
        held = int(fold.loc[valid_index].iloc[0])
        curves[held] = list(model.evals_result_["valid_0"][f"ndcg@{EVAL_K}"])

        per_group = ndcg_at_k(valid["grade"], valid["query_group"], scores)
        rows.append(
            {
                "fold": int(fold.loc[valid_index].iloc[0]),
                "train_rows": len(train),
                "valid_rows": len(valid),
                "valid_groups": int(per_group.notna().sum()),
                "best_iteration": iterations[-1],
                "ndcg@10": float(per_group.mean()),
            }
        )
        print(
            f"  cv fold {index}/{len(splits)}: stopped at {iterations[-1]:>4} trees, "
            f"held-out NDCG@10 {rows[-1]['ndcg@10']:.4f} on {rows[-1]['valid_groups']} groups"
        )

    return (
        pd.concat(blocks).rename("score"),
        iterations,
        pd.DataFrame(rows).set_index("fold"),
        curves,
    )


def refit(
    development: pd.DataFrame,
    n_estimators: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    params: dict[str, object] | None = None,
    features: list[str] | None = None,
) -> dict[int, object]:
    """Fit the final model on the whole development pool, once per seed.

    **No validation set exists here**, so the iteration count is fixed rather than found:
    ``n_estimators`` is the median ``best_iteration_`` across the cross-validation folds, taken
    **unscaled**. Scaling it by ``k/(k-1)`` for the larger training set is folklore, and with
    several seeds the spread will say whether the choice mattered.
    """
    if len(seeds) > 1 and not lm.is_stochastic(params or {}):
        warnings.warn(
            f"{len(seeds)} seeds requested, but this parameter set is deterministic "
            f"({lm.DETERMINISTIC_DEFAULTS}): every seed will produce identical predictions and "
            "the reported spread will be exactly zero. That is an absence of randomness, not "
            "evidence of stability — read the bootstrap interval and the fold-to-fold spread "
            "instead",
            stacklevel=2,
        )
    models = {}
    for seed in seeds:
        models[seed] = lm.fit(
            development,
            validation=None,
            params=params,
            seed=seed,
            n_estimators=n_estimators,
            features=features,
        )
        print(f"  refit seed {seed}: {n_estimators} trees on {len(development):,} rows")
    return models


def summarise_seeds(table: pd.DataFrame, prefix: str = _SEED_PREFIX) -> pd.DataFrame:
    """Collapse the per-seed rows of a comparison table into mean, spread and range.

    The headline is ``ndcg_mean``. ``ndcg_sd`` is the number that says whether a lead is real:
    a 0.01 win with a 0.02 seed spread is not a result, and reporting the best seed instead
    would hide exactly that.
    """
    metric = next(column for column in table.columns if column.startswith("ndcg@"))
    seeds = table[table.index.get_level_values("ranker").str.startswith(prefix)]
    grouped = seeds.groupby("slice", sort=False)[metric]
    summary = pd.DataFrame(
        {
            "seeds": grouped.count(),
            "groups": seeds.groupby("slice", sort=False)["groups"].first(),
            "ndcg_mean": grouped.mean(),
            "ndcg_sd": grouped.std(),
            "ndcg_min": grouped.min(),
            "ndcg_max": grouped.max(),
            "floor": seeds.groupby("slice", sort=False)["floor"].first(),
        }
    )
    # An sd of exactly zero is a statement about the algorithm, not about the model. Marked so a
    # reader of the CSV does not take it for stability.
    summary["deterministic"] = summary["ndcg_sd"].fillna(0.0).eq(0.0)
    difference = [c for c in seeds.columns if c.startswith("vs_")]
    if difference:
        summary[difference[0]] = seeds.groupby("slice", sort=False)[difference[0]].mean()
    return summary


def plot_learning_curves(curves: dict[int, list[float]], iterations: list[int], path: Path) -> Path:
    """Validation NDCG@10 against boosting iteration, one line per development fold.

    **This figure is the argument, not decoration.** The stopping iteration is chosen as the
    median of four ``best_iteration_`` values that ranged 158-718 — a 4.5x spread — while the
    folds' scores stayed inside a 0.06 band. Written as a sentence that is an assertion; drawn as
    four curves that go flat early and stay flat, it is evidence. It is also what makes the
    later finding legible: a 35-configuration search on a surface this flat was never going to
    find much, and it did not.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    for (held, curve), stop in zip(sorted(curves.items()), iterations):
        line = axis.plot(range(1, len(curve) + 1), curve, linewidth=1.4, label=f"fold {held}")[0]
        axis.axvline(stop, color=line.get_color(), linestyle=":", alpha=0.6, linewidth=1)
    axis.axvline(
        int(np.median(iterations)),
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"median = {int(np.median(iterations))} (used for the refit)",
    )
    axis.set_xlabel("boosting iteration")
    axis.set_ylabel("validation NDCG@10 (LightGBM's own)")
    axis.set_title(
        "Held-out NDCG@10 per development fold — dotted lines mark each fold's best iteration"
    )
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _log_model(model: object, directory: Path) -> None:
    """Log the fitted ranker in **MLflow model format**, deployable either way.

    **What Azure ML does not support is narrower than it first appears.** MLflow 3 made
    ``log_model`` register a first-class *LoggedModel* entity through
    ``/api/2.0/mlflow/logged-models``; the workspace's MLflow server implements the 2.x API
    surface and returns **404**, which failed a run whose training had already finished
    (``strong_tiger_9myv0v3105``, 2026-08-18). **The model format is not the problem — only that
    one registration call is.**

    So the fallback is not a downgrade to a bare booster. ``mlflow.lightgbm.save_model`` writes a
    complete MLmodel directory *locally, contacting no server at all*, and logging that directory
    as run artifacts preserves the ``python_function`` flavour intact. Registering it afterwards
    with ``az ml model create --type mlflow_model`` gives no-code deployment to a managed
    endpoint — verified end to end on 2026-08-18. Nothing about serving is blocked.
    """
    try:
        mlflow.lightgbm.log_model(model.booster_, name="model")
        return
    except Exception as error:  # noqa: BLE001 — a server-side refusal must not fail the run
        print(
            f"note: mlflow.lightgbm.log_model failed ({type(error).__name__}: "
            f"{str(error)[:100]}). Saving the MLmodel directory locally and logging it as "
            "artifacts instead — the flavour is preserved and the model stays deployable."
        )

    saved = directory / "model"
    mlflow.lightgbm.save_model(model.booster_, str(saved))
    mlflow.log_artifacts(str(saved), artifact_path="model")
    mlflow.set_tag("model_logging", "MLmodel directory as artifacts (LoggedModel API absent)")


def _log_frame(frame: pd.DataFrame, name: str, directory: Path) -> None:
    path = directory / f"{name}.csv"
    frame.to_csv(path)
    mlflow.log_artifact(str(path))


def run(
    features_path: Path = FEATURE_TABLE_PATH,
    folds: int = DEFAULT_FOLDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    split_seed: int = 0,
    params: dict[str, object] | None = None,
    log_to_mlflow: bool = True,
    version: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the whole protocol and return every table it produced.

    Returns:
        ``fold_report``, ``cv``, ``oof`` (development out-of-fold comparison, city breakdown),
        ``oof_band`` (size-band breakdown), ``sealed`` and ``sealed_vs_price_rating`` (the
        same sealed comparison paired against each frozen baseline in turn), ``seeds``
        (collapsed) and ``importance``.
    """
    table = pd.read_parquet(features_path)
    check_feature_table(table)
    columns = feature_columns(table)
    print(f"feature table  {len(table):,} rows x {len(columns)} features -> {features_path}")

    fold, fold_report = assign_folds(table, folds=folds, seed=split_seed)
    sealed = sealed_mask(fold)
    development = table[~sealed]
    held_out = table[sealed]
    print(
        f"\nsplit — sealed fold {SEALED_FOLD}: {len(held_out):,} rows; pool: {len(development):,}"
    )
    print(fold_report.round(4).to_string())

    print(f"\ncross-validating on folds 1-{folds - 1}")
    oof_scores, iterations, cv_report, curves = cross_validate(
        table, fold, params=params, seed=seeds[0], features=columns
    )
    chosen = int(np.median(iterations))
    print(f"  iterations {iterations} -> median {chosen}, taken unscaled")

    print(f"\nrefitting on the whole pool, {len(seeds)} seeds")
    models = refit(development, chosen, seeds=seeds, params=params, features=columns)

    # One floor for the whole population: a group's floor does not depend on which side it fell.
    floor = random_floor(table["grade"], table["query_group"])
    baselines = {
        "reviews": bl.rank_by_reviews(table),
        "price_rating": bl.rank_by_price_and_rating(table, table["query_group"]),
    }

    oof = comparison_table(
        development["grade"],
        development["query_group"],
        {
            "model_oof": oof_scores.loc[development.index],
            **{k: v[~sealed] for k, v in baselines.items()},
        },
        reference="reviews",
        breakdown=development["city"],
        floor=floor,
    )
    oof_band = comparison_table(
        development["grade"],
        development["query_group"],
        {
            "model_oof": oof_scores.loc[development.index],
            **{k: v[~sealed] for k, v in baselines.items()},
        },
        reference="reviews",
        breakdown=group_size_band(development),
        floor=floor,
    )
    sealed_scores = {
        **{
            f"{_SEED_PREFIX}{s}": lm.predict(m, held_out, features=columns)
            for s, m in models.items()
        },
        **{k: v[sealed] for k, v in baselines.items()},
    }
    # One table per baseline. The sealed fold is where the two frozen baselines swap order —
    # price+rating leads there, review-count leads on the population — so the model is reported
    # against **both**, and a paired interval only exists against the reference its table was
    # built on. Quoting the level against one baseline and the interval against the other is how
    # a report ends up naming whichever comparator flatters it.
    sealed_by_reference = {
        name: comparison_table(
            held_out["grade"],
            held_out["query_group"],
            sealed_scores,
            reference=name,
            breakdown=held_out["city"],
            floor=floor,
        )
        for name in baselines
    }
    sealed_table = sealed_by_reference["reviews"]
    seed_summary = summarise_seeds(sealed_table)
    importance = lm.feature_importance(models[seeds[0]], columns)

    if log_to_mlflow:
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="lambdamart-sealed-fold"):
            mlflow.set_tags(
                {
                    "dataset_version": version or dataset_version(),
                    "dataset_digest": dataset_digest(features_path),
                    "git_commit": git_commit(),
                    "protocol": f"sealed fold {SEALED_FOLD} of {folds}, CV on the rest",
                }
            )
            mlflow.log_params(
                {
                    **{
                        k: v
                        for k, v in {**lm.DEFAULT_PARAMS, **(params or {})}.items()
                        if k != "label_gain"
                    },
                    "label_gain": str(lm.LABEL_GAIN),
                    "folds": folds,
                    "sealed_fold": SEALED_FOLD,
                    "split_seed": split_seed,
                    "seeds": str(list(seeds)),
                    "n_features": len(columns),
                    "cv_iterations": str(iterations),
                    "n_estimators_refit": chosen,
                }
            )
            for slice_name in ("overall", "n>10"):
                key = slice_name.replace(">", "gt")
                mlflow.log_metrics(
                    {
                        f"sealed_{key}_ndcg_mean": seed_summary.loc[slice_name, "ndcg_mean"],
                        f"sealed_{key}_ndcg_sd": seed_summary.loc[slice_name, "ndcg_sd"],
                        f"sealed_{key}_floor": seed_summary.loc[slice_name, "floor"],
                        f"oof_{key}_ndcg": oof.loc[(slice_name, "model_oof"), "ndcg@10"],
                        f"baseline_reviews_{key}": oof.loc[(slice_name, "reviews"), "ndcg@10"],
                    }
                )
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                (out / "features.json").write_text(json.dumps(columns, indent=2))
                mlflow.log_artifact(str(out / "features.json"))
                mlflow.log_artifact(
                    str(plot_learning_curves(curves, iterations, out / "learning_curves.png"))
                )
                _log_frame(
                    pd.DataFrame({f"fold_{k}": pd.Series(v) for k, v in sorted(curves.items())}),
                    "learning_curves",
                    out,
                )
                for name, frame in (
                    ("fold_report", fold_report),
                    ("cv_report", cv_report),
                    ("oof_comparison", oof),
                    ("oof_by_size_band", oof_band),
                    ("sealed_comparison", sealed_table),
                    ("seed_summary", seed_summary),
                    ("feature_importance", importance),
                ):
                    _log_frame(frame, name, out)
                _log_model(models[seeds[0]], out)

    return {
        "fold_report": fold_report,
        "cv": cv_report,
        "oof": oof,
        "oof_band": oof_band,
        "sealed": sealed_table,
        "sealed_vs_price_rating": sealed_by_reference["price_rating"],
        "seeds": seed_summary,
        "importance": importance,
        # The raw out-of-fold score per development row, returned so a caller can analyse the
        # cohort structure of the predictions (notebook 04's cold-start section) without paying
        # for a second four-fold cross-validation.
        "oof_scores": oof_scores.loc[development.index],
        "curves": pd.DataFrame({f"fold_{k}": pd.Series(v) for k, v in sorted(curves.items())}),
    }


def run_sweep(
    features_path: Path = FEATURE_TABLE_PATH,
    n_configs: int = 35,
    seed: int = 0,
    split_seed: int = 0,
    output: Path = SWEEP_RESULTS_PATH,
) -> pd.DataFrame:
    """Run the random search and write the results table, losers included.

    Separated from :func:`run` because it costs 35 cross-validations and its output is a
    *decision*, not a model: which configuration the pre-declared acceptance rule admits. Written
    to disk under the same contract as the feature table — gitignored, rebuilt by one command —
    so notebook 04 displays it rather than recomputing it.
    """
    from rental_ranking.train.sweep import accept, run_search

    table = pd.read_parquet(features_path)
    check_feature_table(table)
    fold, _ = assign_folds(table, seed=split_seed)
    results, _ = run_search(table, fold, n_configs=n_configs, seed=seed)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(output)
    chosen, verdict = accept(results)
    print(f"\nsweep results -> {output}")
    print(f"\nverdict (rule declared before the search ran): {verdict}")
    return results


def _report(tables: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 96)
    print("DEVELOPMENT OUT-OF-FOLD — estimates the procedure; the per-city claim lives here")
    print("=" * 96)
    print(tables["oof"].round(4).to_string())
    print("\nby group-size band:")
    print(tables["oof_band"].round(4).to_string())

    print("\n" + "=" * 96)
    print("SEALED FOLD — estimates the artifact; scored once")
    print("=" * 96)
    print(tables["sealed"].round(4).to_string())
    print("\nacross seeds:")
    print(tables["seeds"].round(4).to_string())

    print("\ntop 15 features by gain (read against notebook 03 §2 — a third are conditioners):")
    print(tables["importance"].head(15).round(4).to_string())

    # Both baselines, both slices. The sealed fold is where the two frozen baselines swap
    # places, so quoting only the one the model beats would name the wrong comparator.
    print("\n" + "-" * 96)
    print("HEADLINE — seed 0 of the sealed-fold refit, against both baselines and each slice's")
    print("own floor. The mean across seeds is in the table above; this is the sentence form.")
    print("-" * 96)
    for key, base in (("sealed", "reviews"), ("sealed_vs_price_rating", "price_rating")):
        for slice_name in ("overall", "n>10"):
            print(
                "\n" + headline(tables[key], f"{_SEED_PREFIX}{DEFAULT_SEEDS[0]}", base, slice_name)
            )


def main() -> None:
    """Train, evaluate and log. Same entry point locally and as an Azure ML command job."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", type=Path, default=FEATURE_TABLE_PATH)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="version tag for the MLflow run. Required on Azure ML, where the asset YAML is "
        "not in the job snapshot; locally it is read from pipelines/data/feature_table.yml",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--min-data-in-leaf", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--sweep",
        type=int,
        metavar="N",
        help="run the hyperparameter search over N configurations instead of training, and "
        "write the results table (losers included) to data/train/sweep_results.csv",
    )
    args = parser.parse_args()

    if args.sweep:
        run_sweep(features_path=args.features, n_configs=args.sweep, split_seed=args.split_seed)
        return

    overrides = {
        key: value
        for key, value in (
            ("learning_rate", args.learning_rate),
            ("num_leaves", args.num_leaves),
            ("min_child_samples", args.min_data_in_leaf),
        )
        if value is not None
    }
    tables = run(
        features_path=args.features,
        folds=args.folds,
        seeds=tuple(args.seeds),
        split_seed=args.split_seed,
        params=overrides or None,
        log_to_mlflow=not args.no_mlflow,
        version=args.dataset_version,
    )
    _report(tables)


if __name__ == "__main__":
    main()
