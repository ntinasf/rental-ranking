"""Random search over LambdaMART hyperparameters, selected on paired out-of-fold NDCG.

**The selection metric is the thing that matters here, not the optimiser.** Measured
2026-08-18, the four development folds score the default configuration at 0.7026 / 0.7632 /
0.7078 / 0.7119 — mean 0.7214, **sd 0.0281**, so the 4-fold mean carries a 95 % interval of
+/- 0.0276. Real hyperparameter differences on this plateau are 0.005-0.015. Any scheme that
picks a winner by comparing 4-fold means is picking noise, however clever the optimiser.

So configurations are compared on the **per-group out-of-fold vector** — one NDCG per
development query group, 311 of them, every configuration evaluated on identically the same
groups by identically the same folds — and ranked by the **paired** bootstrap difference against
the default. Pairing is what makes the comparison possible: two similar configurations agree
closely group by group, so differencing inside the resample removes most of the variance that
swamps the unpaired levels.

**Why random search and not Bayesian optimisation.** TPE's advantage is sample efficiency when
an evaluation is expensive; ours costs about three minutes. Its assumption is that an observed
score is informative about the configuration, and at this noise level it is not — TPE would
converge confidently onto a lucky fold draw. Random search also produces the artefact the
roadmap actually asks for: a table with the losers in it.

**Why no early-termination policy.** Bandit, median-stopping and truncation-selection compare a
trial's *intermediate* metric against its peers. A trial here is four fits pooled into one
number — there is no trajectory to act on, pruning on a single fold would prune on sd 0.0281 of
noise, and 35 trials take 90 minutes, so there is nothing to save. The per-trial termination
that does belong is LightGBM's own early stopping, which every fit already uses.

What survives from that family is :data:`CATASTROPHE_MARGIN` — a guard, not a selection rule. A
configuration is abandoned after its first fold only if it is **three standard deviations** below
the best first fold seen, which catches a genuinely broken configuration and cannot plausibly
discard a plateau winner. Pruned configurations stay in the results table marked ``pruned`` so it
remains a complete record.

**The acceptance rule is declared before the search runs.** Keep the defaults unless the
winner's paired interval against them excludes zero. "Tuning did not clear the noise floor" is a
legitimate outcome and, given a ``best_iteration`` that ranged 158-718 while NDCG stayed inside a
0.06 band, a likely one.

**The winner's out-of-fold gain is optimistically biased** — it is the maximum of ~35 draws, and
the maximum of noise is positive. The sealed fold is the only unbiased measure of what tuning
bought, and by declaration (2026-08-18) it is read exactly twice in this project: once for the
defaults, once for the winner.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from rental_ranking.evaluate.metrics import ndcg_at_k
from rental_ranking.evaluate.report import paired_difference
from rental_ranking.features.assemble import feature_columns
from rental_ranking.train import lambdamart as lm
from rental_ranking.train.split import dev_cv_splits

#: How far below the best first-fold score a configuration must be before it is abandoned, in
#: units of the measured fold-to-fold standard deviation (0.0281). Three is deliberately
#: generous: the guard exists to skip broken configurations, not to choose between plausible
#: ones. Set to ``None`` to evaluate every configuration on every fold.
CATASTROPHE_MARGIN = 3 * 0.0281

#: The search space. Two of the eight are ranking-specific and are the interesting half:
#:
#: * ``lambdarank_truncation_level`` (LightGBM default 30) decides how far down a ranked list
#:   pairs still generate gradient. With query groups up to 2,088 rows and a cut-off of 10, this
#:   is the parameter that speaks directly to the measured gradient/metric inversion — 86.7 % of
#:   the label-differing pairs sit in the 31 groups carrying 7.9 % of the metric.
#: * ``lambdarank_norm`` normalises lambdas per query, which is the other side of the same
#:   asymmetry.
#:
#: ``subsample`` is always paired with ``subsample_freq=1`` because row bagging without a
#: frequency does nothing — see ``lambdamart.is_stochastic``. Turning ``subsample`` and
#: ``colsample_bytree`` below 1 also makes the fit stochastic, which gives the five-seed
#: protocol something to measure for the first time.
SEARCH_SPACE: dict[str, Callable[[np.random.Generator], object]] = {
    "num_leaves": lambda rng: int(np.exp(rng.uniform(np.log(15), np.log(127)))),
    "min_child_samples": lambda rng: int(np.exp(rng.uniform(np.log(5), np.log(100)))),
    "learning_rate": lambda rng: float(np.exp(rng.uniform(np.log(0.02), np.log(0.1)))),
    "colsample_bytree": lambda rng: float(rng.uniform(0.5, 1.0)),
    "subsample": lambda rng: float(rng.uniform(0.5, 1.0)),
    "reg_lambda": lambda rng: float(np.exp(rng.uniform(np.log(0.01), np.log(10.0)))),
    "lambdarank_truncation_level": lambda rng: int(rng.choice([10, 20, 30, 50, 100])),
    "lambdarank_norm": lambda rng: bool(rng.choice([True, False])),
}


def sample_configs(n: int, seed: int = 0, space: dict | None = None) -> list[dict[str, object]]:
    """Draw ``n`` configurations, the first of which is always the defaults.

    The defaults are included as configuration 0 rather than compared to from outside, so the
    reference is evaluated by exactly the same code on exactly the same folds as its challengers.

    Args:
        n: Number of configurations including the defaults. Must be at least 1.
        seed: Draw seed.
        space: Samplers keyed by parameter, defaulting to :data:`SEARCH_SPACE`.

    Returns:
        Configurations as parameter dicts. Index 0 is ``{}``, meaning
        :data:`~rental_ranking.train.lambdamart.DEFAULT_PARAMS` unchanged.
    """
    if n < 1:
        raise ValueError(f"n={n}: the search must include at least the default configuration")
    rng = np.random.default_rng(seed)
    samplers = space if space is not None else SEARCH_SPACE
    configs: list[dict[str, object]] = [{}]
    for _ in range(n - 1):
        drawn = {name: sample(rng) for name, sample in samplers.items()}
        if "subsample" in drawn:
            drawn["subsample_freq"] = 1
        configs.append(drawn)
    return configs


def evaluate_config(
    table: pd.DataFrame,
    fold: pd.Series,
    params: dict[str, object],
    features: Sequence[str] | None = None,
    seed: int = 0,
    guard: float | None = None,
) -> tuple[pd.Series | None, list[float], list[int]]:
    """Cross-validate one configuration, abandoning it if the first fold is catastrophic.

    Args:
        table: The full feature table.
        fold: Fold ids; the sealed fold is excluded by ``dev_cv_splits``.
        params: Overrides on :data:`~rental_ranking.train.lambdamart.DEFAULT_PARAMS`.
        features: Column list, defaulting to ``feature_columns``.
        seed: Fit seed, held constant across configurations so the comparison is not confounded.
        guard: Abandon after fold 1 if its NDCG falls below this. ``None`` evaluates every fold.

    Returns:
        ``(oof, fold_ndcgs, iterations)``. ``oof`` is ``None`` when the guard fired — the
        configuration has no complete out-of-fold vector and cannot be ranked.
    """
    columns = list(features) if features is not None else feature_columns(table)
    blocks, scores, iterations = [], [], []

    for train_index, valid_index in dev_cv_splits(fold):
        train, valid = table.loc[train_index], table.loc[valid_index]
        model = lm.fit(train, valid, params=params, seed=seed, features=columns)
        predicted = lm.predict(model, valid, features=columns)
        blocks.append(predicted)
        iterations.append(int(model.best_iteration_ or model.n_estimators))
        scores.append(float(ndcg_at_k(valid["grade"], valid["query_group"], predicted).mean()))

        if guard is not None and len(scores) == 1 and scores[0] < guard:
            return None, scores, iterations

    return pd.concat(blocks).rename("score"), scores, iterations


def run_search(
    table: pd.DataFrame,
    fold: pd.Series,
    n_configs: int = 35,
    seed: int = 0,
    fit_seed: int = 0,
    features: Sequence[str] | None = None,
    margin: float | None = CATASTROPHE_MARGIN,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Run the search and rank every configuration against the defaults.

    Returns:
        ``(results, oof)``. ``results`` has one row per configuration — its parameters, status,
        per-fold NDCG, out-of-fold NDCG, and the paired difference against configuration 0 with
        its interval — sorted best first, **pruned rows included**. ``oof`` maps configuration
        index to its out-of-fold score vector, for whatever the caller wants to do next.
    """
    columns = list(features) if features is not None else feature_columns(table)
    configs = sample_configs(n_configs, seed=seed)

    rows: list[dict[str, object]] = []
    vectors: dict[int, pd.Series] = {}
    best_first_fold = -np.inf

    for index, params in enumerate(configs):
        guard = None if margin is None or index == 0 else best_first_fold - margin
        oof, folds, iterations = evaluate_config(
            table, fold, params, features=columns, seed=fit_seed, guard=guard
        )
        if folds:
            best_first_fold = max(best_first_fold, folds[0])

        row: dict[str, object] = {
            "config": index,
            "status": "default" if index == 0 else ("complete" if oof is not None else "pruned"),
            **{key: value for key, value in params.items()},
            "folds_run": len(folds),
            "fold_ndcg_1": folds[0] if folds else np.nan,
            "iterations": str(iterations),
        }
        if oof is not None:
            vectors[index] = oof
            per_group = ndcg_at_k(
                table.loc[oof.index, "grade"], table.loc[oof.index, "query_group"], oof
            )
            row["oof_ndcg"] = float(per_group.mean())
            row["groups"] = int(per_group.notna().sum())
            row["fold_sd"] = float(np.std(folds, ddof=1))
        rows.append(row)
        if verbose:
            print(
                f"  [{index:>2}/{len(configs) - 1}] {row['status']:<8} "
                f"oof {row.get('oof_ndcg', float('nan')):.4f}  fold1 {row['fold_ndcg_1']:.4f}",
                flush=True,
            )

    reference = ndcg_at_k(
        table.loc[vectors[0].index, "grade"], table.loc[vectors[0].index, "query_group"], vectors[0]
    )
    for row in rows:
        index = row["config"]
        if index in vectors and index != 0:
            challenger = ndcg_at_k(
                table.loc[vectors[index].index, "grade"],
                table.loc[vectors[index].index, "query_group"],
                vectors[index],
            )
            mean, low, high = paired_difference(challenger, reference)
            row["vs_default"], row["vs_low"], row["vs_high"] = mean, low, high
        elif index == 0:
            row["vs_default"] = row["vs_low"] = row["vs_high"] = 0.0

    results = pd.DataFrame(rows).set_index("config")
    return results.sort_values("oof_ndcg", ascending=False, na_position="last"), vectors


def accept(results: pd.DataFrame) -> tuple[int, str]:
    """Apply the pre-declared acceptance rule: beat the defaults with an interval clear of zero.

    **Declared before the search ran.** The winner's out-of-fold gain is the maximum of many
    draws and the maximum of noise is positive, so a point estimate is not enough — the paired
    interval against the defaults has to exclude zero before a configuration replaces them.

    Returns:
        ``(config index, one-line verdict)``. Index 0 means the defaults were kept.
    """
    complete = results[results["status"].eq("complete") & results["vs_low"].notna()]
    if complete.empty:
        return 0, "no configuration completed; defaults kept"

    best = complete.sort_values("vs_default", ascending=False).iloc[0]
    if best["vs_low"] > 0:
        return int(best.name), (
            f"config {int(best.name)} accepted: paired +{best['vs_default']:.4f} "
            f"[{best['vs_low']:+.4f}, {best['vs_high']:+.4f}] over the defaults, interval clear "
            "of zero"
        )
    return 0, (
        f"defaults kept: the best challenger (config {int(best.name)}) gained "
        f"{best['vs_default']:+.4f} [{best['vs_low']:+.4f}, {best['vs_high']:+.4f}], an interval "
        "that covers zero — the search did not clear the noise floor"
    )
