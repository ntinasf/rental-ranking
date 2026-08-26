"""The comparison table: every ranker, every slice, identically sliced.

``metrics.py`` answers "what did this ranker score". This module answers "did the model beat the
frozen baselines, and by how much, on data it has not seen" — a question with three ways of going
quietly wrong.

**1. The comparison must be same-groups.** A baseline scored on the whole population does not
describe a sealed fold — the baselines move enough between folds to swap order — so the
comparator is always recomputed on the groups being reported, and :func:`comparison_table` slices
every ranker with the same index.

**2. The floor is per slice, not a constant.** A random ranking scores far higher in small groups
than in large ones, so quoting one floor beside a sliced number misstates the result.
:func:`random_floor` is recomputed per slice.

**3. Much of the unsliced metric is measured where nothing can be shown.** In groups of ``k`` or
fewer the cut-off excludes nothing, so NDCG@k scores the whole list and reads high for every
ranker alike — including a random one. Those groups are a small share of the listings and a large
share of the metric's weight, so averaging them in shrinks the measurable improvement. The
``n > k`` slice is therefore produced **automatically**: a report that omits it understates the
result.

Pairing matters for the same reason. Both rankers see the same groups, so their errors are
correlated and an unpaired interval on a *difference* is materially too wide.
:func:`comparison_table` reports the paired difference against a chosen reference beside the
levels.

Pure transforms — no I/O, no ``main()``.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from rental_ranking.evaluate.metrics import DEFAULT_K, bootstrap_ci, ndcg_at_k

#: Random draws averaged into the floor. A single permutation varies across seeds by enough to
#: matter next to the differences being reported.
FLOOR_DRAWS = 20

#: Name the floor is reported under, and the name :func:`comparison_table` gives the random
#: control if the caller does not supply one.
FLOOR_NAME = "random"


def random_floor(
    grades: pd.Series,
    groups: pd.Series,
    k: int = DEFAULT_K,
    draws: int = FLOOR_DRAWS,
    seed: int = 0,
) -> pd.Series:
    """Per-group NDCG@k of a random ranking, averaged over ``draws`` permutations.

    Returns:
        Float Series indexed by group id, degenerate groups NaN — the same index
        :func:`~rental_ranking.evaluate.metrics.ndcg_at_k` returns, so it slices identically.
    """
    rng = np.random.default_rng(seed)
    per_draw = [
        ndcg_at_k(grades, groups, pd.Series(rng.random(len(grades)), index=grades.index), k=k)
        for _ in range(draws)
    ]
    return pd.concat(per_draw, axis=1).mean(axis=1).rename(f"{FLOOR_NAME}@{k}")


def paired_difference(
    treatment: pd.Series,
    reference: pd.Series,
    iterations: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean difference of two per-group metrics with a paired bootstrap CI.

    **Groups are resampled once and both rankers read the same resample.** The two are evaluated
    on identical groups, so their errors are correlated and an unpaired interval on the difference
    is wider than the truth.

    Args:
        treatment: Per-group metric for the ranker under test.
        reference: Per-group metric for the comparator, indexed by the same group ids.
        iterations: Bootstrap resamples.
        confidence: Interval width.
        seed: Bootstrap seed.

    Returns:
        ``(mean_difference, low, high)``, all NaN if no group carries both.
    """
    aligned = pd.DataFrame({"t": treatment, "r": reference}).dropna()
    if aligned.empty:
        return (np.nan, np.nan, np.nan)

    delta = (aligned["t"] - aligned["r"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, delta.size, size=(iterations, delta.size))
    means = delta[draws].mean(axis=1)
    tail = (1 - confidence) / 2
    return (
        float(delta.mean()),
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1 - tail)),
    )


def metric_slices(
    groups: pd.Series,
    k: int = DEFAULT_K,
    breakdown: pd.Series | None = None,
) -> dict[str, pd.Index]:
    """The slices every ranker is reported on, as group-id indexes.

    Always produced: ``overall``, ``n>k`` and ``n<=k``. The last two are not optional — see the
    module docstring.

    Args:
        groups: Query-group id per row.
        k: The NDCG cut-off the slices are defined against.
        breakdown: A per-row label — ``city`` or a group-size band — appended as further slices.
            Must be constant within a group; the group's first value is used.

    Returns:
        Ordered mapping of slice name to the group ids it covers.
    """
    sizes = groups.value_counts()
    slices: dict[str, pd.Index] = {
        "overall": sizes.index,
        f"n>{k}": sizes.index[sizes > k],
        f"n<={k}": sizes.index[sizes <= k],
    }
    if breakdown is not None:
        by_group = breakdown.groupby(groups, observed=True).first()
        for value in sorted(by_group.dropna().unique(), key=str):
            slices[str(value)] = by_group.index[by_group.eq(value)]
    return slices


def comparison_table(
    grades: pd.Series,
    groups: pd.Series,
    scores: Mapping[str, pd.Series],
    reference: str | None = None,
    k: int = DEFAULT_K,
    breakdown: pd.Series | None = None,
    floor: pd.Series | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Score every ranker on every slice, with floors, range shares and paired differences.

    Args:
        grades: Graded relevance per row.
        groups: Query-group id per row.
        scores: ``{ranker name: score Series}``. Pass the model and every baseline together —
            that is what makes their slicing identical rather than merely intended to be.
        reference: Ranker to take paired differences against, typically the leading baseline.
            ``None`` omits the difference columns.
        k: NDCG cut-off.
        breakdown: Extra slices — ``city``, or :func:`~rental_ranking.train.split.group_size_band`.
        floor: Precomputed per-group random floor, to avoid recomputing 20 draws per call. Built
            with :func:`random_floor` when omitted.
        seed: Seed for the floor and both bootstraps.

    Returns:
        A frame indexed by ``(slice, ranker)`` with ``groups``, ``degenerate``, ``ndcg@k`` and
        its interval, ``floor``, ``range_share`` — the share of ``floor``-to-1 the ranker
        traversed, which is the honest denominator — and, when ``reference`` is given,
        ``vs_<reference>`` with a paired interval.

    Raises:
        KeyError: If ``reference`` is not a key of ``scores``.
    """
    if reference is not None and reference not in scores:
        raise KeyError(f"reference {reference!r} is not one of the rankers: {sorted(scores)}")

    per_group = {name: ndcg_at_k(grades, groups, score, k=k) for name, score in scores.items()}
    if floor is None:
        floor = random_floor(grades, groups, k=k, seed=seed)

    rows = []
    for slice_name, index in metric_slices(groups, k=k, breakdown=breakdown).items():
        floor_slice = floor.loc[floor.index.intersection(index)]
        floor_mean = float(floor_slice.mean()) if len(floor_slice) else np.nan

        for name, values in per_group.items():
            held = values.loc[values.index.intersection(index)]
            mean, low, high = bootstrap_ci(held, seed=seed)
            row = {
                "slice": slice_name,
                "ranker": name,
                "groups": int(held.notna().sum()),
                "degenerate": int(held.isna().sum()),
                f"ndcg@{k}": mean,
                "ndcg_low": low,
                "ndcg_high": high,
                "floor": floor_mean,
                "range_share": (mean - floor_mean) / (1 - floor_mean)
                if np.isfinite(mean) and np.isfinite(floor_mean)
                else np.nan,
            }
            if reference is not None:
                delta, delta_low, delta_high = paired_difference(
                    held, per_group[reference].loc[held.index], seed=seed
                )
                row[f"vs_{reference}"] = delta
                row["vs_low"] = delta_low
                row["vs_high"] = delta_high
            rows.append(row)

    return pd.DataFrame(rows).set_index(["slice", "ranker"])


def headline(table: pd.DataFrame, ranker: str, reference: str, slice_name: str = "overall") -> str:
    """The result sentence, written from the table rather than around a number.

    The framing is fixed before the number exists: a bare "0.68 vs 0.64" is true and misleading,
    because the floor is not zero. This renders level, comparator, that slice's own floor, and the
    share of the usable range — so the sentence cannot be tuned to flatter.

    Args:
        table: Output of :func:`comparison_table`, built with ``reference``.
        ranker: The ranker to describe.
        reference: The comparator the table was built against.
        slice_name: Which slice to quote.

    Returns:
        One sentence, ready to paste into a report.
    """
    row = table.loc[(slice_name, ranker)]
    base = table.loc[(slice_name, reference)]
    metric = next(c for c in table.columns if c.startswith("ndcg@"))
    sentence = (
        f"{metric.upper()} {row[metric]:.4f} [{row['ndcg_low']:.4f}, {row['ndcg_high']:.4f}] "
        f"against {reference} at {base[metric]:.4f} and a random floor of {row['floor']:.4f} "
        f"on {int(row['groups'])} groups ({slice_name})"
    )
    # The paired difference exists only against the reference the table was built with. Asking for
    # another baseline is reasonable, so this degrades to the levels rather than raising: the
    # per-group vectors a paired interval needs are not in the table.
    if f"vs_{reference}" in table.columns:
        sentence += (
            f"; the difference is {row[f'vs_{reference}']:+.4f} "
            f"[{row['vs_low']:+.4f}, {row['vs_high']:+.4f}] paired, and it traverses "
            f"{row['range_share']:.1%} of the floor-to-1 range against "
            f"{base['range_share']:.1%} for {reference}."
        )
    else:
        sentence += (
            f"; it traverses {row['range_share']:.1%} of the floor-to-1 range against "
            f"{base['range_share']:.1%} for {reference} (no paired interval: this table was "
            "built against a different reference)."
        )
    return sentence


def unpaired_difference(
    treatment: pd.Series,
    reference: pd.Series,
    iterations: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Difference of two per-group metrics computed on **different** groups.

    The counterpart to :func:`paired_difference`, and the right tool when the two sets share no
    members — comparing one slice of groups against another rather than two rankers on the same
    groups. Each set is resampled independently, because there is no pairing to exploit and
    pretending otherwise would understate the interval.

    Returns:
        ``(mean_difference, low, high)``, all NaN if either side is empty.
    """
    left = treatment.dropna().to_numpy()
    right = reference.dropna().to_numpy()
    if left.size == 0 or right.size == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    means = left[rng.integers(0, left.size, size=(iterations, left.size))].mean(axis=1) - right[
        rng.integers(0, right.size, size=(iterations, right.size))
    ].mean(axis=1)
    tail = (1 - confidence) / 2
    return (
        float(left.mean() - right.mean()),
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1 - tail)),
    )
