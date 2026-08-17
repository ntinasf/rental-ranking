"""NDCG@k, Recall@k, and bootstrap confidence intervals over query groups.

A point estimate without variance is not a result: bootstrap over groups, report CIs, and break
results out per city and by group size.

**Everything is computed per query group and then averaged over groups**, never pooled over
listings. Pooling would weight a 2,088-listing group a thousand times more heavily than a
6-listing one, and the group is the unit a search engine is judged on.

**Gain is exponential** (``2**grade - 1``), matching LightGBM's ``lambdarank`` default, so a
number from here is comparable with what the training job reports rather than merely similar to
it. Linear gain is available for reporting but is not the default.

**Degenerate groups are counted, never quietly scored.** Two kinds arise:

* Every grade equal — any permutation scores NDCG 1.0, so including them inflates the mean
  without any ranker having done anything. Measured on the current key, 10 of 393 groups.
* Every grade zero — the ideal DCG is 0 and NDCG is 0/0, undefined.

Both are returned as NaN and reported as a count, so the headline says how many groups it is
actually about. ``include_degenerate=True`` restores the convention LightGBM uses internally.

**Ties are broken by row order, not favourably.** A tie-break that consults the grade inflates
every metric; ranking is a stable sort on the score alone, and because rows arrive in hashed-id
order the residual order is unrelated to the label.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Cut-off used for the project's headline metrics.
DEFAULT_K = 10

#: Minimum grade counted as relevant by :func:`recall_at_k`. Grades run 0-4 and 3 is the top
#: two classes — "would plausibly satisfy the search" rather than "appeared in it".
RELEVANT_GRADE = 3


def _dcg(gains: np.ndarray, k: int) -> float:
    top = gains[:k]
    return float((top / np.log2(np.arange(2, top.size + 2))).sum())


def _gain(grades: np.ndarray, exponential: bool) -> np.ndarray:
    return np.exp2(grades) - 1 if exponential else grades.astype("float64")


def ndcg_at_k(
    grades: pd.Series,
    groups: pd.Series,
    scores: pd.Series,
    k: int = DEFAULT_K,
    exponential_gain: bool = True,
    include_degenerate: bool = False,
) -> pd.Series:
    """NDCG@k for every query group, as a Series indexed by group.

    Args:
        grades: Graded relevance in ``{0..4}``.
        groups: Query-group id per row.
        scores: The ranker's score per row; higher ranks first.
        k: Cut-off.
        exponential_gain: ``2**g - 1`` when True (LightGBM's default), else ``g``.
        include_degenerate: Score groups whose grades are all equal as 1.0 rather than NaN.

    Returns:
        Float Series indexed by group id. NaN marks a degenerate group unless
        ``include_degenerate``.
    """
    frame = pd.DataFrame({"grade": grades, "group": groups, "score": scores})
    out: dict[object, float] = {}
    for group, block in frame.groupby("group", observed=True, sort=False):
        graded = block["grade"].to_numpy()
        ideal = _gain(np.sort(graded)[::-1], exponential_gain)
        ideal_dcg = _dcg(ideal, k)

        degenerate = ideal_dcg == 0 or np.all(graded == graded[0])
        if degenerate and not include_degenerate:
            out[group] = np.nan
            continue
        if ideal_dcg == 0:
            out[group] = np.nan
            continue

        # Stable sort on the negated score: ties keep row order, which is hashed-id order and
        # therefore unrelated to the grade. A tie-break that consulted the grade would inflate.
        order = np.argsort(-block["score"].to_numpy(), kind="stable")
        out[group] = _dcg(_gain(graded[order], exponential_gain), k) / ideal_dcg
    return pd.Series(out, name=f"ndcg@{k}", dtype="float64")


def recall_at_k(
    grades: pd.Series,
    groups: pd.Series,
    scores: pd.Series,
    k: int = DEFAULT_K,
    relevant_grade: int = RELEVANT_GRADE,
) -> pd.Series:
    """Share of a group's relevant listings that appear in its top ``k``.

    Relevant means ``grade >= relevant_grade``. Groups with no relevant listing return NaN —
    the denominator is zero, and scoring them 0 would punish a ranker for the grading.
    """
    frame = pd.DataFrame({"grade": grades, "group": groups, "score": scores})
    out: dict[object, float] = {}
    for group, block in frame.groupby("group", observed=True, sort=False):
        relevant = block["grade"].to_numpy() >= relevant_grade
        if not relevant.any():
            out[group] = np.nan
            continue
        order = np.argsort(-block["score"].to_numpy(), kind="stable")
        out[group] = float(relevant[order][:k].sum() / relevant.sum())
    return pd.Series(out, name=f"recall@{k}", dtype="float64")


def bootstrap_ci(
    per_group: pd.Series,
    iterations: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of a per-group metric with a percentile bootstrap CI, resampling **groups**.

    The group is the sampling unit because it is the unit the metric is defined on; resampling
    listings would treat a 2,088-listing group as 2,088 independent observations.

    Returns:
        ``(mean, low, high)``. All NaN if nothing survives.
    """
    values = per_group.dropna().to_numpy()
    if values.size == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[draws].mean(axis=1)
    tail = (1 - confidence) / 2
    return (
        float(values.mean()),
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1 - tail)),
    )


def evaluate_ranking(
    grades: pd.Series,
    groups: pd.Series,
    scores: pd.Series,
    k: int = DEFAULT_K,
    breakdown: pd.Series | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """NDCG@k and Recall@k with bootstrap CIs, overall and optionally split by ``breakdown``.

    Args:
        grades: Graded relevance.
        groups: Query-group id per row.
        scores: The ranker's score.
        k: Cut-off.
        breakdown: A per-row label to split on — pass ``city`` for the per-city table, or a
            group-size band. It must be constant within a group; the group's first value is
            used and a mixed group would be reported under whichever came first.
        seed: Bootstrap seed.

    Returns:
        One row per slice (``overall`` first), with ``groups``, ``degenerate``, ``ndcg``,
        ``ndcg_low``, ``ndcg_high``, ``recall`` and its interval.
    """
    ndcg = ndcg_at_k(grades, groups, scores, k)
    recall = recall_at_k(grades, groups, scores, k)
    degenerate = ndcg.isna()

    slices: list[tuple[str, pd.Index]] = [("overall", ndcg.index)]
    if breakdown is not None:
        by_group = breakdown.groupby(groups, observed=True).first()
        for value in sorted(by_group.dropna().unique()):
            slices.append((str(value), by_group.index[by_group.eq(value)]))

    rows = []
    for name, index in slices:
        index = ndcg.index.intersection(index)
        n_mean, n_low, n_high = bootstrap_ci(ndcg.loc[index], seed=seed)
        r_mean, r_low, r_high = bootstrap_ci(recall.loc[index], seed=seed)
        rows.append(
            {
                "slice": name,
                "groups": len(index),
                "degenerate": int(degenerate.loc[index].sum()),
                f"ndcg@{k}": n_mean,
                "ndcg_low": n_low,
                "ndcg_high": n_high,
                f"recall@{k}": r_mean,
                "recall_low": r_low,
                "recall_high": r_high,
            }
        )
    return pd.DataFrame(rows).set_index("slice")


def size_band(groups: pd.Series, edges: Sequence[int] = (0, 10, 50, 200, 10_000)) -> pd.Series:
    """Label each row with its query group's size band — the breakdown the roadmap asks for.

    Group sizes span 2 to 2,088 here, and NDCG@10 over 2,088 documents is a different
    measurement from NDCG@10 over 6. Reporting one average across both hides that.
    """
    sizes = groups.map(groups.value_counts())
    return pd.cut(
        sizes, bins=list(edges), labels=[f"{a + 1}-{b}" for a, b in zip(edges, edges[1:])]
    )
