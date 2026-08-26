"""Who reaches the first screen, against who deserves to — exposure disparity by cohort.

This module exists because **NDCG cannot see the thing it measures.** NDCG@10 asks whether the
top ten are good; it does not ask *who* is in them. A cohort can be displaced systematically
down every group without the first screen getting any worse, because the listings replacing them
are equally good. Measured, listings from large operators sit a tenth of a group lower than their
grade warrants, in most groups — and the score offset that would correct it is **zero** on every
fold. The disparity is real, consistent, dose-responsive, and invisible to the headline metric.

Four decisions are load-bearing, each a way to get this wrong:

* **Truth is the grade, not the raw label.** ``grade`` is what the ranker was trained on;
  ``blocked_fraction_90`` is finer. Commercial listings sit slightly lower *inside* their grade
  bands, so measuring against the label scores the model for a coarsening it was never asked to
  undo and confounds that with a cohort penalty.
* **Position is normalised by group length.** Groups run 11 to 2,088 listings, and ten places in
  a group of twelve is a different event from ten places in a group of two thousand.
* **The gap is paired inside a query group**, so neighbourhood, room type and capacity tier are
  held fixed by construction — they are the group key. Pooling across groups would let cohort
  composition drive the result.
* **Groups of ``k`` or fewer are excluded.** Every listing in them reaches the first screen under
  any ranking whatsoever, so each contributes an exact zero and dilutes every estimate toward
  "no effect".

**Direction is not harm.** The measured disparity runs *toward* single-listing hosts, which few
readers would call an injustice. The finding is about the blind spot rather than the cohort: the
same machinery would have hidden the same disparity along an axis where it did matter.

Pure transforms — no I/O and no ``main()``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rental_ranking.evaluate.metrics import DEFAULT_K, bootstrap_ci

#: What a listing *deserves* is read from here: the training target, deliberately — see the
#: module docstring.
DEFAULT_TRUTH = "grade"

#: Listings a cohort needs inside a group before its rate there means anything. A reach computed
#: over one or two members is a coin flip that the group-level average would carry at full
#: weight.
MIN_COHORT_MEMBERS = 3

#: Host-scale cut for the headline cohort, in ``calculated_host_listings_count``. **The threshold
#: is reported, never selected**: the gap is negative and its interval excludes zero at every cut
#: from 2 to 50, and :func:`dose_response` needs no threshold at all.
COMMERCIAL_THRESHOLD = 5


def within_group_position(
    values: pd.Series, groups: pd.Series, ascending: bool = False
) -> pd.Series:
    """Position within a query group as a fraction: 0.0 is first, 1.0 is last.

    **Ties break on row order, not favourably**, matching ``metrics.ndcg_at_k``. Rows arrive in
    hashed-id order, so the residual order is unrelated to both the grade and the cohort; a
    tie-break that consulted either would manufacture the disparity this module reports.

    Args:
        values: The quantity to order by — a model score, or a truth column.
        groups: Query-group id per row.
        ascending: Order ascending instead of descending.

    Returns:
        Float Series aligned to ``values``. NaN for any group of one, where there is no position
        to hold and a 0.0 would read as "first" rather than "undefined".
    """
    frame = pd.DataFrame({"value": values.to_numpy(), "group": groups.to_numpy()})
    by_group = frame.groupby("group", observed=True)["value"]
    rank = by_group.rank(ascending=ascending, method="first")
    size = by_group.transform("size")
    position = (rank - 1) / (size - 1)
    return pd.Series(np.where(size > 1, position, np.nan), index=values.index, name="position")


def displacement(
    scores: pd.Series,
    groups: pd.Series,
    truth: pd.Series,
) -> pd.Series:
    """How much higher the ranker places a listing than its truth warrants.

    Positive means over-ranked, negative under-ranked, and the unit is a fraction of the group's
    length — so −0.10 reads as "ten percent of the way down this search, whatever its size".

    Args:
        scores: The ranker's score per row.
        groups: Query-group id per row.
        truth: What the listing deserves — normally ``grade``; see :data:`DEFAULT_TRUTH`.

    Returns:
        Float Series aligned to ``scores``, NaN for groups of one.
    """
    ideal = within_group_position(truth, groups)
    actual = within_group_position(scores, groups)
    return (ideal - actual).rename("displacement")


def informative_groups(groups: pd.Series, k: int = DEFAULT_K) -> pd.Series:
    """Rows whose group is large enough for first-screen exposure to be a question at all.

    In a group of ``k`` or fewer every listing reaches the first screen under any ranking, a
    shuffle included, so such groups carry no information about who the ranker favours — and each
    contributes an exact zero to every disparity, pulling the average toward "no effect".

    Returns:
        Boolean Series aligned to ``groups``.
    """
    sizes = groups.groupby(groups, observed=True).transform("size")
    return sizes.gt(k).rename("informative")


def cohort_gap(
    displaced: pd.Series,
    groups: pd.Series,
    cohort: pd.Series,
    min_members: int = MIN_COHORT_MEMBERS,
) -> pd.Series:
    """Per-group difference in mean displacement, cohort minus everyone else.

    **Paired inside the group**, which is what makes the number attributable to the cohort rather
    than to where the cohort happens to live: both sides come from one query group, so the city,
    neighbourhood, room type and capacity tier are identical by construction — they are the key.

    Args:
        displaced: Per-row displacement from :func:`displacement`.
        groups: Query-group id per row.
        cohort: Boolean mask selecting the cohort under examination.
        min_members: Members each side needs before the group contributes.

    Returns:
        Float Series indexed by group id, holding only groups with ``min_members`` on both
        sides. Negative means the cohort is ranked below the rest of its own search.
    """
    frame = pd.DataFrame(
        {
            "displacement": displaced.to_numpy(),
            "group": groups.to_numpy(),
            "cohort": cohort.to_numpy().astype(bool),
        }
    ).dropna(subset=["displacement"])

    stats = frame.groupby(["group", "cohort"], observed=True)["displacement"].agg(["mean", "size"])
    wide = stats.unstack("cohort")
    if wide.empty or True not in wide["mean"].columns or False not in wide["mean"].columns:
        return pd.Series(dtype="float64", name="cohort_gap")

    enough = (wide["size"][True] >= min_members) & (wide["size"][False] >= min_members)
    gap = wide["mean"][True] - wide["mean"][False]
    return gap[enough.fillna(False)].rename("cohort_gap")


def exposure_amplification(
    scores: pd.Series,
    groups: pd.Series,
    truth: pd.Series,
    cohort: pd.Series,
    k: int = DEFAULT_K,
    min_members: int = MIN_COHORT_MEMBERS,
) -> pd.DataFrame:
    """First-screen reach under the ranker, against reach under a perfect one.

    **The reference is the ideal ranking, not a shuffle.** A cohort can reach the first screen
    less often simply by being less in demand, and a shuffle reference would report that as the
    ranker's doing. Ordering by ``truth`` gives the reach the cohort has *earned*; only the
    difference is attributable to the model.

    Args:
        scores: The ranker's score per row.
        groups: Query-group id per row.
        truth: What the listing deserves — normally ``grade``.
        cohort: Boolean mask selecting the cohort.
        k: First-screen size.
        min_members: Members a group needs before it contributes a rate.

    Returns:
        One row per cohort side (``cohort``, ``rest``): ``groups``, ``reach``, ``earned``,
        ``amplification`` (reach minus earned) and its bootstrap interval.
    """
    frame = pd.DataFrame(
        {
            "score": scores.to_numpy(),
            "group": groups.to_numpy(),
            "truth": truth.to_numpy(),
            "cohort": cohort.to_numpy().astype(bool),
        },
        index=scores.index,
    )
    by_group = frame.groupby("group", observed=True)
    frame["reached"] = by_group["score"].rank(ascending=False, method="first").le(k)
    frame["earned"] = by_group["truth"].rank(ascending=False, method="first").le(k)

    rows = []
    for name, side in (("cohort", frame["cohort"]), ("rest", ~frame["cohort"])):
        per_group = (
            frame[side]
            .groupby("group", observed=True)
            .agg(reach=("reached", "mean"), earned=("earned", "mean"), members=("reached", "size"))
        )
        per_group = per_group[per_group["members"] >= min_members]
        mean, low, high = bootstrap_ci(per_group["reach"] - per_group["earned"])
        rows.append(
            {
                "side": name,
                "groups": len(per_group),
                "reach": per_group["reach"].mean(),
                "earned": per_group["earned"].mean(),
                "amplification": mean,
                "amplification_low": low,
                "amplification_high": high,
            }
        )
    return pd.DataFrame(rows).set_index("side")


def dose_response(
    displaced: pd.Series,
    groups: pd.Series,
    dose: pd.Series,
    bands: Sequence[float],
    labels: Sequence[str] | None = None,
    min_members: int = MIN_COHORT_MEMBERS,
) -> pd.DataFrame:
    """Mean displacement per band of a continuous cohort variable — **no threshold at all**.

    The answer to "did you pick the cut that gave you the finding": a monotone trend across bands
    is evidence a single threshold cannot supply.

    Args:
        displaced: Per-row displacement from :func:`displacement`.
        groups: Query-group id per row.
        dose: The continuous cohort variable — here ``calculated_host_listings_count``.
        bands: ``pd.cut`` edges over ``dose``.
        labels: Band labels; defaults to ``pd.cut``'s intervals.
        min_members: Members a group needs before it contributes a band mean.

    Returns:
        One row per band: ``listings``, ``groups``, ``displacement`` and its bootstrap interval.
    """
    # The band stays categorical rather than being flattened to an array: pd.cut returns an
    # *ordered* categorical, and groupby preserves that order. Passing .to_numpy() would drop
    # the dtype and the rows would come back sorted alphabetically by label — turning a
    # monotone dose response into a scramble that still looks like a table.
    banded = pd.cut(dose, bins=list(bands), labels=labels)
    frame = pd.DataFrame(
        {
            "displacement": displaced.to_numpy(),
            "group": groups.to_numpy(),
            "band": pd.Categorical(banded, categories=banded.cat.categories, ordered=True),
        }
    ).dropna(subset=["displacement"])

    rows = []
    for band, block in frame.groupby("band", observed=True):
        per_group = block.groupby("group", observed=True)["displacement"].agg(["mean", "size"])
        per_group = per_group[per_group["size"] >= min_members]
        mean, low, high = bootstrap_ci(per_group["mean"])
        rows.append(
            {
                "band": band,
                "listings": len(block),
                "groups": len(per_group),
                "displacement": mean,
                "low": low,
                "high": high,
            }
        )
    return pd.DataFrame(rows).set_index("band")
