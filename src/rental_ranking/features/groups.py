"""Grouping keys: the capacity tier, the query group built from it, and the near-twin cluster.

Three derivations that all answer "which listings are interchangeable", for three consumers —
the price cascade, the ranker, and the train/test split.

**Capacity tier** is a listing *attribute*, owned by Phase 1 because price imputation uses it
as a cascade rung. **Query groups** are an *assembly* built on top of it, owned by Phase 2.
Attribute here, assembly there — the split is recorded in docs/data_pipeline_design.md.

**The query group is the set LambdaMART compares.** A grade means nothing on its own; it means
"more relevant than the other listings in this group", and NDCG is computed inside the group.
The key is ``city x neighbourhood_cleansed x room_type x capacity_tier``, with a minimum size
of 5 and a fallback cascade shaped like the price-imputation cascade — see :func:`query_group`.
:func:`group_sizes` turns the assignment into the positional array LightGBM wants, and is the
place BUILD_GUIDE gotcha #4 is enforced.

**Cluster id** is not a filter and must never become one. Listings sharing a host, a location
and a capacity look like duplicates, but measured across 1,953 such clusters only 6.5 % share a
calendar and 15 % share a review count, and the median within-cluster spread in review count is
7. They are mostly *distinct inventory* — one operator with several identical flats in a
building — and dropping them would delete 12 % of real supply, concentrated in exactly the
commercial-operator population Phase 5 studies.

What they do create is leakage: two near-identical rows with near-identical labels (median
within-cluster label spread 0.079) let a model memorise the pair instead of learning to rank,
and at Phase 3 they would straddle a random train/test split. The remedy is a **grouped split**
— pass ``cluster_id`` as the grouping variable so every member of a cluster lands wholly in
train or wholly in test. No rows lost, the leakage closed.

Convention, matching ``rental_ranking.data``: pure transforms, no I/O and no ``main()``.
:func:`query_group` returns ``(ids, report)`` like ``filter_listings`` and ``impute_price`` —
the assignment, and the per-rung counts that document it.
"""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from rental_ranking.data.validate import require_columns
from rental_ranking.features.label import DEFAULT_PARTITION_COLS

#: Upper bounds of the capacity tiers, as `pd.cut` edges. `accommodates` takes 16 distinct
#: values with 87 % of listings at 6 or fewer, so raw values fragment the query key into 1,126
#: groups with 237 singletons; these bounds give 512 groups with 66 and a median size of 15.
#: 5-7 / 8+ rather than 5-6 / 7+ because capacity clusters on even values — only 2.7 % of
#: listings sleep exactly 7 — so `8+` opens the top tier on a party size people actually book.
CAPACITY_TIER_BOUNDS: list[float] = [0, 2, 4, 7, 100]

CAPACITY_TIER_LABELS: list[str] = ["1-2", "3-4", "5-7", "8+"]

#: Listings a query group needs before it ranks on its own key. A group of one gives LambdaMART
#: no pair to compare and contributes no gradient, and NDCG@10 over a handful of documents says
#: almost nothing. Five is the roadmap's threshold, unchanged since 2026-08-05.
MIN_GROUP_SIZE = 5

#: The fallback cascade, coarsest-last, in the shape of ``price.CASCADE``: each rung is
#: ``(name, group key)`` and each drops one dimension from the one above.
#:
#: **A group below the minimum is dissolved and its listings re-keyed at the next rung, where
#: they group among themselves.** A healthy group is never re-opened to absorb them — that would
#: dissolve a working group to rescue a broken one, and the fallback is meant to widen the
#: comparison set for the listings that lack one, not to coarsen the whole population. Each rung
#: is re-tested against the minimum, so a pooled group that is still too small falls again.
#:
#: Measured on the ranked population (44,684 listings) on 2026-08-16 — groups / singletons /
#: under 5 / median size: rung 1 alone 516 / 71 / 153 / 14; two rungs 399 / 6 / 15 / 28; all
#: three **393 / 0 / 7 / 29**, with rung usage 44,395 / 255 / 34. The third rung is worth
#: taking for 34 listings (0.08 %) because it is what reaches zero singletons.
#:
#: **Every rung must keep the grading partition's columns** — see :func:`query_group`.
GROUP_CASCADE: list[tuple[str, list[str]]] = [
    ("nbhd_room_tier", ["city", "neighbourhood_cleansed", "room_type", "capacity_tier"]),
    ("city_room_tier", ["city", "room_type", "capacity_tier"]),
    ("city_room", ["city", "room_type"]),
]

_GROUP_REQUIRED_COLUMNS = ("city", "neighbourhood_cleansed", "room_type", "accommodates")

#: Decimal places for the coordinates in the cluster key. Airbnb jitters published coordinates
#: by 0-150 m, so 4 dp (~11 m) matches listings the source has placed at the same point rather
#: than implying a precision the data does not have.
_COORDINATE_PRECISION = 4

_CLUSTER_KEY = ["host_id", "_cluster_lat", "_cluster_lon", "accommodates"]


def capacity_tier(listings: pd.DataFrame) -> pd.Series:
    """Bin ``accommodates`` into the four search-intent tiers.

    Args:
        listings: Frame carrying ``accommodates``.

    Returns:
        An ordered categorical Series aligned to ``listings``.

    Raises:
        KeyError: If ``accommodates`` is absent.
    """
    require_columns(listings, ("accommodates",), "listings")
    return pd.cut(
        listings["accommodates"],
        bins=CAPACITY_TIER_BOUNDS,
        labels=CAPACITY_TIER_LABELS,
    ).rename("capacity_tier")


def query_group(
    listings: pd.DataFrame,
    minimum: int = MIN_GROUP_SIZE,
    cascade: list[tuple[str, list[str]]] = GROUP_CASCADE,
    partition_cols: Sequence[str] = DEFAULT_PARTITION_COLS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Assemble the query groups, widening the key for any group below ``minimum``.

    The group is the search a listing is competing in, so the key is what a guest would have
    typed: a city, a neighbourhood, a room type and a party size. Groups below ``minimum`` are
    **never dropped** — the listings are re-keyed against a broader search instead, so the
    population that gets ranked stays the population the filters chose.

    **The fallback pools the fallers, it does not coarsen the survivors.** See
    :data:`GROUP_CASCADE`. The terminal rung takes whatever it is handed regardless of size:
    on the current snapshots that is 7 groups still under five rows, which is the honest
    residue of a product key, not a failure.

    **No ``Other`` collapse for rare ``room_type``s**, considered twice and rejected twice.
    Rare types are 307 listings (0.69 %), and of the 15 groups still under five rows after one
    rung, seven involve the ``8+`` capacity tier rather than the room type. Thresholding one
    factor cannot fix a *product* — the same argument that rejected a neighbourhood threshold
    on 2026-08-05 — and the second rung beats the collapse outright (393 groups / 0 singletons
    against 391 / 3) while inventing no pseudo-category.

    **The grading partition must stay a coarsening of this key**, at every rung. That is what
    makes the grade a monotone step function of the label inside a group, so the target can
    never contradict the label (docs/data_pipeline_design.md; measured at 145 inverted groups
    of 516 when it was violated). The cascade satisfies it by only ever dropping columns the
    partition does not use, and ``partition_cols`` is checked against every rung rather than
    trusted — a rung that dropped ``room_type`` would put two grading cells in one group and
    corrupt the target silently. Grading's own fallback to ``city`` is harmless here: it
    regrades a whole partition cell at once, so listings inside a group still share cut points.

    Args:
        listings: The **filtered** ranked population — see ``_GROUP_REQUIRED_COLUMNS``.
            ``capacity_tier`` is derived here rather than read, so the groups cannot be built
            on different tier bounds than the price cascade used.
        minimum: Rows a group needs to rank on its own key. Defaults to
            :data:`MIN_GROUP_SIZE`.
        cascade: Rungs as ``(name, group key)``, coarsest last. Defaults to
            :data:`GROUP_CASCADE`.
        partition_cols: The grading partition every rung must retain. Defaults to
            ``label.DEFAULT_PARTITION_COLS``; pass the same value you pass ``assign_grades``.

    Returns:
        ``(ids, report)``. ``ids`` is an integer Series aligned to ``listings``, named
        ``query_group`` and dense from 0; like ``cluster_id`` the ids are positional, so they
        are stable only within one call on one frame. ``report`` is one row per rung, indexed
        by rung name: ``key``, ``n`` (listings whose group was settled there), ``groups``,
        ``singletons``, ``under_minimum`` and ``median_size``. ``n`` sums to ``len(listings)``
        and ``groups`` to the number of query groups. ``singletons`` and ``under_minimum`` can
        only be nonzero on the terminal rung — every other rung settles a group precisely
        because it cleared the minimum.

    Warns:
        UserWarning: Once, with a count, if any listing is alone in its group after the final
            rung. Measured zero today; it is worth hearing about if a future snapshot changes
            that, because a one-document group is invisible to every metric.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If ``cascade`` is empty, or if a rung drops a ``partition_cols`` member.
    """
    require_columns(listings, _GROUP_REQUIRED_COLUMNS, "ranked listings")
    if not cascade:
        raise ValueError("cascade is empty, so no listing would be assigned a query group")
    for name, keys in cascade:
        dropped = sorted(set(partition_cols) - set(keys))
        if dropped:
            raise ValueError(
                f"cascade rung {name!r} drops {dropped}, which the grading partition "
                f"{list(partition_cols)} uses: listings in one query group would then be "
                "graded against different populations, and the grade could oppose the label "
                "inside the group LambdaMART is trained on. A rung may only drop a column the "
                "partition does not use"
            )

    keyed = listings.assign(capacity_tier=capacity_tier(listings))
    ids = pd.Series(-1, index=listings.index, dtype="int64", name="query_group")
    pending = pd.Series(True, index=listings.index)
    rows, next_id = [], 0

    for rung, (name, keys) in enumerate(cascade):
        subset = keyed.loc[pending]
        # dropna=False, and sizes taken from the same grouping that produces the codes: a null
        # neighbourhood must form its own group rather than vanish, because a row groupby drops
        # is not a row that raises — it is a row silently absent from the ranking.
        codes = subset.groupby(keys, observed=True, dropna=False).ngroup()
        settled = (
            pd.Series(True, index=subset.index)
            if rung == len(cascade) - 1
            else codes.map(codes.value_counts()).ge(minimum)
        )

        # Re-densify: the settled groups are a subset of this rung's groups, so their ngroup
        # codes have gaps, and gaps would make `ids` non-contiguous across rungs.
        local, _ = pd.factorize(codes[settled])
        assigned = subset.index[settled.to_numpy()]
        ids.loc[assigned] = local + next_id
        pending.loc[assigned] = False

        sizes = np.bincount(local) if len(local) else np.zeros(0, dtype="int64")
        next_id += len(sizes)
        rows.append(
            {
                "key": " x ".join(keys),
                "n": int(settled.sum()),
                "groups": len(sizes),
                "singletons": int((sizes == 1).sum()),
                "under_minimum": int((sizes < minimum).sum()),
                "median_size": float(np.median(sizes)) if len(sizes) else float("nan"),
            }
        )

    alone = sum(row["singletons"] for row in rows)
    if alone:
        warnings.warn(
            f"{alone} query group(s) hold a single listing after the final cascade rung "
            f"{cascade[-1][1]}: one document is no pair to compare, so those listings "
            "contribute nothing to LambdaMART's gradient and nothing to NDCG",
            stacklevel=2,
        )

    report = pd.DataFrame(rows, index=[name for name, _ in cascade])
    report.index.name = "rung"
    return ids, report


def group_sizes(query_groups: pd.Series) -> np.ndarray:
    """Run lengths of ``query_groups`` in row order — LightGBM's ``group`` argument.

    **BUILD_GUIDE gotcha #4 lives here.** LightGBM reads the group array *positionally*: it
    walks the rows in order, taking the first ``group[0]`` as one query, the next ``group[1]``
    as the next, and so on. It never sees the ids. So a frame that is not sorted by its group
    trains on queries that mix listings from different searches, and every metric still prints
    a plausible number. The sum check is the one everyone quotes; the contiguity check is the
    one that catches the mistake, because a shuffled frame still sums correctly.

    Not to be confused with :func:`cluster_sizes`, which returns a per-listing Series. This
    returns one entry per *group*, in row order, and only makes sense on a sorted frame.

    Args:
        query_groups: The ``query_group`` column of the feature matrix, in matrix row order.

    Returns:
        Group sizes as an integer array, summing to ``len(query_groups)``.

    Raises:
        ValueError: If any id is null, or if the frame is not sorted by query group.
    """
    if query_groups.isna().any():
        raise ValueError(
            f"{int(query_groups.isna().sum())} row(s) carry no query group, so they would be "
            "counted into whichever group happens to precede them"
        )

    runs = query_groups.ne(query_groups.shift()).cumsum()
    sizes = query_groups.groupby(runs).size().to_numpy()
    if len(sizes) != query_groups.nunique():
        raise ValueError(
            f"the frame is not sorted by query group: {len(sizes)} contiguous run(s) over "
            f"{query_groups.nunique()} group(s). LightGBM would slice queries across search "
            "boundaries and report metrics for groups that never existed — sort the matrix by "
            "query_group before building the group array"
        )
    # True by construction above; stated anyway because it is the invariant the roadmap names
    # and the one whose violation no downstream metric reports.
    if sizes.sum() != len(query_groups):
        raise ValueError(f"group sizes sum to {sizes.sum()} against {len(query_groups)} rows")
    return sizes


def cluster_id(listings: pd.DataFrame) -> pd.Series:
    """Identify near-twin listings: same host, same point, same capacity.

    **For splitting, never for filtering.** See the module docstring for the measurement that
    settled that — cluster members are mostly distinct units, not duplicate postings.

    Every listing gets an id, including the ~88 % that sit alone in a cluster of one, so the
    result can be passed straight to a grouped splitter without special-casing singletons.

    Args:
        listings: Frame carrying ``host_id``, ``latitude``, ``longitude`` and ``accommodates``.

    Returns:
        An integer Series aligned to ``listings``. Ids are dense and start at 0; they are
        positional, so they are stable only within one call on one frame.

    Raises:
        KeyError: If a required column is missing.
    """
    require_columns(listings, ("host_id", "latitude", "longitude", "accommodates"), "listings")
    keyed = listings.assign(
        _cluster_lat=listings["latitude"].round(_COORDINATE_PRECISION),
        _cluster_lon=listings["longitude"].round(_COORDINATE_PRECISION),
    )
    # factorize over the tuple rather than groupby.ngroup: it keeps nulls in their own group
    # instead of silently dropping the row, which would leave a listing with no split assignment.
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(keyed[_CLUSTER_KEY]))
    return pd.Series(codes, index=listings.index, name="cluster_id")


def cluster_sizes(listings: pd.DataFrame) -> pd.Series:
    """Size of the cluster each listing belongs to — 1 for a listing with no near-twin.

    Reported rather than acted on: 15.4 / 16.2 / 9.3 % of each market sits in a cluster of more
    than one, which is a finding about market structure as much as a modelling constraint.
    """
    ids = cluster_id(listings)
    return ids.map(ids.value_counts()).rename("cluster_size")
