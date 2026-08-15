"""Grouping keys: the capacity tier, and the cluster a listing shares with its near-twins.

Two derivations that both answer "which listings are interchangeable", for different consumers.

**Capacity tier** is a listing *attribute*, owned by Phase 1 because price imputation uses it
as a cascade rung. **Query groups** are an *assembly* built on top of it, owned by Phase 2.
Attribute here, assembly there — the split is recorded in docs/data_pipeline_design.md.

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

Convention, matching ``rental_ranking.data``: pure ``DataFrame -> Series`` transforms, no I/O.
"""

import pandas as pd

from rental_ranking.data.validate import require_columns

#: Upper bounds of the capacity tiers, as `pd.cut` edges. `accommodates` takes 16 distinct
#: values with 87 % of listings at 6 or fewer, so raw values fragment the query key into 1,126
#: groups with 237 singletons; these bounds give 512 groups with 66 and a median size of 15.
#: 5-7 / 8+ rather than 5-6 / 7+ because capacity clusters on even values — only 2.7 % of
#: listings sleep exactly 7 — so `8+` opens the top tier on a party size people actually book.
CAPACITY_TIER_BOUNDS: list[float] = [0, 2, 4, 7, 100]

CAPACITY_TIER_LABELS: list[str] = ["1-2", "3-4", "5-7", "8+"]

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
