"""Tests for rental_ranking.features.groups.

The cluster id exists to be handed to a grouped splitter, so the properties that matter are
coverage (every listing gets one) and separation (unrelated listings never share one). Both
fail silently if wrong — a dropped row simply never reaches the split.
"""

import pandas as pd
import pytest

from rental_ranking.data.filters import filter_listings
from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import groups
from rental_ranking.features.label import occupancy_label

_DEFAULTS = {"host_id": "h1", "latitude": 40.6401, "longitude": 22.9444, "accommodates": 4}


def _listings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_DEFAULTS, **row} for row in rows])


# --- capacity tiers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("guests", "tier"),
    [(1, "1-2"), (2, "1-2"), (3, "3-4"), (4, "3-4"), (5, "5-7"), (7, "5-7"), (8, "8+"), (16, "8+")],
)
def test_capacity_tier_boundaries(guests: int, tier: str) -> None:
    """7 belongs with 5-6, not with 8+: only 2.7 % of listings sleep exactly 7."""
    assigned = groups.capacity_tier(_listings([{"accommodates": guests}]))
    assert assigned.iloc[0] == tier


def test_capacity_tier_covers_every_listing() -> None:
    frame = _listings([{"accommodates": n} for n in range(1, 17)])
    assert groups.capacity_tier(frame).notna().all()


def test_capacity_tier_requires_its_column() -> None:
    with pytest.raises(KeyError, match="listings"):
        groups.capacity_tier(pd.DataFrame({"city": ["athens"]}))


# --- cluster ids ---------------------------------------------------------------------------


def test_same_host_point_and_capacity_share_a_cluster() -> None:
    ids = groups.cluster_id(_listings([{}, {}]))
    assert ids.nunique() == 1


def test_a_different_host_at_the_same_address_is_a_different_cluster() -> None:
    """A block of flats holds many owners; the address alone does not make them twins."""
    ids = groups.cluster_id(_listings([{}, {"host_id": "h2"}]))
    assert ids.nunique() == 2


def test_a_different_capacity_is_a_different_cluster() -> None:
    ids = groups.cluster_id(_listings([{}, {"accommodates": 6}]))
    assert ids.nunique() == 2


def test_coordinates_are_matched_at_four_decimals() -> None:
    """~11 m. Airbnb jitters published coordinates 0-150 m, so finer would be false precision."""
    same = groups.cluster_id(_listings([{}, {"latitude": 40.640104}]))
    apart = groups.cluster_id(_listings([{}, {"latitude": 40.6501}]))

    assert same.nunique() == 1
    assert apart.nunique() == 2


def test_every_listing_receives_an_id_including_singletons() -> None:
    """A row without an id would silently never reach the grouped split."""
    frame = _listings([{}, {"host_id": "h2"}, {"host_id": "h3", "accommodates": 9}])
    ids = groups.cluster_id(frame)

    assert ids.notna().all()
    assert len(ids) == len(frame)
    assert ids.index.equals(frame.index)


def test_a_null_host_does_not_collapse_rows_into_one_cluster() -> None:
    """groupby drops null keys; two unrelated listings must not be merged by a missing host."""
    ids = groups.cluster_id(
        _listings([{"host_id": None, "latitude": 40.1}, {"host_id": None, "latitude": 41.9}])
    )
    assert ids.nunique() == 2


def test_cluster_sizes_count_the_membership() -> None:
    sizes = groups.cluster_sizes(_listings([{}, {}, {"host_id": "h2"}]))
    assert sizes.tolist() == [2, 2, 1]


# --- against the real snapshots -------------------------------------------------------------


def test_real_clusters_are_distinct_inventory_not_duplicate_postings() -> None:
    """The measurement that kept 12 % of the market out of the filter.

    If cluster members were one unit posted twice they would share a calendar and a review
    history. They do not — which is why `cluster_id` feeds the split and never `filters.py`.
    """
    for name in ("listings", "calendar"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    labels = occupancy_label(pd.read_parquet(PROCESSED_DIR / "calendar.parquet"))
    ranked, _ = filter_listings(listings.merge(labels, left_on="id", right_index=True, how="inner"))

    ranked = ranked.assign(cluster=groups.cluster_id(ranked))
    multi = ranked[ranked["cluster"].map(ranked["cluster"].value_counts()) > 1]
    per_cluster = multi.groupby("cluster").agg(
        label_spread=("blocked_fraction_90", "std"),
        review_counts=("number_of_reviews", "nunique"),
    )

    assert ranked["cluster"].notna().all()
    # Distinct inventory: most clusters differ in both calendar and review history.
    assert per_cluster["label_spread"].eq(0).mean() < 0.15
    assert per_cluster["review_counts"].eq(1).mean() < 0.25
