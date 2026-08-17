"""Tests for rental_ranking.features.spatial.

Distance code fails quietly: a swapped lat/lon, a degrees-for-radians slip, or a self-match left
in a neighbour count all return numbers that look like distances. So the haversine is pinned
against pairs whose separation is known independently, and both leave-one-out features are tested
for the thing that would otherwise be invisible — that the listing never counts itself.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import spatial

_DEFAULTS = {
    "city": "athens",
    "neighbourhood_cleansed": "Kolonaki",
    "latitude": 37.9755,
    "longitude": 23.7348,
}


def _listings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_DEFAULTS, **row} for row in rows])


# --- the haversine itself --------------------------------------------------------------------


def test_haversine_matches_a_known_separation() -> None:
    """Syntagma to the White Tower is about 300 km by great circle."""
    got = spatial.haversine_km(
        np.array([37.9755]), np.array([23.7348]), np.array([40.6264]), np.array([22.9483])
    )
    assert got[0] == pytest.approx(300, abs=5)


def test_haversine_is_zero_at_the_same_point_and_symmetric() -> None:
    a, b = (np.array([35.5]), np.array([24.0]))
    c, d = (np.array([35.9]), np.array([24.6]))

    assert spatial.haversine_km(a, b, a, b)[0] == pytest.approx(0.0)
    assert spatial.haversine_km(a, b, c, d)[0] == pytest.approx(spatial.haversine_km(c, d, a, b)[0])


def test_haversine_broadcasts_to_a_matrix() -> None:
    lat = np.array([[35.5], [36.0]])
    lon = np.array([[24.0], [24.0]])
    other_lat, other_lon = np.array([[35.5, 36.0]]), np.array([[24.0, 24.0]])

    assert spatial.haversine_km(lat, lon, other_lat, other_lon).shape == (2, 2)


# --- distance to the nearest anchor ------------------------------------------------------------


def test_the_nearest_of_a_markets_anchors_wins() -> None:
    """A Chania listing must measure to Chania, not to the island's other three centres."""
    frame = _listings([{"city": "crete", "latitude": 35.5140, "longitude": 24.0185}])
    assert spatial.km_to_nearest_anchor(frame).iloc[0] < 0.1


def test_each_market_uses_its_own_anchor_set() -> None:
    frame = _listings(
        [
            {"city": "athens", "latitude": 37.9755, "longitude": 23.7348},
            {"city": "crete", "latitude": 35.3387, "longitude": 25.1442},
        ]
    )
    assert spatial.km_to_nearest_anchor(frame).lt(0.1).all()


def test_a_market_without_an_anchor_set_raises() -> None:
    """Returning NaN would hide a whole market from the feature."""
    frame = _listings([{"city": "rhodes"}])

    with pytest.raises(ValueError, match="no anchor set"):
        spatial.km_to_nearest_anchor(frame)


def test_distances_are_rounded_to_the_coordinate_precision() -> None:
    """Airbnb jitters coordinates 0-150 m; finer than 10 m is precision the source lacks."""
    frame = _listings([{"city": "crete", "latitude": 35.4, "longitude": 24.3}])
    value = spatial.km_to_nearest_anchor(frame).iloc[0]

    assert value == round(value, spatial.DISTANCE_PRECISION)


# --- distance to the neighbourhood centroid ----------------------------------------------------


def test_the_centroid_leaves_the_listing_out() -> None:
    """Including itself would drag the centroid toward the listing and shrink its own distance."""
    frame = _listings(
        [
            {"latitude": 38.0000},
            {"latitude": 38.0100},
            {"latitude": 38.0200},
        ]
    )
    out = spatial.km_to_neighbourhood_centroid(frame)

    # Row 0's centroid is the mean of rows 1 and 2 (38.015), about 1.67 km away.
    assert out.iloc[0] == pytest.approx(1.67, abs=0.05)
    # Row 1 sits exactly between rows 0 and 2, so its leave-one-out centroid is its own position.
    assert out.iloc[1] == pytest.approx(0.0, abs=0.01)


def test_a_lone_listing_has_no_neighbourhood_centroid() -> None:
    """A centroid of nothing is undefined — never the listing's own position, which reads as 0."""
    assert spatial.km_to_neighbourhood_centroid(_listings([{}])).isna().all()


def test_centroids_are_scoped_by_city_and_neighbourhood() -> None:
    frame = _listings(
        [
            {"city": "athens", "latitude": 38.00},
            {"city": "athens", "latitude": 38.02},
            {"city": "crete", "latitude": 35.50},
        ]
    )
    out = spatial.km_to_neighbourhood_centroid(frame)

    assert out.iloc[:2].notna().all()  # the two Athens listings see each other
    assert pd.isna(out.iloc[2])  # the Crete one is alone in its own unit, despite the same name


# --- local density -----------------------------------------------------------------------------


def test_density_counts_other_listings_inside_the_radius() -> None:
    """0.005 degrees of latitude is ~0.56 km; 0.05 is ~5.6 km."""
    frame = _listings(
        [
            {"latitude": 40.000},
            {"latitude": 40.005},
            {"latitude": 40.050},
        ]
    )
    assert spatial.listing_density(frame, radius_km=1.0).tolist() == [1, 1, 0]


def test_density_never_counts_the_listing_itself() -> None:
    """The self match sits at distance zero and would inflate every count by exactly one."""
    frame = _listings([{"latitude": 40.0}])
    assert spatial.listing_density(frame).tolist() == [0]


def test_density_is_computed_per_market() -> None:
    """Two listings at the same coordinates in different cities are not neighbours."""
    frame = _listings([{"city": "athens"}, {"city": "crete"}])
    assert spatial.listing_density(frame).tolist() == [0, 0]


def test_a_larger_radius_can_only_add_neighbours() -> None:
    frame = _listings([{"latitude": 40.0 + 0.004 * i} for i in range(6)])
    small = spatial.listing_density(frame, radius_km=0.5)
    large = spatial.listing_density(frame, radius_km=2.0)

    assert (large >= small).all()


# --- the assembled block ------------------------------------------------------------------------


def test_the_block_carries_all_three_features() -> None:
    frame = _listings([{"latitude": 38.0}, {"latitude": 38.01}])
    block = spatial.spatial_features(frame)

    assert list(block.columns) == [
        "km_to_nearest_anchor",
        "km_to_neighbourhood_centroid",
        "density_1km",
    ]
    assert len(block) == len(frame)


def test_a_missing_column_raises_a_readable_keyerror() -> None:
    with pytest.raises(KeyError, match="ranked listings"):
        spatial.spatial_features(_listings([{}]).drop(columns=["latitude"]))


# --- against the real snapshots -------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_ranked() -> pd.DataFrame:
    for name in ("listings", "calendar"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    from rental_ranking.data.filters import filter_listings
    from rental_ranking.features.label import occupancy_label

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    labels = occupancy_label(pd.read_parquet(PROCESSED_DIR / "calendar.parquet"))
    ranked, _ = filter_listings(listings.merge(labels, left_on="id", right_index=True, how="inner"))
    return ranked


def test_real_spatial_block_is_complete(real_ranked: pd.DataFrame) -> None:
    block = spatial.spatial_features(real_ranked)

    assert len(block) == len(real_ranked)
    assert block["km_to_nearest_anchor"].notna().all()
    assert block["density_1km"].ge(0).all()
    # Only the single one-listing neighbourhood lacks a centroid.
    assert block["km_to_neighbourhood_centroid"].isna().sum() <= 5


def test_real_anchors_sit_where_the_listings_are(real_ranked: pd.DataFrame) -> None:
    """A mistyped anchor would show up as a market whose listings are all far from it."""
    distance = spatial.km_to_nearest_anchor(real_ranked)
    per_city = distance.groupby(real_ranked["city"]).median()

    assert per_city["athens"] < 3
    assert per_city["thessaloniki"] < 3
    # Crete is a region, not a city — 84 x 248 km, so its median is legitimately far larger.
    assert per_city["crete"] < 20


def test_real_crete_needs_its_four_anchors(real_ranked: pd.DataFrame) -> None:
    """With one centre a Chania listing would read as 150 km out — the reason for the anchor set."""
    crete = real_ranked[real_ranked["city"].eq("crete")]
    four = spatial.km_to_nearest_anchor(crete)
    one = spatial.km_to_nearest_anchor(
        crete, anchors={"crete": (("heraklion", 35.33870, 25.14420),)}
    )

    assert four.max() < one.max() / 2
    assert four.median() < one.median()
