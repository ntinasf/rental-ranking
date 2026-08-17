"""Spatial features: distance to a market's anchors, to its own neighbourhood, and local density.

**Crete is not a city.** Its bounding box is 84 x 248 km against Athens' 9 x 7, so a single
"distance to the centre" is meaningless there — it would place a Chania apartment 150 km from
downtown and rank it as remote. Each market therefore gets an **anchor set**, and the feature is
the distance to the *nearest* one. Verified against listing density on the ranked population,
the four Crete anchors claim 10,089 / 6,201 / 5,920 / 3,567 listings, which is a coherent split
of the island's four sub-markets rather than an artefact of where the points were placed.

**Coordinates are pinned constants, never geocoded at runtime.** No ``geopy``, no network call:
the landmarks do not move, and a feature that depends on a rate-limited service is not
reproducible. Airbnb jitters published coordinates by 0-150 m, so distances are rounded to
:data:`DISTANCE_PRECISION` decimal places of a kilometre — anything finer is precision the
source does not have.

**Two of the three features are conditioners.** Measured within-group variance ratios:
``km_to_nearest_anchor`` and ``density_1km`` are near-constant inside a query group, because a
group is one neighbourhood and geography is smooth across it. They condition rather than rank —
the same role as ``room_type`` and ``city``. The one that discriminates is
``km_to_neighbourhood_centroid``: it measures how peripheral a listing is *within its own
municipality*, which is exactly the variation a group preserves.

**Density is an aggregate over listings, so step 4's rule applies**: it is a leave-one-out count
by construction — a listing is never its own neighbour. It is legitimate and structural (it
counts supply, not the target), unlike the neighbourhood mean-label aggregate that
``features/aggregates.py`` refuses to build.

**A stated limitation in Crete.** Distance to the nearest town centre conflates "remote
countryside villa" with "beach resort 25 km from Heraklion" — the second is a destination, not a
remote spot. ``km_to_neighbourhood_centroid`` is the complement that does not suffer from it,
because it is measured against the listing's own municipality rather than against a town.

Convention, matching ``rental_ranking.data``: pure transforms, no I/O and no ``main()``.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from rental_ranking.data.validate import require_columns

#: Mean Earth radius (km), the standard spherical approximation.
EARTH_RADIUS_KM = 6371.0088

#: Decimal places kept on every distance. Airbnb jitters coordinates by 0-150 m, so 2 dp (10 m)
#: already claims more than the source supports; it exists to stop a float tail being read as
#: signal, and to keep the features stable under an irrelevant recomputation.
DISTANCE_PRECISION = 2

#: Radius of the local-supply count, in km.
DENSITY_RADIUS_KM = 1.0

#: Per-market anchor sets, ``city -> ((name, latitude, longitude), ...)``.
#:
#: One anchor each for the two cities, whose bounding boxes are 13.5 x 17.8 and 9.0 x 7.0 km.
#: **Four for Crete**, which is a region: Chania, Rethymno, Heraklion and Agios Nikolaos are its
#: four population and tourism centres, and the nearest-anchor split reproduces the island's
#: sub-markets. Thessaloniki's anchor sits between the White Tower and Aristotelous Square, the
#: two ends of its seafront core.
#:
#: Precision is deliberate: these are town centres to ~100 m, which is finer than the 0-150 m
#: jitter already present in every listing's own coordinates.
ANCHORS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "thessaloniki": (("thessaloniki_centre", 40.62945, 22.94470),),
    "athens": (("syntagma", 37.97550, 23.73480),),
    "crete": (
        ("chania", 35.51380, 24.01800),
        ("rethymno", 35.36620, 24.47770),
        ("heraklion", 35.33870, 25.14420),
        ("agios_nikolaos", 35.19110, 25.71680),
    ),
}

#: The unit neighbourhood centroids are computed over, matching ``aggregates.NEIGHBOURHOOD_KEY``.
NEIGHBOURHOOD_KEY: tuple[str, ...] = ("city", "neighbourhood_cleansed")

_REQUIRED_COLUMNS = ("city", "latitude", "longitude", *NEIGHBOURHOOD_KEY)


def haversine_km(
    lat: np.ndarray, lon: np.ndarray, other_lat: np.ndarray, other_lon: np.ndarray
) -> np.ndarray:
    """Great-circle distance in km, broadcasting over numpy arrays.

    Args:
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.
        other_lat: Latitudes to measure to, broadcastable against ``lat``.
        other_lon: Longitudes to measure to, broadcastable against ``lon``.

    Returns:
        Distances in km, of the broadcast shape.
    """
    phi1, phi2 = np.radians(lat), np.radians(other_lat)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(np.asarray(other_lon) - np.asarray(lon))
    inner = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(inner))


def km_to_nearest_anchor(
    listings: pd.DataFrame, anchors: dict[str, tuple[tuple[str, float, float], ...]] = ANCHORS
) -> pd.Series:
    """Distance to the closest of the listing's own market's anchors.

    The full ``n x k`` matrix per market, then a row minimum — ``k`` is at most 4, so the matrix
    is trivial and there is no reason to loop over listings.

    **The identity of the nearest anchor is deliberately not returned.** Measured, it varies
    inside only 43 of 393 query groups, it is constant outside Crete by construction, and within
    Crete 79 % of neighbourhoods map to exactly one anchor — so it is a coarsening of
    ``neighbourhood_cleansed``, which is already a query-group key column. The distance carries
    the geography continuously; the label would only repeat a key.

    Args:
        listings: Frame carrying ``city``, ``latitude`` and ``longitude``.
        anchors: Per-market anchor sets. Defaults to :data:`ANCHORS`.

    Returns:
        A float Series aligned to ``listings``, named ``km_to_nearest_anchor``.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If a city in ``listings`` has no anchor set — silently returning NaN would
            hide a whole market from the feature.
    """
    require_columns(listings, ("city", "latitude", "longitude"), "listings")

    missing = sorted(set(listings["city"].unique()) - set(anchors))
    if missing:
        raise ValueError(
            f"no anchor set for {missing}; every market needs one, because a single "
            "'distance to the centre' is meaningless in a region such as Crete"
        )

    out = pd.Series(np.nan, index=listings.index, dtype="float64")
    for city, points in anchors.items():
        rows = listings["city"].eq(city).to_numpy()
        if not rows.any():
            continue
        lat = listings["latitude"].to_numpy()[rows][:, None]
        lon = listings["longitude"].to_numpy()[rows][:, None]
        anchor_lat = np.array([point[1] for point in points])[None, :]
        anchor_lon = np.array([point[2] for point in points])[None, :]
        out.iloc[np.flatnonzero(rows)] = haversine_km(lat, lon, anchor_lat, anchor_lon).min(axis=1)
    return out.round(DISTANCE_PRECISION).rename("km_to_nearest_anchor")


def km_to_neighbourhood_centroid(
    listings: pd.DataFrame, key: tuple[str, ...] = NEIGHBOURHOOD_KEY
) -> pd.Series:
    """Distance to the **leave-one-out** centroid of the listing's own neighbourhood.

    How peripheral a listing is within its own municipality — the one spatial feature that
    genuinely varies inside a query group, because a group *is* a neighbourhood and this measures
    position within it.

    The centroid excludes the listing itself, per step 4's rule: it is an aggregate over
    listings, and including the row moves the target it is measured against. Unlike the
    neighbourhood mean-label aggregate the correction is safe here, because the aggregated
    quantity is position rather than the target. Mean coordinates rather than a true spherical
    centroid: over a municipality the difference is far below the 0-150 m coordinate jitter.

    Returns NaN for a listing alone in its neighbourhood — a centroid of nothing is undefined,
    not the listing's own position, which would silently read as distance zero.
    """
    require_columns(listings, ("latitude", "longitude", *key), "listings")

    unit = listings[list(key)].astype(str).agg("|".join, axis=1)
    grouped = listings.groupby(unit, observed=True, dropna=False)
    size = grouped["latitude"].transform("size")
    centroid_lat = (grouped["latitude"].transform("sum") - listings["latitude"]) / (size - 1)
    centroid_lon = (grouped["longitude"].transform("sum") - listings["longitude"]) / (size - 1)

    distance = haversine_km(
        listings["latitude"].to_numpy(),
        listings["longitude"].to_numpy(),
        centroid_lat.to_numpy(),
        centroid_lon.to_numpy(),
    )
    return (
        pd.Series(distance, index=listings.index)
        .round(DISTANCE_PRECISION)
        .rename("km_to_neighbourhood_centroid")
    )


def listing_density(listings: pd.DataFrame, radius_km: float = DENSITY_RADIUS_KM) -> pd.Series:
    """How many **other** listings sit within ``radius_km`` — local supply, leave-one-out.

    A KD-tree per market on a local equirectangular projection, rather than a 44,684 x 44,684
    haversine matrix, which would need 16 GB. Over a single market at these latitudes the
    projection is accurate to well under a percent at 1 km — far inside the coordinate jitter —
    and the tree turns an O(n^2) scan into O(n log n).

    Leave-one-out by construction: the listing is its own nearest neighbour at distance zero, so
    the self match is subtracted. Zero is a real answer, not a missing one.

    Args:
        listings: Frame carrying ``city``, ``latitude`` and ``longitude``.
        radius_km: Search radius.

    Returns:
        An integer Series aligned to ``listings``, named ``density_<radius>km``.

    Raises:
        KeyError: If a required column is missing.
    """
    require_columns(listings, ("city", "latitude", "longitude"), "listings")

    out = pd.Series(0, index=listings.index, dtype="int64")
    for city in listings["city"].unique():
        rows = listings["city"].eq(city).to_numpy()
        lat = listings["latitude"].to_numpy()[rows]
        lon = listings["longitude"].to_numpy()[rows]

        # Project once per market, about its own mean latitude, so x and y are both in km.
        reference = np.radians(lat.mean())
        x = np.radians(lon) * EARTH_RADIUS_KM * np.cos(reference)
        y = np.radians(lat) * EARTH_RADIUS_KM
        points = np.c_[x, y]

        tree = cKDTree(points)
        counts = tree.query_ball_point(points, r=radius_km, return_length=True) - 1
        out.iloc[np.flatnonzero(rows)] = counts
    return out.rename(f"density_{radius_km:g}km")


def spatial_features(
    listings: pd.DataFrame,
    anchors: dict[str, tuple[tuple[str, float, float], ...]] = ANCHORS,
    radius_km: float = DENSITY_RADIUS_KM,
) -> pd.DataFrame:
    """Assemble the spatial block.

    Args:
        listings: The ranked population — see ``_REQUIRED_COLUMNS``.
        anchors: Passed to :func:`km_to_nearest_anchor`.
        radius_km: Passed to :func:`listing_density`.

    Returns:
        A frame aligned to ``listings`` with ``km_to_nearest_anchor``,
        ``km_to_neighbourhood_centroid`` and the density column.

    Raises:
        KeyError: If a required column is missing.
    """
    require_columns(listings, _REQUIRED_COLUMNS, "ranked listings")
    return pd.concat(
        [
            km_to_nearest_anchor(listings, anchors),
            km_to_neighbourhood_centroid(listings),
            listing_density(listings, radius_km),
        ],
        axis=1,
    )
