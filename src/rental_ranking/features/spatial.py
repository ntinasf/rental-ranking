"""Spatial features: haversine distances to landmarks and transit.

Listings already carry lat/long; only landmark/transit coordinates are needed
(from an offline OSM extract or cached results), then vectorized haversine
locally. No online geocoding — geopy is slow and rate-limited.
"""

# TODO: define landmark/transit coordinate sets per city (computed once, cached).
# TODO: vectorized haversine distance from each listing to each landmark set.
# TODO: derive features such as distance-to-centre and distance-to-nearest-transit.
