"""The listing feature block: structural attributes, host attributes, and the amenity encoding.

Everything here is an attribute of the property or its operator as recorded at the scrape, so the
pre-T rule is satisfied by construction rather than by argument: none of it reads the calendar,
and none of it reads a review.

**Nothing is imputed.** ``bedrooms`` is 7.5 % null and stays null: within a query group the mean
label percentile of a null-bedrooms listing is indistinguishable from the rest, so there is
nothing for the model to exploit. The per-city gap that *is* visible is composition rather than
signal — nulls concentrate in private rooms, and ``room_type`` is part of the query-group key, so
the group differences the confound out. That is the structural difference from ``price``, whose
missingness sat on a mechanical path from the label. LightGBM learns a split direction for NaN,
which is a fitted decision; any fill is an assumed one.

**Two things in the key are conditioners, not discriminators.** ``room_type`` and the city are
constant inside almost every group and can separate no pair. They are kept because a tree uses
them to *condition* — to let price or capacity act differently inside entire homes than inside
private rooms. Expect them near the top of a split count and near the bottom of any honest "what
ranks a listing" reading.

**``property_type`` is decomposed rather than passed raw.** Its 81 values largely restate
``room_type`` plus a building type ("Entire rental unit", "Private room in rental unit"), so the
occupancy prefix is stripped and only :func:`building_type` is carried — 58 values that genuinely
vary inside a query group.

Pure ``DataFrame -> DataFrame`` transforms, no I/O and no ``main()``.
"""

import re
from collections.abc import Sequence

import pandas as pd

from rental_ranking.data.columns import LABEL_ADJACENT_COLUMNS
from rental_ranking.data.validate import require_columns
from rental_ranking.features.amenities import amenity_features

#: Structural attributes passed through unchanged. `price` arrives already imputed and is
#: the only one here that ever was; the rest keep their nulls (`bedrooms` 7.5 %, `beds` 3.7 %,
#: `bathrooms_shared` 0.1 %) and reach LightGBM as NaN.
STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "bathrooms_shared",
    "price",
    "minimum_nights",
    "maximum_nights",
)

#: Host and licence attributes. ``host_id`` is **not** among them: 18,088 near-unique values on
#: 44,684 rows is a direct route to memorising individual operators rather than learning what
#: makes a listing rank. ``license_hash`` is excluded for the same reason.
HOST_COLUMNS: tuple[str, ...] = (
    "host_is_superhost",
    "host_is_local",
    "host_has_about",
    "host_tenure_months",
    "host_listings_count",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "license_status",
)

#: Passed through as pandas categoricals so LightGBM can use its native categorical splits.
#: ``room_type`` and ``city`` are group-key columns and act as conditioners — see the module
#: docstring. The trainer must declare them; a categorical silently cast to codes becomes an
#: ordinal feature and the model reads "Hotel room > Entire home/apt" as an inequality.
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "room_type",
    "building_type",
    "host_is_local",
    "license_status",
    "city",
)

#: Occupancy prefixes stripped from ``property_type`` to leave the building type.
_OCCUPANCY_PREFIX = re.compile(r"^(entire|private room in|shared room in|room in)\s+", re.I)

_REQUIRED_COLUMNS = ("id", "city", "property_type", "room_type", "amenities")


def building_type(listings: pd.DataFrame) -> pd.Series:
    """Strip the occupancy prefix from ``property_type``, leaving what kind of building it is.

    ``property_type`` conflates two things: whether the guest gets the whole place — which
    ``room_type`` already says, and which the query-group key already conditions on — and what the
    place *is*. Only the second is new information, and unlike ``room_type`` it varies inside a
    query group, so a ranker can use it. 81 property types collapse to 58 building types.

    Args:
        listings: Frame carrying ``property_type``.

    Returns:
        A lowercase string Series aligned to ``listings``, named ``building_type``.

    Raises:
        KeyError: If ``property_type`` is absent.
    """
    require_columns(listings, ("property_type",), "listings")
    stripped = listings["property_type"].str.strip().str.replace(_OCCUPANCY_PREFIX, "", regex=True)
    return stripped.str.lower().rename("building_type")


def listing_features(
    listings: pd.DataFrame,
    amenity_scheme: str = "buckets",
    vocabulary: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assemble the structural, host and amenity features for the ranked population.

    Args:
        listings: The **filtered** ranked population with ``price`` already imputed — the frame
            ``features/build.py`` hands on. See ``_REQUIRED_COLUMNS``, ``STRUCTURAL_COLUMNS`` and
            ``HOST_COLUMNS`` for what is read.
        amenity_scheme: Passed to
            :func:`rental_ranking.features.amenities.amenity_features`.
        vocabulary: Passed through for the ``"flags"`` scheme.

    Returns:
        One row per listing, indexed as ``listings``, carrying ``id`` and the feature block.
        Categorical columns arrive as pandas ``category`` dtype; numeric ones keep their nulls.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If a ``LABEL_ADJACENT_COLUMNS`` member reaches the output.
    """
    require_columns(listings, _REQUIRED_COLUMNS, "ranked listings")
    require_columns(listings, STRUCTURAL_COLUMNS, "ranked listings")
    require_columns(listings, HOST_COLUMNS, "ranked listings")

    features = pd.DataFrame({"id": listings["id"]}, index=listings.index)
    features[list(STRUCTURAL_COLUMNS)] = listings[list(STRUCTURAL_COLUMNS)]
    features[list(HOST_COLUMNS)] = listings[list(HOST_COLUMNS)]
    features["building_type"] = building_type(listings)
    features["room_type"] = listings["room_type"]
    features["city"] = listings["city"]

    features = pd.concat([features, amenity_features(listings, amenity_scheme, vocabulary)], axis=1)
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype("category")

    # Checked here as well as at assembly, because this is the module that reads the raw frame: a
    # column added to STRUCTURAL_COLUMNS by hand is exactly how a forward availability window
    # would enter the matrix wearing a structural name.
    blocked = sorted(set(features.columns) & LABEL_ADJACENT_COLUMNS)
    if blocked:
        raise ValueError(
            f"label-adjacent column(s) reached the feature block: {blocked}. These are forward "
            "windows or direct reads of the label, and may never be "
            "model inputs"
        )
    return features
