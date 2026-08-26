"""Tests for rental_ranking.features.listing.

The two guards that matter are the ones a passing pipeline would otherwise hide:

* **Nothing is imputed.** `bedrooms` stays null. A fill would be invisible downstream — every
  shape check still passes, every model still trains — while quietly asserting a bedroom count
  for 3,343 listings whose product has no bedroom count.
* **No blocklist column reaches the matrix.** Checked against ``columns.py`` rather than a
  hand-kept list, so a column moved into ``LABEL_ADJACENT_COLUMNS`` by a future snapshot is
  enforced here without anyone remembering to update this file.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data.columns import LABEL_ADJACENT_COLUMNS
from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import listing

_DEFAULTS = {
    "id": "abc123",
    "city": "athens",
    "property_type": "Entire rental unit",
    "room_type": "Entire home/apt",
    "accommodates": 4,
    "bedrooms": 2.0,
    "beds": 3.0,
    "bathrooms": 1.0,
    "bathrooms_shared": False,
    "price": 100.0,
    "minimum_nights": 2.0,
    "maximum_nights": 365.0,
    "host_is_superhost": True,
    "host_is_local": "local",
    "host_has_about": True,
    "host_tenure_months": 40,
    "host_listings_count": 3,
    "calculated_host_listings_count": 2,
    "calculated_host_listings_count_entire_homes": 2,
    "calculated_host_listings_count_private_rooms": 0,
    "calculated_host_listings_count_shared_rooms": 0,
    "license_status": "registered",
}


def _listings(rows: list[dict] | None = None) -> pd.DataFrame:
    rows = rows or [{}]
    frame = pd.DataFrame([{**_DEFAULTS, **row} for row in rows])
    frame["amenities"] = [np.array(["Wifi", "Kitchen"], dtype=object) for _ in range(len(frame))]
    return frame


# --- building_type -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("property_type", "expected"),
    [
        ("Entire rental unit", "rental unit"),
        ("Private room in rental unit", "rental unit"),
        ("Shared room in hostel", "hostel"),
        ("Room in boutique hotel", "boutique hotel"),
        ("Entire villa", "villa"),
        ("Tiny home", "tiny home"),  # no occupancy prefix to strip
    ],
)
def test_building_type_strips_the_occupancy_prefix(property_type: str, expected: str) -> None:
    """The occupancy half is `room_type`, which is already a group-key column."""
    frame = _listings([{"property_type": property_type}])
    assert listing.building_type(frame).iloc[0] == expected


def test_building_type_separates_what_room_type_conflates() -> None:
    """An entire unit and a private room in the same building share a building type."""
    frame = _listings(
        [{"property_type": "Entire condo"}, {"property_type": "Private room in condo"}]
    )
    assert listing.building_type(frame).nunique() == 1


# --- the feature block -------------------------------------------------------------------------


def test_nothing_is_imputed_so_a_null_bedroom_count_stays_null() -> None:
    """Within a query group the null-bedrooms label percentile is indistinguishable from the rest."""
    frame = _listings([{"bedrooms": float("nan")}, {"bedrooms": 2.0}])
    out = listing.listing_features(frame)

    assert out["bedrooms"].isna().tolist() == [True, False]


def test_null_beds_and_bathrooms_survive_too() -> None:
    frame = _listings([{"beds": float("nan"), "bathrooms": float("nan")}])
    out = listing.listing_features(frame)

    assert out["beds"].isna().all()
    assert out["bathrooms"].isna().all()


def test_no_label_adjacent_column_reaches_the_feature_block() -> None:
    """Read from `columns.py`, never from a hand-kept list — see the module docstring."""
    out = listing.listing_features(_listings())

    assert set(out.columns) & LABEL_ADJACENT_COLUMNS == set()


def test_a_blocklist_column_added_to_the_column_tuple_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The realistic way one gets in: someone appends it to STRUCTURAL_COLUMNS by hand.

    A blocklist column merely *present* on the input frame is already harmless, because the
    block copies named columns rather than everything it is handed. The guard exists for the
    edit that names one — `availability_eoy` reads like a structural attribute.
    """
    blocked = sorted(LABEL_ADJACENT_COLUMNS)[0]
    monkeypatch.setattr(listing, "STRUCTURAL_COLUMNS", (*listing.STRUCTURAL_COLUMNS, blocked))
    frame = _listings()
    frame[blocked] = 1

    with pytest.raises(ValueError, match="label-adjacent"):
        listing.listing_features(frame, amenity_scheme="count")


def test_a_blocklist_column_merely_present_on_the_input_is_not_copied() -> None:
    frame = _listings()
    for blocked in LABEL_ADJACENT_COLUMNS:
        frame[blocked] = 1

    assert set(listing.listing_features(frame).columns) & LABEL_ADJACENT_COLUMNS == set()


def test_the_host_identifier_is_never_a_feature() -> None:
    """18,088 near-unique values is a route to memorising operators, not to ranking."""
    frame = _listings()
    frame["host_id"] = "host-1"
    frame["license_hash"] = "abc"
    out = listing.listing_features(frame)

    assert "host_id" not in out.columns
    assert "license_hash" not in out.columns


def test_categoricals_arrive_as_category_dtype() -> None:
    """Cast to codes instead, LightGBM reads 'Hotel room > Entire home/apt' as an inequality."""
    out = listing.listing_features(_listings())

    for column in listing.CATEGORICAL_COLUMNS:
        assert isinstance(out[column].dtype, pd.CategoricalDtype), column


def test_the_block_carries_the_id_and_the_callers_index() -> None:
    frame = _listings([{"id": "a"}, {"id": "b"}]).set_axis([4, 8])
    out = listing.listing_features(frame)

    assert out["id"].tolist() == ["a", "b"]
    assert out.index.tolist() == [4, 8]


def test_the_amenity_scheme_is_passed_through() -> None:
    out = listing.listing_features(_listings(), amenity_scheme="count")
    amenity_columns = [c for c in out.columns if c.startswith("amenity_")]

    assert amenity_columns == ["amenity_count"]


def test_a_missing_source_column_raises_a_readable_keyerror() -> None:
    with pytest.raises(KeyError, match="ranked listings"):
        listing.listing_features(_listings().drop(columns=["bathrooms"]))


# --- against the real snapshots ------------------------------------------------------------------


def test_the_block_builds_on_the_real_population() -> None:
    """Every listing keeps a row, and nothing on the blocklist arrives with it."""
    for name in ("listings", "calendar"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    from rental_ranking.data.filters import filter_listings
    from rental_ranking.features.label import occupancy_label
    from rental_ranking.features.price import impute_price

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    labels = occupancy_label(pd.read_parquet(PROCESSED_DIR / "calendar.parquet"))
    kept, _ = filter_listings(listings.merge(labels, left_on="id", right_index=True, how="inner"))
    ranked, _ = impute_price(kept)

    out = listing.listing_features(ranked)

    assert len(out) == len(ranked)
    assert out["id"].is_unique
    assert set(out.columns) & LABEL_ADJACENT_COLUMNS == set()
    assert out["price"].notna().all()  # imputed upstream
    assert out["bedrooms"].isna().any()  # and deliberately not imputed here
