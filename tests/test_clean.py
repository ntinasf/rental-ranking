"""Cleaning must be lossless, honest about failure, and faithful to the column spec.

Three properties carry the weight. **Losslessness**: apart from duplicate ids, every row that
goes in comes out — exclusion belongs to ``filters.py``. **Honesty**: a value that cannot be
parsed is reported, never quietly turned into a plausible one. **Spec fidelity**: the columns
dropped and kept are the ones ``columns.py`` says, not a literal that drifts from it.

The synthetic listings frame is built *from* the spec rather than hand-written, so a schema
change surfaces as a failing test instead of a stale fixture. Pure tests always run; snapshot
tests skip when the gitignored raw data is absent, so CI stays green without it.
"""

import json
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data import anonymize as anon
from rental_ranking.data import clean
from rental_ranking.data import columns as cols
from rental_ranking.data.download import SNAPSHOTS
from rental_ranking.data.paths import RAW_DIR

CITY = "thessaloniki"
BOX = clean.BOUNDING_BOXES[CITY]

#: Values for the columns `clean_listings` actually reads. Row 0 is ordinary, row 1 exercises
#: the shared/thousands-separator paths, row 2 the missing-source and half-bath paths.
_READ_VALUES = {
    "id": ["a1", "a2", "a3"],
    "last_scraped": ["2026-06-29", "2026-06-29", "2026-07-02"],
    "first_review": ["2025-01-05", None, "2026-01-01"],
    "last_review": ["2026-06-01", None, "2026-06-20"],
    "price": ["$50.00", "$1,712.00", None],
    "amenities": ['["Wifi", "Kitchen"]', "[]", None],
    "bathrooms": [1.0, np.nan, np.nan],
    "bathrooms_text": ["1 bath", "1.5 shared baths", "Half-bath"],
    "latitude": [40.60, 40.62, 40.64],
    "longitude": [22.95, 22.96, 22.97],
    "host_is_superhost": ["t", "f", "t"],
    "host_has_profile_pic": ["t", "t", "f"],
    "host_identity_verified": ["t", "f", "t"],
    "hosts_time_as_user_years": [3, 0, 10],
    "hosts_time_as_user_months": [4, 0, 11],
    "hosts_time_as_host_years": [2, 0, 9],
    "hosts_time_as_host_months": [1, 5, 0],
}

#: What `anonymize_listings` leaves behind: the raw header minus PII and the consumed
#: derivation sources, plus the four columns it adds.
_POST_ANONYMIZE_COLUMNS = (
    set(cols.LISTINGS_COLUMNS)
    - set(cols.PII_DROP_COLUMNS)
    - {"host_location", "host_about", "license"}
) | {"host_is_local", "host_has_about", "license_status", "license_hash"}


@pytest.fixture
def listings() -> pd.DataFrame:
    """A three-row frame shaped like `anonymize_listings` output, built from the spec."""
    frame = pd.DataFrame({column: [None, None, None] for column in sorted(_POST_ANONYMIZE_COLUMNS)})
    for column, values in _READ_VALUES.items():
        frame[column] = values
    return frame


@pytest.fixture
def calendar() -> pd.DataFrame:
    """Two listings: one with a full contiguous year, one deliberately short."""
    full = pd.date_range("2026-06-29", periods=365, freq="D")
    short = pd.date_range("2026-06-29", periods=300, freq="D")
    dates = list(full) + list(short)
    return pd.DataFrame(
        {
            "listing_id": ["a1"] * len(full) + ["a2"] * len(short),
            "date": [day.strftime("%Y-%m-%d") for day in dates],
            "available": ["t" if index % 2 else "f" for index in range(len(dates))],
            "minimum_nights": 2,
            "maximum_nights": 30,
        }
    )


@pytest.fixture
def reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": ["a1", "a2"],
            "id": [5001, 5002],
            "date": ["2026-01-05", "2026-02-11"],
            "comments": ["Great host", "Lovely place"],
        }
    )


# --- price ------------------------------------------------------------------------------


def test_parse_price_strips_currency_and_thousands_separators() -> None:
    parsed = clean.parse_price(pd.Series(["$50.21", "$1,712.00", "$116.00"]))
    assert parsed.tolist() == [50.21, 1712.00, 116.00]


def test_parse_price_keeps_missing_as_nan() -> None:
    """Price nullity tracks the label, so it must survive to be imputed — never filled here."""
    assert clean.parse_price(pd.Series(["$50.00", None])).isna().tolist() == [False, True]


def test_parse_price_returns_a_float_dtype() -> None:
    assert clean.parse_price(pd.Series(["$50.00"])).dtype == float


# --- dates ------------------------------------------------------------------------------


def test_parse_dates_produces_datetimes() -> None:
    parsed = clean.parse_dates(pd.Series(["2026-06-29", "2025-01-05"]))
    assert parsed.dtype == "datetime64[ns]"
    assert parsed.iloc[0] == pd.Timestamp("2026-06-29")


def test_parse_dates_passes_missing_through_silently() -> None:
    values = pd.Series(["2026-06-29", None], name="last_scraped")
    with _no_warning():
        parsed = clean.parse_dates(values)
    assert parsed.isna().tolist() == [False, True]


def test_parse_dates_warns_when_coercion_destroys_a_value() -> None:
    """`errors="coerce"` alone would empty a reformatted column in silence."""
    values = pd.Series(["2026-06-29", "29/06/2026"], name="last_scraped")
    with pytest.warns(UserWarning, match="unparseable date"):
        parsed = clean.parse_dates(values)
    assert parsed.isna().tolist() == [False, True]


# --- tenure -----------------------------------------------------------------------------


def test_tenure_months_combines_years_and_remainder() -> None:
    combined = clean.tenure_months(pd.Series([3, 0, 10]), pd.Series([4, 0, 11]))
    assert combined.tolist() == [40, 0, 131]


def test_tenure_months_is_nullable_integer() -> None:
    """Plain int64 arithmetic with a null silently yields floats."""
    assert clean.tenure_months(pd.Series([1]), pd.Series([2])).dtype == "Int64"


def test_tenure_months_rejects_a_total_in_the_remainder_field() -> None:
    """A snapshot switching to total months would otherwise inflate every tenure silently."""
    with pytest.raises(ValueError, match="0-11"):
        clean.tenure_months(pd.Series([3]), pd.Series([40]))


# --- booleans ---------------------------------------------------------------------------


def test_to_boolean_maps_the_two_tokens() -> None:
    assert clean.to_boolean(pd.Series(["t", "f"])).tolist() == [True, False]


def test_to_boolean_is_nullable_not_plain_bool() -> None:
    """astype(bool) on an object column maps NaN to True — NaN is truthy in Python."""
    converted = clean.to_boolean(pd.Series(["t", None]))
    assert converted.dtype == "boolean"
    assert converted.isna().tolist() == [False, True]


def test_to_boolean_rejects_an_unknown_token() -> None:
    """np.where(values == "t", ...) would send this to False and invent a clean column."""
    with pytest.raises(ValueError, match="unexpected boolean tokens"):
        clean.to_boolean(pd.Series(["t", "unknown"], name="host_is_superhost"))


# --- amenities --------------------------------------------------------------------------


def test_parse_amenities_returns_lists() -> None:
    parsed = clean.parse_amenities(pd.Series(['["Wifi", "Kitchen"]']))
    assert parsed.iloc[0] == ["Wifi", "Kitchen"]


def test_parse_amenities_treats_an_empty_array_as_valid() -> None:
    """`[]` is well-formed JSON and a real empty amenity list — not a parse failure."""
    with _no_warning():
        parsed = clean.parse_amenities(pd.Series(["[]"]))
    assert parsed.iloc[0] == []


def test_parse_amenities_maps_missing_to_an_empty_list() -> None:
    """Downstream code should be able to call len() without a null check."""
    assert clean.parse_amenities(pd.Series([None])).iloc[0] == []


def test_parse_amenities_warns_on_malformed_json() -> None:
    with pytest.warns(UserWarning, match="unparseable JSON"):
        parsed = clean.parse_amenities(pd.Series(["[not json"]))
    assert parsed.iloc[0] == []


# --- bathrooms --------------------------------------------------------------------------


def test_infer_bathrooms_reads_the_count_from_the_text() -> None:
    count, _ = clean.infer_bathrooms(
        pd.Series([np.nan, np.nan]), pd.Series(["1 bath", "1.5 shared baths"])
    )
    assert count.tolist() == [1.0, 1.5]


def test_infer_bathrooms_reads_half_baths_as_one_half() -> None:
    """These carry no digit; float(text.split()[0]) raises and an except would lose them."""
    count, _ = clean.infer_bathrooms(
        pd.Series([np.nan] * 3),
        pd.Series(["Half-bath", "Shared half-bath", "Private half-bath"]),
    )
    assert count.tolist() == [0.5, 0.5, 0.5]


def test_infer_bathrooms_flags_shared_baths() -> None:
    _, shared = clean.infer_bathrooms(
        pd.Series([np.nan] * 3), pd.Series(["1 bath", "1.5 shared baths", "Shared half-bath"])
    )
    assert shared.tolist() == [False, True, True]


def test_infer_bathrooms_leaves_shared_unknown_when_the_text_is_missing() -> None:
    """Absence of the word 'shared' is not evidence of a private bath."""
    _, shared = clean.infer_bathrooms(pd.Series([2.0]), pd.Series([None]))
    assert shared.isna().all()


def test_infer_bathrooms_falls_back_to_the_numeric_column() -> None:
    """bathrooms_text leads because it is the near-complete field, but it is not always there."""
    count, _ = clean.infer_bathrooms(pd.Series([2.0]), pd.Series([None]))
    assert count.tolist() == [2.0]


# --- bounding box -----------------------------------------------------------------------


def test_within_bounding_box_accepts_a_point_inside() -> None:
    inside = clean.within_bounding_box(pd.Series([40.60]), pd.Series([22.95]), BOX)
    assert inside.all()


def test_within_bounding_box_rejects_swapped_coordinates() -> None:
    """The gross error the check exists for: latitude and longitude the wrong way round."""
    inside = clean.within_bounding_box(pd.Series([22.95]), pd.Series([40.60]), BOX)
    assert not inside.any()


# --- clean_listings ---------------------------------------------------------------------


def test_clean_listings_drops_the_spec_sets(listings: pd.DataFrame) -> None:
    out = clean.clean_listings(listings, CITY)
    assert not (set(out.columns) & set(cols.ALL_NULL_COLUMNS))
    assert not (set(out.columns) & set(cols.REDUNDANT_DROP_COLUMNS))


def test_clean_listings_drops_the_consumed_sources(listings: pd.DataFrame) -> None:
    out = clean.clean_listings(listings, CITY)
    assert "last_scraped" not in out.columns
    assert not any(column.startswith("hosts_time_as_") for column in out.columns)


def test_clean_listings_keeps_the_label_adjacent_columns(listings: pd.DataFrame) -> None:
    """They are banned as model inputs but kept for validation — notebook 02 needs them."""
    out = clean.clean_listings(listings, CITY)
    assert set(cols.LABEL_ADJACENT_COLUMNS) <= set(out.columns)


def test_clean_listings_adds_the_derived_columns(listings: pd.DataFrame) -> None:
    out = clean.clean_listings(listings, CITY)
    expected = {
        "city",
        "scrape_date",
        "user_tenure_months",
        "host_tenure_months",
        "bathrooms_shared",
    }
    assert expected <= set(out.columns)


def test_clean_listings_carries_scrape_date_per_row(listings: pd.DataFrame) -> None:
    """Per row, not per city: last_scraped spans up to four days inside one market."""
    out = clean.clean_listings(listings, CITY)
    assert out["scrape_date"].dtype == "datetime64[ns]"
    assert out["scrape_date"].nunique() == 2


def test_clean_listings_computes_no_label_anchor(listings: pd.DataFrame) -> None:
    """T is min(calendar.date) per listing and belongs to Phase 1, not to a listings frame."""
    out = clean.clean_listings(listings, CITY)
    assert not {"anchor_date", "T", "label_anchor"} & set(out.columns)


def test_clean_listings_parses_the_read_columns(listings: pd.DataFrame) -> None:
    out = clean.clean_listings(listings, CITY)
    assert out["price"].tolist()[:2] == [50.0, 1712.0]
    assert out["bathrooms"].tolist() == [1.0, 1.5, 0.5]
    assert out["bathrooms_shared"].tolist() == [False, True, False]
    assert out["amenities"].tolist() == [["Wifi", "Kitchen"], [], []]
    assert out["user_tenure_months"].tolist() == [40, 0, 131]
    assert out["host_is_superhost"].tolist() == [True, False, True]


def test_clean_listings_is_lossless_in_rows(listings: pd.DataFrame) -> None:
    assert len(clean.clean_listings(listings, CITY)) == len(listings)


def test_clean_listings_does_not_mutate_the_input(listings: pd.DataFrame) -> None:
    before = listings.copy()
    clean.clean_listings(listings, CITY)
    pd.testing.assert_frame_equal(listings, before)


def test_clean_listings_deduplicates_on_id(listings: pd.DataFrame) -> None:
    """The one row-removing act permitted here: a repeated id is a scrape artefact."""
    doubled = pd.concat([listings, listings.iloc[[0]]], ignore_index=True)
    with pytest.warns(UserWarning, match="duplicate id"):
        out = clean.clean_listings(doubled, CITY)
    assert len(out) == len(listings)
    assert out["id"].is_unique


def test_clean_listings_warns_on_a_review_after_its_own_scrape_date(
    listings: pd.DataFrame,
) -> None:
    listings.loc[0, "last_review"] = "2026-12-31"
    with pytest.warns(UserWarning, match="postdates its own scrape_date"):
        clean.clean_listings(listings, CITY)


def test_clean_listings_warns_on_coordinates_outside_the_box(listings: pd.DataFrame) -> None:
    listings.loc[0, "latitude"] = 22.95  # swapped with longitude
    with pytest.warns(UserWarning, match="outside the thessaloniki bounding box"):
        clean.clean_listings(listings, CITY)


def test_clean_listings_rejects_an_unknown_city(listings: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown city"):
        clean.clean_listings(listings, "atlantis")


def test_clean_listings_rejects_a_frame_missing_a_column_it_reads(listings: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="listings"):
        clean.clean_listings(listings.drop(columns=["amenities"]), CITY)


# --- clean_calendar and clean_reviews ---------------------------------------------------


def test_clean_calendar_projects_to_the_kept_columns(calendar: pd.DataFrame) -> None:
    with pytest.warns(UserWarning):  # the deliberately short listing
        out = clean.clean_calendar(calendar)
    assert set(out.columns) == set(cols.CALENDAR_KEEP)


def test_clean_calendar_types_its_columns(calendar: pd.DataFrame) -> None:
    with pytest.warns(UserWarning):
        out = clean.clean_calendar(calendar)
    assert out["date"].dtype == "datetime64[ns]"
    assert out["available"].dtype == "boolean"


def test_clean_calendar_is_lossless_in_rows(calendar: pd.DataFrame) -> None:
    with pytest.warns(UserWarning):
        assert len(clean.clean_calendar(calendar)) == len(calendar)


def test_clean_calendar_warns_on_a_short_span(calendar: pd.DataFrame) -> None:
    with pytest.warns(UserWarning, match="not spanning 365 days"):
        clean.clean_calendar(calendar)


def test_clean_calendar_is_quiet_on_a_full_year(calendar: pd.DataFrame) -> None:
    full_only = calendar[calendar["listing_id"] == "a1"]
    with _no_warning():
        clean.clean_calendar(full_only)


def test_clean_reviews_parses_the_date(reviews: pd.DataFrame) -> None:
    out = clean.clean_reviews(reviews)
    assert out["date"].dtype == "datetime64[ns]"
    assert len(out) == len(reviews)


def test_clean_reviews_rejects_a_calendar_frame(calendar: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="reviews"):
        clean.clean_reviews(calendar)


# --- against the real snapshots ---------------------------------------------------------


def _snapshot(city: str, filename: str):
    path = RAW_DIR / city / SNAPSHOTS[city]["as_of"] / filename
    if not path.exists():
        pytest.skip(f"raw snapshot not on disk: {path}")
    return path


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
def test_real_listings_clean_without_row_loss(city: str) -> None:
    raw = pd.read_csv(_snapshot(city, "listings.csv.gz"))
    out = clean.clean_listings(anon.anonymize_listings(raw, "test-salt"), city)

    assert len(out) == len(raw)
    assert out["city"].eq(city).all()
    assert out["scrape_date"].notna().all()
    assert not (set(out.columns) & set(cols.ALL_NULL_COLUMNS))
    assert set(cols.LABEL_ADJACENT_COLUMNS) <= set(out.columns)


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
def test_real_bathrooms_recovered_from_text(city: str) -> None:
    """The text column is the near-complete one; leading with it is what closes the gap."""
    raw = pd.read_csv(_snapshot(city, "listings.csv.gz"))
    out = clean.clean_listings(anon.anonymize_listings(raw, "test-salt"), city)
    assert out["bathrooms"].isna().sum() < raw["bathrooms"].isna().sum() / 10


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
def test_real_amenities_all_parse(city: str) -> None:
    raw = pd.read_csv(_snapshot(city, "listings.csv.gz"), usecols=["amenities"])
    parsed = clean.parse_amenities(raw["amenities"])
    assert parsed.map(lambda value: isinstance(value, list)).all()
    assert json.loads(raw["amenities"].iloc[0]) == parsed.iloc[0]


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
def test_real_reviews_never_postdate_their_listing_scrape(city: str) -> None:
    """Cross-entity, so it lives here rather than inside clean_reviews."""
    listings_raw = pd.read_csv(_snapshot(city, "listings.csv.gz"), usecols=["id", "last_scraped"])
    reviews_raw = pd.read_csv(_snapshot(city, "reviews.csv.gz"), usecols=["listing_id", "date"])

    scrape = clean.parse_dates(listings_raw["last_scraped"])
    merged = reviews_raw.assign(date=clean.parse_dates(reviews_raw["date"])).merge(
        pd.DataFrame({"listing_id": listings_raw["id"], "scrape_date": scrape}),
        on="listing_id",
        how="inner",
    )
    assert (merged["date"] <= merged["scrape_date"]).all()


@contextmanager
def _no_warning():
    """Assert the block emits no warning. ``pytest.warns`` has no negative form.

    Worth asserting explicitly: a check that fires on clean data is worse than no check,
    because it teaches you to ignore the channel the real problems arrive on.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    assert not caught, f"unexpected warning(s): {[str(w.message) for w in caught]}"
