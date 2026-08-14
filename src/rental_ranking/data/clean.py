"""Typing, parsing and derivation for the raw Inside Airbnb entities.

Runs after :mod:`rental_ranking.data.anonymize` and before concatenation. Like anonymization,
every function is a pure ``DataFrame -> DataFrame`` transform with no I/O, and the step is
**lossless apart from integrity**: the only rows it removes are exact duplicate listing IDs.
Every other suspicious value is reported through a warning and kept, because row exclusion is
an analytical decision that lives in ``filters.py`` and stays revisable without re-processing.

The column dispositions applied here come from :mod:`rental_ranking.data.columns`, never from
a fresh literal — the spec is tested against the real headers, a literal is not.

**No label anchor is computed here.** T is each listing's own ``min(calendar.date)``, which a
listings frame cannot see, and pinning a per-city T would contradict the logged decision.
``scrape_date`` is carried per row so the Phase 1 label step can anchor properly.
"""

import json
from typing import NamedTuple

import pandas as pd

from rental_ranking.data import columns as cols
from rental_ranking.data.validate import out_of_range, require_columns, warn_violations

_BOOLEAN_TOKENS = {"t": True, "f": False}

#: Columns carrying Inside Airbnb's 't'/'f' flags. ``instant_bookable`` is 100% null and
#: ``has_availability`` is constant, so both are dropped rather than parsed.
_LISTINGS_BOOLEAN_COLUMNS = ("host_is_superhost", "host_has_profile_pic", "host_identity_verified")

#: The listings date columns, all ISO-formatted. ``last_scraped`` becomes ``scrape_date``.
#: The two ``price_quote_*`` dates are typed like any other date even though they are
#: label-adjacent and banned as features: notebook 02 subtracts T from the check-in date to
#: measure the leak, and leaving them as strings pushes that parse into every consumer.
_LISTINGS_DATE_COLUMNS = (
    "last_scraped",
    "first_review",
    "last_review",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
)

_DATE_FORMAT = "%Y-%m-%d"

#: Calendar length per listing. Verified 365 for every Thessaloniki listing, but 8 listings
#: across Athens and Crete carry 366-367 rows, so this is a warning threshold, not an assert.
_EXPECTED_CALENDAR_DAYS = 365


class BoundingBox(NamedTuple):
    """Geographic bounds for a market, in decimal degrees."""

    south: float
    north: float
    west: float
    east: float


#: Generous administrative bounds per market — a sanity check, not a precision fence. Sized to
#: catch gross errors (swapped latitude/longitude, a city merged into the wrong frame) while
#: leaving legitimate outlying listings alone: tightening them to the observed extent would
#: flag ~25 real listings per snapshot and train you to ignore the warning.
BOUNDING_BOXES: dict[str, BoundingBox] = {
    "thessaloniki": BoundingBox(south=40.40, north=40.80, west=22.70, east=23.20),
    "athens": BoundingBox(south=37.60, north=38.35, west=23.00, east=24.20),
    "crete": BoundingBox(south=34.75, north=35.75, west=23.40, east=26.40),
}


def parse_price(values: pd.Series) -> pd.Series:
    """Convert Inside Airbnb's ``"$1,712.00"`` price strings to floats.

    Note what this column *is*: a dated quote for the listing's first available stay, not a
    standing nightly rate. Its missingness therefore tracks availability, which tracks the
    label — see the contract. Impute it downstream; never expose a "has price" flag.

    Args:
        values: Raw ``price`` values.

    Returns:
        A float Series; missing and unparseable entries become ``NaN``.
    """
    stripped = values.astype("string").str.replace(r"[$,]", "", regex=True)
    # Plain float64, not nullable Float64: the nullable dtypes earn their keep where a silent
    # coercion lurks (bool from NaN, int to float), and neither applies to a float column.
    return pd.to_numeric(stripped, errors="coerce").astype("float64")


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse an ISO date column, warning if parsing turned any present value into ``NaT``.

    ``errors="coerce"`` on its own is the quiet kind of dangerous: a format change would empty
    the column instead of failing. Comparing null counts before and after makes that visible
    while still keeping the run alive on a handful of bad rows.

    Args:
        values: Raw date strings.

    Returns:
        A ``datetime64`` Series.
    """
    parsed = pd.to_datetime(values, format=_DATE_FORMAT, errors="coerce")
    warn_violations(parsed.isna() & values.notna(), f"{values.name}: unparseable date")
    return parsed


def tenure_months(years: pd.Series, months: pd.Series) -> pd.Series:
    """Combine Inside Airbnb's split tenure fields into a single month count.

    The range check is the point: if a future snapshot puts a *total* in the ``_months`` field
    instead of a 0-11 remainder, ``years * 12 + months`` silently inflates every tenure and
    nothing downstream looks wrong.

    Args:
        years: Whole years of tenure.
        months: Remaining months, expected in 0-11.

    Returns:
        Total months as nullable ``Int64``.

    Raises:
        ValueError: If any month value falls outside 0-11.
    """
    if not months.dropna().between(0, 11).all():
        raise ValueError(f"{months.name}: expected a 0-11 remainder, got values outside it")
    return (years * 12 + months).astype("Int64")


def to_boolean(values: pd.Series) -> pd.Series:
    """Convert Inside Airbnb's 't'/'f' flags to real booleans, failing on unknown tokens.

    Uses ``map`` rather than ``np.where(values == "t", ...)``: the comparison form silently
    sends every unrecognised token — including ``NaN`` and a future ``"unknown"`` — to
    ``False``, producing a clean-looking column full of invented values.

    Args:
        values: Raw ``'t'``/``'f'`` flags.

    Returns:
        A nullable ``boolean`` Series. Nullable, not plain ``bool``, because ``astype(bool)``
        on an object column maps ``NaN`` to ``True`` — ``NaN`` is truthy in Python.

    Raises:
        ValueError: If a token other than ``'t'`` or ``'f'`` is present.
    """
    converted = values.map(_BOOLEAN_TOKENS)
    unknown = converted.isna() & values.notna()
    if unknown.any():
        raise ValueError(f"{values.name}: unexpected boolean tokens {sorted(set(values[unknown]))}")
    return converted.astype("boolean")


def parse_amenities(values: pd.Series) -> pd.Series:
    """Parse the amenities JSON array into a Python list per row.

    Missing entries become an empty list rather than ``NaN`` so downstream code can call
    ``len`` unconditionally. ``json.loads`` cannot be vectorised, but the ``apply`` stays an
    implementation detail — the signature is still column-in, column-out, so a future
    vectorised form would not touch a single call site.

    Args:
        values: Raw ``amenities`` JSON strings.

    Returns:
        A Series of lists; unparseable rows become empty lists and are warned about.
    """

    def _load(raw: object) -> list | None:
        if not isinstance(raw, str):
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None

    parsed = values.apply(_load)
    # Failure is tracked by the sentinel, not by "the result is empty": `[]` is valid JSON and
    # a genuinely empty amenity list, and conflating the two would report clean rows as broken.
    warn_violations(parsed.isna() & values.notna(), "amenities: unparseable JSON")
    return parsed.apply(lambda value: [] if value is None else value)


def infer_bathrooms(numeric: pd.Series, text: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Recover a bathroom count and a shared-bathroom flag from the two source columns.

    ``bathrooms_text`` leads and ``bathrooms`` is the fallback, not the other way round: the
    text column is the near-complete one (7 nulls in Athens against 711 in ``bathrooms``).

    "Half-bath", "Shared half-bath" and "Private half-bath" carry no digit and mean 0.5 — 23
    rows in Athens alone. A plain ``float(text.split()[0])`` raises on them, and swallowing
    that in an ``except`` would turn a known quantity into a missing one.

    Args:
        numeric: Raw ``bathrooms`` values.
        text: Raw ``bathrooms_text`` values.

    Returns:
        ``(count, shared)`` — a float Series and a nullable boolean Series. ``shared`` is null
        where the text is missing, since absence of the word is not evidence of a private bath.
    """
    text_missing = text.isna()

    shared = text.str.contains("shared", case=False, na=False)
    shared = shared.where(~text_missing, pd.NA).astype("boolean")

    extracted = pd.to_numeric(text.str.extract(r"(\d+\.\d+|\.\d+|\d+)")[0], errors="coerce")
    is_half = text.str.contains("half", case=False, na=False)

    count = extracted.where(~is_half, 0.5)
    count = count.fillna(numeric)  # text missing, or present but carrying no parseable count
    return count, shared


def within_bounding_box(latitude: pd.Series, longitude: pd.Series, box: BoundingBox) -> pd.Series:
    """Mask the coordinates falling inside ``box``.

    Returns a mask instead of printing, so the caller can warn once with a count. A per-row
    print in a library is a denial of service on your own terminal at 27k rows.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        box: The market's bounds.

    Returns:
        A boolean Series, ``True`` where the point lies inside the box.
    """
    return latitude.between(box.south, box.north) & longitude.between(box.west, box.east)


def clean_listings(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Type, parse and derive a single city's anonymized listings frame.

    Args:
        df: An anonymized listings frame.
        city: Market key; must be present in :data:`BOUNDING_BOXES`.

    Returns:
        A typed frame with ``city``, ``scrape_date`` and the derived columns added, and the
        empty / redundant / consumed source columns dropped.

    Raises:
        KeyError: If a column this step reads or drops is absent.
        ValueError: If ``city`` is unknown, or a source column fails its value check.
    """
    if city not in BOUNDING_BOXES:
        raise ValueError(f"unknown city {city!r}; expected one of {sorted(BOUNDING_BOXES)}")

    consumed = {
        "hosts_time_as_user_years",
        "hosts_time_as_user_months",
        "hosts_time_as_host_years",
        "hosts_time_as_host_months",
        "last_scraped",
    }
    read = {"amenities", "bathrooms", "bathrooms_text", "id", "latitude", "longitude", "price"}
    dropped = set(cols.ALL_NULL_COLUMNS) | set(cols.REDUNDANT_DROP_COLUMNS) | consumed
    require_columns(df, dropped | read | set(_LISTINGS_BOOLEAN_COLUMNS), "listings")

    out = df.copy()

    # Deduplication is the one row-removing act allowed here: a repeated `id` is a scrape
    # artefact, not an analytical outlier. Everything else is warned about and kept.
    duplicates = out["id"].duplicated()
    if warn_violations(duplicates, "listings: duplicate id, keeping first"):
        out = out[~duplicates]

    for column in _LISTINGS_DATE_COLUMNS:
        out[column] = parse_dates(out[column])
    for column in _LISTINGS_BOOLEAN_COLUMNS:
        out[column] = to_boolean(out[column])

    out["price"] = parse_price(out["price"])
    out["amenities"] = parse_amenities(out["amenities"])
    out["bathrooms"], out["bathrooms_shared"] = infer_bathrooms(
        out["bathrooms"], out["bathrooms_text"]
    )
    out["user_tenure_months"] = tenure_months(
        out["hosts_time_as_user_years"], out["hosts_time_as_user_months"]
    )
    out["host_tenure_months"] = tenure_months(
        out["hosts_time_as_host_years"], out["hosts_time_as_host_months"]
    )

    out["city"] = city
    # Per row, not per city: `last_scraped` spans up to four days inside one market (Crete runs
    # 06-29 to 07-03), so the folder's release date is not this listing's scrape date.
    out["scrape_date"] = out["last_scraped"]

    # A review cannot postdate the scrape that observed it. The upper bound is a Series, so
    # each row is checked against its own scrape date rather than a market-wide constant.
    for column in ("first_review", "last_review"):
        warn_violations(
            out_of_range(out[column], upper=out["scrape_date"]),
            f"listings: {column} postdates its own scrape_date",
        )
    warn_violations(
        out_of_range(out["first_review"], upper=out["last_review"]),
        "listings: first_review is later than last_review",
    )
    warn_violations(
        ~within_bounding_box(out["latitude"], out["longitude"], BOUNDING_BOXES[city]),
        f"listings: coordinates outside the {city} bounding box",
    )

    return out.drop(columns=sorted(dropped))


def clean_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Type an anonymized calendar frame and project it to the kept columns.

    The per-date ``minimum_nights``/``maximum_nights`` are dropped here rather than during
    anonymization: they fall inside the label window, which is an analytical reason, not a
    privacy one.

    Args:
        df: An anonymized calendar frame.

    Returns:
        A typed frame holding ``listing_id``, ``date`` and a boolean ``available``.

    Raises:
        KeyError: If an expected column is absent.
    """
    require_columns(df, cols.CALENDAR_COLUMNS, "calendar")

    out = df.copy()
    out["date"] = parse_dates(out["date"])
    out["available"] = to_boolean(out["available"])

    # Each listing should carry one contiguous year forward from its own scrape date. Verified
    # true for every Thessaloniki listing, but 8 listings across Athens and Crete run 366-367
    # days, so this reports rather than asserts — and the label window is 90 days, so a few
    # extra tail days change nothing.
    span = out.groupby("listing_id")["date"].agg(["size", "min", "max"])
    irregular = (span["size"] != _EXPECTED_CALENDAR_DAYS) | (
        (span["max"] - span["min"]).dt.days + 1 != span["size"]
    )
    warn_violations(irregular, f"calendar: listings not spanning {_EXPECTED_CALENDAR_DAYS} days")

    return out[[column for column in out.columns if column in cols.CALENDAR_KEEP]]


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Type an anonymized reviews frame.

    Args:
        df: An anonymized reviews frame.

    Returns:
        A typed frame with ``date`` parsed to ``datetime64``.

    Raises:
        KeyError: If an expected column is absent.
    """
    require_columns(df, cols.REVIEWS_KEEP, "reviews")

    out = df.copy()
    out["date"] = parse_dates(out["date"])
    return out
