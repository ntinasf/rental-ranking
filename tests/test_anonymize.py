"""Anonymization must be deterministic, join-preserving, and complete.

Three properties carry the weight here. **Determinism**: the same input and salt must give
the same digest on every run, or a re-processed snapshot silently stops joining to the old
one. **Join integrity**: a listing's hashed ``id`` must equal its hashed ``listing_id`` in the
calendar and reviews, or the whole dataset comes apart. **Completeness**: no PII column
survives the transform.

The pure tests build tiny frames and always run. The snapshot tests only run when the raw
data is present (it is gitignored), so CI stays green without it.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data import anonymize as anon
from rental_ranking.data import columns as cols
from rental_ranking.data.download import SNAPSHOTS
from rental_ranking.data.paths import RAW_DIR

SALT = "test-salt-not-the-real-one"

LISTINGS_ROWS = {
    "id": [101, 102, 103],
    "host_id": [901, 901, 902],  # host 901 owns two listings — hashing must preserve that
    "listing_url": ["u1", "u2", "u3"],
    "host_url": ["h1", "h2", "h3"],
    "host_name": ["Maria", "Nikos", "Elena"],
    "host_picture_url": ["p1", "p2", "p3"],
    "host_profile_id": ["pi1", "pi2", "pi3"],
    "host_profile_url": ["pu1", "pu2", "pu3"],
    "host_location": ["Athens, Greece", "London, United Kingdom", None],
    "host_about": ["I love hosting", "   ", None],
    "license": ["0206K13000181400", "Exempt", None],
    "name": ["Cosy flat", "Sea view", "Studio"],
}


@pytest.fixture
def listings() -> pd.DataFrame:
    return pd.DataFrame(LISTINGS_ROWS)


@pytest.fixture
def calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": [101, 101, 102],
            "date": ["2026-07-01", "2026-07-02", "2026-07-01"],
            "available": ["t", "f", "t"],
            "minimum_nights": [2, 2, 1],
            "maximum_nights": [30, 30, 60],
        }
    )


@pytest.fixture
def reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": [101, 102],
            "id": [5001, 5002],
            "date": ["2026-01-05", "2026-02-11"],
            "reviewer_id": [77, 78],
            "reviewer_name": ["Giorgos", "Anna"],
            "comments": ["Maria was a great host", "Lovely place"],
        }
    )


# --- the salt ---------------------------------------------------------------------------


def test_missing_salt_raises_rather_than_hashing_unsalted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failing open here would publish reversible digests of public, enumerable IDs."""
    monkeypatch.delenv("ANON_SALT", raising=False)
    with pytest.raises(ValueError, match="salt"):
        anon.hash_value(101)


def test_blank_salt_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANON_SALT", "   ")
    with pytest.raises(ValueError, match="salt"):
        anon.hash_value(101)


def test_salt_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANON_SALT", SALT)
    assert anon.hash_value(101) == anon.hash_value(101, SALT)


def test_a_different_salt_gives_a_different_digest() -> None:
    assert anon.hash_value(101, SALT) != anon.hash_value(101, "another-salt")


# --- the hash primitive -----------------------------------------------------------------


def test_digest_shape_is_twelve_hex_chars() -> None:
    digest = anon.hash_value(101, SALT)
    assert len(digest) == anon.HASH_LENGTH == 12
    assert set(digest) <= set("0123456789abcdef")


def test_hashing_is_deterministic_across_calls() -> None:
    assert anon.hash_value(101, SALT) == anon.hash_value(101, SALT)


def test_int_and_float_forms_of_an_id_agree() -> None:
    """One null widens an int column to float64; the digest must survive that."""
    assert anon.hash_value(101, SALT) == anon.hash_value(101.0, SALT)
    assert anon.hash_value(101, SALT) == anon.hash_value(np.int64(101), SALT)


def test_distinct_ids_give_distinct_digests() -> None:
    digests = {anon.hash_value(value, SALT) for value in range(500)}
    assert len(digests) == 500


def test_hash_series_preserves_nulls() -> None:
    """A hashed null would give every unlicensed listing one shared 'operator' digest."""
    hashed = anon.hash_series(pd.Series([101, None, 103]), SALT)
    assert hashed.isna().tolist() == [False, True, False]


def test_hash_series_matches_the_scalar_helper() -> None:
    values = pd.Series([101, 102, 101])
    hashed = anon.hash_series(values, SALT)
    assert hashed.tolist() == [anon.hash_value(v, SALT) for v in values]


def test_hash_series_maps_equal_values_to_equal_digests() -> None:
    hashed = anon.hash_series(pd.Series([901, 901, 902]), SALT)
    assert hashed.iloc[0] == hashed.iloc[1] != hashed.iloc[2]


def test_hash_series_preserves_a_non_default_index() -> None:
    values = pd.Series([101, 102], index=[7, 9])
    assert anon.hash_series(values, SALT).index.tolist() == [7, 9]


# --- the derivations --------------------------------------------------------------------


def test_host_is_local_is_three_way() -> None:
    derived = anon.host_is_local(pd.Series(["Athens, Greece", "London, United Kingdom", None]))
    assert derived.tolist() == ["local", "foreign", "unknown"]


def test_host_is_local_ignores_case() -> None:
    assert anon.host_is_local(pd.Series(["athens, greece"])).tolist() == ["local"]


def test_host_has_about_treats_blank_as_absent() -> None:
    derived = anon.host_has_about(pd.Series(["I love hosting", "", "   ", None]))
    assert derived.tolist() == [True, False, False, False]


def test_license_status_is_three_way() -> None:
    derived = anon.license_status(pd.Series(["0206K13000181400", "Exempt", None]))
    assert derived.tolist() == ["registered", "exempt", "missing"]


def test_license_status_ignores_case_and_padding() -> None:
    assert anon.license_status(pd.Series([" EXEMPT "])).tolist() == ["exempt"]


# --- listings ---------------------------------------------------------------------------


def test_listings_drops_every_pii_column(listings: pd.DataFrame) -> None:
    out = anon.anonymize_listings(listings, SALT)
    assert not (set(out.columns) & set(cols.PII_DROP_COLUMNS))


def test_listings_drops_the_derivation_sources(listings: pd.DataFrame) -> None:
    """Keeping host_location alongside host_is_local would re-expose what was generalized."""
    out = anon.anonymize_listings(listings, SALT)
    assert not ({"host_location", "host_about", "license"} & set(out.columns))


def test_listings_adds_the_derived_columns(listings: pd.DataFrame) -> None:
    out = anon.anonymize_listings(listings, SALT)
    expected = {"host_is_local", "host_has_about", "license_status", "license_hash"}
    assert expected <= set(out.columns)


def test_listings_hashes_the_id_columns(listings: pd.DataFrame) -> None:
    out = anon.anonymize_listings(listings, SALT)
    assert out["id"].tolist() == [anon.hash_value(v, SALT) for v in LISTINGS_ROWS["id"]]
    assert out["host_id"].tolist() == [anon.hash_value(v, SALT) for v in LISTINGS_ROWS["host_id"]]


def test_listings_keeps_shared_hosts_linkable(listings: pd.DataFrame) -> None:
    """The commercial-operator signal depends on one host keeping one digest."""
    out = anon.anonymize_listings(listings, SALT)
    assert out["host_id"].iloc[0] == out["host_id"].iloc[1] != out["host_id"].iloc[2]


def test_listings_leaves_a_missing_licence_unhashed(listings: pd.DataFrame) -> None:
    out = anon.anonymize_listings(listings, SALT)
    assert pd.isna(out["license_hash"].iloc[2])
    assert out["license_status"].iloc[2] == "missing"


def test_listings_is_lossless_in_rows(listings: pd.DataFrame) -> None:
    """Row exclusion is filters.py's job, never anonymization's."""
    assert len(anon.anonymize_listings(listings, SALT)) == len(listings)


def test_listings_does_not_mutate_the_input(listings: pd.DataFrame) -> None:
    before = listings.copy()
    anon.anonymize_listings(listings, SALT)
    pd.testing.assert_frame_equal(listings, before)


def test_listings_keeps_the_marketing_name(listings: pd.DataFrame) -> None:
    """`name` is marketing copy, not host PII — the contract keeps it raw."""
    assert anon.anonymize_listings(listings, SALT)["name"].tolist() == LISTINGS_ROWS["name"]


def test_listings_rejects_a_frame_missing_a_policy_column(listings: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="listings"):
        anon.anonymize_listings(listings.drop(columns=["host_about"]), SALT)


# --- calendar and reviews ---------------------------------------------------------------


def test_calendar_hashes_listing_id(calendar: pd.DataFrame) -> None:
    out = anon.anonymize_calendar(calendar, SALT)
    assert out["listing_id"].tolist() == [anon.hash_value(v, SALT) for v in [101, 101, 102]]


def test_calendar_drops_nothing(calendar: pd.DataFrame) -> None:
    """Trimming to CALENDAR_KEEP is a cleaning decision, not an anonymization one."""
    out = anon.anonymize_calendar(calendar, SALT)
    assert list(out.columns) == list(calendar.columns)
    assert len(out) == len(calendar)


def test_reviews_drops_reviewer_identity(reviews: pd.DataFrame) -> None:
    out = anon.anonymize_reviews(reviews, SALT)
    assert not (set(out.columns) & set(cols.REVIEWS_PII_DROP))


def test_reviews_keeps_its_own_id_raw(reviews: pd.DataFrame) -> None:
    """The reviews `id` identifies a review, not a person, and is the dedup key."""
    assert anon.anonymize_reviews(reviews, SALT)["id"].tolist() == [5001, 5002]


def test_reviews_keeps_comments_for_sentiment(reviews: pd.DataFrame) -> None:
    out = anon.anonymize_reviews(reviews, SALT)
    assert out["comments"].tolist() == reviews["comments"].tolist()


def test_reviews_rejects_a_listings_frame(listings: pd.DataFrame) -> None:
    """Passing the wrong entity should fail by name, not as a bare pandas KeyError."""
    with pytest.raises(KeyError, match="reviews"):
        anon.anonymize_reviews(listings, SALT)


# --- the property that matters most -----------------------------------------------------


def test_ids_still_join_across_the_three_entities(
    listings: pd.DataFrame, calendar: pd.DataFrame, reviews: pd.DataFrame
) -> None:
    """One salt, one helper — or every downstream join silently returns zero rows."""
    hashed_listings = anon.anonymize_listings(listings, SALT)
    hashed_calendar = anon.anonymize_calendar(calendar, SALT)
    hashed_reviews = anon.anonymize_reviews(reviews, SALT)

    known = set(hashed_listings["id"])
    assert set(hashed_calendar["listing_id"]) <= known
    assert set(hashed_reviews["listing_id"]) <= known


def test_a_mismatched_salt_breaks_the_join(listings: pd.DataFrame, calendar: pd.DataFrame) -> None:
    """The failure mode this module exists to prevent, asserted explicitly."""
    hashed_listings = anon.anonymize_listings(listings, SALT)
    hashed_calendar = anon.anonymize_calendar(calendar, "a-different-salt")
    assert not (set(hashed_calendar["listing_id"]) & set(hashed_listings["id"]))


# --- against the real snapshots ---------------------------------------------------------


def _snapshot(city: str, filename: str):
    path = RAW_DIR / city / SNAPSHOTS[city]["as_of"] / filename
    if not path.exists():
        pytest.skip(f"raw snapshot not on disk: {path}")
    return path


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
def test_real_listings_anonymize_without_loss(city: str) -> None:
    raw = pd.read_csv(_snapshot(city, "listings.csv.gz"), nrows=3000)
    out = anon.anonymize_listings(raw, SALT)

    assert len(out) == len(raw)
    assert not (set(out.columns) & set(cols.PII_DROP_COLUMNS))
    assert set(out["host_is_local"]) <= {"local", "foreign", "unknown"}
    assert set(out["license_status"]) <= {"registered", "exempt", "missing"}
    assert out["id"].is_unique


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
def test_real_calendar_ids_resolve_against_real_listings(city: str) -> None:
    """The 'joins break' failure would show up here first, on real ID magnitudes (>2**53)."""
    listings_ids = pd.read_csv(_snapshot(city, "listings.csv.gz"), usecols=["id"])
    calendar_ids = pd.read_csv(
        _snapshot(city, "calendar.csv.gz"), usecols=["listing_id"], nrows=200_000
    )

    hashed_listings = anon.hash_series(listings_ids["id"], SALT)
    hashed_calendar = anon.hash_series(calendar_ids["listing_id"], SALT)

    raw_orphans = set(calendar_ids["listing_id"]) - set(listings_ids["id"])
    hashed_orphans = set(hashed_calendar) - set(hashed_listings)
    assert len(hashed_orphans) == len(raw_orphans)
