"""Loading must pin the dtypes it promises and resolve paths the way the downloader wrote them.

The dtype pinning is the reason this module exists, so most of these tests are about it: that
``int64`` refuses a null instead of accepting one, that a leading-zero licence survives, and
that a pinned column silently vanishing is caught rather than read back inferred. The path
tests guard the other half — reader and writer deriving the same filename from the same URL.

Pure tests always run. Snapshot tests skip when the gitignored raw data is absent.
"""

import gzip
from pathlib import Path
from typing import get_args

import pandas as pd
import pytest

from rental_ranking.data import io as data_io
from rental_ranking.data.download import SNAPSHOTS
from rental_ranking.data.paths import RAW_DIR

ENTITIES = get_args(data_io.Entity)


# --- the entity/city contract -----------------------------------------------------------


def test_entity_type_and_dtype_table_agree() -> None:
    """The import-time assert made executable: three lists that must not drift."""
    assert set(ENTITIES) == set(data_io._DTYPES)


def test_every_entity_is_a_downloaded_file() -> None:
    """A loadable entity that the downloader never fetches would be unloadable by definition."""
    for city in SNAPSHOTS:
        assert set(ENTITIES) == set(SNAPSHOTS[city]["files"])


def test_raw_path_rejects_an_unknown_entity() -> None:
    with pytest.raises(ValueError, match="unknown entity"):
        data_io.raw_path("athens", "hosts")


def test_raw_path_rejects_an_unknown_city() -> None:
    with pytest.raises(ValueError, match="unknown city"):
        data_io.raw_path("atlantis", "listings")


def test_load_raw_rejects_an_unknown_city() -> None:
    with pytest.raises(ValueError, match="unknown city"):
        data_io.load_raw("atlantis", "listings")


# --- path derivation --------------------------------------------------------------------


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
@pytest.mark.parametrize("entity", ENTITIES)
def test_raw_path_matches_the_download_url_basename(city: str, entity: str) -> None:
    """Reader and writer derive the filename the same way, so they cannot disagree."""
    url = SNAPSHOTS[city]["files"][entity]
    assert data_io.raw_path(city, entity).name == url.rsplit("/", 1)[-1]


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
def test_raw_path_uses_the_snapshot_release_date(city: str) -> None:
    """Callers pass a city, never a date — the one place the date is resolved."""
    path = data_io.raw_path(city, "listings")
    assert path.parent == RAW_DIR / city / SNAPSHOTS[city]["as_of"]


def test_raw_path_uses_the_data_source_spelling() -> None:
    """Inside Airbnb ships `neighbourhoods.csv`; a hardcoded US spelling would miss the file."""
    assert data_io.raw_path("athens", "neighbourhoods").name == "neighbourhoods.csv"


# --- dtype pinning ----------------------------------------------------------------------


def _write_csv(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    if name.endswith(".gz"):
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return path


@pytest.fixture
def fake_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A throwaway snapshot on disk, so dtype behaviour is tested through the real reader."""

    def _build(listings_csv: str):
        directory = tmp_path / "athens" / SNAPSHOTS["athens"]["as_of"]
        directory.mkdir(parents=True)
        _write_csv(directory, "listings.csv.gz", listings_csv)
        monkeypatch.setattr(data_io, "RAW_DIR", tmp_path)
        return directory

    return _build


def test_load_raw_reports_a_missing_snapshot_helpfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The message must name the fix; a bare pandas FileNotFoundError does not."""
    monkeypatch.setattr(data_io, "RAW_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="download"):
        data_io.load_raw("athens", "listings")


def test_ids_are_read_as_int_not_float(fake_snapshot) -> None:
    fake_snapshot("id,host_id,license\n1717622154020343858,901,0206K1\n")
    frame = data_io.load_raw("athens", "listings")
    assert frame["id"].dtype == "int64"
    assert frame["id"].iloc[0] == 1717622154020343858  # past 2**53; float would lose this


def test_a_null_id_raises_rather_than_widening_the_column(fake_snapshot) -> None:
    """int64, not nullable Int64, precisely so this fails: a widened column rehashes."""
    fake_snapshot("id,host_id,license\n101,901,0206K1\n,902,0206K2\n")
    with pytest.raises(ValueError, match="NA values"):
        data_io.load_raw("athens", "listings")


def test_leading_zero_licences_survive(fake_snapshot) -> None:
    """Numeric inference turns 00000364602 into 364602 — a different licence entirely."""
    fake_snapshot("id,host_id,license\n101,901,00000364602\n102,902,1359294\n")
    frame = data_io.load_raw("athens", "listings")
    assert frame["license"].tolist() == ["00000364602", "1359294"]


def test_a_renamed_pinned_column_is_caught(fake_snapshot) -> None:
    """pandas ignores a dtype key for an absent column, so nothing else would notice."""
    fake_snapshot("listing_ident,host_id,license\n101,901,0206K1\n")
    with pytest.raises(KeyError, match="athens/listings"):
        data_io.load_raw("athens", "listings")


def test_unpinned_columns_are_left_to_inference(fake_snapshot) -> None:
    """Only join keys and meaning-carrying identifiers are pinned; the rest stays pandas'."""
    fake_snapshot("id,host_id,license,accommodates\n101,901,0206K1,4\n")
    frame = data_io.load_raw("athens", "listings")
    assert frame["accommodates"].dtype == "int64"


# --- against the real snapshots ---------------------------------------------------------


def _require_snapshot(city: str, entity: str) -> None:
    if not data_io.raw_path(city, entity).exists():
        pytest.skip(f"raw snapshot not on disk: {city}/{entity}")


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
@pytest.mark.parametrize("entity", ENTITIES)
def test_real_snapshots_load_with_their_dtypes_pinned(city: str, entity: str) -> None:
    _require_snapshot(city, entity)
    frame = data_io.load_raw(city, entity)

    assert len(frame) > 0
    for column, dtype in data_io._DTYPES[entity].items():
        assert frame[column].dtype == dtype, (
            f"{city}/{entity}: {column} read as {frame[column].dtype}"
        )


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
def test_real_listings_match_the_column_spec(city: str) -> None:
    """Pinning must not disturb the header the schema tests assert against."""
    from rental_ranking.data import columns as cols

    _require_snapshot(city, "listings")
    assert tuple(data_io.load_raw(city, "listings").columns) == cols.LISTINGS_COLUMNS


#: Known orphan `listing_id`s per city — present in calendar/reviews, absent from listings.
#: Athens' five are recorded in the contract; one of them (…756012) also carries 2 reviews.
#: Listed rather than tolerated by threshold so that a *new* orphan fails the test.
_KNOWN_ORPHANS: dict[str, set[int]] = {
    "thessaloniki": set(),
    "crete": set(),
    "athens": {
        1340401991391301891,
        1349183601787358006,
        1361558511345756012,
        1556435361099913696,
        1556435361190155127,
    },
}


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
@pytest.mark.parametrize("entity", ["calendar", "reviews"])
def test_real_ids_join_against_listings(city: str, entity: str) -> None:
    """Pinned consistently, the raw join resolves apart from the recorded orphans.

    A dtype mismatch between the two files would not orphan a handful of ids — it would
    orphan every one of them, so this catches the failure the pinning exists to prevent
    while still describing the data honestly.
    """
    _require_snapshot(city, "listings")
    _require_snapshot(city, entity)

    listings = data_io.load_raw(city, "listings")
    child = data_io.load_raw(city, entity)

    assert listings["id"].dtype == child["listing_id"].dtype
    assert set(child["listing_id"]) - set(listings["id"]) <= _KNOWN_ORPHANS[city]


@pytest.mark.parametrize("city", sorted(SNAPSHOTS))
def test_real_licences_keep_their_leading_zeros(city: str) -> None:
    _require_snapshot(city, "listings")
    licences = data_io.load_raw(city, "listings")["license"].dropna()
    assert pd.api.types.is_string_dtype(licences)
    assert licences.str.startswith("0").any()
