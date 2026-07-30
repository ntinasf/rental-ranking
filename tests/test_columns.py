"""The column spec must partition the schema and must match the files on disk.

The partition tests are pure and always run. The snapshot tests only run when the raw
data is present (it is gitignored), so CI stays green without it.
"""

import csv
import gzip
from pathlib import Path

import pytest

from rental_ranking.data import columns as cols
from rental_ranking.data.download import SNAPSHOTS
from rental_ranking.data.paths import RAW_DIR

DISPOSITIONS = {
    "all_null": cols.ALL_NULL_COLUMNS,
    "pii_drop": cols.PII_DROP_COLUMNS,
    "redundant_drop": cols.REDUNDANT_DROP_COLUMNS,
    "derived": frozenset(cols.DERIVED_FROM),
    "label_adjacent": cols.LABEL_ADJACENT_COLUMNS,
    "keep": cols.KEEP_COLUMNS,
}


def test_listings_header_has_no_duplicates() -> None:
    assert len(cols.LISTINGS_COLUMNS) == len(set(cols.LISTINGS_COLUMNS)) == 90


def test_dispositions_cover_every_listings_column() -> None:
    covered = set().union(*DISPOSITIONS.values())
    assert covered == set(cols.LISTINGS_COLUMNS)


def test_dispositions_are_pairwise_disjoint() -> None:
    overlaps = {
        (a, b): DISPOSITIONS[a] & DISPOSITIONS[b]
        for i, a in enumerate(DISPOSITIONS)
        for b in list(DISPOSITIONS)[i + 1 :]
        if DISPOSITIONS[a] & DISPOSITIONS[b]
    }
    assert overlaps == {}


def test_label_adjacent_columns_are_never_features() -> None:
    """The blocklist is the whole point of this module — it must not leak into KEEP."""
    assert not (cols.LABEL_ADJACENT_COLUMNS & cols.KEEP_COLUMNS)
    for leaky in ("price_quote_checkin_date", "price_quote_checkout_date", "availability_eoy"):
        assert leaky in cols.LABEL_ADJACENT_COLUMNS


def test_hashed_columns_are_kept_not_dropped() -> None:
    """Hashing is a transformation of a kept column, not a disposition of its own."""
    assert cols.HASH_COLUMNS <= cols.KEEP_COLUMNS


def test_calendar_and_reviews_keep_sets_are_subsets_of_their_headers() -> None:
    assert cols.CALENDAR_KEEP <= set(cols.CALENDAR_COLUMNS)
    assert cols.REVIEWS_KEEP <= set(cols.REVIEWS_COLUMNS)
    assert cols.REVIEWS_PII_DROP <= set(cols.REVIEWS_COLUMNS)
    assert not (cols.REVIEWS_KEEP & cols.REVIEWS_PII_DROP)


def test_calendar_carries_no_price() -> None:
    """v4.7 removed the per-date price schedule; nothing downstream may assume it."""
    assert "price" not in cols.CALENDAR_COLUMNS
    assert "adjusted_price" not in cols.CALENDAR_COLUMNS


def _snapshot_dirs() -> list[tuple[str, Path]]:
    return [(city, RAW_DIR / city / snap["as_of"]) for city, snap in SNAPSHOTS.items()]


def _header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


@pytest.mark.parametrize(
    ("entity", "filename", "expected"),
    [
        ("listings", "listings.csv.gz", cols.LISTINGS_COLUMNS),
        ("calendar", "calendar.csv.gz", cols.CALENDAR_COLUMNS),
        ("reviews", "reviews.csv.gz", cols.REVIEWS_COLUMNS),
    ],
)
@pytest.mark.parametrize(("city", "snapshot_dir"), _snapshot_dirs())
def test_downloaded_headers_match_the_spec(
    city: str, snapshot_dir: Path, entity: str, filename: str, expected: tuple[str, ...]
) -> None:
    path = snapshot_dir / filename
    if not path.exists():
        pytest.skip(f"{path} not downloaded")
    assert tuple(_header(path)) == expected, f"{city} {entity} schema drifted from the spec"
