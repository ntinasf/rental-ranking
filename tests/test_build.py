"""The orchestrator's job is the checks no single-entity transform can perform.

Concatenation itself is pandas' problem. What is tested here is everything guarding it: that
a schema difference between cities *stops* the run rather than being unioned away, that a
mis-salted entity is caught instead of producing a table nothing joins to, and that a hash
shared between two cities fails rather than silently merging two listings.

Each guard is tested in both directions — that it fires when it should, and stays quiet when
it should not. A guard only proven to pass on good data has not been proven to work at all.

The pure tests always run. The end-to-end test needs the gitignored raw snapshots and writes
to a tmp directory, never to the real ``data/processed/``.
"""

import warnings
from contextlib import contextmanager

import pandas as pd
import pytest

from rental_ranking.data import build
from rental_ranking.data.download import SNAPSHOTS

SALT = "test-salt-not-the-real-one"


@contextmanager
def _no_warning():
    """Assert the block emits no warning; ``pytest.warns`` has no negative form."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    assert not caught, f"unexpected warning(s): {[str(w.message) for w in caught]}"


def _listings(ids: list[str], city: str = "thessaloniki") -> pd.DataFrame:
    return pd.DataFrame({"id": ids, "city": city, "price": 50.0})


def _calendar(listing_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"listing_id": listing_ids, "date": "2026-06-29", "available": True})


# --- the entity contract ----------------------------------------------------------------


def test_processed_entities_are_all_loadable() -> None:
    """The import-time assert made visible: you cannot process what io.py cannot read."""
    from typing import get_args

    from rental_ranking.data.io import Entity

    assert set(build.PROCESSED_ENTITIES) <= set(get_args(Entity))


def test_neighbourhoods_is_not_a_processed_entity() -> None:
    """It is a lookup table for the notebook, not a modelling entity — no parquet for it."""
    assert "neighbourhoods" not in build.PROCESSED_ENTITIES


def test_cities_come_from_the_snapshot_table() -> None:
    assert set(build.CITIES) == set(SNAPSHOTS)


# --- the salt ---------------------------------------------------------------------------


def test_missing_salt_fails_before_any_file_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failing here costs nothing; failing inside anonymize costs a 5M-row CSV parse first."""
    monkeypatch.delenv("ANON_SALT", raising=False)
    with pytest.raises(ValueError, match="ANON_SALT"):
        build._resolve_salt()


def test_blank_salt_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANON_SALT", "   ")
    with pytest.raises(ValueError, match="ANON_SALT"):
        build._resolve_salt()


def test_salt_is_returned_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANON_SALT", SALT)
    assert build._resolve_salt() == SALT


# --- prepare: the single dispatch point -------------------------------------------------


@pytest.mark.parametrize(
    ("entity", "anonymizer", "cleaner"),
    [
        ("listings", "anonymize_listings", "clean_listings"),
        ("calendar", "anonymize_calendar", "clean_calendar"),
        ("reviews", "anonymize_reviews", "clean_reviews"),
    ],
)
def test_prepare_routes_each_entity_to_its_own_pair(
    monkeypatch: pytest.MonkeyPatch, entity: str, anonymizer: str, cleaner: str
) -> None:
    """Wiring an entity to the wrong transform is the mistake this dispatch exists to fix."""
    called = []
    monkeypatch.setattr(build, "load_raw", lambda city, ent: pd.DataFrame({"x": [1]}))
    for name in ("anonymize_listings", "anonymize_calendar", "anonymize_reviews"):
        monkeypatch.setattr(
            build.anonymize, name, lambda df, salt, _n=name: called.append(_n) or df
        )
    for name in ("clean_listings", "clean_calendar", "clean_reviews"):
        monkeypatch.setattr(build.clean, name, lambda df, *a, _n=name: called.append(_n) or df)

    build.prepare("thessaloniki", entity, SALT)
    assert called == [anonymizer, cleaner]


def test_prepare_passes_the_city_to_clean_listings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only listings needs it — for the `city` column and the right bounding box."""
    seen = {}
    monkeypatch.setattr(build, "load_raw", lambda city, ent: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(build.anonymize, "anonymize_listings", lambda df, salt: df)
    monkeypatch.setattr(
        build.clean, "clean_listings", lambda df, city: seen.update(city=city) or df
    )

    build.prepare("crete", "listings", SALT)
    assert seen == {"city": "crete"}


def test_prepare_rejects_an_unknown_entity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build, "load_raw", lambda city, ent: pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="unknown entity"):
        build.prepare("athens", "hosts", SALT)


# --- concat_cities ----------------------------------------------------------------------


def test_concat_cities_stacks_every_city() -> None:
    frames = {"athens": _listings(["a1", "a2"]), "crete": _listings(["c1"])}
    combined = build.concat_cities(frames, "listings")
    assert len(combined) == 3
    assert combined["id"].tolist() == ["a1", "a2", "c1"]


def test_concat_cities_resets_the_index() -> None:
    """Duplicate index labels would break every downstream .loc and alignment."""
    frames = {"athens": _listings(["a1", "a2"]), "crete": _listings(["c1", "c2"])}
    assert build.concat_cities(frames, "listings").index.tolist() == [0, 1, 2, 3]


def test_concat_cities_rejects_an_extra_column() -> None:
    """pd.concat would union the columns and null-fill — a silently wider table, not an error."""
    frames = {"athens": _listings(["a1"]), "crete": _listings(["c1"]).assign(extra=1)}
    with pytest.raises(ValueError, match="crete schema differs from athens"):
        build.concat_cities(frames, "listings")


def test_concat_cities_names_the_offending_columns() -> None:
    """'schemas differ' on a 58-column table is not a debuggable message."""
    frames = {"athens": _listings(["a1"]), "crete": _listings(["c1"]).assign(extra=1)}
    with pytest.raises(ValueError, match="extra"):
        build.concat_cities(frames, "listings")


def test_concat_cities_rejects_reordered_columns() -> None:
    """Column *order* matters: the schema tests assert against an ordered tuple."""
    athens = _listings(["a1"])
    frames = {"athens": athens, "crete": athens[["price", "city", "id"]]}
    with pytest.raises(ValueError, match="different order"):
        build.concat_cities(frames, "listings")


def test_concat_cities_accepts_matching_schemas() -> None:
    frames = {"athens": _listings(["a1"]), "crete": _listings(["c1"])}
    with _no_warning():
        build.concat_cities(frames, "listings")


# --- cross-city uniqueness --------------------------------------------------------------


def test_disjoint_city_ids_pass() -> None:
    with _no_warning():
        build.check_ids_unique_across_cities({"athens": {"a1", "a2"}, "crete": {"c1"}})


def test_a_shared_id_between_cities_raises() -> None:
    """Two listings sharing a 12-hex digest would merge silently in every downstream join."""
    with pytest.raises(ValueError, match="not unique across cities"):
        build.check_ids_unique_across_cities({"athens": {"a1", "x"}, "crete": {"x"}})


def test_the_shared_id_is_named_in_the_error() -> None:
    with pytest.raises(ValueError, match="x9"):
        build.check_ids_unique_across_cities({"athens": {"x9"}, "crete": {"x9"}})


def test_uniqueness_check_handles_a_single_city() -> None:
    with _no_warning():
        build.check_ids_unique_across_cities({"athens": {"a1", "a2"}})


# --- join integrity ---------------------------------------------------------------------


def test_a_fully_resolving_join_is_silent() -> None:
    with _no_warning():
        unresolved = build.check_join_integrity(
            _calendar(["a1", "a1", "a2"]), {"a1", "a2"}, "athens/calendar"
        )
    assert unresolved == 0


def test_a_mis_salted_entity_raises_rather_than_warns() -> None:
    """The failure this module exists to catch: both frames look fine, nothing joins."""
    with pytest.raises(ValueError, match="salt or dtype mismatch"):
        build.check_join_integrity(_calendar(["z1", "z2"]), {"a1", "a2"}, "athens/calendar")


def test_the_error_reports_the_resolved_share() -> None:
    with pytest.raises(ValueError, match="0.00%"):
        build.check_join_integrity(_calendar(["z1"]), {"a1"}, "athens/calendar")


def test_a_handful_of_orphans_warns_and_is_counted() -> None:
    """Athens genuinely ships orphans; that is a data fact, not a pipeline fault."""
    child = _calendar(["a1"] * 999 + ["orphan"])
    with pytest.warns(UserWarning, match="no listings row"):
        unresolved = build.check_join_integrity(child, {"a1"}, "athens/calendar")
    assert unresolved == 1


def test_an_empty_child_frame_does_not_divide_by_zero() -> None:
    with _no_warning():
        assert build.check_join_integrity(_calendar([]), {"a1"}, "athens/calendar") == 0


# --- writing --------------------------------------------------------------------------


def test_write_creates_the_output_directory(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh clone has no data/processed/; to_parquet would fail rather than create it."""
    target = tmp_path / "processed"
    monkeypatch.setattr(build, "PROCESSED_DIR", target)
    build.write(_listings(["a1"]), "listings")
    assert (target / "listings.parquet").exists()


def test_write_roundtrips_without_an_index_column(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build, "PROCESSED_DIR", tmp_path)
    frame = _listings(["a1", "a2"])
    build.write(frame, "listings")

    restored = pd.read_parquet(tmp_path / "listings.parquet")
    assert list(restored.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(restored, frame)


# --- end to end -------------------------------------------------------------------------


def test_main_builds_the_processed_layer(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One real city through the whole chain, writing to tmp rather than data/processed/.

    Thessaloniki is the smallest market and the only one with no known orphans, so this runs
    in a few seconds and must complete without a single warning.
    """
    from rental_ranking.data.io import raw_path

    if not raw_path("thessaloniki", "listings").exists():
        pytest.skip("raw snapshots not on disk")

    monkeypatch.setenv("ANON_SALT", SALT)
    monkeypatch.setattr(build, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(build, "CITIES", ("thessaloniki",))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # this city must be clean end to end
        build.main()

    listings = pd.read_parquet(tmp_path / "listings.parquet")
    calendar = pd.read_parquet(tmp_path / "calendar.parquet")
    reviews = pd.read_parquet(tmp_path / "reviews.parquet")

    assert len(listings) == 4965
    assert listings["id"].is_unique
    assert listings["city"].eq("thessaloniki").all()
    assert set(calendar.columns) == {"listing_id", "date", "available"}

    known = set(listings["id"])
    assert calendar["listing_id"].isin(known).all()
    assert reviews["listing_id"].isin(known).all()
