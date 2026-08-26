"""Orchestrate the raw → processed pipeline: the only module that writes ``data/processed/``.

Everything upstream is a pure transform; this is where they are chained, where the filesystem is
touched, and where the checks that need *more than one frame* live:

- **Schema equality across cities before concat.** Column drift between Inside Airbnb city files
  is a known failure, and ``pd.concat`` would paper over it by unioning columns and filling the
  gaps with nulls — a silently wider table rather than an error.
- **Join integrity after hashing.** Salt or dtype trouble upstream shows up nowhere else: the
  frames all look fine individually and every join downstream just returns nothing.

Entities are processed one at a time and released, so the 17M-row calendar is never resident
alongside the other two. Preprocessing runs locally and produces a local artifact; a pipeline
job that wanted this step would call ``python -m rental_ranking.data.build`` and hold no logic.
"""

import os
import warnings
from typing import Literal, get_args

import pandas as pd
from dotenv import load_dotenv

from rental_ranking.data import anonymize, clean
from rental_ranking.data.download import SNAPSHOTS
from rental_ranking.data.io import Entity, load_raw
from rental_ranking.data.paths import PROCESSED_DIR

CITIES: tuple[str, ...] = tuple(SNAPSHOTS)

#: The entities that reach ``data/processed/``. ``neighbourhoods`` is loadable but is a lookup
#: table, not a modelling entity, so it is not concatenated.
ProcessedEntity = Literal["listings", "calendar", "reviews"]

PROCESSED_ENTITIES: tuple[ProcessedEntity, ...] = get_args(ProcessedEntity)

# Narrower than `Entity`, so every processed entity must also be a loadable one.
assert set(PROCESSED_ENTITIES) <= set(get_args(Entity)), "PROCESSED_ENTITIES is not loadable"

#: Below this share of child rows resolving to a listing, the run fails instead of warning. Not a
#: quality bar: real orphans are ~0.0002 % of rows, while a salt mismatch or an unpinned ID dtype
#: resolves *nothing*. Anything in between is a bug worth stopping for.
_MIN_JOIN_RESOLUTION = 0.99


def _resolve_salt() -> str:
    """Read ``ANON_SALT`` up front so a missing salt fails before any file is read.

    ``anonymize`` would raise on its own, but only after the first CSV is parsed.

    Returns:
        The salt.

    Raises:
        ValueError: If ``ANON_SALT`` is unset or blank.
    """
    salt = os.environ.get("ANON_SALT", "")
    if not salt.strip():
        raise ValueError(
            "ANON_SALT is unset. Put it in .env (`openssl rand -hex 32`) — note that .env is "
            "loaded by the VS Code Python extension, so a terminal run needs load_dotenv()."
        )
    return salt


def prepare(city: str, entity: ProcessedEntity, salt: str) -> pd.DataFrame:
    """Load one city-entity file and run it through anonymization and cleaning.

    The one place entity-specific wiring lives: the transforms themselves take a frame and return
    a frame, and knowing which function goes with which entity belongs to the orchestrator.

    Args:
        city: Market key.
        entity: One of :data:`PROCESSED_ENTITIES`.
        salt: The anonymization salt, identical for every entity or the joins break.

    Returns:
        The anonymized, typed frame for that city.

    Raises:
        ValueError: If ``entity`` is not a processed entity.
    """
    frame = load_raw(city, entity)
    if entity == "listings":
        return clean.clean_listings(anonymize.anonymize_listings(frame, salt), city)
    if entity == "calendar":
        return clean.clean_calendar(anonymize.anonymize_calendar(frame, salt))
    if entity == "reviews":
        return clean.clean_reviews(anonymize.anonymize_reviews(frame, salt))
    raise ValueError(f"unknown entity {entity!r}; expected one of {PROCESSED_ENTITIES}")


def concat_cities(frames: dict[str, pd.DataFrame], entity: ProcessedEntity) -> pd.DataFrame:
    """Assert every city's schema matches, then concatenate.

    Raises rather than asserts — ``assert`` is stripped under ``python -O``. The message names the
    offending city and the symmetric difference, because "schemas differ" on a 58-column table is
    not a debuggable error.

    Args:
        frames: City name → prepared frame. Insertion order is preserved in the output.
        entity: Entity name, for the error message.

    Returns:
        One frame with a fresh index, carrying every city's rows.

    Raises:
        ValueError: If any city's columns differ from the first city's, in name or order.
    """
    reference_city, reference = next(iter(frames.items()))
    for city, frame in frames.items():
        if not frame.columns.equals(reference.columns):
            difference = sorted(set(frame.columns).symmetric_difference(reference.columns))
            raise ValueError(
                f"{entity}: {city} schema differs from {reference_city} — "
                f"columns only in one of them: {difference or '(same set, different order)'}"
            )

    combined = pd.concat(frames.values(), ignore_index=True)
    expected = sum(len(frame) for frame in frames.values())
    if len(combined) != expected:
        raise ValueError(f"{entity}: concat produced {len(combined)} rows, expected {expected}")
    return combined


def check_ids_unique_across_cities(ids_by_city: dict[str, set[str]]) -> None:
    """Verify no hashed listing id appears in two cities.

    Two distinct listings sharing a digest would silently merge in every downstream join, and the
    12-hex truncation makes that a small but real risk (~1e-4 at this scale). Comparing the summed
    sizes against the size of the union states "pairwise disjoint" without materialising every
    pairwise intersection.

    Args:
        ids_by_city: Market → the set of hashed listing ids it contributed.

    Raises:
        ValueError: If any id is shared between cities.
    """
    total = sum(len(ids) for ids in ids_by_city.values())
    union = set().union(*ids_by_city.values())
    if total != len(union):
        shared = {
            city: sorted(ids & set().union(*(o for c, o in ids_by_city.items() if c != city)))[:5]
            for city, ids in ids_by_city.items()
        }
        raise ValueError(
            f"listing ids are not unique across cities ({total - len(union)} shared); "
            f"examples per city: { {c: s for c, s in shared.items() if s} }"
        )


def check_join_integrity(child: pd.DataFrame, listing_ids: set[str], label: str) -> int:
    """Verify a child entity's ``listing_id`` resolves against the listings it belongs to.

    The check that catches the failure nothing else can see: if the salt differed between two
    entities, or an ID column read back as float, both frames still look well-formed on their own
    while every downstream join silently returns zero rows.

    A handful of orphans is a real property of the data, so those warn. A collapse below
    :data:`_MIN_JOIN_RESOLUTION` is a pipeline bug and raises.

    Args:
        child: A prepared calendar or reviews frame.
        listing_ids: Hashed ids from the listings frame of the same scope.
        label: Identifier for messages, e.g. ``"athens/calendar"``.

    Returns:
        The number of unresolved rows.

    Raises:
        ValueError: If the resolved share falls below :data:`_MIN_JOIN_RESOLUTION`.
    """
    resolved = child["listing_id"].isin(listing_ids)
    rate = float(resolved.mean()) if len(child) else 1.0
    if rate < _MIN_JOIN_RESOLUTION:
        raise ValueError(
            f"{label}: only {rate:.2%} of rows resolve to a listing. This is a salt or dtype "
            f"mismatch, not a data gap — every entity must be hashed with the same salt."
        )

    unresolved = int((~resolved).sum())
    if unresolved:
        orphans = sorted(set(child.loc[~resolved, "listing_id"]))
        warnings.warn(
            f"{label}: {unresolved} row(s) across {len(orphans)} listing_id(s) have no "
            f"listings row — expected for Athens, investigate elsewhere",
            stacklevel=2,
        )
    return unresolved


def write(frame: pd.DataFrame, entity: ProcessedEntity) -> None:
    """Write one processed entity to ``data/processed/<entity>.parquet``.

    Args:
        frame: The concatenated frame.
        entity: Entity name; also the file stem.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    destination = PROCESSED_DIR / f"{entity}.parquet"
    frame.to_parquet(destination, index=False)
    print(f"{entity:9s} {len(frame):>10,} rows x {frame.shape[1]:>2} cols -> {destination}")


def main() -> None:
    """Build the processed layer from the raw snapshots.

    Listings are built first so their hashed ids are available to check the other two against,
    then each remaining entity is built, verified, written and released — the calendar alone is
    17M rows.
    """
    load_dotenv()
    salt = _resolve_salt()

    listings_by_city = {city: prepare(city, "listings", salt) for city in CITIES}
    ids_by_city = {city: set(frame["id"]) for city, frame in listings_by_city.items()}
    check_ids_unique_across_cities(ids_by_city)
    write(concat_cities(listings_by_city, "listings"), "listings")
    del listings_by_city

    for entity in PROCESSED_ENTITIES[1:]:
        frames = {}
        for city in CITIES:
            frames[city] = prepare(city, entity, salt)
            check_join_integrity(frames[city], ids_by_city[city], f"{city}/{entity}")
        write(concat_cities(frames, entity), entity)
        del frames


if __name__ == "__main__":
    main()
