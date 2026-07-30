"""Read raw snapshot files with their ID dtypes pinned.

The only module that reads from ``data/raw/``, and the counterpart to ``download.py``: that
one writes the layout, this one reads it. Both derive the filename from the same URL in
``SNAPSHOTS`` rather than spelling it out, so the reader cannot drift from the writer — the
British/American ``neighbourhoods`` split is exactly the bug that arrangement prevents.

Pinning dtypes is this module's real job, not the path convenience. A hashed ID must not
depend on how pandas happened to infer a column: one null widens an int64 column to float64,
``str(12345.0)`` hashes differently from ``str(12345)``, and the calendar/listings join then
returns nothing with no error anywhere. Reading IDs as ``int64`` rather than nullable
``Int64`` is deliberate — ``int64`` *raises* on a null, which is the failure we want.

Writing is not here. ``data/processed/`` is written once, by the orchestrator, at the end of
the run; a module that both reads raw and writes processed would invite loading a
half-finished layer.
"""

from pathlib import Path
from typing import Literal, get_args
from urllib.parse import urlparse

import pandas as pd

from rental_ranking.data.download import SNAPSHOTS
from rental_ranking.data.paths import RAW_DIR
from rental_ranking.data.validate import require_columns

#: The four files downloaded per city snapshot.
Entity = Literal["listings", "calendar", "reviews", "neighbourhoods"]

#: Columns read at a fixed dtype, per entity. Everything else is left to pandas' inference —
#: only the join keys and the identifiers whose text form carries meaning are pinned.
#:
#: - ``id`` / ``host_id`` / ``listing_id``: hashed downstream, so their string form must be
#:   stable. Athens listing IDs reach 1.7e18, past 2**53, so a float column has already lost
#:   precision by the time anything can check it.
#: - ``license``: 12,464 Athens values carry leading zeros ("00000364602"). Numeric inference
#:   silently turns that into 364602 — a different licence, and a corrupted operator signal.
_DTYPES: dict[str, dict[str, str]] = {
    "listings": {"id": "int64", "host_id": "int64", "license": "string"},
    "calendar": {"listing_id": "int64"},
    "reviews": {"listing_id": "int64", "id": "int64"},
    "neighbourhoods": {},
}

# The entity names live in three places that must agree: the type, the dtype table, and the
# URL table in download.py. Checking at import turns a drift into a startup failure instead
# of a KeyError somewhere down the call stack.
assert set(get_args(Entity)) == set(_DTYPES), "Entity and _DTYPES disagree"


def raw_path(city: str, entity: Entity) -> Path:
    """Return the on-disk path of one raw snapshot file.

    The filename is taken from the basename of the download URL, the same rule
    ``download.py`` applies when writing, so the two cannot disagree.

    Args:
        city: Market key, as used in ``SNAPSHOTS``.
        entity: Which of the four snapshot files.

    Returns:
        The path, whether or not it exists.

    Raises:
        ValueError: If ``city`` or ``entity`` is unknown.
    """
    if entity not in _DTYPES:
        raise ValueError(f"unknown entity {entity!r}; expected one of {sorted(_DTYPES)}")
    if city not in SNAPSHOTS:
        raise ValueError(f"unknown city {city!r}; expected one of {sorted(SNAPSHOTS)}")

    snapshot = SNAPSHOTS[city]
    filename = Path(urlparse(snapshot["files"][entity]).path).name
    return RAW_DIR / city / snapshot["as_of"] / filename


def load_raw(city: str, entity: Entity) -> pd.DataFrame:
    """Read one raw snapshot file with its ID dtypes pinned.

    Args:
        city: Market key, as used in ``SNAPSHOTS``.
        entity: Which of the four snapshot files.

    Returns:
        The raw frame, untouched apart from the pinned dtypes. Anonymization and cleaning
        are separate steps.

    Raises:
        ValueError: If ``city`` or ``entity`` is unknown.
        FileNotFoundError: If the snapshot has not been downloaded.
        KeyError: If a pinned column is missing from the file.
    """
    path = raw_path(city, entity)
    if not path.exists():
        raise FileNotFoundError(
            f"{city}/{entity} not on disk at {path} — run `python -m rental_ranking.data.download`"
        )

    frame = pd.read_csv(path, dtype=_DTYPES[entity])

    # pandas silently ignores a `dtype` key naming a column the file does not have, so a
    # renamed ID column would read back inferred and unpinned — the exact thing this module
    # exists to prevent, and invisible without this check.
    require_columns(frame, _DTYPES[entity], f"{city}/{entity}")
    return frame
