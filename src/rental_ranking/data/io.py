"""Read raw snapshot files with their ID dtypes pinned.

The only module that reads ``data/raw/``, and the counterpart to ``download.py``: both derive
the filename from the same URL in ``SNAPSHOTS``, so the reader cannot drift from the writer.
Writing is not here — ``data/processed/`` is written once by the orchestrator at the end of the
run.

Pinning dtypes is this module's real job. A hashed ID must not depend on how pandas inferred a
column: one null widens int64 to float64, ``str(12345.0)`` hashes differently from
``str(12345)``, and the calendar/listings join then returns nothing with no error anywhere.
IDs are read as ``int64`` rather than nullable ``Int64`` on purpose — ``int64`` raises on a
null, which is the failure we want.
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

#: Columns read at a fixed dtype: the join keys and the identifiers whose text form carries
#: meaning. Everything else is left to pandas' inference.
#:
#: - ``id`` / ``host_id`` / ``listing_id``: hashed downstream, so their string form must be
#:   stable. Listing IDs reach 1.7e18, past 2**53, so a float column has already lost precision
#:   by the time anything can check it.
#: - ``license``: values carry leading zeros ("00000364602"), which numeric inference turns into
#:   a different licence and a corrupted operator signal.
_DTYPES: dict[str, dict[str, str]] = {
    "listings": {"id": "int64", "host_id": "int64", "license": "string"},
    "calendar": {"listing_id": "int64"},
    "reviews": {"listing_id": "int64", "id": "int64"},
    "neighbourhoods": {},
}

# The entity names live in three places that must agree: the type, the dtype table, and the URL
# table in download.py. Checking at import turns a drift into a startup failure rather than a
# KeyError somewhere down the call stack.
assert set(get_args(Entity)) == set(_DTYPES), "Entity and _DTYPES disagree"


def raw_path(city: str, entity: Entity) -> Path:
    """Return the on-disk path of one raw snapshot file.

    The filename is the basename of the download URL, the same rule ``download.py`` applies when
    writing, so the two cannot disagree.

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
        The raw frame, untouched apart from the pinned dtypes.

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
