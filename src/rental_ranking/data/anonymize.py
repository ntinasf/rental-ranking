"""Strip and hash host and reviewer PII before anything leaves the local environment.

The threat model is the *publication* boundary — the repo, notebook outputs, the README.
Local disk and the private Blob container are storage, not publication, so raw snapshots
keep their PII untouched; these transforms gate what reaches ``data/processed/`` and
everything downstream of it. The policy is fixed by docs/data_pipeline_design.md
§Anonymization and encoded as the column sets in :mod:`rental_ranking.data.columns` —
change the doc, that module, and this one together.

Every function here is a pure ``DataFrame -> DataFrame`` transform with no I/O; loading and
writing belong to the orchestrator. Anonymization is **lossless in rows**: it removes columns
and replaces identifying values, but never drops a listing. Row exclusion lives in
``filters.py`` and nowhere else.

One deliberate exception to "strip the text": reviews ``comments`` is kept raw, because the
Phase 4 sentiment features need it. The free text does contain reviewer first names, which is
acceptable under the threat model above — only aggregate sentiment scores are ever published —
on one standing condition: **a published notebook cell must never render raw comment rows.**
"""

import hashlib
import os

import numpy as np
import pandas as pd

from rental_ranking.data import columns as cols
from rental_ranking.data.validate import require_columns

#: Truncation length of the hex digest. 12 hex chars = 48 bits; at ~47k listings the
#: birthday-collision probability is ~1e-4, and a collision merges two listings rather than
#: leaking one, so the risk is tolerable. Do not shorten it further.
HASH_LENGTH = 12

_SALT_ENV_VAR = "ANON_SALT"


def _resolve_salt(salt: str | None = None) -> str:
    """Return the salt to hash with, falling back to the ``ANON_SALT`` environment variable.

    Raises rather than defaulting to an empty salt. An unsalted digest of a public,
    enumerable listing ID is reversible by dictionary attack, and the failure is invisible:
    the output still looks like a correctly anonymized dataset. Failing loudly here is the
    only point at which that mistake is catchable.

    Args:
        salt: Explicit salt. When ``None``, ``$ANON_SALT`` is read instead.

    Returns:
        The resolved, non-empty salt.

    Raises:
        ValueError: If neither an explicit salt nor ``$ANON_SALT`` provides a non-blank value.
    """
    resolved = salt if salt is not None else os.environ.get(_SALT_ENV_VAR, "")
    if not resolved.strip():
        raise ValueError(
            f"No anonymization salt: pass salt= explicitly or set ${_SALT_ENV_VAR}. "
            "Hashing public listing IDs without a salt produces reversible digests."
        )
    return resolved


def _canonical_key(value: object) -> str:
    """Render a value as the exact string that gets hashed, independent of pandas dtypes.

    A single null anywhere in an integer ID column makes pandas widen the whole column to
    ``float64``, and ``str(12345.0)`` is ``"12345.0"`` — a different digest for the same
    listing. The calendar/listings join then returns zero rows with no error anywhere.
    Collapsing integral floats back to integers keeps the digest stable across that widening.

    This is a second line of defence, not the first. Listing IDs here exceed 2**53
    (max observed 1.7e18), so a column genuinely parsed as float has *already* lost
    precision before this function sees it. Pin ID dtypes at load time.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def hash_value(value: object, salt: str | None = None) -> str:
    """Salted SHA-256 of a single value, truncated to :data:`HASH_LENGTH` hex characters.

    Args:
        value: The value to hash. Canonicalized first, so ``12345`` and ``12345.0`` agree.
        salt: Explicit salt; falls back to ``$ANON_SALT``.

    Returns:
        A lowercase hex digest of length :data:`HASH_LENGTH`.
    """
    salt = _resolve_salt(salt)
    payload = f"{salt}:{_canonical_key(value)}".encode()
    return hashlib.sha256(payload).hexdigest()[:HASH_LENGTH]


def hash_series(values: pd.Series, salt: str | None = None) -> pd.Series:
    """Salted-hash a column, hashing each *distinct* value once and preserving nulls.

    Hashing per row would recompute the same digest millions of times: the calendar is 17.0M
    rows across the three cities but holds only 46,640 distinct ``listing_id`` values. Building
    the digest map from the uniques and calling :meth:`pandas.Series.map` gives the identical
    result far more cheaply — measured on Athens' 5.2M calendar rows, 0.06s here against 2.8s
    for the equivalent ``Series.apply``.

    Nulls stay null. That matters most for ``license``: hashing a missing licence would give
    every unlicensed listing the *same* digest, which reads downstream as one operator holding
    hundreds of properties and corrupts the duplicate-licence commercial-operator signal.

    Args:
        values: The column to hash.
        salt: Explicit salt; falls back to ``$ANON_SALT``.

    Returns:
        A Series of digests aligned to the input index, with nulls passed through.
    """
    salt = _resolve_salt(salt)
    digests = {value: hash_value(value, salt) for value in values.dropna().unique()}
    return values.map(digests)


def host_is_local(host_location: pd.Series) -> pd.Series:
    """Derive the three-way home-market flag from ``host_location``.

    Three-way, not boolean: the source column is 34-37% null, so "unknown" is a real category
    and collapsing it into "foreign" would invent an observation that was never made.

    Args:
        host_location: Raw ``host_location`` values.

    Returns:
        A Series of ``"local"`` / ``"foreign"`` / ``"unknown"``.
    """
    text = host_location.astype("string")
    in_greece = text.str.contains("greece", case=False, na=False)
    return pd.Series(
        np.where(text.isna(), "unknown", np.where(in_greece, "local", "foreign")),
        index=host_location.index,
        dtype="object",
    )


def host_has_about(host_about: pd.Series) -> pd.Series:
    """Derive a presence flag for the host bio, treating blank and whitespace-only as absent.

    Args:
        host_about: Raw ``host_about`` values.

    Returns:
        A boolean Series: ``True`` where a non-blank bio exists.
    """
    return host_about.astype("string").fillna("").str.strip().ne("").astype(bool)


def license_status(license_values: pd.Series) -> pd.Series:
    """Derive the three-way registration status from the raw ``license`` column.

    Greece's AMA registry makes this near-complete (only 2.8-4.3% null), so the three
    categories are all well populated and "missing" is informative rather than noise.

    Args:
        license_values: Raw ``license`` values.

    Returns:
        A Series of ``"registered"`` / ``"exempt"`` / ``"missing"``.
    """
    normalized = license_values.astype("string").str.strip().str.lower().fillna("")
    return pd.Series(
        np.where(
            normalized == "", "missing", np.where(normalized == "exempt", "exempt", "registered")
        ),
        index=license_values.index,
        dtype="object",
    )


def anonymize_listings(df: pd.DataFrame, salt: str | None = None) -> pd.DataFrame:
    """Apply the listings anonymization policy: drop PII, hash IDs, derive-then-drop.

    Drops :data:`~rental_ranking.data.columns.PII_DROP_COLUMNS`, hashes
    :data:`~rental_ranking.data.columns.HASH_COLUMNS` (``id`` and ``host_id``), and replaces
    ``host_location`` / ``host_about`` / ``license`` with the derived columns recorded in
    :data:`~rental_ranking.data.columns.DERIVED_FROM`.

    The licence is both categorized *and* hashed: the hash is what makes duplicate licences
    linkable across listings (369 / 2,204 / 10,925 rows across the three cities), which is a
    strong commercial-operator signal, while the raw number is a registry identifier.

    Args:
        df: A raw listings frame with the v4.7 schema.
        salt: Explicit salt; falls back to ``$ANON_SALT``.

    Returns:
        A new frame with the same rows, PII removed and IDs hashed.

    Raises:
        KeyError: If any column the policy touches is absent.
        ValueError: If no salt is available.
    """
    salt = _resolve_salt(salt)
    derived_sources = ["host_location", "host_about", "license"]
    require_columns(
        df, set(cols.PII_DROP_COLUMNS) | set(cols.HASH_COLUMNS) | set(derived_sources), "listings"
    )

    out = df.copy()
    for column in sorted(cols.HASH_COLUMNS):
        out[column] = hash_series(out[column], salt)

    out["host_is_local"] = host_is_local(out["host_location"])
    out["host_has_about"] = host_has_about(out["host_about"])
    out["license_status"] = license_status(out["license"])
    out["license_hash"] = hash_series(out["license"], salt)

    return out.drop(columns=sorted(set(cols.PII_DROP_COLUMNS) | set(derived_sources)))


def anonymize_calendar(df: pd.DataFrame, salt: str | None = None) -> pd.DataFrame:
    """Hash ``listing_id`` in a calendar frame. No columns are dropped here.

    The per-date ``minimum_nights`` / ``maximum_nights`` columns also leave the pipeline, but
    that is a *cleaning* decision (they fall inside the label window) encoded in
    :data:`~rental_ranking.data.columns.CALENDAR_KEEP`, not an anonymization one. Keeping the
    two concerns in separate steps means a change to either is reviewable on its own.

    Args:
        df: A raw calendar frame.
        salt: Explicit salt; falls back to ``$ANON_SALT``. Must match the listings salt.

    Returns:
        A new frame with ``listing_id`` hashed.

    Raises:
        KeyError: If ``listing_id`` is absent.
        ValueError: If no salt is available.
    """
    require_columns(df, {"listing_id"}, "calendar")
    out = df.copy()
    out["listing_id"] = hash_series(out["listing_id"], salt)
    return out


def anonymize_reviews(df: pd.DataFrame, salt: str | None = None) -> pd.DataFrame:
    """Drop reviewer identity from a reviews frame and hash ``listing_id``.

    Only ``listing_id`` is hashed. The reviews ``id`` is the *review's* own identifier, not a
    person's, and it is the natural dedup key for the review table, so it stays raw per the
    contract. ``comments`` also stays raw — see the module docstring for the condition attached.

    Args:
        df: A raw reviews frame.
        salt: Explicit salt; falls back to ``$ANON_SALT``. Must match the listings salt.

    Returns:
        A new frame with reviewer columns removed and ``listing_id`` hashed.

    Raises:
        KeyError: If ``listing_id`` or a reviewer column is absent.
        ValueError: If no salt is available.
    """
    require_columns(df, {"listing_id"} | set(cols.REVIEWS_PII_DROP), "reviews")
    out = df.copy()
    out["listing_id"] = hash_series(out["listing_id"], salt)
    return out.drop(columns=sorted(cols.REVIEWS_PII_DROP))
