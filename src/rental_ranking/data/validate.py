"""Shape and value checks shared across the data layer.

Each check encodes an assumption the pipeline rests on, so a future snapshot that breaks one
fails at the boundary rather than producing a plausible-looking wrong number downstream. Inside
Airbnb rotates its snapshots and the schema has shifted before.
"""

import warnings
from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, required: Iterable[str], entity: str) -> None:
    """Fail with a readable message when an expected column is absent.

    Names the entity that was expected, which a bare ``KeyError`` from inside ``drop`` does not.

    Args:
        df: The frame to check.
        required: Column names that must be present.
        entity: Entity name used in the error message ("listings", "calendar", "reviews").

    Raises:
        KeyError: If any required column is missing.
    """
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"{entity}: expected columns are absent from the frame: {missing}")


def warn_violations(mask: pd.Series, message: str) -> int:
    """Emit one aggregate warning when ``mask`` flags anything, and return the count.

    One warning carrying a count, never one per row: a systematic problem would otherwise emit
    tens of thousands of messages and bury the signal it was meant to raise.

    Warns rather than raises because preprocessing is lossless: a suspicious value is reported
    and kept, and the decision to exclude a row belongs to ``filters.py``.

    Args:
        mask: Boolean Series where ``True`` marks a violation. Nulls count as no violation.
        message: Human-readable description; the count is appended.

    Returns:
        The number of flagged rows.
    """
    count = int(mask.fillna(False).sum())
    if count:
        warnings.warn(f"{message}: {count} row(s)", stacklevel=2)
    return count


def out_of_range(
    values: pd.Series,
    lower: object | pd.Series | None = None,
    upper: object | pd.Series | None = None,
) -> pd.Series:
    """Mask the values falling outside an inclusive ``[lower, upper]`` range.

    Either bound may be a scalar *or* a Series; a Series compares row-wise, which is what "no
    review may postdate the listing's own scrape date" needs, since the scrape date differs per
    row.

    Nulls are never flagged — a missing value is a separate concern from an out-of-range one.

    Args:
        values: The column to check.
        lower: Inclusive lower bound, scalar or row-aligned Series. ``None`` to skip.
        upper: Inclusive upper bound, scalar or row-aligned Series. ``None`` to skip.

    Returns:
        A boolean Series, ``True`` where the value is present and outside the range.
    """
    flagged = pd.Series(False, index=values.index)
    if lower is not None:
        flagged |= values < lower
    if upper is not None:
        flagged |= values > upper
    return flagged & values.notna()
