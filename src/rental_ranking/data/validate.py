"""Shape and value checks shared across the data layer.

These exist to turn silent corruption into a loud failure. The snapshots on disk are clean
today, but Inside Airbnb rotates them and the schema has already shifted once (v4.7 dropped
the calendar price and redefined ``price``). Every check here encodes an assumption the
pipeline rests on, so that a future snapshot which breaks it fails at the boundary rather
than producing a plausible-looking wrong number downstream.
"""

import warnings
from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, required: Iterable[str], entity: str) -> None:
    """Fail with a readable message when an expected column is absent.

    Guards against passing the wrong frame to an entity transform, or applying a column set
    that belongs to a different entity — a mistake pandas otherwise reports as a bare
    ``KeyError`` raised from inside ``drop``, listing columns without saying which entity
    was expected.

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

    One warning carrying a count, never one per row: a systematic problem in Crete would
    otherwise emit 27,333 messages and bury the signal it was meant to raise. Returning the
    count lets the caller record it — per-city violation counts are what the inventory
    notebook reports.

    Warns rather than raises because preprocessing is lossless by contract: a suspicious
    value is reported and kept, and the decision to exclude a row belongs to ``filters.py``.

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

    Both bounds are optional, so one function covers a lower-only check, an upper-only check,
    or both — an ``if`` per call site would multiply as the checks do. Either bound may be a
    scalar *or* a Series: passing a Series compares row-wise, which is what "no review may
    postdate the listing's own scrape date" needs, since the scrape date differs per row.

    Nulls are never flagged. A missing value is a separate concern from an out-of-range one,
    and conflating them would report the same row twice under two different problems.

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
