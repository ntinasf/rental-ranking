"""Leave-one-out neighbourhood aggregates, and the one aggregate this project will not build.

**No neighbourhood mean-label aggregate, at any scale.** Leave-one-out is the standard fix for
an including-self aggregate, but against *this* group key it does not fix a label aggregate — it
creates the leak. 365 of 393 query groups sit inside a single neighbourhood, so within a group
the neighbourhood's total ``S`` and size ``n`` are constants, and the leave-one-out mean
``S/(n-1) - x_i/(n-1)`` is an exact, strictly decreasing function of the listing's **own label**
— within-group Spearman of -1.000 in 100 % of those groups. The include-self version has the
opposite problem and is merely useless: one distinct value per group, so it separates no pair.

**What is built instead:** aggregates of things that are not the target — price and listing count
— always leave-one-out, plus the *relative* features that make them earn their place.

**Leave-one-out is unconditional.** The gap from the include-self mean is exactly
``(x_i - mu) / (n - 1)``, so it decays as ``1/(n-1)`` and is negligible in a large neighbourhood
anyway. Applying it only below some size would cost a threshold to justify, a branch to maintain,
and a feature whose definition changes discontinuously at that threshold.

**What the levels do.** A neighbourhood aggregate is constant inside a rung-0 query group, so
like ``room_type`` and ``city`` it is a *conditioner*, not a discriminator. The ratio features
are what vary within a group, and even they are monotone in ``price`` there — so their value is
**generalisation across groups**: one split on "1.3x the local median" transfers between
neighbourhoods, where a split on "price > 150" does not.

Pure transforms, no I/O and no ``main()``.
"""

import numpy as np
import pandas as pd

from rental_ranking.data.validate import require_columns

#: The aggregation unit. ``city`` is part of it because neighbourhood names collide across cities.
#: 75 units on the ranked population, and **one of them holds a single listing** — see
#: :func:`leave_one_out_median`.
NEIGHBOURHOOD_KEY: tuple[str, ...] = ("city", "neighbourhood_cleansed")

_REQUIRED_COLUMNS = ("price", *NEIGHBOURHOOD_KEY)


def _loo_median_within(values: np.ndarray) -> np.ndarray:
    """Leave-one-out medians for one group, vectorised over its members.

    Sorting once and indexing by rank is what makes this exact rather than approximate. For
    sorted ``v`` with element ``i`` removed, the remaining array is ``v`` with every position at
    or after ``rank(i)`` shifted down by one, so the median is read straight off ``v`` at a
    shifted index — no per-row recomputation.
    """
    n = len(values)
    if n < 2:
        return np.full(n, np.nan)

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    remaining = n - 1
    if remaining % 2 == 1:
        middle = (remaining - 1) // 2
        return ordered[middle + (rank <= middle).astype(np.int64)]

    lower, upper = remaining // 2 - 1, remaining // 2
    return (
        ordered[lower + (rank <= lower).astype(np.int64)]
        + ordered[upper + (rank <= upper).astype(np.int64)]
    ) / 2.0


def leave_one_out_median(values: pd.Series, by: pd.Series) -> pd.Series:
    """Median of each row's group, computed **without that row**.

    The median rather than the mean, for the reason ``price.py`` gives: price is heavily skewed
    and `filters.is_extreme_price` keeps everything below a 20x data-error guard, so a group mean
    is dragged by rows the filter chose not to remove.

    Unlike a mean, a leave-one-out median has no closed form in the group total, so it is computed
    per group — cheap here, with 75 neighbourhoods.

    Args:
        values: The column to aggregate. Must be non-null.
        by: Group labels aligned to ``values``.

    Returns:
        A float Series aligned to ``values``. **NaN where the group holds one row** — the median
        of an empty set is undefined, not zero, and a sentinel would be read as a real price.

    Raises:
        ValueError: If ``values`` holds nulls, which would silently shift every median.
    """
    if values.isna().any():
        raise ValueError(
            f"{int(values.isna().sum())} null value(s) reached leave_one_out_median; a null "
            "shifts its group's median without ever appearing in it. Impute or drop first"
        )

    out = pd.Series(np.nan, index=values.index, dtype="float64")
    for _, positions in values.groupby(by, observed=True, dropna=False).indices.items():
        out.iloc[positions] = _loo_median_within(values.to_numpy()[positions])
    return out


def leave_one_out_count(by: pd.Series) -> pd.Series:
    """How many **other** rows share each row's group.

    Zero is a real answer here, unlike for the median: a listing alone in its neighbourhood has
    no neighbours, which is a fact rather than a missing value.
    """
    return by.groupby(by, observed=True, dropna=False).transform("size").sub(1).astype("int64")


def neighbourhood_features(
    listings: pd.DataFrame, key: tuple[str, ...] = NEIGHBOURHOOD_KEY
) -> pd.DataFrame:
    """Leave-one-out neighbourhood aggregates and the relative features built from them.

    Args:
        listings: The **filtered** ranked population with ``price`` already imputed.
        key: The aggregation unit. Defaults to :data:`NEIGHBOURHOOD_KEY`.

    Returns:
        A frame aligned to ``listings``:

        * ``nbhd_listings`` — other listings in the neighbourhood (leave-one-out count).
        * ``nbhd_median_price`` — leave-one-out median price. NaN for a lone listing.
        * ``price_vs_nbhd`` — the listing's price over that median. NaN follows through, and
          this is the column that carries the aggregate's value: it makes one split transfer
          across neighbourhoods rather than being relearned in each.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If ``price`` holds nulls — it must be imputed first.
    """
    require_columns(listings, _REQUIRED_COLUMNS, "ranked listings")

    unit = listings[list(key)].astype(str).agg("|".join, axis=1)
    median_price = leave_one_out_median(listings["price"], unit)

    return pd.DataFrame(
        {
            "nbhd_listings": leave_one_out_count(unit),
            "nbhd_median_price": median_price,
            "price_vs_nbhd": (listings["price"] / median_price).astype("float64"),
        },
        index=listings.index,
    )
