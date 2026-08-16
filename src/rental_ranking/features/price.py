"""Price imputation over a structural fallback cascade.

``price`` in Inside Airbnb v4.7 is not a standing nightly rate: it equals
``price_quote_price_per_night``, the per-night figure from a quote for a *dated* stay, and the
scraper chooses that date by walking forward to the listing's first opening. So price is
missing exactly when there was nothing to quote — **its missingness tracks the label**, and
both of the obvious shortcuts are closed:

* **Dropping the rows** deletes a non-random slice of the ranked population.
* **A "has price" flag**, or passing NaN to LightGBM and letting its native missing handling
  split on it, hands the model a label proxy. Measured on the filtered population, mean label
  is 0.273 / 0.231 / 0.458 for price-null listings against 0.300 / 0.379 / 0.575 for the rest —
  Athens' gap alone is as wide as its entire room-type gradient.

So the price is imputed, and this module is the only place that happens. It cannot live in
``data/clean.py``: the processed layer is lossless by contract and cannot see the filters,
while the median that fills a row must be computed on the population that will be ranked.
Ordering is therefore fixed — **filters, then imputation** — and is recorded in
docs/data_pipeline_design.md.

**The key is structural, never behavioural.** City, neighbourhood, room type, capacity: what
the property *is*. Never cohort, listing age or ``rating_shrunk``: how it has *performed*.
The full argument is in the decisions log; the short form is that conditioning the imputation
on the label's dominant driver would let performance leak into a column the model reads as an
attribute.

**There is no price tier here.** The grading partition is ``city x room_type`` — see
``label.assign_grades`` — because a price tier cross-cuts the query-group key instead of
coarsening it, and grading inside a cross-cut partition inverts the label against the grade
in 145 of 516 query groups. Rank-within-market survives only as a possible Phase 2 *feature*
(a continuous within-city percentile), which is not this module's business.

Convention, matching ``rental_ranking.data``: a pure ``DataFrame -> DataFrame`` transform with
no I/O and no ``main()``.
"""

import pandas as pd

from rental_ranking.data.validate import require_columns
from rental_ranking.features.groups import capacity_tier

#: The fallback cascade, coarsest-last. Each rung is ``(name, group key)``; a row is filled by
#: the first rung whose stratum has a median, so every rung reads "same city, same room type,
#: and as much location and capacity as this listing still has a priced peer for".
#:
#: On the current snapshots only the first two fire (692 and 2 rows), but the cascade is a
#: contract term rather than a convenience: a thinner neighbourhood would need the lower rungs,
#: and ``city`` is the guaranteed terminator. Rung 2 is why Phase 1 owns the capacity tier.
CASCADE: list[tuple[str, list[str]]] = [
    ("nbhd_room_accommodates", ["city", "neighbourhood_cleansed", "room_type", "accommodates"]),
    ("nbhd_room_tier", ["city", "neighbourhood_cleansed", "room_type", "capacity_tier"]),
    ("nbhd_room", ["city", "neighbourhood_cleansed", "room_type"]),
    ("city_room", ["city", "room_type"]),
    ("city", ["city"]),
]

_REQUIRED_COLUMNS = (
    "city",
    "neighbourhood_cleansed",
    "room_type",
    "accommodates",
    "price",
)


def impute_price(
    listings: pd.DataFrame,
    cascade: list[tuple[str, list[str]]] = CASCADE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing prices from progressively coarser structural strata.

    The median, not the mean: ``price`` has a skew of 6.5 and a maximum of 9,243 against a
    median of 120, and ``filters.is_extreme_price`` deliberately keeps everything below a 20x
    data-error guard, so a stratum mean would be dragged by the very rows the filter chose not
    to remove. It also makes the fill idempotent under any monotone rescaling of price.

    No leave-one-out is needed here, unlike when the cascade is benchmarked against held-out
    prices: a row being filled has a null price, so it contributes nothing to its own stratum
    median in the first place.

    The filled values are written **into** ``price`` rather than into a second column, so no
    NaN-bearing price column survives into the feature matrix. To recover which rows were
    imputed — for a grading sensitivity check, say — take ``listings["price"].isna()`` from the
    input *before* calling; it is deliberately not returned per row, because a row-level rung
    marker is a "has price" flag by another name.

    Args:
        listings: The **filtered** ranked population — see ``_REQUIRED_COLUMNS``. Passing the
            unfiltered frame computes medians over listings that will never be ranked.
        cascade: Rungs as ``(name, group key)``, coarsest last. Defaults to :data:`CASCADE`.

    Returns:
        ``(frame, counts)``. ``frame`` is a copy of ``listings`` with ``price`` filled.
        ``counts`` is one row per city: ``n``, ``missing``, then one column per rung holding
        the rows that rung filled. The rung columns sum to ``missing``.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If any price is still null after the final rung — the terminator failed,
            which means a whole stratum at the coarsest key had no price at all.
    """
    require_columns(listings, _REQUIRED_COLUMNS, "filtered listings")

    # Derived here rather than required as a column so the cascade cannot be run against a
    # capacity tier built with different bounds than the query groups use.
    keyed = listings.assign(capacity_tier=capacity_tier(listings))
    filled = listings["price"].astype("float64")
    city = listings["city"]

    counts = pd.DataFrame({"n": city.groupby(city).size()})
    counts["missing"] = filled.isna().groupby(city).sum()

    for name, keys in cascade:
        stratum_median = keyed.groupby(keys, observed=True)["price"].transform("median")
        counts[name] = (filled.isna() & stratum_median.notna()).groupby(city).sum()
        filled = filled.fillna(stratum_median)

    unresolved = filled.isna()
    if unresolved.any():
        by_city = unresolved.groupby(city).sum()
        raise ValueError(
            f"price is still null after the final cascade rung {cascade[-1][1]}: "
            f"{by_city[by_city > 0].to_dict()} — that stratum holds no priced listing at all"
        )

    return listings.assign(price=filled), counts.astype("int64")
