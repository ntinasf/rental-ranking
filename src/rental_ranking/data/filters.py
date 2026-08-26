"""Dead/implausible-listing removal, applied before the label is trusted.

Identical criteria across every city, with per-city per-rule removal counts returned for
reporting. **Four rules:**

1. ``is_inactive`` — zero reviews ever **and** a fully blocked label window.
2. ``is_long_term`` — ``minimum_nights`` above 30.
3. ``is_dormant`` — blocked for essentially the whole forward year.
4. ``is_extreme_price`` — a price wildly out of line with the listing's own stratum.

**Not a rule: duplicate clusters.** Listings sharing a host, a location and a capacity look like
duplicates but are mostly distinct inventory — one operator with several identical flats in a
building — and dropping them would delete 12 % of real supply, precisely the commercial-operator
population. The leakage they create is a *splitting* problem, handled by
:func:`rental_ranking.features.groups.cluster_id` and a grouped train/test split.

Filters run **before** price imputation, tiering and grading, so quantile boundaries are computed
on the population that will actually be ranked. ``is_inactive`` and ``is_dormant`` both read the
label window, which mechanically raises the label's correlation with review signals — validation
must report that correlation both before and after filtering.

Deliberately **not** filtered: listings whose first review falls shortly before T. That is the
cold-start cohort, flagged by ``has_reviews`` and studied for fairness.
"""

import pandas as pd

from rental_ranking.data.validate import require_columns

#: A `minimum_nights` above this is a de facto long-term rental, out of scope for a short-stay
#: ranker. A safety net rather than a material filter — well under 1 % of rows in every city.
MAX_MINIMUM_NIGHTS = 30

#: Blocked share of the whole calendar above which a listing is treated as withdrawn.
DORMANT_BLOCKED_SHARE = 0.99

#: Multiple of the stratum median price above which a price is treated as a data error.
MAX_PRICE_MULTIPLE = 20

#: Structural key the price sanity check compares within — never behavioural. Same rule as the
#: imputation strata: what the property *is*, not how it has performed.
_PRICE_STRATUM = ["city", "room_type", "accommodates"]

_REQUIRED_COLUMNS = (
    "city",
    "number_of_reviews",
    "blocked_fraction_90",
    "blocked_fraction_calendar",
    "minimum_nights",
    "price",
    "room_type",
    "accommodates",
)


def is_inactive(listings: pd.DataFrame) -> pd.Series:
    """Zero reviews **ever** and a fully blocked label window — inactive or personal use.

    Both halves are load-bearing. Zero reviews alone is cold start, which the project keeps; a
    fully blocked window alone is a plausibly booked-out listing. Only the conjunction says
    "nothing has ever happened here and nothing is on offer".

    Reads the lifetime ``number_of_reviews``, never ``number_of_reviews_ly`` — under the latter a
    listing reviewed steadily until last year would read as never-reviewed.
    """
    return listings["number_of_reviews"].eq(0) & listings["blocked_fraction_90"].eq(1.0)


def is_long_term(listings: pd.DataFrame, threshold: int = MAX_MINIMUM_NIGHTS) -> pd.Series:
    """A stay minimum above ``threshold`` nights — not a short-stay rental."""
    return listings["minimum_nights"] > threshold


def is_dormant(listings: pd.DataFrame, blocked_share: float = DORMANT_BLOCKED_SHARE) -> pd.Series:
    """Blocked for essentially the entire forward year — withdrawn, not booked.

    Defined over the **whole** calendar rather than only the months after the label window, and
    that is the point: a seasonal operator is open during the summer by definition, so a
    whole-year rule cannot mistake one for a dead listing. The narrower "blocked across days
    90-359" flags 1,493 listings that are actively selling in the window.

    These listings mostly carry review histories, which ``is_inactive`` never sees, but their last
    review is far older than the ranked population's, they offer at most 3 bookable nights across
    the entire forward year, and ~96 % sit at a label of exactly 1.0 — the top grade.
    """
    return listings["blocked_fraction_calendar"] >= blocked_share


def is_extreme_price(listings: pd.DataFrame, multiple: float = MAX_PRICE_MULTIPLE) -> pd.Series:
    """A price more than ``multiple`` times its own stratum's median — a data error.

    Not an outlier cut: a ranker must rank expensive listings too. This catches the handful of
    rows where the figure cannot be a real nightly rate. Nulls are never flagged — missing price
    is imputed rather than filtered, because its missingness tracks the label.
    """
    stratum_median = listings.groupby(_PRICE_STRATUM, observed=True)["price"].transform("median")
    return (listings["price"] > multiple * stratum_median).fillna(False)


def filter_listings(
    listings: pd.DataFrame,
    threshold: int = MAX_MINIMUM_NIGHTS,
    blocked_share: float = DORMANT_BLOCKED_SHARE,
    price_multiple: float = MAX_PRICE_MULTIPLE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop implausible listings, returning the survivors and the per-city removal counts.

    The rules are independent predicates over the same input, never chained: chaining would make
    each rule's count depend on the ones before it and hide the overlaps, which are reported.

    Args:
        listings: Listings joined with the label — see ``_REQUIRED_COLUMNS``.
        threshold: Passed to :func:`is_long_term`.
        blocked_share: Passed to :func:`is_dormant`.
        price_multiple: Passed to :func:`is_extreme_price`.

    Returns:
        ``(kept, counts)``. ``counts`` is one row per city: ``n``, one column per rule,
        ``multi_rule`` (rows caught by more than one), ``removed``, ``kept``.

    Raises:
        KeyError: If a required column is missing.
    """
    require_columns(listings, _REQUIRED_COLUMNS, "listings joined with the label")

    flags = pd.DataFrame(
        {
            "inactive": is_inactive(listings),
            "long_term": is_long_term(listings, threshold),
            "dormant": is_dormant(listings, blocked_share),
            "extreme_price": is_extreme_price(listings, price_multiple),
        },
        index=listings.index,
    )
    removed = flags.any(axis=1)
    city = listings["city"]

    counts = pd.DataFrame({"n": city.groupby(city).size()})
    for rule in flags.columns:
        counts[rule] = flags[rule].groupby(city).sum()
    counts["multi_rule"] = flags.sum(axis=1).gt(1).groupby(city).sum()
    counts["removed"] = removed.groupby(city).sum()
    counts["kept"] = counts["n"] - counts["removed"]

    # .copy() because callers add columns (imputed price, tier, grade) to the result, and a
    # view would raise SettingWithCopyWarning or silently write to the parent frame.
    return listings.loc[~removed].copy(), counts.astype("int64")
