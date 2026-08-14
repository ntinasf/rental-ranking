"""Dead/implausible-listing removal, applied before the label is trusted.

Thresholds are re-derived against the forward-90 label definition (not copied from the
Thessaloniki project); identical criteria apply to every city, and per-city per-rule
removal counts are returned for reporting.

**Two rules, not three.** The contract's original rule 2 — first review inside the label
window — is void under a forward window and was dropped on 2026-08-01: the reviews file
ends at the scrape date and T is that listing's first calendar day, so ``first_review > T``
is impossible by construction. See docs/data_pipeline_design.md and the decisions log.

Filters run **before** price imputation, tiering and grading, so quantile boundaries are
computed on the population that will actually be ranked. Note that the inactive rule removes
exactly the (zero reviews, blocked == 1.0) corner, which mechanically raises the label's
correlation with review signals — validation must report that correlation both before and
after filtering.

Deliberately **not** filtered: listings whose first review falls shortly before T (59 / 202 /
518 within 30 days). That is the cold-start cohort Phase 2 flags with ``has_reviews`` and
Phase 5 studies for fairness.
"""

import pandas as pd

from rental_ranking.data.validate import require_columns

#: A `minimum_nights` above this is a de facto long-term rental, out of scope for a
#: short-stay ranker. A safety net rather than a material filter — it covers 0.50 / 0.01 /
#: 0.16 % of rows — but identical thresholds across cities are worth more than the rows.
MAX_MINIMUM_NIGHTS = 30

#: Everything the rules read. Checked once at the boundary, before any rule runs.
_REQUIRED_COLUMNS = ("city", "number_of_reviews", "blocked_fraction_90", "minimum_nights")


def is_inactive(listings: pd.DataFrame) -> pd.Series:
    """Zero reviews **ever** and a fully blocked label window — inactive or personal use.

    Both halves are load-bearing. Zero reviews alone is cold start, which the project keeps
    and later studies; a fully blocked window alone is a plausibly booked-out listing, and
    after this rule that spike still holds 304 / 379 / 1,607 listings with review histories.
    Only the conjunction says "nothing has ever happened here and nothing is on offer".

    ``number_of_reviews`` is the lifetime count. ``number_of_reviews_ly`` is the calendar-2025
    count, so a listing reviewed steadily until 2024 would read as never-reviewed under it.
    """
    return listings["number_of_reviews"].eq(0) & listings["blocked_fraction_90"].eq(1.0)


def is_long_term(listings: pd.DataFrame, threshold: int = MAX_MINIMUM_NIGHTS) -> pd.Series:
    """A stay minimum above ``threshold`` nights — not a short-stay rental."""
    return listings["minimum_nights"] > threshold


def filter_listings(
    listings: pd.DataFrame, threshold: int = MAX_MINIMUM_NIGHTS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop implausible listings, returning the survivors and the per-city removal counts.

    The rules are independent predicates over the same input, never chained: chaining would
    make the second rule's count depend on the first and hide the overlap, and the overlap is
    part of what gets reported (2 rows, in the current snapshots).

    Counts are a return value rather than a print or an optional extra, because per-city
    filter counts are a named reporting deliverable of Phase 1 (docs/BUILD_GUIDE.md).

    Args:
        listings: Listings joined with the label — needs ``city``, ``number_of_reviews``,
            ``blocked_fraction_90`` and ``minimum_nights``.
        threshold: Passed to :func:`is_long_term`.

    Returns:
        ``(kept, counts)``. ``counts`` is one row per city: ``n``, ``inactive``,
        ``long_term``, ``both_rules``, ``removed``, ``kept``.

    Raises:
        KeyError: If a required column is missing.
    """
    require_columns(listings, _REQUIRED_COLUMNS, "listings joined with the label")

    inactive = is_inactive(listings)
    long_term = is_long_term(listings, threshold)
    removed = inactive | long_term
    city = listings["city"]

    counts = pd.DataFrame(
        {
            "n": city.groupby(city).size(),
            "inactive": inactive.groupby(city).sum(),
            "long_term": long_term.groupby(city).sum(),
            "both_rules": (inactive & long_term).groupby(city).sum(),
            "removed": removed.groupby(city).sum(),
        }
    )
    counts["kept"] = counts["n"] - counts["removed"]

    # .copy() because callers add columns (imputed price, tier, grade) to the result, and a
    # view would raise SettingWithCopyWarning or silently write to the parent frame.
    return listings.loc[~removed].copy(), counts.astype("int64")
