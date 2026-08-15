"""Forward-90-day occupancy demand proxy, bucketed into graded relevance 0-4.

The label is a **demand proxy**, never "bookings": the Inside Airbnb calendar is
forward-looking availability, and blocked days include personal use, maintenance and
seasonality. Features may only use data available before the label window starts.

The window points **forward** from T = ``min(calendar.date)`` for that listing — per
listing, never per city, because scrape dates spread over up to four days inside one
market (decisions log 2026-07-24 and 2026-07-25). ``availability_90`` is the label in
column form: the standing cross-check, never a feature.

Convention, matching ``rental_ranking.data``: pure ``DataFrame -> DataFrame`` transforms,
no I/O, no ``main()``. Reading the parquets belongs to the caller — the notebook, or a
Phase 2 orchestrator.
"""

import pandas as pd

from rental_ranking.data.validate import require_columns, warn_violations

#: The label window in days. Fixed by contract (docs/data_pipeline_design.md), not a knob —
#: the default is the label. The parameter exists so notebook 02 can run a window-length
#: sensitivity check without a second implementation to keep in step.
LABEL_WINDOW_DAYS = 90

#: Share of listings whose derived availability must match the shipped ``availability_90``.
#: Measured 99.96-99.99 %; the residual is scrape timing, not a defect. Enforced against the
#: real snapshots by ``tests/test_label.py``, not at runtime — see ``crosscheck_availability_90``.
MIN_AVAILABILITY_AGREEMENT = 0.999


def occupancy_label(calendar: pd.DataFrame, window_days: int = LABEL_WINDOW_DAYS) -> pd.DataFrame:
    """Per-listing occupancy demand proxy over the forward window from each listing's own T.

    Args:
        calendar: Processed calendar frame with ``listing_id``, ``date`` and ``available``.
        window_days: Length of the label window. Leave at the default; see
            ``LABEL_WINDOW_DAYS``.

    Returns:
        A frame indexed by ``listing_id`` carrying ``T`` (the anchor), ``calendar_days``,
        ``calendar_span``, ``avail_<window_days>`` (available nights in the window),
        ``blocked_fraction_<window_days>`` — the label, in ``[0, 1]``, where 1.0 means
        nothing in the window was bookable — and ``blocked_fraction_calendar``, the same
        fraction over the listing's **entire** calendar rather than the label window.
        Listings absent from ``calendar`` are absent here; the caller's merge is where that
        surfaces.

        ``blocked_fraction_calendar`` exists for the dormancy filter and is **never a
        feature**: it spans the label window and then some. A listing blocked for its whole
        forward year has been withdrawn, not booked — and unlike a rule defined only outside
        the window, this one cannot mistake a summer-only operator for a dead listing,
        because a seasonal listing is open during the summer by definition.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If any listing's window does not hold exactly ``window_days`` rows.
    """
    require_columns(calendar, ("listing_id", "date", "available"), "calendar")

    # transform("min") broadcasts each listing's anchor back over its own rows, so the
    # window is a row-wise date comparison. Positional slicing (head(window_days)) would be
    # correct only if the frame happened to be sorted, which no parquet round-trip promises.
    anchor = calendar.groupby("listing_id")["date"].transform("min")
    window = calendar.loc[calendar["date"] < anchor + pd.Timedelta(days=window_days)]

    # Gate the arithmetic, do not merely report on it afterwards: a short window divided by
    # window_days understates the blocked fraction, and the result still looks like a
    # plausible number in [0, 1]. Raise rather than assert — asserts vanish under python -O.
    sizes = window.groupby("listing_id").size()
    wrong = sizes[sizes.ne(window_days)]
    if not wrong.empty:
        raise ValueError(
            f"{len(wrong)} listing(s) do not have exactly {window_days} calendar rows in "
            f"the label window, so the denominator would move: "
            f"{dict(list(wrong.items())[:5])}"
        )

    by_listing = calendar.groupby("listing_id")["date"]
    available = f"avail_{window_days}"
    blocked = f"blocked_fraction_{window_days}"

    labels = pd.DataFrame(
        {
            "T": by_listing.min(),
            "calendar_days": by_listing.size(),
            "calendar_span": (by_listing.max() - by_listing.min()).dt.days + 1,
            # `available` is a nullable BooleanDtype, so the sum arrives as Int64 and the
            # division would propagate to a nullable Float64. Cast out of the nullable
            # family here: scipy and LightGBM both handle Float64 inconsistently downstream.
            available: window.groupby("listing_id")["available"].sum().astype("int64"),
        }
    )
    labels[blocked] = (1 - labels[available] / window_days).astype("float64")

    # Over the whole calendar, not the window. Its denominator is per-listing (365, or 366-367
    # for eight of them), which is why it is computed as a mean rather than a fixed division.
    labels["blocked_fraction_calendar"] = (
        1 - calendar.groupby("listing_id")["available"].mean()
    ).astype("float64")

    # Reported, not raised: a gap outside the window cannot affect the label, and a gap
    # inside it already failed the size check above. Eight listings run 366-367 contiguous
    # days, so length alone is not the signal — span disagreeing with count is.
    warn_violations(
        labels["calendar_days"].ne(labels["calendar_span"]),
        "calendar is not contiguous (row count disagrees with first-to-last span)",
    )
    return labels


def crosscheck_availability_90(labels: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    """Per-city agreement between the derived availability and the shipped ``availability_90``.

    The one sanctioned use of a ``LABEL_ADJACENT_COLUMNS`` member: Inside Airbnb computes
    ``availability_90`` at the moment of the scrape, independently of anything this package
    does, so it is the standing evidence that the window is anchored and counted correctly.
    A break here means the anchor drifted or the calendar changed shape — not that a listing
    behaved oddly.

    Returns a frame rather than raising, matching the data layer's report-don't-delete rule.
    ``MIN_AVAILABILITY_AGREEMENT`` is the threshold; the assertion lives in the test suite,
    where it runs against the real snapshots, so a notebook can display the number without
    a runtime cost on every call.

    Args:
        labels: Output of :func:`occupancy_label` at the default 90-day window.
        listings: Processed listings frame with ``id``, ``city`` and ``availability_90``.

    Returns:
        One row per city: ``n``, ``exact_agreement``, ``n_mismatched``, ``mean_abs_diff``.
    """
    require_columns(listings, ("id", "city", "availability_90"), "listings")
    require_columns(labels, ("avail_90",), "labels")

    merged = listings.merge(labels["avail_90"], left_on="id", right_index=True, how="inner")
    merged["_diff"] = merged["avail_90"] - merged["availability_90"]

    return merged.groupby("city").agg(
        n=("_diff", "size"),
        exact_agreement=("_diff", lambda s: float(s.eq(0).mean())),
        n_mismatched=("_diff", lambda s: int(s.ne(0).sum())),
        mean_abs_diff=("_diff", lambda s: float(s.abs().mean())),
    )


# TODO — assign_grades. Measured evidence and counts: NEXT_STEPS step 5.
#   - assign_grades(frame, label_col, partition_cols) takes partition *column names*, not a
#     price frame, so this module never imports features/price.py — the price tier is just
#     another string column. Price must be imputed first.
#   - Scheme decided 2026-08-04: **reserve the atoms, quantile the interior.** label == 0.0 ->
#     grade 0; label == 1.0 -> grade 4; the interior quantiled into grades 1-3 within partition.
#     Chosen over pure quantile grading (which buries 38.5 % of never-reviewed listings in
#     grade 0, against 8.0 % here — re-measured 2026-08-15 on the four-rule population) and over
#     fixed global cut points (which put 42.2 % of Thessaloniki in grade 0 against 11.4 % of
#     Crete, destroying cross-city comparability; measured pre-audit, rejected structurally).
#   - An interior tie rule is still needed: the label lives on a 91-value grid
#     (blocked_days / 90), so listings share values straddling a cut point. Not because the
#     spikes are heavy — the atoms sit at the extremes, where a boundary rarely lands.
#   - The partition rule is explicit: collapse rare room_types, set a minimum size, then fall
#     back city x room_type -> city, returning which level each row used. Pin it with tests.
