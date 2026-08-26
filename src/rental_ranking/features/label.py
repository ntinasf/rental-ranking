"""Forward-90-day occupancy demand proxy, bucketed into graded relevance 0-4.

The label is a **demand proxy**, never "bookings": the Inside Airbnb calendar is forward-looking
availability, and blocked days include personal use, maintenance and seasonality. Features may
only use data available before the label window starts.

The window points **forward** from T = ``min(calendar.date)``, per listing and never per city,
because scrape dates spread over several days inside one market. ``availability_90`` is the same
quantity in column form: the standing cross-check, never a feature.

Pure ``DataFrame -> DataFrame`` transforms, no I/O, no ``main()``; reading the parquets belongs
to the caller.
"""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from rental_ranking.data.validate import require_columns, warn_violations

#: The label window in days. Not a knob — the default *is* the label. The parameter exists so a
#: window-length sensitivity check needs no second implementation.
LABEL_WINDOW_DAYS = 90

#: Share of listings whose derived availability must match the shipped ``availability_90``.
#: Observed at 99.96-99.99 %, the residual being scrape timing rather than a defect. Enforced
#: against the real snapshots by ``tests/test_label.py``, not at runtime.
MIN_AVAILABILITY_AGREEMENT = 0.999

#: Quantile bins above the zero atom. Grades run 0-4: grade 0 is the atom itself, grades 1-4
#: are quartiles of everything above it. See :func:`assign_grades` for why the 1.0 atom is not
#: reserved as well.
GRADES_ABOVE_ATOM = 4

#: Rows above the atom a partition cell needs before it cuts its own quantiles. Three is the
#: arithmetic floor for quartiles; 30 is where the cut points stop moving with a handful of rows.
#: Cells below it grade within ``DEFAULT_FALLBACK_COLS`` instead.
MIN_PARTITION_ROWS = 30

#: The grading partition. **A coarsening of the query-group key, never a cross-cut of it** — see
#: :func:`assign_grades`. Room type earns its place on the label gradient across its levels; a
#: price tier does not, and would break the coarsening rule besides.
DEFAULT_PARTITION_COLS = ("city", "room_type")

#: Terminator for cells below ``MIN_PARTITION_ROWS``. Applied regardless of its own size.
DEFAULT_FALLBACK_COLS = ("city",)


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

        ``blocked_fraction_calendar`` exists for the dormancy filter and is **never a feature**:
        it spans the label window and then some. A listing blocked for its whole forward year has
        been withdrawn rather than booked, and a whole-year rule cannot mistake a summer-only
        operator for a dead listing.

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

    # Over the whole calendar, not the window. Its denominator is per-listing (365, or a little
    # more for a handful), which is why it is a mean rather than a fixed division.
    labels["blocked_fraction_calendar"] = (
        1 - calendar.groupby("listing_id")["available"].mean()
    ).astype("float64")

    # Reported, not raised: a gap outside the window cannot affect the label, and a gap inside it
    # already failed the size check above. A few listings run 366-367 contiguous days, so length
    # alone is not the signal — span disagreeing with count is.
    warn_violations(
        labels["calendar_days"].ne(labels["calendar_span"]),
        "calendar is not contiguous (row count disagrees with first-to-last span)",
    )
    return labels


def crosscheck_availability_90(labels: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    """Per-city agreement between the derived availability and the shipped ``availability_90``.

    The one sanctioned use of a ``LABEL_ADJACENT_COLUMNS`` member: Inside Airbnb computes
    ``availability_90`` at the moment of the scrape, independently of this package, so it is
    standing evidence that the window is anchored and counted correctly. A break here means the
    anchor drifted or the calendar changed shape.

    Returns a frame rather than raising. ``MIN_AVAILABILITY_AGREEMENT`` is the threshold, and the
    assertion lives in the test suite, where it runs against the real snapshots.

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


def _quartiles(values: pd.Series, reference: pd.Series, cell: object) -> pd.Series:
    """Grade ``values`` 1-4 against the quartiles of ``reference``.

    Split from a plain ``qcut`` because the two are not always the same population: a cell cutting
    its own quantiles passes ``values`` twice, while an undersized cell passes its own rows and
    the *fallback* pool as the reference.
    """
    try:
        _, edges = pd.qcut(reference, GRADES_ABOVE_ATOM, labels=False, retbins=True)
    except ValueError as exc:  # duplicate edges — the distribution is too concentrated
        raise ValueError(
            f"grading cell {cell!r} ({len(reference)} rows above the atom) cannot be cut into "
            f"{GRADES_ABOVE_ATOM} quantiles on value: {exc}. Raising rather than merging bins, "
            "which would hand back fewer grades than the scale promises"
        ) from exc

    # Open the outer edges so a value outside the reference range still bins. Only reachable
    # when reference is a superset of values, but a silent NaN grade is not worth the risk.
    edges[0], edges[-1] = -np.inf, np.inf
    return pd.cut(values, edges, labels=False).astype("int64") + 1


def assign_grades(
    listings: pd.DataFrame,
    label_col: str = "blocked_fraction_90",
    partition_cols: Sequence[str] = DEFAULT_PARTITION_COLS,
    fallback_cols: Sequence[str] = DEFAULT_FALLBACK_COLS,
    min_rows: int = MIN_PARTITION_ROWS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Bucket the demand proxy into relevance grades 0-4 within a partition.

    ``label == 0.0`` is reserved as grade 0; everything above it is quartiled into grades 1-4
    within partition. The zero atom is a qualitatively different state — nothing in 90 peak-season
    days was blocked — and reserving it is what keeps cold-start listings out of the bottom grade,
    where plain quintiles would strand several times as many of them.
    The 1.0 atom is deliberately **not** reserved; every listing at 1.0 carries reviews, so it
    reaches the top quartile on its merits.

    **Cuts are on the label's value, not on its rank.** ``pd.qcut`` yields left-open,
    right-closed intervals, so every listing sharing a label value gets the same grade and the
    grade is non-decreasing in the label inside each cell. Cutting on ``rank(method="first")``
    would split identical listings by row order instead. The cost is that the quartiles are not
    exactly equal: the label lives on a 91-value grid, so a boundary value takes its whole tie
    group into the lower bin.

    **The partition must be a coarsening of the query-group key** — which is why this takes
    column *names*: `city x room_type` is nested inside
    `city x neighbourhood x room_type x capacity_tier`, so grade order can never oppose label
    order inside a group. A cross-cutting partition such as a price tier breaks that in 145 of
    516 groups. ``tests/test_label.py`` pins the invariant.

    Not special-cased, deliberately: listings with **zero reviews and a zero label** land in
    grade 0 like any other zero. Undiscovered and undesirable are indistinguishable in this data,
    and a neutral grade would fabricate a target.

    Args:
        listings: The **filtered** ranked population, with price already imputed, carrying
            ``label_col`` and every partition and fallback column.
        label_col: The demand proxy column, in ``[0, 1]``.
        partition_cols: Columns whose combination defines a grading cell.
        fallback_cols: Coarser columns used by cells that cannot carry their own quantiles.
            This is the terminator — applied regardless of its own size, and the one place a
            failure to cut is fatal rather than recoverable.
        min_rows: Rows above the atom a cell needs before it cuts its own quartiles. It is only
            a proxy: a cell can clear it and still be too concentrated to cut, in which case it
            takes the same fallback route and the call warns once with a count.

    Returns:
        ``(grades, report)``. ``grades`` is an integer Series aligned to ``listings``, named
        ``grade``, with values in ``{0, 1, 2, 3, 4}``. ``report`` is one row per partition
        cell: ``n``, ``above_atom``, and ``level`` — ``"partition"``, ``"fallback"``, or
        ``"empty"`` for a cell with nothing above the atom to cut.

    Warns:
        UserWarning: Once, with a count, if any cell cleared ``min_rows`` but could not be cut
            into quantiles and was graded within ``fallback_cols`` instead.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If the label is null or outside ``[0, 1]``, or if a **fallback** cell's
            quantile edges are not unique — there is no coarser population left, and returning
            fewer grades than the scale promises is worse than failing.
    """
    partition_cols, fallback_cols = list(partition_cols), list(fallback_cols)
    require_columns(listings, [label_col, *partition_cols, *fallback_cols], "listings")

    label = listings[label_col]
    if label.isna().any():
        raise ValueError(
            f"{label_col} is null for {int(label.isna().sum())} row(s); grading "
            "every row is the point, so there is no sensible grade to assign"
        )
    if not label.between(0, 1).all():
        raise ValueError(
            f"{label_col} must lie in [0, 1]; found "
            f"[{label.min()}, {label.max()}] — this is not a blocked fraction"
        )

    grades = pd.Series(-1, index=listings.index, dtype="int64", name="grade")

    # Exact equality is safe: the label is 1 - k/90 for integer k, and k == 90 gives exactly
    # 0.0 in IEEE 754. A tolerance here would silently absorb 1/90 = 0.011 into the atom.
    at_atom = label.eq(0.0)
    grades[at_atom] = 0

    above = listings.loc[~at_atom]
    cell_size = above.groupby(partition_cols, observed=True)[label_col].transform("size")
    fell_back = cell_size < min_rows

    # `min_rows` is only a proxy for "this cell can carry its own quantiles"; a cell can clear it
    # and still be too concentrated to cut, which is the same condition detected exactly. Such a
    # cell takes the same route as an undersized one rather than killing the call — but it warns,
    # because regrading a cell against a coarser population is a decision worth seeing.
    concentrated = []
    for cell, idx in above.loc[~fell_back].groupby(partition_cols, observed=True).groups.items():
        try:
            grades.loc[idx] = _quartiles(label.loc[idx], label.loc[idx], cell)
        except ValueError:
            fell_back.loc[idx] = True
            concentrated.append(cell)
    if concentrated:
        warnings.warn(
            f"{len(concentrated)} partition cell(s) had {min_rows}+ rows above the atom but "
            f"could not be cut into {GRADES_ABOVE_ATOM} quantiles on value, so they were graded "
            f"within {fallback_cols} instead: {concentrated[:5]}",
            stacklevel=2,
        )

    # Undersized cells are graded against the *whole* fallback population, never against each
    # other. Pooling five shared rooms and quantiling them among themselves would spread them
    # across all four grades on no evidence — exactly the noise `min_rows` exists to prevent.
    # So the cut points come from every above-atom row sharing the fallback key, and only the
    # undersized rows are assigned from them.
    # Both mappings come from `.groups` so their keys have the same shape — a groupby over a
    # one-element column list yields tuple keys when iterated but scalars from `.groups`.
    pools = above.groupby(fallback_cols, observed=True).groups
    for cell, idx in above.loc[fell_back].groupby(fallback_cols, observed=True).groups.items():
        grades.loc[idx] = _quartiles(label.loc[idx], label.loc[pools[cell]], cell)

    counts = listings.groupby(partition_cols, observed=True).size().rename("n")
    report = pd.DataFrame({"n": counts})
    report["above_atom"] = (
        above.groupby(partition_cols, observed=True).size().reindex(counts.index).fillna(0)
    ).astype("int64")
    used_fallback = above.loc[fell_back].groupby(partition_cols, observed=True).size()
    report["level"] = np.where(
        used_fallback.reindex(counts.index).fillna(0) > 0, "fallback", "partition"
    )
    report.loc[report["above_atom"].eq(0), "level"] = "empty"
    return grades, report
