"""Tests for rental_ranking.features.label.

The guards here exist because the failure mode is a *plausible* wrong number, not a crash:
a short window still divides to something in [0, 1], and a positionally-sliced window still
returns 90 rows. Each guard is therefore tested in both directions — it fires when it should,
and stays quiet when it should not.
"""

import warnings

import pandas as pd
import pytest

from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import label

WINDOW = label.LABEL_WINDOW_DAYS


def _calendar(
    available_nights: dict[str, int],
    days: int = 365,
    start: str = "2026-06-29",
    offsets: dict[str, int] | None = None,
) -> pd.DataFrame:
    """A calendar where each listing's first ``available_nights`` days are bookable.

    Everything after that is blocked, so the blocked fraction over the first ``WINDOW`` days
    is exactly ``1 - available_nights / WINDOW`` — a value the test can assert on directly
    rather than recomputing with the same logic under test.
    """
    frames = []
    for listing_id, avail in available_nights.items():
        first = pd.Timestamp(start) + pd.Timedelta(days=(offsets or {}).get(listing_id, 0))
        dates = pd.date_range(first, periods=days, freq="D")
        frames.append(
            pd.DataFrame(
                {
                    "listing_id": listing_id,
                    "date": dates,
                    "available": pd.array([i < avail for i in range(len(dates))], dtype="boolean"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def _no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --- occupancy_label: the arithmetic ------------------------------------------------------


@pytest.mark.parametrize(("available", "expected"), [(0, 1.0), (45, 0.5), (90, 0.0), (9, 0.9)])
def test_blocked_fraction_follows_the_available_count(available: int, expected: float) -> None:
    out = label.occupancy_label(_calendar({"a": available}))
    assert out.loc["a", "blocked_fraction_90"] == pytest.approx(expected)
    assert out.loc["a", "avail_90"] == available


def test_blocked_fraction_is_bounded_between_zero_and_one() -> None:
    out = label.occupancy_label(_calendar({"a": 0, "b": 30, "c": 90}))
    assert out["blocked_fraction_90"].between(0.0, 1.0).all()


def test_blocked_fraction_is_plain_float64_not_nullable() -> None:
    """`available` is BooleanDtype, so the sum is Int64 and the division would carry a
    nullable Float64 through to scipy and LightGBM, both of which handle it inconsistently."""
    out = label.occupancy_label(_calendar({"a": 45}))
    assert out["blocked_fraction_90"].dtype == "float64"
    assert out["avail_90"].dtype == "int64"


# --- occupancy_label: the anchor ----------------------------------------------------------


def test_anchor_is_per_listing_not_a_shared_date() -> None:
    """Scrape dates spread over four days inside one city; a global anchor would shift the
    window for every listing but the earliest."""
    calendar = _calendar({"early": 30, "late": 30}, offsets={"late": 4})
    out = label.occupancy_label(calendar)

    assert out.loc["early", "T"] == pd.Timestamp("2026-06-29")
    assert out.loc["late", "T"] == pd.Timestamp("2026-07-03")
    # Each listing is measured against its own anchor, so the labels agree despite the shift.
    assert out.loc["early", "blocked_fraction_90"] == out.loc["late", "blocked_fraction_90"]


def test_rows_beyond_the_window_are_excluded() -> None:
    """A listing available for its whole second half must still read as fully blocked."""
    calendar = _calendar({"a": 0})
    calendar.loc[
        calendar["date"] >= pd.Timestamp("2026-06-29") + pd.Timedelta(days=90), "available"
    ] = True

    out = label.occupancy_label(calendar)
    assert out.loc["a", "blocked_fraction_90"] == 1.0


def test_row_order_does_not_change_the_result() -> None:
    """The reason the window is a date comparison and not head(90): nothing guarantees row
    order after a parquet round-trip."""
    calendar = _calendar({"a": 20, "b": 70})
    shuffled = calendar.sample(frac=1.0, random_state=17).reset_index(drop=True)

    pd.testing.assert_frame_equal(label.occupancy_label(calendar), label.occupancy_label(shuffled))


# --- occupancy_label: the guards, both directions -----------------------------------------


def test_short_calendar_raises_instead_of_dividing_by_ninety() -> None:
    """60 available days out of a 60-day calendar is not a 0.33 blocked fraction."""
    with pytest.raises(ValueError, match="exactly 90 calendar rows"):
        label.occupancy_label(_calendar({"a": 60}, days=60))


def test_the_short_calendar_error_names_the_offenders() -> None:
    """A bare count sends you hunting through 17M rows; the id is what makes it actionable."""
    mixed = pd.concat(
        [_calendar({"healthy": 90}), _calendar({"truncated-42": 10}, days=10)],
        ignore_index=True,
    )
    with pytest.raises(ValueError) as raised:
        label.occupancy_label(mixed)

    assert "truncated-42" in str(raised.value)
    assert "healthy" not in str(raised.value)


def test_calendar_longer_than_365_days_still_yields_a_90_day_window() -> None:
    """Eight real listings run 366-367 days; length alone must not be treated as the signal."""
    out = label.occupancy_label(_calendar({"a": 45}, days=400))
    assert out.loc["a", "calendar_days"] == 400
    assert out.loc["a", "blocked_fraction_90"] == pytest.approx(0.5)


def test_non_contiguous_calendar_warns_once_with_a_count() -> None:
    """A gap outside the window cannot affect the label, so it is reported, not raised."""
    calendar = _calendar({"a": 45}, days=90)
    stray = pd.DataFrame(
        {
            "listing_id": ["a"],
            "date": [pd.Timestamp("2027-06-29")],
            "available": pd.array([True], dtype="boolean"),
        }
    )
    with pytest.warns(UserWarning, match="not contiguous"):
        out = label.occupancy_label(pd.concat([calendar, stray], ignore_index=True))
    assert out.loc["a", "blocked_fraction_90"] == pytest.approx(0.5)


def test_contiguous_calendar_does_not_warn(_no_warning) -> None:
    label.occupancy_label(_calendar({"a": 45, "b": 10}))


def test_window_days_renames_the_derived_columns() -> None:
    """A sensitivity run must not be able to mislabel its own output as the 90-day label."""
    out = label.occupancy_label(_calendar({"a": 30}), window_days=60)
    assert "blocked_fraction_60" in out.columns
    assert "blocked_fraction_90" not in out.columns
    assert out.loc["a", "blocked_fraction_60"] == pytest.approx(0.5)


def test_missing_column_raises_a_readable_keyerror() -> None:
    calendar = _calendar({"a": 45}).drop(columns=["available"])
    with pytest.raises(KeyError, match="calendar"):
        label.occupancy_label(calendar)


# --- crosscheck_availability_90 -----------------------------------------------------------


def _listings(rows: dict[str, int], city: str = "thessaloniki") -> pd.DataFrame:
    return pd.DataFrame({"id": list(rows), "city": city, "availability_90": list(rows.values())})


def test_crosscheck_reports_full_agreement_when_the_counts_match() -> None:
    labels = label.occupancy_label(_calendar({"a": 30, "b": 60}))
    out = label.crosscheck_availability_90(labels, _listings({"a": 30, "b": 60}))

    assert out.loc["thessaloniki", "exact_agreement"] == 1.0
    assert out.loc["thessaloniki", "n_mismatched"] == 0
    assert out.loc["thessaloniki", "mean_abs_diff"] == 0.0


def test_crosscheck_detects_a_shifted_anchor() -> None:
    """The negative direction: if the window drifted, this is the instrument that says so."""
    labels = label.occupancy_label(_calendar({"a": 30, "b": 60}))
    out = label.crosscheck_availability_90(labels, _listings({"a": 30, "b": 44}))

    assert out.loc["thessaloniki", "exact_agreement"] == pytest.approx(0.5)
    assert out.loc["thessaloniki", "n_mismatched"] == 1
    assert out.loc["thessaloniki", "mean_abs_diff"] == pytest.approx(8.0)


def test_crosscheck_excludes_listings_with_no_listings_row() -> None:
    """Athens ships five calendar orphans; they cannot be scored against a row that is absent."""
    labels = label.occupancy_label(_calendar({"a": 30, "orphan": 60}))
    out = label.crosscheck_availability_90(labels, _listings({"a": 30}))

    assert out.loc["thessaloniki", "n"] == 1


def test_crosscheck_requires_the_ninety_day_columns() -> None:
    labels = label.occupancy_label(_calendar({"a": 30}), window_days=60)
    with pytest.raises(KeyError, match="labels"):
        label.crosscheck_availability_90(labels, _listings({"a": 30}))


# --- against the real snapshots -----------------------------------------------------------


def _processed(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"processed layer not on disk: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def real_labels() -> pd.DataFrame:
    return label.occupancy_label(_processed("calendar"))


def test_real_label_is_bounded_and_complete(real_labels: pd.DataFrame) -> None:
    assert real_labels["blocked_fraction_90"].between(0.0, 1.0).all()
    assert real_labels["blocked_fraction_90"].notna().all()
    assert real_labels["T"].notna().all()


def test_real_label_lands_on_the_ninety_one_value_grid(real_labels: pd.DataFrame) -> None:
    """blocked_days/90 has 91 attainable values — the fact the grading tie rule rests on."""
    assert real_labels["blocked_fraction_90"].nunique() <= WINDOW + 1
    assert real_labels["avail_90"].between(0, WINDOW).all()


def test_real_availability_crosscheck_meets_the_threshold(real_labels: pd.DataFrame) -> None:
    """The standing evidence that the anchor is right, enforced rather than merely displayed."""
    report = label.crosscheck_availability_90(real_labels, _processed("listings"))

    assert (report["exact_agreement"] >= label.MIN_AVAILABILITY_AGREEMENT).all()
    assert (report["mean_abs_diff"] < 0.05).all()
    assert report["n"].sum() == len(_processed("listings"))


# --- assign_grades ------------------------------------------------------------------------
#
# Scheme E: the 0.0 atom is grade 0, everything above it is quartiled into 1-4 within
# partition. The failure modes are all silent — a grade scale missing its top class, a tie
# split by row order, a partition that reverses the label inside a query group — so each is
# pinned in both directions rather than merely exercised.


def _graded(
    labels: list[float],
    cities: list[str] | None = None,
    room_types: list[str] | None = None,
) -> pd.DataFrame:
    """A minimal frame carrying just what ``assign_grades`` reads."""
    n = len(labels)
    return pd.DataFrame(
        {
            "blocked_fraction_90": labels,
            "city": cities or ["athens"] * n,
            "room_type": room_types or ["Entire home/apt"] * n,
        }
    )


def _spread(count: int, first_blocked: int = 1) -> list[float]:
    """``count`` distinct labels above the atom, as ``blocked_days / 90``.

    Expressed in blocked *days* rather than fractions so a test can say where in the 91-value
    grid its listings sit, which is what the tie and partition tests actually depend on.
    """
    assert first_blocked + count <= WINDOW + 1, "labels would run past the top of the grid"
    return [k / WINDOW for k in range(first_blocked, first_blocked + count)]


def test_the_zero_atom_is_grade_zero_and_nothing_else_is() -> None:
    frame = _graded([0.0] * 5 + _spread(40))
    grades, _ = label.assign_grades(frame)

    assert (grades[frame["blocked_fraction_90"].eq(0.0)] == 0).all()
    assert (grades[frame["blocked_fraction_90"].gt(0.0)] > 0).all()


def test_the_one_atom_is_not_reserved_and_lands_in_the_top_quartile() -> None:
    """Scheme E, reversing scheme B: 1.0 earns grade 4, it is not handed one."""
    frame = _graded(_spread(40) + [1.0] * 4)
    grades, _ = label.assign_grades(frame)

    assert (grades[frame["blocked_fraction_90"].eq(1.0)] == 4).all()
    assert (grades == 4).sum() > int(frame["blocked_fraction_90"].eq(1.0).sum())


def test_grades_use_the_whole_scale() -> None:
    grades, _ = label.assign_grades(_graded([0.0] * 4 + _spread(60)))

    assert sorted(grades.unique()) == [0, 1, 2, 3, 4]


def test_identical_labels_always_get_identical_grades() -> None:
    """The tie rule: cuts land on the label's value, never on ``rank(method="first")``."""
    frame = _graded(_spread(20) * 3)
    grades, _ = label.assign_grades(frame)

    per_value = frame.assign(grade=grades).groupby("blocked_fraction_90")["grade"].nunique()
    assert (per_value == 1).all()


def test_row_order_does_not_change_any_grade() -> None:
    frame = _graded([0.0] * 3 + _spread(50))
    forward, _ = label.assign_grades(frame)
    shuffled = frame.sample(frac=1.0, random_state=7)
    reversed_, _ = label.assign_grades(shuffled)

    pd.testing.assert_series_equal(forward, reversed_.reindex(frame.index))


def test_grade_never_decreases_as_the_label_rises_within_a_partition() -> None:
    """The coarsening invariant, at partition scope — cross-cut partitions break it."""
    frame = _graded([0.0] * 5 + _spread(60))
    grades, _ = label.assign_grades(frame)

    ordered = grades[frame["blocked_fraction_90"].sort_values().index].to_numpy()
    assert (ordered[1:] >= ordered[:-1]).all()


def test_each_partition_cell_cuts_its_own_quantiles() -> None:
    """The Crete/Thessaloniki effect: one label, two cities, two grades.

    Crete spans 1-80 blocked days; Thessaloniki tops out at 20. A listing blocked 18 days is
    near the bottom of Crete's distribution and at the top of Thessaloniki's, which is the
    whole point of grading within a market rather than globally.
    """
    frame = _graded(
        _spread(80) + _spread(20) * 2,
        cities=["crete"] * 80 + ["thessaloniki"] * 40,
    )
    grades, _ = label.assign_grades(frame)
    at_eighteen = frame["blocked_fraction_90"].eq(18 / WINDOW)

    by_city = frame.assign(grade=grades)[at_eighteen].groupby("city")["grade"].max()
    assert by_city["crete"] == 1
    assert by_city["thessaloniki"] == 4


def test_undersized_cell_falls_back_to_the_coarser_partition() -> None:
    """The five shared rooms are the cheapest in the city, so pooling collapses them to one
    grade — where quantiling them on their own would have spread them over all four."""
    frame = _graded(
        _spread(50, first_blocked=41) + _spread(5),
        room_types=["Entire home/apt"] * 50 + ["Shared room"] * 5,
    )
    grades, report = label.assign_grades(frame, min_rows=30)

    assert report.loc[("athens", "Shared room"), "level"] == "fallback"
    assert report.loc[("athens", "Entire home/apt"), "level"] == "partition"
    assert sorted(grades[frame["room_type"].eq("Shared room")].unique()) == [1]


def test_a_cell_at_the_minimum_keeps_its_own_quantiles() -> None:
    """Both directions of the threshold: 30 rows is enough, 29 is not."""
    frame = _graded(
        _spread(60) + _spread(30),
        room_types=["Entire home/apt"] * 60 + ["Private room"] * 30,
    )
    _, report = label.assign_grades(frame, min_rows=30)

    assert report.loc[("athens", "Private room"), "level"] == "partition"


def test_report_counts_every_row_and_separates_the_atom() -> None:
    frame = _graded([0.0] * 7 + _spread(50))
    _, report = label.assign_grades(frame)

    assert report["n"].sum() == len(frame)
    assert report["above_atom"].sum() == int(frame["blocked_fraction_90"].gt(0).sum())


def test_a_null_label_raises_rather_than_being_graded() -> None:
    frame = _graded(_spread(40) + [float("nan")])

    with pytest.raises(ValueError, match="null"):
        label.assign_grades(frame)


def test_a_label_outside_the_unit_interval_raises() -> None:
    frame = _graded(_spread(40) + [1.4])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        label.assign_grades(frame)


def test_concentrated_cell_falls_back_rather_than_failing() -> None:
    """`min_rows` is a proxy; a cell can clear it and still be uncuttable.

    Thirty-plus rows on a single label value pass the size test and then produce duplicate
    quantile edges. That is the same condition the minimum exists for, detected exactly, so the
    cell takes the fallback route — loudly, because it is a decision worth seeing.
    """
    frame = _graded(
        [0.9] * 35 + _spread(80) * 3,
        room_types=["Shared room"] * 35 + ["Entire home/apt"] * 240,
    )

    with pytest.warns(UserWarning, match="could not be cut"):
        grades, report = label.assign_grades(frame)

    assert report.loc[("athens", "Shared room"), "level"] == "fallback"
    assert report.loc[("athens", "Shared room"), "above_atom"] == 35
    assert grades[frame["room_type"].eq("Shared room")].nunique() == 1


def test_uncuttable_fallback_raises_instead_of_returning_fewer_grades() -> None:
    """The terminator has nowhere left to go, so a degenerate scale must fail loudly."""
    frame = _graded([0.5] * 40)

    with pytest.raises(ValueError, match="cannot be cut into 4 quantiles"):
        label.assign_grades(frame)


def test_a_healthy_partition_grades_without_warning(_no_warning) -> None:
    """The fallback warning must stay silent on a population that does not need it."""
    label.assign_grades(_graded([0.0] * 5 + _spread(60)))


def test_missing_partition_column_raises_a_readable_keyerror() -> None:
    frame = _graded(_spread(40)).drop(columns=["room_type"])

    with pytest.raises(KeyError, match="room_type"):
        label.assign_grades(frame)


def test_grading_never_reads_a_price_column() -> None:
    """`label.py` must stay uncoupled from `price.py`; the partition is column names only."""
    frame = _graded([0.0] * 4 + _spread(60))
    grades, _ = label.assign_grades(frame)
    with_price, _ = label.assign_grades(frame.assign(price=range(len(frame))))

    pd.testing.assert_series_equal(grades, with_price)


# --- assign_grades against the real snapshots ---------------------------------------------


@pytest.fixture(scope="module")
def real_ranked() -> pd.DataFrame:
    """The ranked population as Phase 1 hands it to grading: filtered, price imputed."""
    from rental_ranking.data.filters import filter_listings
    from rental_ranking.features.groups import capacity_tier
    from rental_ranking.features.price import impute_price

    listings = _processed("listings")
    labels = label.occupancy_label(_processed("calendar"))
    kept, _ = filter_listings(listings.merge(labels, left_on="id", right_index=True, how="inner"))
    priced, _ = impute_price(kept)
    return priced.assign(capacity_tier=capacity_tier(priced))


def test_real_grades_cover_the_scale_and_every_row(real_ranked: pd.DataFrame) -> None:
    grades, report = label.assign_grades(real_ranked)

    assert sorted(grades.unique()) == [0, 1, 2, 3, 4]
    assert grades.notna().all()
    assert report["n"].sum() == len(real_ranked)


def test_real_grading_leaves_no_tie_split_across_grades(real_ranked: pd.DataFrame) -> None:
    """0 % on the real snapshots — the whole reason cuts land on value rather than rank."""
    per_value = (
        real_ranked.assign(grade=label.assign_grades(real_ranked)[0])
        .groupby(["city", "room_type", "blocked_fraction_90"], observed=True)["grade"]
        .nunique()
    )

    assert (per_value == 1).all()


def test_real_grade_never_opposes_the_label_inside_a_query_group(
    real_ranked: pd.DataFrame,
) -> None:
    """The coarsening rule that rejected the price tier, enforced against the real key."""
    grades = label.assign_grades(real_ranked)[0].to_numpy()
    labels = real_ranked["blocked_fraction_90"].to_numpy()
    key = ["city", "neighbourhood_cleansed", "room_type", "capacity_tier"]

    for _, positions in real_ranked.groupby(key, observed=True).indices.items():
        ordered = grades[positions][labels[positions].argsort(kind="stable")]
        assert (ordered[1:] >= ordered[:-1]).all()


def test_real_cold_start_listings_are_not_buried_in_grade_zero(
    real_ranked: pd.DataFrame,
) -> None:
    """8.0 % under scheme E against 38.6 % under plain quintiles — the reason for the atom."""
    grades = label.assign_grades(real_ranked)[0]
    never_reviewed = real_ranked["number_of_reviews"].eq(0)

    assert (grades[never_reviewed] == 0).mean() < 0.10
