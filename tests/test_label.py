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
