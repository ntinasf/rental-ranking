"""Tests for the exposure-disparity measurement.

The one that carries the most weight is
``test_a_cohort_ranked_exactly_as_it_deserves_shows_no_amplification``. A cohort can reach the
first screen less often purely by being less in demand, and the obvious implementation — compare
the cohort's reach against a shuffle, or against the other cohort's reach — reports that as the
ranker's doing. Referencing the *ideal* ranking is what separates the market from the model, and
it is the difference between a finding and an artefact.

The rest pin cases whose answer is known by construction: a ranker that reproduces the truth is
displaced by exactly zero, groups of one have no position rather than the first one, and groups
of ``k`` or fewer must be excluded rather than counted as evidence of no effect.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.evaluate import fairness

# --- the rules the module exists to keep --------------------------------------------------------


def test_truth_defaults_to_the_training_target_not_the_raw_label() -> None:
    """Measuring against a finer truth than the ranker was trained on scores it for a coarsening
    it was never asked to undo, and confounds that with a cohort penalty."""
    assert fairness.DEFAULT_TRUTH == "grade"


def test_a_cohort_ranked_exactly_as_it_deserves_shows_no_amplification() -> None:
    """The reference is the ideal ranking, not a shuffle and not the other cohort.

    Here the cohort is genuinely less in demand — every one of its listings sits below every
    other listing on the truth — and the ranker orders by the truth exactly. Its reach is
    therefore low, and its *amplification* must be zero: the ranker did nothing to it.
    """
    n = 40
    truth = pd.Series(np.arange(n, dtype="float64"))
    cohort = pd.Series(truth < 10)  # the ten least in demand
    groups = pd.Series(np.zeros(n, dtype="int64"))

    table = fairness.exposure_amplification(truth, groups, truth, cohort, k=10)

    assert table.loc["cohort", "reach"] == 0.0, "the cohort genuinely reaches nothing"
    assert table.loc["cohort", "amplification"] == pytest.approx(0.0)
    assert table.loc["rest", "amplification"] == pytest.approx(0.0)


def test_groups_no_larger_than_k_are_excluded_rather_than_counted_as_no_effect() -> None:
    """Every listing in such a group reaches the first screen under any ranking, so the group
    reports an exact zero disparity — including them shrinks every estimate toward nothing."""
    groups = pd.Series([0] * 8 + [1] * 30)
    keep = fairness.informative_groups(groups, k=10)

    assert not keep[:8].any(), "a group of 8 cannot inform a top-10 disparity"
    assert keep[8:].all()


# --- position and displacement --------------------------------------------------------------------


def test_a_ranker_that_reproduces_the_truth_is_displaced_by_zero() -> None:
    truth = pd.Series([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    groups = pd.Series([0, 0, 0, 1, 1, 1])

    assert fairness.displacement(truth, groups, truth).abs().max() == pytest.approx(0.0)


def test_burying_the_best_listing_displaces_it_by_the_full_group_length() -> None:
    truth = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    inverted = -truth
    groups = pd.Series(np.zeros(5, dtype="int64"))

    displaced = fairness.displacement(inverted, groups, truth)

    assert displaced.iloc[0] == pytest.approx(-1.0), "best listing sent to last place"
    assert displaced.iloc[-1] == pytest.approx(1.0)
    assert displaced.iloc[2] == pytest.approx(0.0), "the middle cannot move"


def test_position_is_a_fraction_so_groups_of_different_sizes_are_comparable() -> None:
    values = pd.Series([2.0, 1.0, 4.0, 3.0, 2.0, 1.0])
    groups = pd.Series([0, 0, 1, 1, 1, 1])

    position = fairness.within_group_position(values, groups)

    assert position.iloc[0] == pytest.approx(0.0)
    assert position.iloc[1] == pytest.approx(1.0)
    assert position.iloc[2] == pytest.approx(0.0)
    assert position.iloc[3] == pytest.approx(1 / 3)


def test_a_group_of_one_has_no_position_rather_than_the_first_one() -> None:
    """0.0 would read as 'ranked first', which is a claim; there is no position to hold."""
    position = fairness.within_group_position(pd.Series([1.0]), pd.Series([0]))

    assert position.isna().all()


def test_ties_break_on_row_order_not_on_the_truth() -> None:
    """A tie-break that consulted the grade would manufacture the disparity being measured."""
    truth = pd.Series([1.0, 5.0, 1.0, 5.0])
    flat = pd.Series([0.0, 0.0, 0.0, 0.0])
    groups = pd.Series([0, 0, 0, 0])

    position = fairness.within_group_position(flat, groups)

    assert list(position) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert fairness.displacement(flat, groups, truth).iloc[1] < 0, (
        "a genuinely good listing left in row order must read as under-ranked"
    )


# --- the paired cohort gap --------------------------------------------------------------------


def _two_groups(bury: bool) -> tuple[pd.Series, ...]:
    """Two groups of eight, four cohort members each, optionally ranked below their truth.

    Members are **interleaved** through the truth order rather than sitting at the bottom of it.
    A cohort already ranked last cannot be buried any further, so a fixture built that way
    reports zero displacement no matter how broken the ranker is — and would pass a test of a
    function that did nothing.
    """
    rows = []
    for group in (0, 1):
        for index in range(8):
            member = index % 2 == 0
            truth = float(index)
            score = truth - (10.0 if (member and bury) else 0.0)
            rows.append({"group": group, "cohort": member, "truth": truth, "score": score})
    frame = pd.DataFrame(rows)
    return frame["score"], frame["group"], frame["truth"], frame["cohort"]


def test_a_cohort_buried_in_every_group_shows_a_negative_gap() -> None:
    score, group, truth, cohort = _two_groups(bury=True)
    displaced = fairness.displacement(score, group, truth)

    gap = fairness.cohort_gap(displaced, group, cohort)

    assert len(gap) == 2
    assert (gap < 0).all()


def test_a_cohort_ranked_faithfully_shows_no_gap() -> None:
    score, group, truth, cohort = _two_groups(bury=False)
    displaced = fairness.displacement(score, group, truth)

    assert fairness.cohort_gap(displaced, group, cohort).abs().max() == pytest.approx(0.0)


def test_a_group_without_enough_of_either_side_contributes_nothing() -> None:
    """A rate over one or two members is a coin flip the group average would carry at full
    weight."""
    score, group, truth, cohort = _two_groups(bury=True)
    cohort = cohort & (score.index % 8 == 0)  # one member per group

    displaced = fairness.displacement(score, group, truth)

    assert fairness.cohort_gap(displaced, group, cohort, min_members=3).empty


def test_the_gap_is_paired_so_group_composition_cannot_drive_it() -> None:
    """One group is all-cohort and one is all-rest. Pooled, that is a large spurious difference;
    paired, neither group qualifies and the answer is empty."""
    frame = pd.DataFrame(
        {
            "group": [0] * 6 + [1] * 6,
            "cohort": [True] * 6 + [False] * 6,
            "truth": list(range(6)) * 2,
        }
    )
    frame["score"] = frame["truth"] - frame["group"] * 100  # a huge between-group score shift
    displaced = fairness.displacement(frame["score"], frame["group"], frame["truth"])

    assert fairness.cohort_gap(displaced, frame["group"], frame["cohort"]).empty


# --- dose response --------------------------------------------------------------------------


def test_dose_response_recovers_a_monotone_trend_without_any_threshold() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for group in range(12):
        for index in range(12):
            dose = float(index)
            truth = float(rng.integers(0, 5))
            rows.append(
                {
                    "group": group,
                    "dose": dose,
                    "truth": truth,
                    # the more listings the host has, the harder this ranker buries it
                    "score": truth - 0.5 * dose,
                }
            )
    frame = pd.DataFrame(rows)
    displaced = fairness.displacement(frame["score"], frame["group"], frame["truth"])

    table = fairness.dose_response(
        displaced,
        frame["group"],
        frame["dose"],
        bands=[-1, 3, 7, 11],
        labels=["low", "mid", "high"],
    )

    assert table["displacement"].is_monotonic_decreasing
    assert table.loc["low", "displacement"] > 0 > table.loc["high", "displacement"]
