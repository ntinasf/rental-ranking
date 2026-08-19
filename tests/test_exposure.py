"""Tests for the exposure analysis behind the A/B design.

The one that carries the most weight is
``test_the_module_exposes_no_way_to_compare_ndcg_across_groupings``. This module's whole reason to
exist is that NDCG is not comparable across grouping schemes — the coarsest rung *is* the grading
partition, so its grade distribution is fixed by construction and a model could "improve" there by
doing nothing. A docstring saying so is a wish; a test is a rule.

The rest pin the two metrics against cases whose answer is known by construction, because both are
easy to write plausibly and wrong: ``coverage`` needs the denominator to be what the top-k *could*
have reached rather than what existed, and ``cohort_reach``'s reference is analytic
(``min(k, n) / n``) rather than the cohort's population share.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.cloud.score import MAX_LISTINGS
from rental_ranking.evaluate import exposure
from rental_ranking.features import groups

# --- the rule the module exists to keep ---------------------------------------------------------


def test_the_module_exposes_no_way_to_compare_ndcg_across_groupings() -> None:
    """Changing the key changes the candidate set, the ideal DCG and the floor at once, so a
    rung-1 NDCG beside a rung-3 NDCG compares two quantities that share a name."""
    public = {name for name in vars(exposure) if not name.startswith("_")}
    assert not {name for name in public if "ndcg" in name.lower()}


# --- candidate set profile ------------------------------------------------------------------------


def _population(sizes: dict[str, int]) -> pd.DataFrame:
    rows = []
    for neighbourhood, count in sizes.items():
        rows += [
            {
                "city": "athens",
                "neighbourhood_cleansed": neighbourhood,
                "room_type": "Entire home/apt",
                "accommodates": 2,
            }
        ] * count
    return pd.DataFrame(rows)


def test_the_profile_flags_a_rung_the_endpoint_could_not_serve() -> None:
    """A rung whose largest group exceeds the scoring cap describes a ranking the deployed
    service refuses — that is a serving fact, not a modelling one, and it belongs in the table."""
    population = _population({"a": MAX_LISTINGS + 1})
    profile = exposure.candidate_set_profile(population)

    assert profile.loc["nbhd_room_tier", "over_cap"] == 1
    assert not profile.loc["nbhd_room_tier", "serviceable"]


def test_the_profile_widens_monotonically_as_the_key_drops_columns() -> None:
    population = _population({"a": 30, "b": 20, "c": 10})
    profile = exposure.candidate_set_profile(population)

    assert profile["groups"].is_monotonic_decreasing
    assert profile.loc["city_room", "max"] == 60
    assert profile.loc["nbhd_room_tier", "max"] == 30


def test_the_profile_takes_its_cap_from_the_scoring_script() -> None:
    """Restating 5,000 here would let the two drift apart silently."""
    assert exposure.candidate_set_profile(_population({"a": 5})).loc["city_room", "serviceable"]
    assert MAX_LISTINGS == 5_000


def test_capacity_tier_is_derived_not_read() -> None:
    """A profile computed on different tier bounds than the groups were built with is a lie about
    the groups; deriving it is the same defence ``groups.query_group`` uses."""
    population = _population({"a": 4}).assign(accommodates=[1, 3, 5, 9])
    assert exposure.candidate_set_profile(population).loc["nbhd_room_tier", "groups"] == 4
    assert len(groups.CAPACITY_TIER_LABELS) == 4


def test_a_missing_key_column_raises_rather_than_grouping_on_what_is_left() -> None:
    with pytest.raises(KeyError):
        exposure.candidate_set_profile(_population({"a": 3}).drop(columns="room_type"))


# --- geographic concentration -----------------------------------------------------------------------


def _candidates(geos: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"neighbourhood_cleansed": geos})


def test_a_ranking_that_shows_one_neighbourhood_scores_zero_spread() -> None:
    """The large-area failure mode: the first screen collapses onto wherever the features score
    highest and the guest never learns the wider area had anything else in it."""
    listings = _candidates(["plaka"] * 4 + ["exarchia"] * 4)
    scores = pd.Series([9, 8, 7, 6, 1, 1, 1, 1], dtype="float64")
    out = exposure.geographic_concentration(listings, scores, pd.Series(["g"] * 8), k=4)

    assert out.loc["g", "geos_in_top_k"] == 1
    assert out.loc["g", "coverage"] == 0.5  # one of the two it could have reached
    assert out.loc["g", "entropy"] == 0.0


def test_a_perfectly_spread_top_k_scores_one() -> None:
    listings = _candidates(["plaka", "exarchia", "plaka", "exarchia"])
    scores = pd.Series([9, 8, 1, 1], dtype="float64")
    out = exposure.geographic_concentration(listings, scores, pd.Series(["g"] * 4), k=2)

    assert out.loc["g", "coverage"] == 1.0
    assert out.loc["g", "entropy"] == pytest.approx(1.0)


def test_coverage_is_measured_against_what_the_top_k_could_reach_not_what_exists() -> None:
    """Ten neighbourhoods and three slots: reaching three is perfect, not 30 %."""
    listings = _candidates([f"n{i}" for i in range(10)])
    scores = pd.Series(np.arange(10, 0, -1), dtype="float64")
    out = exposure.geographic_concentration(listings, scores, pd.Series(["g"] * 10), k=3)

    assert out.loc["g", "geos_in_top_k"] == 3
    assert out.loc["g", "coverage"] == 1.0


def test_a_single_neighbourhood_group_reports_no_entropy_rather_than_a_number() -> None:
    """Every rung-1 group is confined to one neighbourhood by construction. Scoring that 0.0 would
    read as maximal collapse when in fact there was nothing to spread."""
    listings = _candidates(["plaka"] * 5)
    out = exposure.geographic_concentration(
        listings, pd.Series(np.arange(5.0)), pd.Series(["g"] * 5), k=3
    )
    assert out.loc["g", "coverage"] == 1.0
    assert np.isnan(out.loc["g", "entropy"])


def test_concentration_is_reported_per_group_not_pooled() -> None:
    listings = _candidates(["a", "b", "c", "d"])
    out = exposure.geographic_concentration(
        listings, pd.Series([2.0, 1.0, 2.0, 1.0]), pd.Series(["g1", "g1", "g2", "g2"]), k=1
    )
    assert out.index.tolist() == ["g1", "g2"]
    assert (out["n"] == 2).all()


# --- cohort reach ------------------------------------------------------------------------------------


def test_the_random_reference_is_the_exact_shuffle_probability() -> None:
    """``min(k, n) / n``, not the cohort's population share. With ten listings and three slots any
    given listing reaches the screen 30 % of the time however large the cohort is."""
    scores = pd.Series(np.arange(10, 0, -1), dtype="float64")
    out = exposure.cohort_reach(scores, pd.Series(["g"] * 10), pd.Series([True] + [False] * 9), k=3)
    assert out["random_rate"] == pytest.approx(0.3)


def test_a_cohort_the_ranker_buries_shows_a_negative_lift() -> None:
    """The measured cold-start shape: the cohort exists, it is orderable, and it sits below where
    a shuffle would put it."""
    scores = pd.Series([9.0, 8.0, 7.0, 2.0, 1.0])
    cohort = pd.Series([False, False, False, True, True])
    out = exposure.cohort_reach(scores, pd.Series(["g"] * 5), cohort, k=2)

    assert out["reach_rate"] == 0.0
    assert out["random_rate"] == pytest.approx(0.4)
    assert out["lift"] == pytest.approx(-0.4)


def test_a_cohort_the_ranker_favours_shows_a_positive_lift() -> None:
    scores = pd.Series([9.0, 8.0, 7.0, 2.0, 1.0])
    cohort = pd.Series([True, True, False, False, False])
    out = exposure.cohort_reach(scores, pd.Series(["g"] * 5), cohort, k=2)
    assert out["lift"] == pytest.approx(0.6)


def test_reach_is_counted_across_groups_not_within_one() -> None:
    scores = pd.Series([2.0, 1.0, 2.0, 1.0])
    out = exposure.cohort_reach(
        scores, pd.Series(["g1", "g1", "g2", "g2"]), pd.Series([True, False, False, True]), k=1
    )
    assert out["cohort"] == 2
    assert out["reached"] == 1  # the g1 member ranks first, the g2 member does not


def test_an_empty_cohort_returns_nan_rather_than_dividing_by_zero() -> None:
    out = exposure.cohort_reach(
        pd.Series([1.0, 2.0]), pd.Series(["g", "g"]), pd.Series([False, False])
    )
    assert out["cohort"] == 0
    assert np.isnan(out["reach_rate"])


def test_the_reference_saturates_when_the_group_is_smaller_than_k() -> None:
    """Six slots and four listings: everybody is on the first screen, so the shuffle reference is
    1.0 and no ranker can show a lift."""
    out = exposure.cohort_reach(
        pd.Series([4.0, 3.0, 2.0, 1.0]), pd.Series(["g"] * 4), pd.Series([True] * 4), k=6
    )
    assert out["random_rate"] == 1.0
    assert out["lift"] == 0.0


# --- re-keying ------------------------------------------------------------------------------------


def test_rung_labels_are_a_plain_rekeying_with_no_minimum_and_no_fallback() -> None:
    """``groups.query_group`` widens the key for groups below the minimum; this asks the different
    question the design document needs — what if the key had been this for everyone — so a group
    of one must survive rather than be pooled."""
    population = _population({"a": 1, "b": 1, "c": 8})
    labels = exposure.rung_labels(population, ["city", "neighbourhood_cleansed", "room_type"])

    assert labels.nunique() == 3
    assert (labels.value_counts() == 1).sum() == 2
    assert groups.MIN_GROUP_SIZE == 5


def test_rung_labels_align_to_the_frame_they_were_given() -> None:
    population = _population({"a": 3, "b": 2})
    labels = exposure.rung_labels(population, ["city", "room_type"])
    assert labels.index.equals(population.index)
    assert labels.nunique() == 1


# --- the demand prior ---------------------------------------------------------------------------


def _training() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["athens"] * 6,
            "neighbourhood_cleansed": ["plaka"] * 3 + ["exarchia"] * 3,
            "grade": [4, 4, 4, 1, 1, 1],
        }
    )


def test_the_prior_ranks_neighbourhoods_by_historical_demand() -> None:
    prior = exposure.demand_prior(_training())
    assert prior[("athens", "plaka")] == 4.0
    assert prior[("athens", "exarchia")] == 1.0


def test_the_prior_carries_a_city_fallback_for_unseen_neighbourhoods() -> None:
    """A neighbourhood the training data never saw must be treated as average, not silently
    excluded from every search the system runs."""
    prior = exposure.demand_prior(_training())
    assert prior[("athens", None)] == pytest.approx(2.5)


def test_the_prior_refuses_a_frame_without_the_target() -> None:
    with pytest.raises(KeyError):
        exposure.demand_prior(_training().drop(columns="grade"))


# --- narrowing ------------------------------------------------------------------------------------


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["athens"] * 6,
            "neighbourhood_cleansed": [
                "plaka",
                "plaka",
                "exarchia",
                "exarchia",
                "kypseli",
                "kypseli",
            ],
        }
    )


def test_narrowing_keeps_only_the_highest_prior_neighbourhoods() -> None:
    prior = pd.Series(
        {
            ("athens", "plaka"): 4.0,
            ("athens", "exarchia"): 3.0,
            ("athens", "kypseli"): 1.0,
            ("athens", None): 2.0,
        }
    )
    kept = exposure.select_geographies(_universe(), prior, k_geo=2)
    assert kept.tolist() == [True, True, True, True, False, False]


def test_a_neighbourhood_absent_from_the_prior_falls_back_rather_than_vanishing() -> None:
    """Dropping it would remove that neighbourhood from every search the system ever serves —
    a supply-side harm invisible in any guest-side metric."""
    prior = pd.Series({("athens", "plaka"): 0.1, ("athens", None): 9.0})
    kept = exposure.select_geographies(_universe(), prior, k_geo=1)
    assert kept.sum() == 2
    assert not kept.iloc[0]  # plaka's own low prior loses to the two fallbacks


def test_narrowing_to_more_geographies_than_exist_keeps_everything() -> None:
    prior = pd.Series({("athens", None): 1.0})
    assert exposure.select_geographies(_universe(), prior, k_geo=99).all()


# --- the first screen ---------------------------------------------------------------------------------


def _screen() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "grade": [4, 3, 2, 0],
            "has_reviews": [True, False, True, False],
            "neighbourhood_cleansed": ["plaka", "plaka", "exarchia", "exarchia"],
        }
    )


def test_the_screen_is_composition_not_a_normalised_metric() -> None:
    """The point of the whole module: a normalised metric cannot compare two retrieval policies
    because it divides by the candidate set. Ten listings shown is ten listings shown."""
    out = exposure.screen_composition(
        _screen(), pd.Series([4.0, 3.0, 2.0, 1.0]), pd.Series(["g"] * 4), k=2
    )
    assert out.loc["g", "shown"] == 2
    assert out.loc["g", "mean_grade"] == 3.5
    assert out.loc["g", "relevant_share"] == 1.0
    assert out.loc["g", "distinct_geos"] == 1
    assert out.loc["g", "cold_share"] == 0.5
    assert out.loc["g", "deserving_cold_share"] == 0.5


def test_a_screen_shorter_than_k_reports_what_it_actually_showed() -> None:
    out = exposure.screen_composition(
        _screen(), pd.Series([4.0, 3.0, 2.0, 1.0]), pd.Series(["g"] * 4), k=99
    )
    assert out.loc["g", "shown"] == 4
    assert out.loc["g", "mean_grade"] == 2.25


# --- the two arms ---------------------------------------------------------------------------------


def _broad() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    candidates = pd.DataFrame(
        {
            "city": ["athens"] * 6,
            "neighbourhood_cleansed": [
                "plaka",
                "plaka",
                "exarchia",
                "exarchia",
                "kypseli",
                "kypseli",
            ],
            "grade": [4, 3, 2, 2, 1, 0],
            "has_reviews": [True, False, True, True, True, True],
        }
    )
    scores = pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    universe = pd.Series(["broad"] * 6)
    prior = pd.Series(
        {
            ("athens", "plaka"): 3.5,
            ("athens", "exarchia"): 2.0,
            ("athens", "kypseli"): 0.5,
            ("athens", None): 2.0,
        }
    )
    return candidates, scores, universe, prior


def test_both_arms_are_ranked_by_the_same_model() -> None:
    """The experiment attributes its effect to the retrieval policy. If the arms differed in the
    ranker too, nothing could be attributed to either."""
    candidates, scores, universe, prior = _broad()
    out = exposure.simulate_arms(candidates, scores, universe, prior, k_geo=1, k=2)

    assert out.loc["control", "searches"] == out.loc["treatment", "searches"] == 1
    assert out.loc["control", "median_candidates"] == 6
    assert out.loc["treatment", "median_candidates"] == 2  # plaka only


def test_narrowing_bounds_the_geographies_the_screen_can_show() -> None:
    """Worth pinning because the intuition runs the other way: narrowing to k geographies cannot
    increase geographic spread, it caps it."""
    candidates, scores, universe, prior = _broad()
    for k_geo in (1, 2, 3):
        out = exposure.simulate_arms(candidates, scores, universe, prior, k_geo=k_geo, k=6)
        assert out.loc["treatment", "distinct_geos"] <= k_geo


def test_narrowing_to_everything_reproduces_the_control_arm() -> None:
    candidates, scores, universe, prior = _broad()
    out = exposure.simulate_arms(candidates, scores, universe, prior, k_geo=99, k=3)
    assert out.loc["treatment", "mean_grade"] == out.loc["control", "mean_grade"]
