"""Tests for the ablation table's feature bookkeeping.

The one that carries the most weight is ``test_a_renamed_establishment_feature_raises``. Every
other failure in this module is loud; that one is silent. If a column is renamed and the block
quietly resolves to seven features instead of eight, `minus establishment` still runs, still
produces a plausible number, and still gets reported as "denied all 8 establishment features" —
which is the sentence the provenance argument rests on.

The fits themselves are not tested here: they are four cross-validations per ablation and the
thing worth pinning is which columns go into them, not that LightGBM works.
"""

import pandas as pd
import pytest

from rental_ranking.train import ablations

# --- a matrix with one column of every kind, and nothing else ---------------------------------


def _matrix(extra: dict[str, list] | None = None) -> pd.DataFrame:
    columns = {
        "id": ["a", "b"],
        "query_group": [0, 0],
        "cluster_id": [0, 1],
        "grade": [1, 2],
        "blocked_fraction_90": [0.1, 0.2],
        # establishment, all eight
        **{name: [0, 1] for name in ablations.ESTABLISHMENT},
        # quality, which is deliberately NOT establishment
        "review_scores_value": [4.0, 5.0],
        "rating_shrunk": [4.5, 4.6],
        # the derivable blocks
        "amenity_count": [3, 4],
        "amenity_kitchen": [1, 1],
        "km_to_nearest_anchor": [1.0, 2.0],
        "density_1km": [10, 20],
        "nbhd_listings": [50, 60],
        "price_vs_nbhd": [0.9, 1.1],
        # a plain structural feature
        "price": [80.0, 90.0],
    }
    columns.update(extra or {})
    return pd.DataFrame(columns)


# --- the rule this module exists to keep -------------------------------------------------------


def test_a_renamed_establishment_feature_raises() -> None:
    """A block that silently resolves to seven of eight still produces a number, and the number
    gets reported as the eight-feature result."""
    frame = _matrix().drop(columns=["host_tenure_months"])

    with pytest.raises(ValueError, match="host_tenure_months"):
        ablations.block_members(list(frame.columns), "establishment")


def test_establishment_excludes_quality_and_includes_tenure() -> None:
    """Establishment is how long the listing has run and how much traffic it has seen — not how
    good it is. Taking `rating_shrunk` instead of `host_tenure_months` gives a different set that
    also has eight members, so the count alone does not pin it."""
    assert "host_tenure_months" in ablations.ESTABLISHMENT
    assert "rating_shrunk" not in ablations.ESTABLISHMENT
    assert not [c for c in ablations.ESTABLISHMENT if c.startswith("review_scores")]
    assert len(ablations.ESTABLISHMENT) == 8


def test_the_table_offers_no_way_to_include_a_baseline() -> None:
    """A heuristic has no feature set, so it cannot answer what the model loses without a block —
    and its difference against the full model is the reported headline with the sign flipped."""
    names = set(ablations.feature_sets(_matrix()))

    assert not {n for n in names if "review" == n or n.startswith("baseline")}
    assert all(isinstance(v, list) and v for v in ablations.feature_sets(_matrix()).values())


# --- blocks are derived, not typed out ---------------------------------------------------------


def test_a_new_amenity_column_joins_the_block_without_being_listed() -> None:
    frame = _matrix({"amenity_sauna": [0, 1]})

    assert "amenity_sauna" in ablations.block_members(list(frame.columns), "amenities")
    assert "amenity_sauna" not in ablations.feature_sets(frame)["minus amenities"]


def test_the_neighbourhood_block_catches_the_column_without_the_prefix() -> None:
    members = ablations.block_members(list(_matrix().columns), "neighbourhood")

    assert set(members) == {"nbhd_listings", "price_vs_nbhd"}


def test_a_stale_prefix_rule_raises_rather_than_emptying_an_ablation() -> None:
    frame = _matrix().drop(columns=["km_to_nearest_anchor", "density_1km"])

    with pytest.raises(ValueError, match="matched no column"):
        ablations.block_members(list(frame.columns), "spatial")


def test_an_unknown_block_raises() -> None:
    with pytest.raises(KeyError, match="unknown block"):
        ablations.block_members(list(_matrix().columns), "host")


# --- the feature sets themselves ---------------------------------------------------------------


def test_the_reference_name_carries_the_count_it_was_fitted_on() -> None:
    frame = _matrix()
    n = len(frame.columns) - 5  # five identifier/target columns

    assert ablations.full_name(frame) == f"full ({n})"
    assert len(ablations.feature_sets(frame)[f"full ({n})"]) == n


def test_each_minus_set_drops_exactly_its_own_block() -> None:
    frame = _matrix()
    sets = ablations.feature_sets(frame)
    full = set(sets[ablations.full_name(frame)])

    for block, name in (
        ("spatial", "minus spatial"),
        ("neighbourhood", "minus neighbourhood"),
        ("amenities", "minus amenities"),
        ("establishment", "minus establishment"),
    ):
        members = set(ablations.block_members(list(frame.columns), block))
        assert full - set(sets[name]) == members, name


def test_the_amenity_count_variant_keeps_the_control_and_drops_the_buckets() -> None:
    sets = ablations.feature_sets(_matrix())

    kept = [c for c in sets["amenities: count"] if c.startswith("amenity_")]
    assert kept == ["amenity_count"]


def test_establishment_only_is_exactly_the_block() -> None:
    frame = _matrix()

    assert ablations.feature_sets(frame)["establishment only"] == list(
        ablations.block_members(list(frame.columns), "establishment")
    )


def test_establishment_only_is_the_complement_of_minus_establishment() -> None:
    """The two halves of the provenance argument have to partition the feature set, or they are
    not measuring the same block from two sides."""
    frame = _matrix()
    sets = ablations.feature_sets(frame)
    full = set(sets[ablations.full_name(frame)])

    assert set(sets["establishment only"]) | set(sets["minus establishment"]) == full
    assert not set(sets["establishment only"]) & set(sets["minus establishment"])


def test_run_ablations_refuses_a_set_without_its_reference() -> None:
    frame = _matrix()

    with pytest.raises(KeyError, match="reference"):
        ablations.run_ablations(frame, pd.Series([1, 1]), sets={"minus spatial": ["price"]})


def test_extra_scores_on_a_different_index_are_refused() -> None:
    """A rebuilt matrix supplies scores from its own fits. If its rows do not correspond to this
    table's, the paired difference pairs each listing's score with another listing's grade — and
    returns a number rather than an error."""
    frame = _matrix()
    stranger = pd.Series([0.1, 0.2, 0.3], index=[7, 8, 9])

    with pytest.raises(ValueError, match="different index"):
        ablations.run_ablations(
            frame,
            pd.Series([1, 1]),
            sets={ablations.full_name(frame): ["price"]},
            extra_scores={"rebuilt": (stranger, 92)},
        )


def test_the_flag_vocabulary_is_pinned_by_criterion_not_by_luck() -> None:
    """k=50 by frequency reproduces the reported 0.7260; within_group_variance gives 0.7188. The
    pair is a recovered fact, so a silent change to either would break reproduction."""
    assert ablations.FLAG_VOCABULARY_SIZE == 50
    assert ablations.FLAG_VOCABULARY_CRITERION == "frequency"
