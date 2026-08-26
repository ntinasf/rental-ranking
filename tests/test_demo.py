"""Tests for the endpoint demonstration harness.

Two of these guard the demonstration's *meaning* rather than its mechanics, which is the point of
the module. ``build_payload`` must refuse to send the target: a request carrying ``grade`` would
still return a beautiful ranking, and nothing in the output would reveal that the demonstration
had become circular. And ``explain`` must refuse a response whose ids it cannot find, because
joining a ranking to the wrong truth produces a plausible table rather than an error.

The rest pin the request shapes against a scoring script that has already been surprised once:
the cold-start body crashed the container with an unhandled LightGBM ``ValueError`` because a
column that is present-and-null for every listing arrives as ``object``, not ``float64``.
"""

import json

import numpy as np
import pandas as pd
import pytest

from rental_ranking.cloud import demo
from rental_ranking.features import groups
from rental_ranking.train.lambdamart import restore_dtypes

# --- fixtures ---------------------------------------------------------------------------------


def _listings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "grade": [4, 2, 0],
            "blocked_fraction_90": [0.9, 0.3, 0.0],
            "query_group": [7, 7, 7],
            "number_of_reviews": [100, 50, 1],
            "rating_shrunk": [4.9, 4.7, 4.6],
            "reviews_per_month": [2.0, 1.0, np.nan],
            "price": [80.0, 90.0, 70.0],
            "host_is_superhost": [True, False, False],
            "listing_age_days": [900.0, 400.0, 30.0],
        }
    )


_FEATURES = ("number_of_reviews", "rating_shrunk", "reviews_per_month", "price")


def _response(order: list[str], scores: list[float]) -> dict:
    return {
        "ranked": [
            {"id": i, "score": s, "rank": r + 1} for r, (i, s) in enumerate(zip(order, scores))
        ],
        "n_listings": len(order),
    }


# --- building requests --------------------------------------------------------------------------


@pytest.mark.parametrize("leaked", ["grade", "blocked_fraction_90", "query_group", "cluster_id"])
def test_the_target_can_never_be_put_in_a_request(leaked: str) -> None:
    """The failure this prevents is silent: the ranking would look excellent."""
    with pytest.raises(ValueError, match="refusing to send"):
        demo.build_payload(_listings(), [*_FEATURES, leaked])


def test_a_payload_carries_the_id_plus_the_features_in_model_order() -> None:
    body = demo.build_payload(_listings(), _FEATURES)
    assert list(body["listings"][0]) == ["id", *_FEATURES]
    assert len(body["listings"]) == 3


def test_a_payload_is_json_serialisable_with_nan_as_null() -> None:
    body = demo.build_payload(_listings(), _FEATURES)
    round_tripped = json.loads(json.dumps(body))
    assert round_tripped["listings"][2]["reviews_per_month"] is None


def test_numpy_scalars_do_not_survive_into_the_body() -> None:
    body = demo.build_payload(_listings(), _FEATURES)
    assert all(
        not isinstance(value, np.generic) for row in body["listings"] for value in row.values()
    )


def test_perturb_changes_one_listing_and_leaves_the_original_alone() -> None:
    body = demo.build_payload(_listings(), _FEATURES)
    changed = demo.perturb(body, "b", {"price": 5.0})

    assert changed["listings"][1]["price"] == 5.0
    assert body["listings"][1]["price"] == 90.0  # the counterfactual must not edit the control


def test_perturb_names_the_listing_it_could_not_find() -> None:
    with pytest.raises(KeyError, match="zzz"):
        demo.perturb(demo.build_payload(_listings(), _FEATURES), "zzz", {"price": 1.0})


def test_blank_history_nulls_review_fields_and_touches_nothing_else() -> None:
    body = demo.build_payload(_listings(), _FEATURES)
    cold = demo.blank_history(body)

    assert [row["number_of_reviews"] for row in cold["listings"]] == [0, 0, 0]
    assert [row["rating_shrunk"] for row in cold["listings"]] == [None, None, None]
    assert [row["price"] for row in cold["listings"]] == [80.0, 90.0, 70.0]


def test_blank_history_ignores_columns_the_request_does_not_carry() -> None:
    """``COLD_START_BLANKS`` names 14 columns; a request with 4 must not gain the other 10."""
    body = demo.build_payload(_listings(), _FEATURES)
    cold = demo.blank_history(body)
    assert set(cold["listings"][0]) == set(body["listings"][0])


# --- reading responses ----------------------------------------------------------------------------


def test_truth_frame_is_indexed_by_id_and_holds_the_grades() -> None:
    truth = demo.truth_frame(_listings())
    assert truth.index.tolist() == ["a", "b", "c"]
    assert truth.loc["a", "grade"] == 4


def test_rank_of_reads_the_position_back() -> None:
    response = _response(["b", "a", "c"], [2.0, 1.0, 0.0])
    assert demo.rank_of(response, "a") == 2
    with pytest.raises(KeyError, match="zzz"):
        demo.rank_of(response, "zzz")


def test_explain_orders_by_rank_and_flags_the_cut_off() -> None:
    table = demo.explain(
        _response(["b", "a", "c"], [2.0, 1.0, 0.0]), demo.truth_frame(_listings()), k=2
    )
    assert table["id"].tolist() == ["b", "a", "c"]
    assert table["grade"].tolist() == [2, 4, 0]
    assert table["in_top_k"].tolist() == [True, True, False]


def test_explain_refuses_an_error_response_instead_of_rendering_it() -> None:
    with pytest.raises(ValueError, match="not a ranking"):
        demo.explain({"error": "bad level"}, demo.truth_frame(_listings()))


def test_explain_refuses_ids_the_truth_frame_does_not_have() -> None:
    """Joining a ranking to the wrong truth returns a table, not an exception. So check here."""
    response = _response(["a", "b", "zzz"], [2.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="absent from the truth frame"):
        demo.explain(response, demo.truth_frame(_listings()))


def test_query_quality_scores_all_four_rankers_on_the_same_candidates() -> None:
    perfect = _response(["a", "b", "c"], [2.0, 1.0, 0.0])
    quality = demo.query_quality(perfect, demo.truth_frame(_listings()), k=10)

    assert quality["endpoint"] == pytest.approx(1.0)
    assert quality["n_listings"] == 3
    assert quality["n_relevant"] == 1
    assert set(quality.index) >= {"endpoint", "baseline_reviews", "baseline_price_rating", "random"}


def test_query_quality_puts_a_reversed_ranking_below_the_floor() -> None:
    reversed_order = _response(["c", "b", "a"], [2.0, 1.0, 0.0])
    quality = demo.query_quality(reversed_order, demo.truth_frame(_listings()), k=10)
    assert quality["endpoint"] < quality["random"]


# --- the network half -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (
            "https://x.italynorth.inference.ml.azure.com/swagger.json",
            "https://x.italynorth.inference.ml.azure.com/score",
        ),
        (
            "https://x.italynorth.inference.ml.azure.com",
            "https://x.italynorth.inference.ml.azure.com/score",
        ),
        (
            "https://x.italynorth.inference.ml.azure.com/",
            "https://x.italynorth.inference.ml.azure.com/score",
        ),
    ],
)
def test_a_uri_that_is_not_the_scoring_path_is_caught_before_the_call(
    monkeypatch, given: str, expected: str
) -> None:
    """The Studio page lists the Swagger URI one line below the REST endpoint. Posting to it
    returns HTTP 424 wrapping a 405, which names neither the URL nor the mistake."""
    monkeypatch.setattr(demo, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv(demo.URI_VAR, given)
    monkeypatch.setenv(demo.KEY_VAR, "not-a-real-key")

    with pytest.raises(RuntimeError) as caught:
        demo.endpoint_address()
    assert expected in str(caught.value)


def test_a_scoring_uri_is_accepted_and_normalised(monkeypatch) -> None:
    monkeypatch.setattr(demo, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv(demo.URI_VAR, "  https://x.inference.ml.azure.com/score/ ")
    monkeypatch.setenv(demo.KEY_VAR, "k")
    assert demo.endpoint_address() == ("https://x.inference.ml.azure.com/score", "k")


def test_a_missing_address_names_both_variables_and_the_commands(monkeypatch) -> None:
    monkeypatch.setattr(demo, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv(demo.URI_VAR, raising=False)
    monkeypatch.delenv(demo.KEY_VAR, raising=False)

    with pytest.raises(RuntimeError) as caught:
        demo.endpoint_address()
    message = str(caught.value)
    assert demo.URI_VAR in message and demo.KEY_VAR in message
    assert "az ml online-endpoint show" in message
    assert "never commit" in message


# --- the bug the cold-start request found ----------------------------------------------------------


def _metadata() -> dict:
    return {
        "features": ["price", "rating_shrunk", "room_type"],
        "categories": {"room_type": ["a", "b"]},
    }


def test_a_column_present_and_null_for_every_row_scores_rather_than_crashing() -> None:
    """Regression. ``json.loads`` gives an all-null column ``object`` dtype and
    LightGBM raises ``pandas dtypes must be int, float or bool`` — unhandled, so a 500. An absent
    column and an all-null column are the same request; only the serialiser differs."""
    frame = pd.DataFrame(
        {"price": [80.0, 90.0], "rating_shrunk": [None, None], "room_type": ["a", "b"]}
    )
    assert frame["rating_shrunk"].dtype == object

    out = restore_dtypes(frame, _metadata())
    assert out["rating_shrunk"].dtype == "float64"
    assert out["rating_shrunk"].isna().all()


def test_a_non_numeric_value_in_a_numeric_column_is_named_not_coerced() -> None:
    frame = pd.DataFrame(
        {"price": ["cheap", 90.0], "rating_shrunk": [4.9, 4.8], "room_type": ["a", "b"]}
    )
    with pytest.raises(ValueError, match="'price' carries 1 non-numeric"):
        restore_dtypes(frame, _metadata())


def test_an_absent_column_and_an_all_null_column_produce_the_same_matrix() -> None:
    metadata = _metadata()
    absent = restore_dtypes(pd.DataFrame({"price": [80.0], "room_type": ["a"]}), metadata)
    explicit = restore_dtypes(
        pd.DataFrame({"price": [80.0], "rating_shrunk": [None], "room_type": ["a"]}), metadata
    )
    pd.testing.assert_frame_equal(absent, explicit)


# --- the search key -----------------------------------------------------------------------------


def test_every_offered_party_size_lands_in_the_tier_it_is_offered_under() -> None:
    """The drift guard. ``tier_guest_choices`` derives the party sizes and ``guests_to_tier``
    re-derives the tier; if either stopped reading ``CAPACITY_TIER_BOUNDS`` the console would
    offer a guest count under a label that sends it somewhere else."""
    for tier, party_sizes in demo.tier_guest_choices().items():
        for guests in party_sizes:
            assert demo.guests_to_tier(guests) == tier


def test_the_guest_choices_cover_every_tier_the_groups_were_built_on() -> None:
    assert set(demo.tier_guest_choices()) == set(groups.CAPACITY_TIER_LABELS)


def test_a_party_size_outside_the_bounds_is_refused_not_clamped() -> None:
    with pytest.raises(ValueError, match="outside the capacity tiers"):
        demo.guests_to_tier(0)


def test_guests_to_tier_agrees_with_the_module_that_built_the_groups() -> None:
    """Two implementations of one rule is one too many; this pins them together."""
    frame = pd.DataFrame({"accommodates": [1, 2, 3, 4, 5, 7, 8, 30]})
    expected = groups.capacity_tier(frame).astype(str).tolist()
    assert [demo.guests_to_tier(n) for n in frame["accommodates"]] == expected


def _group_frame(neighbourhoods: list[str], tiers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["athens"] * len(neighbourhoods),
            "room_type": ["Entire home/apt"] * len(neighbourhoods),
            "neighbourhood_cleansed": neighbourhoods,
            "capacity_tier": tiers,
        }
    )


def test_group_key_names_the_neighbourhood_when_the_group_has_only_one() -> None:
    key = demo.group_key(_group_frame(["Plaka", "Plaka"], ["3-4", "3-4"]))
    assert key["neighbourhood"] == "Plaka"
    assert key["neighbourhoods"] == 1


def test_group_key_refuses_to_name_a_neighbourhood_for_a_pooled_group() -> None:
    """A group formed at a fallback rung spans several; naming the first would be a lie."""
    key = demo.group_key(_group_frame(["Plaka", "Exarchia", "Kolonaki"], ["3-4"] * 3))
    assert key["neighbourhood"] is None
    assert key["neighbourhoods"] == 3


def test_the_readable_description_columns_are_not_features() -> None:
    """``name`` and the neighbourhood are joined back for a human. If either reached the payload
    the demonstration would be sending the endpoint something the model never trained on."""
    listings = _listings().assign(name=["a title", "b title", "c title"])
    body = demo.build_payload(listings, _FEATURES)
    assert all("name" not in row for row in body["listings"])
