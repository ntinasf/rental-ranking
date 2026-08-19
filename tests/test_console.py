"""Tests for the demonstration console.

The one that matters most is ``test_every_preset_column_is_an_editable_field``. A preset writes
into the *form*, and the form is what gets sent — so a preset naming a field the form does not
render would silently drop it: the page would show one edit and the endpoint would receive
another, and the rank move would be attributed to the wrong change. That is exactly what happened
on the first pass (``reviews_same_season_ly`` was in the preset and not in the form, and the
counterfactual reported 1 -> 8 where the CLI reported 1 -> 15).

The second is ``FIXED_CONTEXT``: ``city`` and ``room_type`` are the query-group key and constant
inside a group, so an editable one would build a candidate the search never contained.
"""

import pytest

from rental_ranking.cloud import console, demo

# --- the form ------------------------------------------------------------------------------


def test_every_preset_column_is_an_editable_field() -> None:
    """A preset that names a field the form does not render is silently dropped on submit."""
    assert console.preset_columns() <= console.editable_columns()


def test_the_query_group_key_is_never_editable() -> None:
    for column in console.FIXED_CONTEXT:
        assert column not in console.editable_columns()


def _metadata() -> dict:
    return {
        "features": ["price", "number_of_reviews", "host_is_superhost", "host_is_local", "city"],
        "categories": {"host_is_local": ["foreign", "local", "unknown"], "city": ["athens"]},
    }


def _listing() -> dict:
    return {
        "id": "a",
        "price": 80.0,
        "number_of_reviews": 12,
        "host_is_superhost": True,
        "host_is_local": "local",
        "city": "athens",
    }


def test_field_spec_reads_the_kind_from_the_value_and_the_metadata() -> None:
    kinds = {f["name"]: f["kind"] for f in console.field_spec(_listing(), _metadata())}
    assert kinds["price"] == "number"
    assert kinds["number_of_reviews"] == "number"
    assert kinds["host_is_superhost"] == "boolean"
    assert kinds["host_is_local"] == "choice"


def test_field_spec_drops_the_query_group_key_even_though_the_model_serves_it() -> None:
    assert "city" not in {f["name"] for f in console.field_spec(_listing(), _metadata())}


def test_field_spec_never_offers_an_input_the_model_does_not_carry() -> None:
    """A rendered field the endpoint ignores is a control that lies about having an effect."""
    served = set(_metadata()["features"])
    assert {f["name"] for f in console.field_spec(_listing(), _metadata())} <= served


def test_a_choice_field_carries_its_levels() -> None:
    spec = {f["name"]: f for f in console.field_spec(_listing(), _metadata())}
    assert spec["host_is_local"]["choices"] == ["foreign", "local", "unknown"]


# --- coercion ------------------------------------------------------------------------------


def _spec() -> list[dict]:
    return console.field_spec(_listing(), _metadata())


def test_an_empty_box_means_null_not_an_error() -> None:
    """A caller saying "this listing has no value here" is describing the world, not erring."""
    assert console.coerce_edits({"price": ""}, _spec()) == {"price": None}


def test_numbers_arrive_as_numbers_and_checkboxes_as_booleans() -> None:
    out = console.coerce_edits({"price": "95.5", "host_is_superhost": "on"}, _spec())
    assert out == {"price": 95.5, "host_is_superhost": True}


def test_a_level_the_model_never_saw_is_refused_before_it_reaches_the_endpoint() -> None:
    with pytest.raises(ValueError, match="not one of"):
        console.coerce_edits({"host_is_local": "martian"}, _spec())


def test_a_non_numeric_box_is_refused_locally() -> None:
    with pytest.raises(ValueError, match="is not a number"):
        console.coerce_edits({"price": "cheap"}, _spec())


def test_a_field_that_is_not_editable_cannot_be_smuggled_in_by_a_hand_made_request() -> None:
    with pytest.raises(ValueError, match="not an editable field"):
        console.coerce_edits({"grade": "4"}, _spec())


# --- presets -------------------------------------------------------------------------------


def test_strip_review_history_is_the_transcript_counterfactual() -> None:
    assert console.preset_edits("strip review history", _listing()) == demo.COUNTERFACTUAL_BLANKS


def test_double_the_price_is_resolved_against_the_listing() -> None:
    assert console.preset_edits("double the price", _listing()) == {"price": 160.0}
    assert console.preset_edits("double the price", {"price": None}) == {"price": None}


def test_an_unknown_preset_raises_rather_than_doing_nothing() -> None:
    with pytest.raises(KeyError, match="unknown preset"):
        console.preset_edits("make it good", _listing())


def test_a_preset_returns_a_copy_the_caller_cannot_use_to_edit_the_preset() -> None:
    edits = console.preset_edits("strip review history", _listing())
    edits["number_of_reviews"] = 999
    assert console.PRESETS["strip review history"]["number_of_reviews"] == 0


# --- what the page draws ---------------------------------------------------------------------


def _truth():
    import pandas as pd

    return pd.DataFrame(
        {
            "grade": [4, 2, 0],
            "blocked_fraction_90": [0.9, 0.3, 0.0],
            "number_of_reviews": [100, 50, 1],
            "rating_shrunk": [4.9, 4.7, 4.6],
            "reviews_per_month": [2.0, 1.0, 0.1],
            "price": [80.0, 90.0, 70.0],
            "host_is_superhost": [True, False, False],
            "listing_age_days": [900.0, 400.0, 30.0],
        },
        index=pd.Index(["a", "b", "c"], name="id"),
    )


def _resp(order):
    return {"ranked": [{"id": i, "score": -n, "rank": n + 1} for n, i in enumerate(order)]}


def test_the_view_is_json_serialisable_and_marks_the_edited_row() -> None:
    import json

    view = console.ranking_view(_resp(["a", "b", "c"]), _truth(), "b")
    json.dumps(view)
    assert view["edited"] == "b"
    assert [r["rank"] for r in view["rows"]] == [1, 2, 3]


def test_movement_is_null_when_there_is_nothing_to_compare_against() -> None:
    assert console.ranking_view(_resp(["a", "b", "c"]), _truth(), "a")["moved"] is None


def test_movement_reports_both_the_rank_and_the_metric() -> None:
    before, after = _resp(["a", "b", "c"]), _resp(["c", "b", "a"])
    moved = console.ranking_view(after, _truth(), "a", before=before)["moved"]

    assert (moved["rank_before"], moved["rank_after"]) == (1, 3)
    assert moved["ndcg_after"] < moved["ndcg_before"]


# --- the display cap ----------------------------------------------------------------------------


def test_the_table_is_capped_but_reports_the_true_size() -> None:
    """The largest sealed group is 2,042 listings; the page draws a hundred."""
    view = console.ranking_view(_resp(["a", "b", "c"]), _truth(), "a", limit=2)
    assert len(view["rows"]) == 2
    assert view["n_rows"] == 3


def test_the_edited_listing_is_shown_even_when_it_falls_past_the_cap() -> None:
    """The largest real query group is 2,088 listings; an edit that buries a listing must still
    show where it went, or the demonstration silently loses its own subject."""
    view = console.ranking_view(_resp(["a", "b", "c"]), _truth(), "c", limit=1)
    assert [row["id"] for row in view["rows"]] == ["a", "c"]
    assert view["n_rows"] == 3
