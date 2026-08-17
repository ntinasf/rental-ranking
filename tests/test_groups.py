"""Tests for rental_ranking.features.groups.

Every failure this file guards against is silent. A cluster id that drops a row simply never
reaches the split; a query group that drops one is absent from the ranking; a group array that
is not sorted trains LightGBM on queries which mix two searches and still prints a metric. So
coverage (every listing gets exactly one id) and separation (unrelated listings never share
one) are asserted in both directions throughout.
"""

import warnings

import pandas as pd
import pytest

from rental_ranking.data.filters import filter_listings
from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import groups
from rental_ranking.features.label import occupancy_label

_DEFAULTS = {"host_id": "h1", "latitude": 40.6401, "longitude": 22.9444, "accommodates": 4}

#: The four columns a query group is keyed on, before the capacity tier is derived from
#: ``accommodates``. Kept apart from ``_DEFAULTS`` so the cluster fixtures stay minimal.
_GROUP_DEFAULTS = {
    "city": "athens",
    "neighbourhood_cleansed": "Kolonaki",
    "room_type": "Entire home/apt",
    "accommodates": 4,
}


def _listings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_DEFAULTS, **row} for row in rows])


def _ranked(*blocks: tuple[int, dict]) -> pd.DataFrame:
    """A ranked population from ``(row count, key overrides)`` blocks, in block order.

    Row order is what the assertions read — block one occupies the first rows — so a test can
    say "these six share a group and those three do not" positionally.
    """
    return pd.DataFrame(
        [{**_GROUP_DEFAULTS, **overrides} for count, overrides in blocks for _ in range(count)]
    )


@pytest.fixture
def _no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --- capacity tiers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("guests", "tier"),
    [(1, "1-2"), (2, "1-2"), (3, "3-4"), (4, "3-4"), (5, "5-7"), (7, "5-7"), (8, "8+"), (16, "8+")],
)
def test_capacity_tier_boundaries(guests: int, tier: str) -> None:
    """7 belongs with 5-6, not with 8+: only 2.7 % of listings sleep exactly 7."""
    assigned = groups.capacity_tier(_listings([{"accommodates": guests}]))
    assert assigned.iloc[0] == tier


def test_capacity_tier_covers_every_listing() -> None:
    frame = _listings([{"accommodates": n} for n in range(1, 17)])
    assert groups.capacity_tier(frame).notna().all()


def test_capacity_tier_requires_its_column() -> None:
    with pytest.raises(KeyError, match="listings"):
        groups.capacity_tier(pd.DataFrame({"city": ["athens"]}))


# --- query groups: assembly ----------------------------------------------------------------


def test_one_key_is_one_group_and_every_listing_lands_in_one(_no_warning) -> None:
    frame = _ranked((5, {}), (5, {"neighbourhood_cleansed": "Exarchia"}))
    ids, report = groups.query_group(frame)

    assert ids.iloc[:5].nunique() == 1
    assert ids.iloc[5:].nunique() == 1
    assert ids.nunique() == 2
    assert (ids >= 0).all()
    assert report["n"].sum() == len(frame)


def test_an_undersized_group_pools_with_the_other_fallers(_no_warning) -> None:
    """Two three-row neighbourhoods become one six-row group once neighbourhood is dropped."""
    frame = _ranked(
        (6, {}),
        (3, {"neighbourhood_cleansed": "Exarchia"}),
        (3, {"neighbourhood_cleansed": "Pagrati"}),
    )
    ids, report = groups.query_group(frame)

    assert ids.iloc[6:].nunique() == 1
    assert ids.iloc[0] != ids.iloc[6]
    assert report["n"].tolist() == [6, 6, 0]


def test_a_healthy_group_is_never_reopened_to_absorb_a_faller() -> None:
    """The rule the whole cascade rests on.

    Merging the two fallers into the six-row group would dissolve a working group to rescue a
    broken one. They pool among themselves instead, fall again for want of company, and are
    accepted below the minimum by the terminal rung.
    """
    frame = _ranked((6, {}), (2, {"neighbourhood_cleansed": "Exarchia"}))
    ids, report = groups.query_group(frame)

    assert ids.iloc[0] != ids.iloc[6]
    assert ids.iloc[6:].nunique() == 1
    assert report["n"].tolist() == [6, 0, 2]
    assert report.loc["city_room", "under_minimum"] == 1


def test_the_last_rung_merges_across_capacity_tiers(_no_warning) -> None:
    """Neither three-row tier reaches five until capacity itself is dropped."""
    frame = _ranked(
        (3, {"neighbourhood_cleansed": "Exarchia"}),
        (3, {"neighbourhood_cleansed": "Pagrati", "accommodates": 6}),
    )
    ids, report = groups.query_group(frame)

    assert ids.nunique() == 1
    assert report["n"].tolist() == [0, 0, 6]


def test_the_terminal_rung_accepts_a_group_below_the_minimum(_no_warning) -> None:
    """Seven such groups on the real snapshots — the residue of a product key, not a failure."""
    ids, report = groups.query_group(_ranked((3, {})))

    assert ids.nunique() == 1
    assert report.loc["city_room", "under_minimum"] == 1


def test_a_listing_left_alone_after_the_final_rung_warns() -> None:
    """A one-document group is invisible to NDCG, so it has to be audible somewhere."""
    frame = _ranked((6, {}), (1, {"room_type": "Hotel room"}))

    with pytest.warns(UserWarning, match="single listing"):
        ids, report = groups.query_group(frame)

    assert ids.nunique() == 2
    assert report["singletons"].sum() == 1


def test_a_null_neighbourhood_keeps_its_listings_in_the_ranking(_no_warning) -> None:
    """groupby drops null keys, and a dropped row is silently absent from the ranking."""
    frame = _ranked((5, {}), (5, {"neighbourhood_cleansed": None}))
    ids, report = groups.query_group(frame)

    assert (ids >= 0).all()
    assert ids.nunique() == 2
    assert report["n"].sum() == len(frame)


def test_the_capacity_tier_is_derived_here_not_read_from_the_frame(_no_warning) -> None:
    """Same guard as the price cascade: groups cannot be built on divergent tier bounds.

    5 and 7 share the ``5-7`` tier, so a stale ``capacity_tier`` column that disagrees must be
    overwritten rather than honoured — otherwise the two cascades key on different tiers.
    """
    frame = _ranked((5, {"accommodates": 5}), (5, {"accommodates": 7}))
    frame["capacity_tier"] = ["stale"] * 5 + ["bounds"] * 5

    assert groups.query_group(frame)[0].nunique() == 1


def test_a_cascade_that_drops_a_grading_partition_column_raises() -> None:
    """The grade would stop being a monotone step function of the label inside the group."""
    with pytest.raises(ValueError, match="grading partition"):
        groups.query_group(_ranked((5, {})), cascade=[("nbhd", ["city", "neighbourhood_cleansed"])])


def test_an_empty_cascade_raises_rather_than_leaving_rows_unassigned() -> None:
    with pytest.raises(ValueError, match="cascade is empty"):
        groups.query_group(_ranked((5, {})), cascade=[])


def test_missing_group_column_raises_a_readable_keyerror() -> None:
    with pytest.raises(KeyError, match="ranked listings"):
        groups.query_group(_ranked((5, {})).drop(columns=["room_type"]))


# --- the group array LightGBM reads ---------------------------------------------------------


def test_group_sizes_are_the_run_lengths_in_row_order() -> None:
    assert groups.group_sizes(pd.Series([0, 0, 1, 1, 1, 2])).tolist() == [2, 3, 1]


def test_group_sizes_sum_to_the_row_count(_no_warning) -> None:
    """BUILD_GUIDE gotcha #4, on the assignment rather than on a hand-built array."""
    ids, _ = groups.query_group(_ranked((5, {}), (5, {"neighbourhood_cleansed": "Exarchia"})))

    assert groups.group_sizes(ids.sort_values()).sum() == len(ids)


def test_group_sizes_reject_a_frame_not_sorted_by_group() -> None:
    """The sum still matches on a shuffled frame — contiguity is what catches the mistake."""
    with pytest.raises(ValueError, match="not sorted"):
        groups.group_sizes(pd.Series([0, 1, 0, 1]))


def test_group_sizes_reject_an_unassigned_row() -> None:
    """A null id would be counted into whichever group happens to precede it."""
    with pytest.raises(ValueError, match="no query group"):
        groups.group_sizes(pd.Series([0.0, float("nan"), 1.0]))


# --- cluster ids ---------------------------------------------------------------------------


def test_same_host_point_and_capacity_share_a_cluster() -> None:
    ids = groups.cluster_id(_listings([{}, {}]))
    assert ids.nunique() == 1


def test_a_different_host_at_the_same_address_is_a_different_cluster() -> None:
    """A block of flats holds many owners; the address alone does not make them twins."""
    ids = groups.cluster_id(_listings([{}, {"host_id": "h2"}]))
    assert ids.nunique() == 2


def test_a_different_capacity_is_a_different_cluster() -> None:
    ids = groups.cluster_id(_listings([{}, {"accommodates": 6}]))
    assert ids.nunique() == 2


def test_coordinates_are_matched_at_four_decimals() -> None:
    """~11 m. Airbnb jitters published coordinates 0-150 m, so finer would be false precision."""
    same = groups.cluster_id(_listings([{}, {"latitude": 40.640104}]))
    apart = groups.cluster_id(_listings([{}, {"latitude": 40.6501}]))

    assert same.nunique() == 1
    assert apart.nunique() == 2


def test_every_listing_receives_an_id_including_singletons() -> None:
    """A row without an id would silently never reach the grouped split."""
    frame = _listings([{}, {"host_id": "h2"}, {"host_id": "h3", "accommodates": 9}])
    ids = groups.cluster_id(frame)

    assert ids.notna().all()
    assert len(ids) == len(frame)
    assert ids.index.equals(frame.index)


def test_a_null_host_does_not_collapse_rows_into_one_cluster() -> None:
    """groupby drops null keys; two unrelated listings must not be merged by a missing host."""
    ids = groups.cluster_id(
        _listings([{"host_id": None, "latitude": 40.1}, {"host_id": None, "latitude": 41.9}])
    )
    assert ids.nunique() == 2


def test_cluster_sizes_count_the_membership() -> None:
    sizes = groups.cluster_sizes(_listings([{}, {}, {"host_id": "h2"}]))
    assert sizes.tolist() == [2, 2, 1]


# --- against the real snapshots -------------------------------------------------------------


@pytest.fixture(scope="module")
def real_ranked() -> pd.DataFrame:
    """The ranked population Phase 2 groups: filtered, with the label attached."""
    for name in ("listings", "calendar"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    labels = occupancy_label(pd.read_parquet(PROCESSED_DIR / "calendar.parquet"))
    kept, _ = filter_listings(listings.merge(labels, left_on="id", right_index=True, how="inner"))
    return kept


@pytest.fixture(scope="module")
def real_query_groups(real_ranked: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    return groups.query_group(real_ranked)


def test_real_query_groups_cover_every_listing_exactly_once(
    real_query_groups: tuple[pd.Series, pd.DataFrame], real_ranked: pd.DataFrame
) -> None:
    ids, report = real_query_groups

    assert (ids >= 0).all()
    assert report["n"].sum() == len(real_ranked)
    assert groups.group_sizes(ids.sort_values()).sum() == len(real_ranked)


def test_real_query_groups_leave_no_singleton(
    real_query_groups: tuple[pd.Series, pd.DataFrame],
) -> None:
    """What the third rung buys: 34 listings (0.08 %) moved, the last six singletons closed."""
    ids, report = real_query_groups

    assert (ids.value_counts() == 1).sum() == 0
    assert report["singletons"].sum() == 0


def test_real_two_rung_cascade_still_leaves_singletons(real_ranked: pd.DataFrame) -> None:
    """The measurement that chose the third rung, pinned so it cannot quietly stop being true."""
    with pytest.warns(UserWarning, match="single listing"):
        ids, _ = groups.query_group(real_ranked, cascade=groups.GROUP_CASCADE[:2])

    assert (ids.value_counts() == 1).sum() > 0


def test_real_fallback_moves_a_negligible_share(
    real_query_groups: tuple[pd.Series, pd.DataFrame],
) -> None:
    """289 listings, 0.65 % — the fallback widens a few searches, it does not reshape the key."""
    _, report = real_query_groups

    assert report["n"].iloc[1:].sum() / report["n"].sum() < 0.01


def test_real_query_group_never_spans_two_grading_cells(
    real_query_groups: tuple[pd.Series, pd.DataFrame], real_ranked: pd.DataFrame
) -> None:
    """The coarsening rule, on the *assembled* key rather than the raw one.

    `tests/test_label.py` pins the other half — that the grade is monotone in the label inside
    a raw key. Together they are the guarantee, which is a relationship between two keys and
    survives the cascade only because no rung drops a partition column.
    """
    ids, _ = real_query_groups
    cells = real_ranked.assign(_g=ids).groupby("_g", observed=True)[["city", "room_type"]].nunique()

    assert (cells == 1).all().all()


def test_real_clusters_are_distinct_inventory_not_duplicate_postings(
    real_ranked: pd.DataFrame,
) -> None:
    """The measurement that kept 12 % of the market out of the filter.

    If cluster members were one unit posted twice they would share a calendar and a review
    history. They do not — which is why `cluster_id` feeds the split and never `filters.py`.
    """
    ranked = real_ranked.assign(cluster=groups.cluster_id(real_ranked))
    multi = ranked[ranked["cluster"].map(ranked["cluster"].value_counts()) > 1]
    per_cluster = multi.groupby("cluster").agg(
        label_spread=("blocked_fraction_90", "std"),
        review_counts=("number_of_reviews", "nunique"),
    )

    assert ranked["cluster"].notna().all()
    # Distinct inventory: most clusters differ in both calendar and review history.
    assert per_cluster["label_spread"].eq(0).mean() < 0.15
    assert per_cluster["review_counts"].eq(1).mean() < 0.25
