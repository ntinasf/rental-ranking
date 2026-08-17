"""Tests for rental_ranking.features.aggregates.

The three skeleton tests this file replaced named the right properties, and they are kept as the
first three below. What they could not name is the reason the module refuses to build a
neighbourhood mean-label aggregate at all — that a leave-one-out label aggregate is an exact
inverse of the target inside a query group — so that is pinned here too, against the real
snapshots, as the evidence for a design decision rather than as a property of shipped code.

Leave-one-out is a silent failure in both directions: an include-self aggregate still returns a
plausible price, and so does an aggregate that leaves out the wrong row.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import aggregates

_DEFAULTS = {"city": "athens", "neighbourhood_cleansed": "Kolonaki", "price": 100.0}


def _listings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_DEFAULTS, **row} for row in rows])


# --- the three properties the skeleton named -------------------------------------------------


def test_neighbourhood_median_excludes_the_listing_itself() -> None:
    """Leave-one-out: the listing's own value never enters its neighbourhood aggregate."""
    frame = _listings([{"price": 1.0}, {"price": 2.0}, {"price": 3.0}])
    out = aggregates.neighbourhood_features(frame)

    # Row 0 sees [2, 3] -> 2.5; row 1 sees [1, 3] -> 2.0; row 2 sees [1, 2] -> 1.5.
    assert out["nbhd_median_price"].tolist() == [2.5, 2.0, 1.5]


def test_single_listing_neighbourhood_has_no_self_referential_aggregate() -> None:
    """With one listing, leave-one-out must yield NaN — never its own value, never zero."""
    out = aggregates.neighbourhood_features(_listings([{"price": 100.0}]))

    assert out["nbhd_median_price"].isna().all()
    assert out["price_vs_nbhd"].isna().all()
    assert out["nbhd_listings"].tolist() == [0]  # a real count: no neighbours


def test_aggregate_changes_when_own_value_changes_only_for_neighbours() -> None:
    """Perturbing listing A's price changes neighbours' aggregates but not A's own."""
    frame = _listings([{"price": 10.0}, {"price": 20.0}, {"price": 30.0}, {"price": 40.0}])
    moved = frame.copy()
    moved.loc[0, "price"] = 1_000.0

    before = aggregates.neighbourhood_features(frame)["nbhd_median_price"]
    after = aggregates.neighbourhood_features(moved)["nbhd_median_price"]

    assert before.iloc[0] == after.iloc[0]
    assert (before.iloc[1:] != after.iloc[1:]).all()


# --- the arithmetic --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prices", "expected"),
    [
        ([1.0, 2.0, 3.0], [2.5, 2.0, 1.5]),  # odd n -> even remainder, averages two
        ([1.0, 2.0, 3.0, 4.0], [3.0, 3.0, 2.0, 2.0]),  # even n -> odd remainder, single middle
        ([5.0, 5.0], [5.0, 5.0]),
    ],
)
def test_loo_median_is_exact_for_both_parities(prices: list[float], expected: list[float]) -> None:
    """The index arithmetic is where an off-by-one would hide behind a plausible number."""
    frame = _listings([{"price": p} for p in prices])
    assert aggregates.neighbourhood_features(frame)["nbhd_median_price"].tolist() == expected


def test_loo_median_matches_a_brute_force_recomputation() -> None:
    """The vectorised version against the definition, on unsorted values with ties."""
    rng = np.random.default_rng(0)
    prices = rng.integers(10, 200, size=60).astype("float64")
    frame = _listings([{"price": p} for p in prices])

    got = aggregates.neighbourhood_features(frame)["nbhd_median_price"].to_numpy()
    want = np.array([np.median(np.delete(prices, i)) for i in range(len(prices))])

    np.testing.assert_allclose(got, want)


def test_neighbourhoods_are_scoped_by_city() -> None:
    """Neighbourhood names collide across cities; the key carries `city` for that reason."""
    frame = _listings(
        [
            {"city": "athens", "price": 10.0},
            {"city": "athens", "price": 20.0},
            {"city": "crete", "price": 1_000.0},
        ]
    )
    out = aggregates.neighbourhood_features(frame)

    assert out["nbhd_median_price"].iloc[0] == 20.0
    assert out["nbhd_listings"].tolist() == [1, 1, 0]


def test_a_null_price_is_refused_rather_than_skipped() -> None:
    """A null shifts its group's median without ever appearing in it."""
    frame = _listings([{"price": float("nan")}, {"price": 10.0}, {"price": 20.0}])

    with pytest.raises(ValueError, match="null value"):
        aggregates.neighbourhood_features(frame)


def test_the_relative_feature_is_the_price_over_the_local_median() -> None:
    frame = _listings([{"price": 100.0}, {"price": 50.0}, {"price": 50.0}])
    out = aggregates.neighbourhood_features(frame)

    assert out["price_vs_nbhd"].iloc[0] == pytest.approx(2.0)


# --- against the real snapshots ---------------------------------------------------------------


@pytest.fixture(scope="module")
def real_ranked() -> pd.DataFrame:
    for name in ("listings", "calendar"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    from rental_ranking.data.filters import filter_listings
    from rental_ranking.features.label import occupancy_label
    from rental_ranking.features.price import impute_price

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    labels = occupancy_label(pd.read_parquet(PROCESSED_DIR / "calendar.parquet"))
    kept, _ = filter_listings(listings.merge(labels, left_on="id", right_index=True, how="inner"))
    ranked, _ = impute_price(kept)
    return ranked


def test_real_aggregates_cover_every_listing_but_the_lone_one(real_ranked: pd.DataFrame) -> None:
    """One neighbourhood of size 1 on these snapshots; its median is undefined, not zero."""
    out = aggregates.neighbourhood_features(real_ranked)
    lone = out["nbhd_listings"].eq(0)

    assert len(out) == len(real_ranked)
    assert out["nbhd_median_price"].isna().equals(lone)
    assert lone.sum() <= 5


def test_real_leave_one_out_is_correctness_not_rescue(real_ranked: pd.DataFrame) -> None:
    """The measured magnitude, so the write-up states a number instead of the folklore.

    At neighbourhood scale the include-self and leave-one-out means differ by 0.0004 on average.
    The rule is followed because it is one line and correct, not because it would otherwise bite.
    """
    label = real_ranked["blocked_fraction_90"]
    unit = real_ranked[list(aggregates.NEIGHBOURHOOD_KEY)].astype(str).agg("|".join, axis=1)
    grouped = label.groupby(unit)
    include_self = grouped.transform("mean")
    loo = (grouped.transform("sum") - label) / (grouped.transform("size") - 1)

    assert (include_self - loo).abs().mean() < 0.001


def test_real_a_leave_one_out_label_aggregate_would_invert_the_target(
    real_ranked: pd.DataFrame,
) -> None:
    """**Why this module builds no mean-label aggregate.**

    Inside a query group nested in one neighbourhood, the group's total and size are constants,
    so the leave-one-out mean is an exact affine decreasing function of the listing's own label.
    This is not a small bias to be tolerated — it is a perfect rank inversion of the target, and
    it is *created* by the leave-one-out correction rather than fixed by it.
    """
    from scipy.stats import spearmanr

    from rental_ranking.features.groups import query_group

    groups, _ = query_group(real_ranked)
    label = real_ranked["blocked_fraction_90"]
    unit = real_ranked[list(aggregates.NEIGHBOURHOOD_KEY)].astype(str).agg("|".join, axis=1)
    grouped = label.groupby(unit)
    loo_label = (grouped.transform("sum") - label) / (grouped.transform("size") - 1)

    frame = pd.DataFrame({"g": groups, "loo": loo_label, "label": label, "unit": unit})
    single_neighbourhood = frame.groupby("g")["unit"].transform("nunique").eq(1)
    frame = frame[single_neighbourhood]

    inverted = frame.groupby("g").apply(
        lambda d: spearmanr(d["loo"], d["label"]).statistic if len(d) > 2 else np.nan,
        include_groups=False,
    )
    assert inverted.dropna().round(6).eq(-1.0).all()

    # And the include-self version is the mirror failure: constant, so it ranks nothing.
    include_self = grouped.transform("mean")[single_neighbourhood]
    assert include_self.groupby(frame["g"]).nunique().eq(1).all()
