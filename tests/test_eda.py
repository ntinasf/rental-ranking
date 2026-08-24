"""Smoke tests for rental_ranking.eda — every function returns (fig, stats) with sane values."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rental_ranking import eda


@pytest.fixture()
def df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(50, 10, n)
    frame = pd.DataFrame(
        {
            "price": x,
            "rating": 0.05 * x + rng.normal(0, 0.5, n),
            "room_type": rng.choice(["entire", "private"], n, p=[0.7, 0.3]),
            "city": rng.choice(["athens", "crete", "thessaloniki"], n),
        }
    )
    frame.loc[:9, "price"] = np.nan
    return frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_analyze_numeric_variable_returns_fig_and_stats(df):
    fig, stats = eda.analyze_numeric_variable(df["price"])
    assert isinstance(fig, plt.Figure)
    assert stats["n_missing"] == 10
    assert stats["n_valid"] == 290
    assert stats["min"] <= stats["median"] <= stats["max"]


def test_analyze_numeric_variable_raises_on_all_nan():
    with pytest.raises(ValueError):
        eda.analyze_numeric_variable(pd.Series([np.nan, np.nan], name="empty"))


def test_analyze_categorical_variable_counts_add_up(df):
    fig, stats = eda.analyze_categorical_variable(df["room_type"])
    assert isinstance(fig, plt.Figure)
    assert stats["cardinality"] == 2
    assert sum(stats["counts"].values()) == len(df)
    assert stats["most_common"] == "entire"


def test_plot_scatter_reports_positive_correlation(df):
    fig, stats = eda.plot_scatter(df["price"], df["rating"], hue=df["room_type"])
    assert isinstance(fig, plt.Figure)
    assert stats["pearson_r"] > 0.5
    assert stats["n"] == 290


def test_quick_correlation_matrix_shapes_and_significance(df):
    fig, stats = eda.quick_correlation_matrix(df)
    corr, p_values = stats["correlations"], stats["p_values"]
    assert corr.shape == (2, 2)
    assert p_values.loc["price", "rating"] < 0.05
    assert np.isnan(p_values.loc["price", "price"])


def test_quick_correlation_matrix_insufficient_pairs_gives_nan_p(df):
    sparse = df.copy()
    sparse.loc[sparse.index[5:], "rating"] = np.nan
    _, stats = eda.quick_correlation_matrix(sparse, min_periods=30)
    assert np.isnan(stats["p_values"].loc["price", "rating"])


def test_analyze_categorical_categorical_independent_vars(df):
    fig, stats = eda.analyze_categorical_categorical(df["room_type"], df["city"])
    assert isinstance(fig, plt.Figure)
    assert 0 <= stats["cramers_v"] <= 1
    assert stats["contingency"].to_numpy().sum() == len(df)


def test_analyze_categorical_numerical_two_groups_uses_welch(df):
    fig, stats = eda.analyze_categorical_numerical(df["room_type"], df["price"])
    assert stats["test"] == "Welch's t-test"
    assert "cohens_d" in stats
    assert set(stats["group_descriptives"]) == {"entire", "private"}


def test_analyze_categorical_numerical_three_groups_uses_anova(df):
    fig, stats = eda.analyze_categorical_numerical(df["city"], df["price"])
    assert stats["test"] == "One-way ANOVA"
    assert "eta_squared" in stats


def test_analyze_numerical_numerical_recovers_relationship(df):
    fig, stats = eda.analyze_numerical_numerical(df["price"], df["rating"])
    assert stats["pearson_r"] > 0.5
    assert stats["significant"] is True
    assert 0 <= stats["r_squared"] <= 1


def test_plot_binned_relationship_returns_per_group_stats() -> None:
    rng = np.random.default_rng(0)
    x = pd.Series(rng.exponential(20, 600), name="lead days")
    y = pd.Series(x / 100 + rng.normal(0, 0.05, 600), name="blocked fraction")
    group = pd.Series(rng.choice(["a", "b"], 600), name="city")

    fig, stats = eda.plot_binned_relationship(x, y, group=group)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert set(stats["groups"]) == {"a", "b"}
    assert stats["groups"]["a"]["spearman_rho"] > 0.8  # y is built monotone in x
    assert sum(stats["groups"][g]["n"] for g in stats["groups"]) == stats["n_observations"]
    plt.close(fig)


def test_plot_binned_relationship_works_without_a_group() -> None:
    x = pd.Series(np.arange(100.0), name="x")
    fig, stats = eda.plot_binned_relationship(x, x * 2)
    assert list(stats["groups"]) == ["all"]
    plt.close(fig)


def test_plot_binned_relationship_refuses_a_fourth_series() -> None:
    """The palette validates for three; a fourth hue would fail CVD separation."""
    x = pd.Series(np.arange(80.0), name="x")
    group = pd.Series(["a", "b", "c", "d"] * 20, name="g")
    with pytest.raises(ValueError, match="palette cap"):
        eda.plot_binned_relationship(x, x * 2, group=group)


def test_plot_binned_relationship_needs_enough_observations() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        eda.plot_binned_relationship(pd.Series([1.0]), pd.Series([2.0]))


def test_discrete_distribution_renders_every_attainable_level() -> None:
    """Unobserved values must appear as zeros — a missing level would close a visible gap."""
    values = pd.Series([0, 0, 5, 5, 5, 10], name="blocked_days")
    fig, stats = eda.plot_discrete_distribution(values, levels=range(11))
    assert stats["levels"] == 11
    assert stats["groups"]["all"]["distinct_values"] == 3
    plt.close(fig)


def test_discrete_distribution_reports_the_share_at_each_atom() -> None:
    values = pd.Series([0] * 10 + [5] * 80 + [10] * 10, name="v")
    _, stats = eda.plot_discrete_distribution(values, levels=range(11), highlight=(0, 10))
    assert stats["groups"]["all"]["share_at"] == {0: 10.0, 10: 10.0}


def test_discrete_distribution_counts_on_integers_not_the_scaled_axis() -> None:
    """x_divisor is display-only: rescaling must not perturb the counted shares."""
    values = pd.Series([0] * 5 + [90] * 5, name="v")
    _, plain = eda.plot_discrete_distribution(values, levels=range(91), highlight=(0, 90))
    _, scaled = eda.plot_discrete_distribution(
        values, levels=range(91), highlight=(0, 90), x_divisor=90
    )
    assert plain["groups"]["all"]["share_at"] == scaled["groups"]["all"]["share_at"]


def test_discrete_distribution_refuses_a_fourth_series() -> None:
    values = pd.Series(list(range(8)), name="v")
    group = pd.Series(["a", "b", "c", "d"] * 2, name="g")
    with pytest.raises(ValueError, match="palette cap"):
        eda.plot_discrete_distribution(values, group=group)


def test_group_composition_normalises_each_bucket_to_one_hundred() -> None:
    """Bucket sizes differ by orders of magnitude; raw counts would hide the small ones."""
    category = pd.Series(["big"] * 90 + ["small"] * 10, name="bucket")
    breakdown = pd.Series(["x"] * 45 + ["y"] * 45 + ["x"] * 2 + ["y"] * 8, name="cohort")
    _, stats = eda.plot_group_composition(category, breakdown)
    panel = stats["panels"]["all"]
    assert panel["sizes"] == {"big": 90, "small": 10}
    assert panel["composition_pct"]["x"]["small"] == 20.0
    assert panel["composition_pct"]["y"]["small"] == 80.0


def test_group_composition_facets_without_dropping_empty_buckets() -> None:
    """A bucket empty in one panel must still hold its slot, or panels stop being comparable."""
    category = pd.Series(pd.Categorical(["a", "b", "a", "a"], categories=["a", "b", "c"]))
    category.name = "bucket"
    breakdown = pd.Series(["x", "y", "x", "y"], name="cohort")
    group = pd.Series(["p1", "p1", "p2", "p2"], name="city")
    _, stats = eda.plot_group_composition(category, breakdown, group=group)
    assert set(stats["panels"]) == {"p1", "p2"}
    assert stats["panels"]["p2"]["sizes"] == {"a": 2, "b": 0, "c": 0}


def test_group_composition_refuses_a_fourth_breakdown_level() -> None:
    category = pd.Series(["a", "b"] * 4, name="bucket")
    breakdown = pd.Series(["w", "x", "y", "z"] * 2, name="cohort")
    with pytest.raises(ValueError, match="palette cap"):
        eda.plot_group_composition(category, breakdown)


def test_plot_scores_against_floor_reports_range_share() -> None:
    fig, stats = eda.plot_scores_against_floor(
        {"model": 0.75, "baseline": 0.65},
        floor=0.55,
        intervals={"model": (0.71, 0.79)},
        highlight="model",
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    assert stats["floor"] == 0.55
    assert stats["ceiling"] == 1.0
    # The whole point of the chart: the share of the range above chance, not of 0-to-1.
    # range_share is rounded to 4 dp, the precision the report quotes it at.
    assert stats["rankers"]["model"]["range_share"] == pytest.approx(0.2 / 0.45, abs=1e-4)
    assert stats["rankers"]["baseline"]["range_share"] == pytest.approx(0.1 / 0.45, abs=1e-4)
    plt.close(fig)


def test_plot_scores_against_floor_honours_a_ceiling_below_one() -> None:
    """A cohort's reach is capped by how many of it can fit on a first screen at all."""
    fig, stats = eda.plot_scores_against_floor(
        {"model": 0.058}, floor=0.096, ceiling=0.172, highlight="model"
    )
    assert stats["ceiling"] == 0.172
    # Below the floor, so the share is negative — worse than chance, and it must read that way.
    assert stats["rankers"]["model"]["range_share"] < 0
    plt.close(fig)


def test_plot_scores_against_floor_rejects_a_floor_above_the_ceiling() -> None:
    with pytest.raises(ValueError, match="must sit below"):
        eda.plot_scores_against_floor({"model": 0.5}, floor=0.9, ceiling=0.8)


def test_plot_scores_against_floor_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="No scores"):
        eda.plot_scores_against_floor({}, floor=0.5)


def test_plot_dose_response_returns_per_series_range_and_reference() -> None:
    fig, stats = eda.plot_dose_response(
        [1, 2, 3, 5],
        {"grade": [2.78, 2.99, 3.03, 3.11], "cold share": [0.091, 0.067, 0.049, 0.048]},
        reference={"grade": 3.09, "cold share": 0.048},
    )
    assert stats["dose"] == ["1", "2", "3", "5"]
    assert stats["series"]["grade"]["reference"] == 3.09
    assert stats["series"]["cold share"]["range"] == pytest.approx(0.043)
    plt.close(fig)


def test_plot_dose_response_allows_one_series_without_a_reference() -> None:
    fig, stats = eda.plot_dose_response([1, 2, 3], {"grade": [1.0, 2.0, 3.0]})
    assert stats["series"]["grade"]["reference"] is None
    plt.close(fig)


def test_plot_dose_response_caps_at_two_series() -> None:
    """A third unit has no axis left to be honest on."""
    with pytest.raises(ValueError, match="two-axis cap"):
        eda.plot_dose_response([1, 2], {"a": [1, 2], "b": [1, 2], "c": [1, 2]})


def test_plot_dose_response_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="values for"):
        eda.plot_dose_response([1, 2, 3], {"grade": [1.0, 2.0]})
