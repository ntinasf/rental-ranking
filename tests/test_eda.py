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
