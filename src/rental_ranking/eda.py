"""Reusable EDA utilities for notebooks: univariate and bivariate analysis.

Every public function returns ``(fig, stats)``: a matplotlib Figure and a plain
dict of computed statistics. Nothing prints and nothing calls ``plt.show()`` —
in notebooks the inline backend renders the returned figure automatically, and
in scripts/tests the stats dict is directly assertable.

Example:
    from rental_ranking import eda

    fig, stats = eda.analyze_numeric_variable(df["price"])
    stats["median"]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    gaussian_kde,
    pearsonr,
    spearmanr,
    ttest_ind,
)

__all__ = [
    "analyze_numeric_variable",
    "analyze_categorical_variable",
    "plot_scatter",
    "quick_correlation_matrix",
    "analyze_categorical_categorical",
    "analyze_categorical_numerical",
    "analyze_numerical_numerical",
]

_SHAPIRO_MAX_N = 5000  # Shapiro-Wilk is unreliable/warns above this sample size


def _series_name(data: pd.Series, fallback: str) -> str:
    return str(data.name) if data.name is not None else fallback


def analyze_numeric_variable(
    data: pd.Series, include_outliers: bool = True
) -> tuple[plt.Figure, dict]:
    """Univariate analysis of a numeric variable.

    Args:
        data: Numeric series to analyze (NaNs are excluded).
        include_outliers: Whether the box plot shows outlier fliers.

    Returns:
        (fig, stats) — fig holds histogram, box plot, KDE, Q-Q plot, CDF, and a
        text summary panel; stats holds central tendency, dispersion, quantiles,
        shape, and sample-size fields.

    Raises:
        ValueError: If the series is empty after dropping NaNs.
    """
    clean = data.dropna()
    if clean.empty:
        raise ValueError("No data left after removing NaN values.")

    name = _series_name(data, "Variable")
    mode_values = clean.mode()
    stats: dict = {
        "n_valid": int(len(clean)),
        "n_missing": int(data.isna().sum()),
        "n_total": int(len(data)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "mode": float(mode_values.iloc[0]) if not mode_values.empty else np.nan,
        "std": float(clean.std()),
        "variance": float(clean.var()),
        "range": float(clean.max() - clean.min()),
        "iqr": float(clean.quantile(0.75) - clean.quantile(0.25)),
        "min": float(clean.min()),
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "skewness": float(clean.skew()),
        "kurtosis": float(clean.kurtosis()),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.hist(clean, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax.axvline(stats["mean"], color="red", ls="--", lw=2, label=f"Mean: {stats['mean']:.2f}")
    ax.axvline(
        stats["median"], color="green", ls="--", lw=2, label=f"Median: {stats['median']:.2f}"
    )
    ax.set_xlabel(name)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {name}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.boxplot(
        clean,
        vert=True,
        patch_artist=True,
        showfliers=include_outliers,
        boxprops=dict(facecolor="lightgreen", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="red", markersize=5, alpha=0.5),
    )
    ax.set_ylabel(name)
    ax.set_title(f"Box Plot of {name}")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    if clean.nunique() > 1:
        clean.plot.kde(ax=ax, color="purple", linewidth=2)
    ax.axvline(stats["mean"], color="red", ls="--", lw=2)
    ax.axvline(stats["median"], color="green", ls="--", lw=2)
    ax.set_xlabel(name)
    ax.set_title(f"KDE of {name}")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    scipy_stats.probplot(clean, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot of {name}")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    sorted_values = np.sort(clean.to_numpy())
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax.plot(sorted_values, cumulative, linewidth=2, color="darkblue")
    for q, style in ((0.25, ":"), (0.5, "--"), (0.75, ":")):
        ax.axhline(q, color="red" if q == 0.5 else "orange", ls=style, lw=1, alpha=0.7)
    ax.set_xlabel(name)
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"CDF of {name}")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    summary_lines = [
        f"n = {stats['n_valid']:,} (missing {stats['n_missing']:,})",
        f"mean = {stats['mean']:.2f}   median = {stats['median']:.2f}",
        f"std = {stats['std']:.2f}   IQR = {stats['iqr']:.2f}",
        f"min = {stats['min']:.2f}   max = {stats['max']:.2f}",
        f"skew = {stats['skewness']:.3f}   kurtosis = {stats['kurtosis']:.3f}",
    ]
    ax.text(0.0, 0.95, "\n".join(summary_lines), va="top", family="monospace", fontsize=11)
    ax.set_title(f"Summary: {name}", loc="left")

    fig.tight_layout()
    return fig, stats


def analyze_categorical_variable(data: pd.Series) -> tuple[plt.Figure, dict]:
    """Univariate analysis of a categorical variable.

    Args:
        data: Categorical series (object/category dtype, or discrete values).

    Returns:
        (fig, stats) — fig is a count bar chart (horizontal when cardinality is
        high); stats holds counts, percentages, cardinality, missingness, the
        most common category, and rare categories (<5%).

    Raises:
        ValueError: If the series is empty after dropping NaNs.
    """
    if data.dropna().empty:
        raise ValueError("No data left after removing NaN values.")

    name = _series_name(data, "Variable")
    counts = data.value_counts()
    percentages = data.value_counts(normalize=True) * 100

    stats: dict = {
        "cardinality": int(data.nunique()),
        "n_missing": int(data.isna().sum()),
        "n_total": int(len(data)),
        "counts": counts.to_dict(),
        "percentages": {k: float(v) for k, v in percentages.items()},
        "most_common": counts.index[0],
        "rare_categories": percentages[percentages < 5].index.tolist(),
    }

    horizontal = len(counts) > 8
    fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(counts)) if horizontal else 6))
    if horizontal:
        counts.iloc[::-1].plot.barh(ax=ax, color="skyblue", edgecolor="black")
        for i, (category, count) in enumerate(counts.iloc[::-1].items()):
            pct = percentages[category]
            ax.text(count, i, f" {count:,} ({pct:.1f}%)", va="center", fontsize=9)
        ax.set_xlabel("Frequency")
    else:
        counts.plot.bar(ax=ax, color="skyblue", edgecolor="black")
        for i, (category, count) in enumerate(counts.items()):
            pct = percentages[category]
            ax.text(i, count, f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Distribution of {name} ({stats['cardinality']} categories)")
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)

    fig.tight_layout()
    return fig, stats


def plot_scatter(
    x_data: pd.Series,
    y_data: pd.Series,
    hue: pd.Series | None = None,
    alpha: float = 0.6,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, dict]:
    """Scatter plot of two numeric variables with a linear trend line.

    Args:
        x_data: X-axis series.
        y_data: Y-axis series.
        hue: Optional categorical series for color coding.
        alpha: Point transparency.
        figsize: Figure size.

    Returns:
        (fig, stats) — stats holds the Pearson correlation, trend-line slope and
        intercept, and the number of complete observations.

    Raises:
        ValueError: If fewer than 3 complete (x, y) pairs remain.
    """
    x_name = _series_name(x_data, "X Variable")
    y_name = _series_name(y_data, "Y Variable")

    columns = {x_name: x_data, y_name: y_data}
    hue_name = None
    if hue is not None:
        hue_name = _series_name(hue, "Category")
        columns[hue_name] = hue
    combined = pd.DataFrame(columns).dropna()
    if len(combined) < 3:
        raise ValueError(f"Need at least 3 complete observations, found {len(combined)}.")

    slope, intercept = np.polyfit(combined[x_name], combined[y_name], 1)
    correlation = float(combined[x_name].corr(combined[y_name]))
    stats = {
        "pearson_r": correlation,
        "slope": float(slope),
        "intercept": float(intercept),
        "n": int(len(combined)),
    }

    fig, ax = plt.subplots(figsize=figsize)
    if hue_name is not None:
        categories = combined[hue_name].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        for color, category in zip(colors, categories):
            mask = combined[hue_name] == category
            ax.scatter(
                combined.loc[mask, x_name],
                combined.loc[mask, y_name],
                alpha=alpha,
                s=50,
                color=color,
                edgecolors="black",
                linewidth=0.5,
                label=str(category),
            )
    else:
        ax.scatter(
            combined[x_name],
            combined[y_name],
            alpha=alpha,
            s=50,
            color="steelblue",
            edgecolors="black",
            linewidth=0.5,
        )

    trend_x = np.array([combined[x_name].min(), combined[x_name].max()])
    ax.plot(
        trend_x,
        slope * trend_x + intercept,
        "r--",
        linewidth=2,
        label=f"Trend: y = {slope:.2f}x + {intercept:.2f}",
    )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    title = f"{x_name} vs {y_name} (r = {correlation:.3f})"
    if hue_name is not None:
        title += f", colored by {hue_name}"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig, stats


def quick_correlation_matrix(
    df: pd.DataFrame, method: str = "pearson", min_periods: int = 30, alpha: float = 0.05
) -> tuple[plt.Figure, dict]:
    """Correlation matrix over numeric columns with pairwise significance tests.

    Args:
        df: DataFrame; non-numeric columns are ignored.
        method: "pearson" or "spearman".
        min_periods: Minimum complete pairs required to test a column pair.
        alpha: Significance level for the significance panel.

    Returns:
        (fig, stats) — stats holds "correlations" and "p_values" DataFrames.
        Pairs with fewer than ``min_periods`` complete observations have NaN
        p-values (NaN is never reported as significant).

    Raises:
        ValueError: If fewer than two numeric columns are present.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        raise ValueError("Need at least two numeric columns.")

    corr = numeric.corr(method=method, min_periods=min_periods)
    columns = corr.columns
    p_values = pd.DataFrame(np.nan, index=columns, columns=columns)
    test = pearsonr if method == "pearson" else spearmanr

    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1 :]:
            pair = numeric[[col1, col2]].dropna()
            if len(pair) >= min_periods:
                _, p = test(pair[col1], pair[col2])
                p_values.loc[col1, col2] = p
                p_values.loc[col2, col1] = p

    stats = {"correlations": corr, "p_values": p_values}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=axes[0],
        cbar_kws={"label": f"{method.capitalize()} correlation"},
    )
    axes[0].set_title(f"{method.capitalize()} Correlation Matrix")

    significant = (p_values < alpha).astype(float).where(p_values.notna())
    sns.heatmap(
        significant,
        mask=mask,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=axes[1],
        cbar_kws={"label": f"Significant (p < {alpha})"},
    )
    axes[1].set_title("Statistical Significance (blank = insufficient data)")

    fig.tight_layout()
    return fig, stats


def analyze_categorical_categorical(
    cat_data1: pd.Series, cat_data2: pd.Series, alpha: float = 0.05
) -> tuple[plt.Figure, dict]:
    """Chi-square independence test between two categorical variables.

    Args:
        cat_data1: First categorical series.
        cat_data2: Second categorical series.
        alpha: Significance level.

    Returns:
        (fig, stats) — fig shows the contingency heatmap and proportional
        stacked bars; stats holds chi2, p_value, dof, Cramér's V with an effect
        interpretation, the contingency table, and ``low_expected_cells`` (cells
        with expected frequency < 5, where chi-square is unreliable).

    Raises:
        ValueError: If either variable has fewer than 2 categories after
            dropping NaNs.
    """
    col1 = _series_name(cat_data1, "Variable 1")
    col2 = _series_name(cat_data2, "Variable 2")
    combined = pd.DataFrame({col1: cat_data1, col2: cat_data2}).dropna()
    contingency = pd.crosstab(combined[col1], combined[col2])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise ValueError("Both variables need at least 2 categories after dropping NaNs.")

    chi2, p_value, dof, expected = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / (n * min_dim)))

    if cramers_v < 0.1:
        effect = "negligible"
    elif cramers_v < 0.3:
        effect = "weak"
    elif cramers_v < 0.5:
        effect = "moderate"
    else:
        effect = "strong"

    stats = {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "effect": effect,
        "significant": bool(p_value < alpha),
        "contingency": contingency,
        "low_expected_cells": int((expected < 5).sum()),
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    sns.heatmap(contingency, annot=True, fmt="d", cmap="YlOrRd", ax=axes[0])
    axes[0].set_title(
        f"{col1} vs {col2} — chi2 p = {p_value:.4g}, Cramér's V = {cramers_v:.3f} ({effect})"
    )

    proportions = contingency.div(contingency.sum(axis=1), axis=0)
    proportions.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis", alpha=0.85)
    axes[1].set_title(f"Proportional distribution of {col2} within {col1}")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(title=col2, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    return fig, stats


def analyze_categorical_numerical(
    cat_data: pd.Series,
    num_data: pd.Series,
    alpha: float = 0.05,
    include_outliers: bool = True,
) -> tuple[plt.Figure, dict]:
    """Group comparison of a numeric variable across categories.

    Uses Welch's t-test (2 groups, Cohen's d) or one-way ANOVA (3+ groups,
    eta-squared). Assumption checks (Shapiro-Wilk per group on at most
    5000 sampled values, Levene's test) are returned in stats, not printed.

    Args:
        cat_data: Grouping variable.
        num_data: Numeric variable.
        alpha: Significance level.
        include_outliers: Whether the box plot shows outlier fliers.

    Returns:
        (fig, stats) — fig shows box and violin plots; stats holds the test
        name, statistic, p_value, effect size, per-group descriptives, and
        assumption-check results.

    Raises:
        ValueError: If fewer than 2 groups remain after dropping NaNs.
    """
    cat_name = _series_name(cat_data, "Group")
    num_name = _series_name(num_data, "Value")
    combined = pd.DataFrame({cat_name: cat_data, num_name: num_data}).dropna()
    combined[cat_name] = combined[cat_name].astype(str)

    groups = combined[cat_name].unique()
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups, found {len(groups)}.")
    group_values = [combined.loc[combined[cat_name] == g, num_name].to_numpy() for g in groups]

    if len(groups) == 2:
        statistic, p_value = ttest_ind(group_values[0], group_values[1], equal_var=False)
        test_name = "Welch's t-test"
        n1, n2 = len(group_values[0]), len(group_values[1])
        s1, s2 = np.std(group_values[0], ddof=1), np.std(group_values[1], ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        effect_size = float((np.mean(group_values[0]) - np.mean(group_values[1])) / pooled)
        effect_measure = "cohens_d"
        thresholds = (0.2, 0.5, 0.8)
    else:
        statistic, p_value = f_oneway(*group_values)
        test_name = "One-way ANOVA"
        grand_mean = combined[num_name].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_values)
        ss_total = float(((combined[num_name] - grand_mean) ** 2).sum())
        effect_size = float(ss_between / ss_total) if ss_total > 0 else np.nan
        effect_measure = "eta_squared"
        thresholds = (0.01, 0.06, 0.14)

    magnitude = abs(effect_size)
    if magnitude < thresholds[0]:
        effect = "negligible"
    elif magnitude < thresholds[1]:
        effect = "small"
    elif magnitude < thresholds[2]:
        effect = "medium"
    else:
        effect = "large"

    shapiro_p = {}
    for group, values in zip(groups, group_values):
        if len(values) >= 3:
            sample = values
            if len(sample) > _SHAPIRO_MAX_N:
                rng = np.random.default_rng(0)
                sample = rng.choice(sample, _SHAPIRO_MAX_N, replace=False)
            shapiro_p[group] = float(scipy_stats.shapiro(sample)[1])
    levene_p = float(scipy_stats.levene(*group_values)[1])

    stats = {
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        effect_measure: effect_size,
        "effect": effect,
        "significant": bool(p_value < alpha),
        "group_descriptives": combined.groupby(cat_name)[num_name]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .to_dict(orient="index"),
        "shapiro_p_by_group": shapiro_p,
        "levene_p": levene_p,
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    combined.boxplot(
        column=num_name,
        by=cat_name,
        ax=axes[0],
        patch_artist=True,
        grid=False,
        showfliers=include_outliers,
    )
    axes[0].set_title(f"{num_name} by {cat_name} — {test_name} p = {p_value:.4g} ({effect})")
    axes[0].set_xlabel(cat_name)
    axes[0].set_ylabel(num_name)
    fig.suptitle("")

    sns.violinplot(
        data=combined,
        x=cat_name,
        y=num_name,
        hue=cat_name,
        ax=axes[1],
        palette="Set2",
        inner="box",
        legend=False,
    )
    axes[1].set_title(f"Violin plot of {num_name} by {cat_name}")

    fig.tight_layout()
    return fig, stats


def analyze_numerical_numerical(
    x_data: pd.Series, y_data: pd.Series, alpha: float = 0.05
) -> tuple[plt.Figure, dict]:
    """Correlation and simple linear-fit analysis of two numeric variables.

    Args:
        x_data: Independent variable.
        y_data: Dependent variable.
        alpha: Significance level.

    Returns:
        (fig, stats) — fig shows the scatter with regression line and a 2D
        density contour (skipped when the KDE is degenerate); stats holds
        Pearson and Spearman coefficients with p-values, R², slope, and
        intercept.

    Raises:
        ValueError: If fewer than 3 complete (x, y) pairs remain.
    """
    x_name = _series_name(x_data, "X Variable")
    y_name = _series_name(y_data, "Y Variable")
    combined = pd.DataFrame({x_name: x_data, y_name: y_data}).dropna()
    if len(combined) < 3:
        raise ValueError(f"Need at least 3 complete observations, found {len(combined)}.")

    x = combined[x_name].to_numpy()
    y = combined[y_name].to_numpy()

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    stats = {
        "n": int(len(combined)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "r_squared": float(r_squared),
        "slope": float(slope),
        "intercept": float(intercept),
        "significant": bool(pearson_p < alpha),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.scatter(x, y, alpha=0.6, s=50, color="steelblue", edgecolors="black", linewidth=0.5)
    order = np.argsort(x)
    ax.plot(x[order], y_pred[order], "r--", lw=2, label=f"y = {slope:.3f}x + {intercept:.3f}")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"Pearson r = {pearson_r:.3f}, R² = {r_squared:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    try:
        kernel = gaussian_kde(np.vstack([x, y]))
        x_margin = (x.max() - x.min()) * 0.1 or 1.0
        y_margin = (y.max() - y.min()) * 0.1 or 1.0
        xx, yy = np.mgrid[
            x.min() - x_margin : x.max() + x_margin : 100j,
            y.min() - y_margin : y.max() + y_margin : 100j,
        ]
        density = kernel(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        contour = ax.contourf(xx, yy, density, levels=15, cmap="YlOrRd", alpha=0.7)
        fig.colorbar(contour, ax=ax, label="Density")
        ax.scatter(x, y, alpha=0.4, s=30, color="black", edgecolors="white", linewidth=0.5)
    except np.linalg.LinAlgError:
        ax.text(0.5, 0.5, "Density estimate unavailable\n(degenerate data)", ha="center")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"2D density — Spearman ρ = {spearman_rho:.3f}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig, stats
