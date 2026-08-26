"""The grouped train/test split: connected components, and a fold assignment over them.

**The split is the one irreversible decision in training.** Every number computed afterwards is
conditional on it, so the reasoning is recorded here rather than in a notebook.

**The conflict.** Near-twin listings (same host, point and capacity —
``features.groups.cluster_id``) must not straddle the split, or the model memorises the pair
instead of learning the feature. Query groups must not be broken, or NDCG on the test half is
computed over a partial candidate set and is no longer the measurement the frozen baselines made.
Some clusters span more than one query group, so neither constraint can hold alone.

:func:`split_component` dissolves it by splitting on the connected components of the
``cluster_id`` x ``query_group`` bipartite graph — the coarsest units that respect both exactly,
with no listing dropped and no group broken. The coarsening costs almost nothing in granularity,
because the lumpiness is imposed by the group-size skew rather than by the components: the
largest component is barely larger than the largest query group already is.

**The protocol the folds encode.** :func:`assign_folds` cuts the components into ``k`` equal
folds. Fold :data:`SEALED_FOLD` is the test half, touched once at the end; the rest are the
development pool, and every selection decision — amenity scheme, hyperparameters, the stopping
iteration — is made by cross-validating inside it via :func:`dev_cv_splits`, never against the
sealed fold.

Four development folds rather than one validation split, because a single 20 % split is too noisy
to choose on. The difference between the two frozen baselines is a *constant* — neither is fitted
and neither reads the label — yet read off five candidate test halves it swings by a factor of
ten. At that size the interval on a level is as wide as the model-versus-baseline effect this
exists to detect, so a single split would pick hyperparameters out of noise.

**Stratification is on city and group-size band, and the band matters more**: per-group NDCG
varies far more across size bands than any model will, so a test half drawn heavy in small groups
scores higher for every ranker in the table. Band is a pre-label structural property, so
balancing on it is not conditioning on the target. **Grade is deliberately not in the objective**
— :func:`fold_balance` reports it and nothing optimises it, because balancing a split on the
target is how a split starts choosing its own answer.

Pure transforms, no I/O and no ``main()``.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from rental_ranking.data.validate import require_columns

#: Number of folds the population is cut into. Five gives a 20 % sealed test half and a
#: four-fold development pool, and makes the sealed fold structurally identical to the inner
#: validation folds — the same size, drawn by the same rule.
DEFAULT_FOLDS = 5

#: The sealed test fold. **Chosen by rule, not by inspection.** Declaring the index before any
#: metric is computed on it is what stops the test half from being selected for a number that
#: flatters the model; picking "whichever fold the baseline is weakest on" would be exactly the
#: silent version of tuning on test.
SEALED_FOLD = 0

#: Group-size band edges, as ``pd.cut`` edges, tracking where NDCG@10 actually moves: below 10
#: the cut-off does not cut, so the metric scores full-list ordering and reads high for everyone.
GROUP_SIZE_BANDS: list[float] = [0, 10, 30, 100, 400, np.inf]

GROUP_SIZE_BAND_LABELS: list[str] = ["<10", "10-30", "30-100", "100-400", "400+"]

#: Columns whose composition each fold must reproduce, alongside the group-size band. City is
#: the generalisability claim; the band is the one that moves the metric.
DEFAULT_STRATA_COLS: tuple[str, ...] = ("city",)

_SPLIT_REQUIRED_COLUMNS = ("query_group", "cluster_id")


def split_component(listings: pd.DataFrame) -> pd.Series:
    """Connected components of the ``cluster_id`` x ``query_group`` bipartite graph.

    The coarsest unit that keeps every near-twin cluster whole *and* every query group whole.
    See the module docstring for why this is preferred to breaking either.

    Args:
        listings: Frame carrying ``query_group`` and ``cluster_id``.

    Returns:
        An integer Series aligned to ``listings``, named ``split_component``. Ids are dense
        from 0 and positional, so — like ``query_group`` and ``cluster_id`` — they are stable
        only within one call on one frame.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If either id column holds a null, which would silently drop the row from
            the graph and leave it with no fold.
    """
    require_columns(listings, _SPLIT_REQUIRED_COLUMNS, "listings")
    for column in _SPLIT_REQUIRED_COLUMNS:
        missing = int(listings[column].isna().sum())
        if missing:
            raise ValueError(
                f"{missing} row(s) carry no {column}, so they would form no edge in the "
                "cluster/group graph and would be assigned to no fold"
            )

    groups, _ = pd.factorize(listings["query_group"])
    clusters, cluster_labels = pd.factorize(listings["cluster_id"])
    n_groups = int(groups.max()) + 1 if len(groups) else 0
    n_nodes = n_groups + len(cluster_labels)

    # One edge per listing, group node -> cluster node. Weights are irrelevant; only
    # connectivity is read.
    graph = coo_matrix(
        (np.ones(len(listings)), (groups, clusters + n_groups)),
        shape=(n_nodes, n_nodes),
    )
    _, labels = connected_components(graph, directed=False)
    components, _ = pd.factorize(labels[groups])
    return pd.Series(components, index=listings.index, name="split_component")


def group_size_band(listings: pd.DataFrame, groups: pd.Series | None = None) -> pd.Series:
    """Band each listing by the size of the query group it sits in.

    Args:
        listings: Frame carrying ``query_group``, unless ``groups`` is passed.
        groups: Query-group ids, if not read from ``listings``.

    Returns:
        An ordered categorical Series aligned to ``listings``, named ``group_size_band``.
    """
    if groups is None:
        require_columns(listings, ("query_group",), "listings")
        groups = listings["query_group"]
    sizes = groups.map(groups.value_counts())
    return pd.cut(sizes, bins=GROUP_SIZE_BANDS, labels=GROUP_SIZE_BAND_LABELS).rename(
        "group_size_band"
    )


def assign_folds(
    listings: pd.DataFrame,
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
    strata_cols: Sequence[str] = DEFAULT_STRATA_COLS,
    row_weight: float = 1.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Cut the split components into ``folds`` balanced, stratified folds.

    Components are assigned largest-first to whichever fold currently leaves the population
    least imbalanced, over two things at once: total rows, and the count of *query groups* in
    each ``strata_cols`` x group-size-band cell. Largest-first matters — the 2,240-row component
    has to be placed while there is still room to compensate for it, and a random assignment
    that placed it last would be stuck with it.

    **The objective is grade-blind by construction.** ``grade`` is neither read nor accepted
    here; balancing the split on the target is how a split begins to choose its own answer. Use
    :func:`fold_balance` to *report* the grade distribution afterwards.

    Args:
        listings: Frame carrying ``query_group``, ``cluster_id`` and every ``strata_cols``
            member. ``split_component`` is derived here rather than read, so the folds cannot
            be built on a stale component assignment.
        folds: Number of folds. Defaults to :data:`DEFAULT_FOLDS`.
        seed: Shuffles the components before the stable size sort, so the many equal-sized
            small components are not always placed in frame order. The large components, which
            determine the balance, are ordered by size regardless.
        strata_cols: Columns whose composition each fold should reproduce, alongside the
            group-size band. Defaults to :data:`DEFAULT_STRATA_COLS`.
        row_weight: Relative weight of row balance against stratum balance in the objective.
            1.0 puts them on the same scale; both terms are mean squared deviations from an
            equal share.

    Returns:
        ``(ids, report)``. ``ids`` is an integer Series aligned to ``listings``, named ``fold``
        and running ``0..folds-1``. ``report`` is one row per fold: ``rows``, ``row_share``,
        ``groups``, ``components``, ``clusters`` and ``median_group_size``.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If ``folds`` is below 2, if there are fewer components than folds, or if
            the resulting assignment breaks a cluster or a query group across folds.
    """
    require_columns(listings, (*_SPLIT_REQUIRED_COLUMNS, *strata_cols), "listings")
    if folds < 2:
        raise ValueError(f"folds={folds}: a split needs at least two sides")

    component = split_component(listings)
    n_components = int(component.nunique())
    if n_components < folds:
        raise ValueError(
            f"{n_components} split component(s) cannot fill {folds} folds; the components are "
            "the atoms of this split and cannot be divided"
        )

    band = group_size_band(listings, listings["query_group"])
    cell = pd.Series(
        [
            "|".join(parts)
            for parts in zip(*(listings[c].astype(str) for c in strata_cols), band.astype(str))
        ],
        index=listings.index,
        name="stratum",
    )

    # One row per query group: its component and its stratum cell. The group, not the listing,
    # is the unit the metric averages over, so it is the unit the strata count.
    per_group = pd.DataFrame(
        {"component": component, "cell": cell, "group": listings["query_group"]}
    )
    per_group = per_group.groupby("group", observed=True).first()
    comp_cells = pd.crosstab(per_group["component"], per_group["cell"]).astype("float64")
    comp_rows = component.value_counts().reindex(comp_cells.index).astype("float64")

    rng = np.random.default_rng(seed)
    shuffled = comp_rows.sample(frac=1.0, random_state=rng.integers(2**32))
    order = shuffled.sort_values(ascending=False, kind="stable").index

    cell_totals = comp_cells.sum(axis=0).to_numpy()
    total_rows = float(comp_rows.sum())
    target = 1.0 / folds
    fold_rows = np.zeros(folds)
    fold_cells = np.zeros((folds, comp_cells.shape[1]))
    cells_matrix = comp_cells.to_numpy()
    positions = {c: i for i, c in enumerate(comp_cells.index)}
    assignment = {}

    for component_id in order:
        rows = comp_rows[component_id]
        cells = cells_matrix[positions[component_id]]

        # Cost of sending this component to each fold: the total squared deviation from an
        # equal share that would result. Only the receiving fold's term changes, so each is the
        # running total with that fold's contribution swapped out.
        row_dev = fold_rows / total_rows - target
        row_cost = (row_dev**2).sum() - row_dev**2 + (row_dev + rows / total_rows) ** 2

        cell_dev = fold_cells / cell_totals - target
        weighted = cell_dev**2 * cell_totals
        cell_cost = (
            weighted.sum()
            - weighted.sum(axis=1)
            + (((cell_dev + cells / cell_totals) ** 2) * cell_totals).sum(axis=1)
        ) / cell_totals.sum()

        chosen = int(np.argmin(row_weight * row_cost + cell_cost))
        assignment[component_id] = chosen
        fold_rows[chosen] += rows
        fold_cells[chosen] += cells

    ids = component.map(assignment).astype("int64").rename("fold")

    # The whole point of the component unit. Checked rather than assumed: a bug here is silent,
    # and every number downstream would be computed on a leaked split.
    for column, unit in (("cluster_id", "cluster"), ("query_group", "query group")):
        spanning = listings.groupby(column, observed=True).apply(
            lambda block, ids=ids: ids.loc[block.index].nunique(), include_groups=False
        )
        broken = int((spanning > 1).sum())
        if broken:
            raise ValueError(
                f"{broken} {unit}(s) span more than one fold, which the split component was "
                "supposed to make impossible — the assignment is leaked and must not be used"
            )

    sizes = listings.groupby("query_group", observed=True).size()
    group_fold = ids.groupby(listings["query_group"], observed=True).first()
    report = pd.DataFrame(
        {
            "rows": ids.value_counts().reindex(range(folds), fill_value=0),
            "row_share": ids.value_counts(normalize=True).reindex(range(folds), fill_value=0.0),
            "groups": group_fold.value_counts().reindex(range(folds), fill_value=0),
            "components": component.groupby(ids).nunique().reindex(range(folds), fill_value=0),
            "clusters": listings["cluster_id"]
            .groupby(ids)
            .nunique()
            .reindex(range(folds), fill_value=0),
            "median_group_size": sizes.groupby(group_fold).median().reindex(range(folds)),
        }
    )
    report.index.name = "fold"
    return ids, report


def fold_balance(
    fold: pd.Series, values: pd.Series, groups: pd.Series | None = None
) -> pd.DataFrame:
    """Cross-tabulate fold against any per-row label, as shares — the balance report.

    Pass ``city`` or :func:`group_size_band` to check what the split balanced on, and ``grade``
    to check what it deliberately did not. A grade table that looks lopsided is information, not
    a reason to re-run the split with grade in the objective.

    Args:
        fold: Fold ids per row.
        values: The label to break down by.
        groups: If given, count *query groups* rather than rows — the right unit for city and
            band, since the metric averages over groups. Leave ``None`` for grade, which is a
            per-listing property.

    Returns:
        A fold x value frame of shares, each row summing to 1.
    """
    frame = pd.DataFrame({"fold": fold, "value": values})
    if groups is not None:
        frame = frame.assign(group=groups).groupby("group", observed=True).first()
    counts = pd.crosstab(frame["fold"], frame["value"])
    return counts.div(counts.sum(axis=1), axis=0)


def constant_grade_groups(grades: pd.Series, groups: pd.Series) -> pd.Series:
    """Flag query groups where every listing carries the same grade.

    Any permutation of such a group scores NDCG 1.0, so they are unrankable by construction —
    ``evaluate.metrics`` returns them as NaN and counts them. Checking the sealed fold before
    training is what stops a degenerate test half being discovered after the fact.

    Args:
        grades: Graded relevance per row.
        groups: Query-group id per row.

    Returns:
        A boolean Series indexed by group id, True where the group is degenerate.
    """
    return grades.groupby(groups, observed=True).nunique().eq(1).rename("constant_grade")


def sealed_mask(fold: pd.Series, sealed: int = SEALED_FOLD) -> pd.Series:
    """Rows belonging to the sealed test fold. Everything else is the development pool."""
    return fold.eq(sealed).rename("is_sealed")


def dev_cv_splits(fold: pd.Series, sealed: int = SEALED_FOLD) -> list[tuple[pd.Index, pd.Index]]:
    """Cross-validation folds over the development pool, with the sealed fold excluded.

    The list every selection decision is made against: amenity scheme, hyperparameters, and the
    stopping iteration. Each entry trains on the rest of the pool and validates on one fold, so a
    decision rests on the mean and spread of several estimates rather than on one small draw.

    **The sealed fold appears in neither side of any pair.** That is the invariant worth a test:
    a sealed fold leaking into a training index is invisible in every metric it produces.

    Args:
        fold: Fold ids per row.
        sealed: The fold to withhold entirely. Defaults to :data:`SEALED_FOLD`.

    Returns:
        ``(train_index, validation_index)`` pairs, one per development fold, in fold order.
    """
    pool = fold[fold.ne(sealed)]
    return [
        (pool.index[pool.ne(held)], pool.index[pool.eq(held)]) for held in sorted(pool.unique())
    ]


def geography_coverage(
    listings: pd.DataFrame,
    fold: pd.Series,
    geo_column: str = "neighbourhood_cleansed",
    sealed: int = SEALED_FOLD,
) -> pd.DataFrame:
    """How much of each geography the sealed fold holds — the split's representativeness report.

    ``fold_balance`` answers what the split balanced on. This answers what it did **not**: the
    strata are city and group-size band, so neighbourhood coverage is incidental, and because whole
    connected components move together a large neighbourhood tends to land in one fold entire.

    Args:
        listings: The ranked population, carrying ``city`` and ``geo_column``.
        fold: Fold id per row.
        geo_column: Geography to report on.
        sealed: The held-out fold.

    Returns:
        One row per ``(city, geography)``: ``rows``, ``sealed_rows`` and ``sealed_share``, sorted
        by size. A geography with ``sealed_rows == 0`` contributes nothing to the held-out estimate.
    """
    require_columns(listings, ("city", geo_column), "ranked listings")
    frame = listings.assign(_fold=fold.to_numpy())
    coverage = frame.groupby(["city", geo_column], observed=True).agg(
        rows=(geo_column, "size"),
        sealed_rows=("_fold", lambda values: int((values == sealed).sum())),
    )
    coverage["sealed_share"] = coverage["sealed_rows"] / coverage["rows"]
    return coverage.sort_values("rows", ascending=False)


def unrepresented_groups(
    listings: pd.DataFrame,
    fold: pd.Series,
    groups: pd.Series,
    geo_column: str = "neighbourhood_cleansed",
    sealed: int = SEALED_FOLD,
) -> pd.Series:
    """Query groups sitting mostly in geographies the sealed fold does not cover.

    The instrument for testing whether the coverage gap **biases** the held-out estimate rather
    than merely narrowing it. Every group flagged here is in the development pool by construction —
    its geography has no sealed rows — so cross-validation has already scored it out-of-fold, and
    the two sets can be compared inside one estimate rather than across two.

    "Mostly" is a majority of the group's listings, because a pooled group can span both kinds.

    Returns:
        Boolean Series indexed by group id.
    """
    require_columns(listings, ("city", geo_column), "ranked listings")
    coverage = geography_coverage(listings, fold, geo_column, sealed)
    absent = set(coverage.index[coverage["sealed_rows"] == 0])

    keys = list(zip(listings["city"], listings[geo_column], strict=True))
    in_absent = pd.Series([key in absent for key in keys], index=listings.index)
    return (
        in_absent.groupby(groups.to_numpy(), observed=True).mean().gt(0.5).rename("unrepresented")
    )
