"""Assemble the four feature blocks into the matrix LightGBM trains on.

One row per ranked listing, **sorted by ``query_group``**, because LightGBM reads its group
array positionally and never sees the ids: an unsorted matrix trains on queries that mix
listings from different searches and every metric still prints a plausible number.

**Three column roles, kept apart on purpose.** :data:`IDENTIFIER_COLUMNS` join and group,
:data:`TARGET_COLUMNS` are what the model is trained against, and everything else is a feature.
The trainer must take its feature list from :func:`feature_columns` rather than from
``table.columns``, so the label cannot arrive as an input by omission — the single most
expensive mistake available at this point in the project.

**The blocklist is read from ``columns.py``, never from a list kept here.** A column moved into
``LABEL_ADJACENT_COLUMNS`` by a future snapshot is enforced without anyone remembering to update
this module. Alongside it, :data:`PHASE1_DIAGNOSTIC_COLUMNS` blocks the working columns Phase 1
produced for validation — ``avail_90`` is the label's own numerator, ``blocked_fraction_calendar``
spans the label window and then some, and ``T``/``scrape_date`` would let a model learn which
scrape day a listing came from.

Convention: a pure ``DataFrame -> DataFrame`` transform. Reading the parquets and writing the
matrix belong to ``features/build.py``, which is to this layer what ``data/build.py`` is to the
processed one.
"""

import pandas as pd

from rental_ranking.data.columns import LABEL_ADJACENT_COLUMNS
from rental_ranking.data.validate import require_columns
from rental_ranking.features.aggregates import neighbourhood_features
from rental_ranking.features.amenities import amenity_features  # noqa: F401  (re-exported scheme)
from rental_ranking.features.groups import group_sizes
from rental_ranking.features.listing import listing_features
from rental_ranking.features.reviews import review_features
from rental_ranking.features.spatial import spatial_features

#: Join keys and grouping ids. Never features: ``id`` and ``cluster_id`` are near-unique, and
#: ``query_group`` is the unit the metric is computed over, so feeding it back would let the
#: model memorise which search it is in.
IDENTIFIER_COLUMNS: tuple[str, ...] = ("id", "query_group", "cluster_id")

#: What the model is trained and scored against. ``grade`` is the LambdaMART target;
#: ``blocked_fraction_90`` is the underlying demand proxy, carried for analysis and **never** an
#: input.
TARGET_COLUMNS: tuple[str, ...] = ("grade", "blocked_fraction_90")

#: Phase-1 working columns that are not on the contract's blocklist but must not reach a model.
#: ``avail_90`` is the label's numerator; ``blocked_fraction_calendar`` covers the label window
#: and the rest of the year; ``calendar_days``/``calendar_span`` describe the label's own
#: denominator; ``T`` and ``scrape_date`` identify the scrape batch; ``level`` is the grading
#: report's rung marker.
PHASE1_DIAGNOSTIC_COLUMNS: frozenset[str] = frozenset(
    {
        "avail_90",
        "blocked_fraction_calendar",
        "calendar_days",
        "calendar_span",
        "T",
        "scrape_date",
        "level",
    }
)

_REQUIRED_COLUMNS = (*IDENTIFIER_COLUMNS, *TARGET_COLUMNS)


def feature_columns(table: pd.DataFrame) -> list[str]:
    """The model inputs: everything that is neither an identifier nor a target.

    Take the feature list from here rather than from ``table.columns``. The difference is one
    line and it is the difference between training on the features and training on the answer.
    """
    reserved = set(IDENTIFIER_COLUMNS) | set(TARGET_COLUMNS)
    return [column for column in table.columns if column not in reserved]


def check_feature_table(table: pd.DataFrame) -> pd.Series:
    """Run every structural check the matrix must satisfy, or raise.

    Five checks, each guarding a failure that is silent rather than loud:

    1. **No blocklist column**, read live from ``columns.py``.
    2. **No Phase-1 diagnostic column** — see :data:`PHASE1_DIAGNOSTIC_COLUMNS`.
    3. **One row per listing.** A duplicated id would double a listing's weight in its group.
    4. **Sorted by query group, in contiguous runs** — BUILD_GUIDE gotcha #4. Delegated to
       ``group_sizes``, which is the module that owns the rule.
    5. **No null identifiers or targets.** A null grade trains against nothing; a null group id
       silently joins whichever query precedes it.

    Returns:
        A Series of headline counts for reporting: rows, groups, features, and the group array's
        sum, which must equal the row count.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If any check fails.
    """
    require_columns(table, _REQUIRED_COLUMNS, "feature table")

    blocked = sorted(set(table.columns) & LABEL_ADJACENT_COLUMNS)
    if blocked:
        raise ValueError(
            f"label-adjacent column(s) reached the feature table: {blocked}. These are forward "
            "availability windows or direct reads of the label and may never be model inputs"
        )

    diagnostics = sorted(set(table.columns) & PHASE1_DIAGNOSTIC_COLUMNS)
    if diagnostics:
        raise ValueError(
            f"Phase-1 diagnostic column(s) reached the feature table: {diagnostics}. They exist "
            "for validation and every one of them is derived from the label window"
        )

    if not table["id"].is_unique:
        duplicated = int(table["id"].duplicated().sum())
        raise ValueError(
            f"{duplicated} duplicated listing id(s) in the feature table; a repeated listing "
            "carries double weight inside its query group"
        )

    reserved = [*IDENTIFIER_COLUMNS, *TARGET_COLUMNS]
    nulls = table[reserved].isna().sum()
    if nulls.any():
        raise ValueError(
            f"null identifier or target value(s): {nulls[nulls > 0].to_dict()}. A null grade "
            "trains against nothing and a null group id joins whichever query precedes it"
        )

    # Raises if the matrix is not sorted into contiguous runs, which is the half of gotcha #4
    # that a sum check cannot catch.
    sizes = group_sizes(table["query_group"])

    return pd.Series(
        {
            "rows": len(table),
            "query_groups": len(sizes),
            "features": len(feature_columns(table)),
            "group_array_sum": int(sizes.sum()),
        }
    )


def assemble_feature_table(
    ranked: pd.DataFrame,
    reviews: pd.DataFrame | None = None,
    amenity_scheme: str = "buckets",
    vocabulary: list[str] | None = None,
) -> pd.DataFrame:
    """Build the sorted, checked feature matrix from the ranked population.

    Args:
        ranked: The filtered population with ``price`` imputed and carrying ``grade``,
            ``query_group`` and ``cluster_id`` — the frame ``features/build.py`` prepares.
        reviews: Processed reviews. When given, ``reviews_same_season_ly`` joins the block;
            when omitted it is left out rather than filled.
        amenity_scheme: Passed through to the amenity encoding, so the Phase 3 comparison
            varies one parameter rather than one code path.
        vocabulary: Passed through for the ``"flags"`` scheme.

    Returns:
        One row per listing, sorted by ``query_group``, with a fresh ``RangeIndex`` — the index
        is positional from here on, matching how LightGBM reads the group array.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If any check in :func:`check_feature_table` fails.
    """
    require_columns(ranked, _REQUIRED_COLUMNS, "ranked listings")

    blocks = [
        ranked[list(IDENTIFIER_COLUMNS)],
        ranked[list(TARGET_COLUMNS)],
        listing_features(ranked, amenity_scheme, vocabulary).drop(columns=["id"]),
        review_features(ranked, reviews).drop(columns=["id"]),
        neighbourhood_features(ranked),
        spatial_features(ranked),
    ]
    table = pd.concat(blocks, axis=1)

    # Stable, so listings keep their incoming order inside a group. That order is hashed-id and
    # therefore unrelated to the label — the same neutrality the metric's tie-break relies on.
    table = table.sort_values("query_group", kind="stable").reset_index(drop=True)

    check_feature_table(table)
    return table
