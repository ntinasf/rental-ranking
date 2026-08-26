"""Orchestrate processed → feature table: the only module that writes ``data/features/``.

What ``data/build.py`` is to the processed layer, this is to the modelling one. Every module
beside it in ``features/`` is a pure transform; this is where they are chained, where the
filesystem is touched, and where the per-stage reports are printed so a run leaves a record of
what it did rather than only an artifact.

**The order is not negotiable:**

``label → filters → price imputation → grading → grouping → feature blocks``

Filters run before imputation and grading so every quantile is computed on the population that
will actually be ranked; grouping runs after grading so the coarsening check has both keys.

Run it with ``uv run python -m rental_ranking.features.build``.
"""

import pandas as pd

from rental_ranking.data.filters import filter_listings
from rental_ranking.data.paths import FEATURE_TABLE_PATH, FEATURES_DIR, PROCESSED_DIR
from rental_ranking.features.assemble import assemble_feature_table, feature_columns
from rental_ranking.features.groups import cluster_id, query_group
from rental_ranking.features.label import assign_grades, occupancy_label
from rental_ranking.features.price import impute_price

#: Columns read from the reviews parquet. It is 327 MB on disk and the review *text* belongs to
#: the sentiment demo, not to the matrix, so only the two columns the windows need are loaded.
_REVIEW_COLUMNS = ["listing_id", "date"]


def prepare_ranked(listings: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Label, filter, price, grade and group, returning the frame the feature blocks consume.

    Args:
        listings: Processed listings.
        calendar: Processed calendar.

    Returns:
        The filtered, priced, graded and grouped population.
    """
    labels = occupancy_label(calendar)
    joined = listings.merge(labels, left_on="id", right_index=True, how="inner")

    kept, filter_counts = filter_listings(joined)
    print(f"\nfilters — {len(joined):,} listings in, {len(kept):,} kept")
    print(filter_counts.to_string())

    ranked, price_counts = impute_price(kept)
    print("\nprice imputation — rows filled per cascade rung")
    print(price_counts.to_string())

    ranked["grade"], grade_report = assign_grades(ranked)
    print(
        f"\ngrading — shares {ranked['grade'].value_counts(normalize=True).sort_index().round(3).to_dict()}"
    )

    ranked["cluster_id"] = cluster_id(ranked)
    ranked["query_group"], rung_report = query_group(ranked)
    print("\nquery groups — per cascade rung")
    print(rung_report.to_string())
    return ranked


def write(table: pd.DataFrame) -> None:
    """Write the feature table to ``data/features/feature_table.parquet``.

    ``index=False`` because the matrix is positional from assembly onward: the row order *is*
    the group array's order, and a persisted index would invite someone to reorder by it.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FEATURE_TABLE_PATH, index=False)
    print(
        f"\nfeature table  {len(table):>10,} rows x {table.shape[1]:>3} cols "
        f"({len(feature_columns(table))} features) -> {FEATURE_TABLE_PATH}"
    )


def main() -> None:
    """Build the feature table from the processed layer."""
    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    calendar = pd.read_parquet(PROCESSED_DIR / "calendar.parquet")
    ranked = prepare_ranked(listings, calendar)
    del calendar, listings

    reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet", columns=_REVIEW_COLUMNS)
    table = assemble_feature_table(ranked, reviews)
    del reviews

    write(table)


if __name__ == "__main__":
    main()
