"""Tests for rental_ranking.features.assemble.

Every check in this module guards a failure that produces a *working* model rather than an
error: a label-adjacent column trains a ranker that cannot be deployed, a duplicated id doubles
a listing's weight in its group, and an unsorted matrix hands LightGBM queries that mix two
searches. All of them print plausible metrics. So each is tested in both directions — it fires
when it should, and stays quiet on a clean table.
"""

import numpy as np
import pandas as pd
import pytest

from rental_ranking.data.columns import LABEL_ADJACENT_COLUMNS
from rental_ranking.data.paths import PROCESSED_DIR
from rental_ranking.features import assemble


def _table(rows: int = 6, **overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(rows)],
            "query_group": [0] * (rows // 2) + [1] * (rows - rows // 2),
            "cluster_id": range(rows),
            "grade": [i % 5 for i in range(rows)],
            "blocked_fraction_90": np.linspace(0, 1, rows),
            "price": np.linspace(50, 200, rows),
        }
    )
    return base.assign(**overrides)


# --- the column roles --------------------------------------------------------------------------


def test_feature_columns_excludes_identifiers_and_targets() -> None:
    """The one line between training on the features and training on the answer."""
    features = assemble.feature_columns(_table())

    assert features == ["price"]
    for reserved in (*assemble.IDENTIFIER_COLUMNS, *assemble.TARGET_COLUMNS):
        assert reserved not in features


def test_the_label_is_carried_but_is_never_a_feature() -> None:
    table = _table()

    assert "blocked_fraction_90" in table.columns
    assert "blocked_fraction_90" not in assemble.feature_columns(table)


# --- the checks --------------------------------------------------------------------------------


def test_a_clean_table_passes_and_reports_its_shape() -> None:
    report = assemble.check_feature_table(_table())

    assert report["rows"] == 6
    assert report["query_groups"] == 2
    assert report["group_array_sum"] == 6


def test_a_label_adjacent_column_is_caught() -> None:
    """Read live from columns.py, so a future snapshot's blocklist is enforced for free."""
    blocked = sorted(LABEL_ADJACENT_COLUMNS)[0]

    with pytest.raises(ValueError, match="label-adjacent"):
        assemble.check_feature_table(_table(**{blocked: 1}))


def test_every_blocklist_member_is_caught_not_just_the_first() -> None:
    for blocked in LABEL_ADJACENT_COLUMNS:
        with pytest.raises(ValueError, match="label-adjacent"):
            assemble.check_feature_table(_table(**{blocked: 1}))


def test_a_phase_one_diagnostic_column_is_caught() -> None:
    """`avail_90` is the label's own numerator; it is not on the contract blocklist."""
    with pytest.raises(ValueError, match="diagnostic"):
        assemble.check_feature_table(_table(avail_90=3))


def test_every_diagnostic_column_is_caught() -> None:
    for column in assemble.DIAGNOSTIC_COLUMNS:
        with pytest.raises(ValueError, match="diagnostic"):
            assemble.check_feature_table(_table(**{column: 1}))


def test_a_duplicated_listing_is_caught() -> None:
    table = _table()
    table.loc[1, "id"] = table.loc[0, "id"]

    with pytest.raises(ValueError, match="duplicated listing id"):
        assemble.check_feature_table(table)


def test_an_unsorted_matrix_is_caught() -> None:
    """The half of gotcha #4 a sum check cannot see: the sum is still correct here."""
    table = _table()
    shuffled = table.iloc[[0, 3, 1, 4, 2, 5]].reset_index(drop=True)

    with pytest.raises(ValueError, match="not sorted"):
        assemble.check_feature_table(shuffled)


def test_a_null_target_is_caught() -> None:
    table = _table()
    table.loc[2, "grade"] = np.nan

    with pytest.raises(ValueError, match="null identifier or target"):
        assemble.check_feature_table(table)


def test_a_missing_identifier_raises_a_readable_keyerror() -> None:
    with pytest.raises(KeyError, match="feature table"):
        assemble.check_feature_table(_table().drop(columns=["cluster_id"]))


# --- against the real snapshots ------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_table() -> pd.DataFrame:
    for name in ("listings", "calendar", "reviews"):
        if not (PROCESSED_DIR / f"{name}.parquet").exists():
            pytest.skip(f"processed layer not on disk: {name}.parquet")

    from rental_ranking.features.build import prepare_ranked

    listings = pd.read_parquet(PROCESSED_DIR / "listings.parquet")
    calendar = pd.read_parquet(PROCESSED_DIR / "calendar.parquet")
    ranked = prepare_ranked(listings, calendar)
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet", columns=["listing_id", "date"])
    return assemble.assemble_feature_table(ranked, reviews)


def test_the_real_matrix_is_one_row_per_ranked_listing(real_table: pd.DataFrame) -> None:
    report = assemble.check_feature_table(real_table)

    assert report["rows"] == 44_684
    assert report["query_groups"] == 393
    assert report["group_array_sum"] == report["rows"]


def test_the_real_matrix_is_sorted_into_contiguous_groups(real_table: pd.DataFrame) -> None:
    """LightGBM reads the group array positionally and never sees the ids."""
    assert real_table["query_group"].is_monotonic_increasing


def test_every_feature_block_reached_the_matrix(real_table: pd.DataFrame) -> None:
    """One representative column per block, so a silently dropped block fails here."""
    for column in (
        "accommodates",  # structural
        "host_is_superhost",  # host
        "amenity_kitchen",  # amenity buckets
        "rating_shrunk",  # review
        "price_vs_nbhd",  # neighbourhood aggregate
        "km_to_neighbourhood_centroid",  # spatial
        "reviews_same_season_ly",  # review window, needs the reviews frame
    ):
        assert column in real_table.columns, column


def test_the_six_aspect_sub_scores_reached_the_matrix(real_table: pd.DataFrame) -> None:
    """The columns that made a sentiment feature unnecessary — measured 2026-08-17."""
    for aspect in ("accuracy", "cleanliness", "checkin", "communication", "location", "value"):
        assert f"review_scores_{aspect}" in real_table.columns


def test_the_derived_review_columns_are_null_for_exactly_the_never_reviewed(
    real_table: pd.DataFrame,
) -> None:
    """16.3 % never reviewed, one clean cohort — not a data-quality problem."""
    never = ~real_table["has_reviews"]

    for column in ("reviews_per_month", "listing_age_days", "days_since_last_review"):
        assert real_table[column].isna().equals(never), column


def test_the_sub_scores_add_a_handful_of_nulls_beyond_that_cohort(
    real_table: pd.DataFrame,
) -> None:
    """The one place the cohort is not exact, and it is worth knowing rather than asserting away.

    ``review_scores_rating`` is null for exactly the never-reviewed, zero exceptions. The six
    **sub**-scores are not: two or three listings carry a single review that gave an overall
    score and left the sub-categories blank. It is a rounding error, but "null iff never
    reviewed" is true of the headline rating and false of its parts.
    """
    never = ~real_table["has_reviews"]

    for aspect in ("accuracy", "cleanliness", "checkin", "communication", "location", "value"):
        nulls = real_table[f"review_scores_{aspect}"].isna()
        assert (nulls | never).equals(nulls), aspect  # every never-reviewed row is null
        assert 0 < (nulls & ~never).sum() <= 5, aspect  # and a handful more


def test_nothing_that_reads_the_label_window_is_present(real_table: pd.DataFrame) -> None:
    forbidden = LABEL_ADJACENT_COLUMNS | assemble.DIAGNOSTIC_COLUMNS
    assert set(real_table.columns) & forbidden == set()
