"""Regenerate the ablation table: what the ranker loses when a block of features is withheld.

**Gain importance answers a different question than the one people read it for.** It says which
features the trees split on, not what the model would lose without them — and when features are
correlated those diverge sharply. The establishment block carries 29.5 % of total gain, yet a
model denied it entirely still reaches 0.7031 against the full model's 0.7209. Only a refit on a
restricted feature set settles it, which is what this module runs.

Four decisions, each of which was a way to get the table wrong:

* **Every ablation is cross-validated, never refitted-and-rescored.** The numbers are out-of-fold
  on the development pool, so each group is scored by the fold model that held it out. That is
  what makes them comparable to the out-of-fold headline rather than to the sealed one, and it is
  why ``full (61)`` here reproduces that headline exactly.
* **Differences are paired over the same groups.** ``comparison_table`` takes every ranker at once
  and takes ``vs_full`` per group, so the columns compare two orderings of the same listings
  rather than two separately-computed averages.
* **The frozen baselines are not ablations and are excluded.** A heuristic has no feature set, so
  it cannot answer "what does the model lose without this block". It also adds nothing: the
  review-count baseline's difference against the full model is the reported headline with the sign
  flipped. It sat in this table until 2026-08-22 and could not be read as anything but a ninth
  ablation.
* **Blocks are derived from the shipped columns by rule, not typed out by hand** — except
  :data:`ESTABLISHMENT`, which is a definition rather than a prefix. A new feature therefore joins
  its block automatically instead of quietly escaping every ablation in the table.

**Establishment is not the review block**, and the difference is the whole point of the provenance
argument in notebook 04 §7. It excludes the six ``review_scores_*`` columns and includes
``host_tenure_months``: establishment is *how long this listing has been running and how much
traffic it has seen*, not *how good it is*. Quality is a separate thing and stays in the model.
The eight below carry **0.2945** of total gain, which is the 29.5 % the notebook reports; taking
``rating_shrunk`` instead of ``host_tenure_months`` gives 0.3080 and is the wrong set.

**One row is a different matrix, not a different subset.** ``amenities: flags`` re-encodes the
amenity block as binaries over a 50-amenity vocabulary, so it cannot be expressed as a column
selection and needs the raw ``amenities`` lists the feature table does not carry. Which vocabulary
was used had never been written down; it was recovered on 2026-08-22 by rebuilding both candidates
and scoring them — ``frequency`` reproduces the reported **0.7260** exactly, and
``within_group_variance`` gives 0.7188, so the criterion is identified rather than assumed.
:func:`flags_out_of_fold` rebuilds it, and :func:`main` folds the row back in when the processed
layer is available.

What is pinned here is the **criterion and the size**, not the fifty strings. ``fit_vocabulary``
asks to be pinned so the feature set cannot depend on the split; fitting on the whole ranked
population satisfies that already, and re-fitting is the behaviour a rebuild should have when the
snapshot changes.

Convention, matching ``train/sweep.py``: pure functions, with I/O confined to :func:`main`.
"""

from __future__ import annotations

import pandas as pd

from rental_ranking.evaluate.report import comparison_table, random_floor
from rental_ranking.features.assemble import feature_columns
from rental_ranking.train.split import SEALED_FOLD

#: The eight establishment features, in gain order. **A definition, not a derivation.** Listing
#: age and host tenure say how long the operation has been running; the four review-volume and
#: recency columns say how much traffic it has seen; ``has_reviews`` names the cohort. What is
#: deliberately absent is every measure of how *good* the listing is — the six ``review_scores_*``
#: aspect scores and ``rating_shrunk`` — because the feedback loop this block exists to test runs
#: through exposure, not through quality.
ESTABLISHMENT: tuple[str, ...] = (
    "number_of_reviews",
    "days_since_last_review",
    "number_of_reviews_ltm",
    "listing_age_days",
    "host_tenure_months",
    "reviews_per_month",
    "reviews_same_season_ly",
    "has_reviews",
)

#: Prefix rules for the blocks that *are* derivable, mirroring the split used in notebook 03 §3.
#: ``price_vs_nbhd`` is a neighbourhood feature that does not carry the prefix, so it is named.
_BLOCK_RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "amenities": (("amenity_",), ()),
    "spatial": (("km_to", "density_"), ()),
    "neighbourhood": (("nbhd_",), ("price_vs_nbhd",)),
}

#: The flag encoding's vocabulary, as parameters rather than as fifty pasted strings. Recovered by
#: reproduction on 2026-08-22: at k=50, ``frequency`` gives the reported 0.7260 on 92 features and
#: ``within_group_variance`` gives 0.7188, so this is the pair that was actually used.
FLAG_VOCABULARY_SIZE = 50
FLAG_VOCABULARY_CRITERION = "frequency"

#: What the flag-encoded ablation reports under.
FLAGS_NAME = "amenities: flags"

#: ``amenity_count`` is emitted by every amenity scheme and is the control the buckets have to
#: beat, so the ``amenities: count`` variant keeps it and drops only the concept buckets.
_AMENITY_CONTROL = "amenity_count"


def block_members(features: list[str], block: str) -> list[str]:
    """Columns of ``features`` belonging to ``block``, in the order they appear.

    Args:
        features: The model's input columns, from ``assemble.feature_columns``.
        block: A key of :data:`_BLOCK_RULES`, or ``"establishment"``.

    Returns:
        The block's members. Order follows ``features`` so the result is stable.

    Raises:
        KeyError: If ``block`` is unknown.
        ValueError: If the block resolves to nothing, which means a rename has silently emptied
            an ablation rather than failing it.
    """
    if block == "establishment":
        members = [column for column in features if column in set(ESTABLISHMENT)]
        missing = sorted(set(ESTABLISHMENT) - set(members))
        if missing:
            raise ValueError(
                f"establishment features absent from the matrix: {missing}. The block is a "
                "definition, so a renamed column must be corrected here rather than dropped "
                "silently — an ablation that withholds seven of eight features is not the "
                "ablation the write-up reports"
            )
        return members

    if block not in _BLOCK_RULES:
        raise KeyError(f"unknown block {block!r}; known: {['establishment', *_BLOCK_RULES]}")

    prefixes, exact = _BLOCK_RULES[block]
    members = [c for c in features if c.startswith(prefixes) or c in exact]
    if not members:
        raise ValueError(f"block {block!r} matched no column; the prefix rule has gone stale")
    return members


def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    """The feature list each ablation trains on, keyed by the name it reports under.

    The reference is named for its size — ``full (61)`` — and that count is read from the matrix
    rather than written down, so the label cannot drift from what was actually fitted.

    Args:
        table: The feature matrix.

    Returns:
        ``{ranker name: feature columns}``, reference first.
    """
    features = feature_columns(table)
    amenities = block_members(features, "amenities")
    spatial = block_members(features, "spatial")
    neighbourhood = block_members(features, "neighbourhood")
    establishment = block_members(features, "establishment")

    def without(*blocks: list[str]) -> list[str]:
        drop = {column for block in blocks for column in block}
        return [column for column in features if column not in drop]

    return {
        full_name(table): features,
        "minus spatial": without(spatial),
        "minus neighbourhood": without(neighbourhood),
        "minus amenities": without(amenities),
        "minus establishment": without(establishment),
        "minus spatial + neighbourhood": without(spatial, neighbourhood),
        # The control kept, the 19 concept buckets dropped.
        "amenities: count": without([c for c in amenities if c != _AMENITY_CONTROL]),
        # The mirror of `minus establishment`, and the other half of the provenance argument:
        # denying the block answers "does the model need it", keeping only the block answers
        # "is the block enough on its own".
        "establishment only": establishment,
    }


def full_name(table: pd.DataFrame) -> str:
    """The reference ranker's name, carrying the feature count it was fitted on."""
    return f"full ({len(feature_columns(table))})"


def run_ablations(
    table: pd.DataFrame,
    fold: pd.Series,
    sets: dict[str, list[str]] | None = None,
    extra_scores: dict[str, tuple[pd.Series, int]] | None = None,
    params: dict[str, object] | None = None,
    seed: int = 0,
    sealed: int = SEALED_FOLD,
) -> pd.DataFrame:
    """Cross-validate every ablation and compare them on identical groups.

    Args:
        table: The feature matrix.
        fold: Fold ids from ``split.assign_folds``.
        sets: ``{name: features}``; defaults to :func:`feature_sets`.
        extra_scores: Rankers whose *matrix* differs rather than whose columns do, as
            ``{name: (out-of-fold scores, feature count)}``. The scores must already be aligned
            to this table's development rows — see :func:`flags_out_of_fold`.
        params: LightGBM parameters, defaulting to the project defaults.
        seed: Seed for the fits, the floor and both bootstraps.
        sealed: The held-out fold, excluded from every number here.

    Returns:
        One row per ranker: ``groups``, ``degenerate``, ``ndcg@10`` and its interval, ``floor``,
        ``range_share``, the paired ``vs_<reference>`` with its interval, and ``n_features``.
    """
    from rental_ranking.train.train import cross_validate

    sets = sets if sets is not None else feature_sets(table)
    reference = full_name(table)
    if reference not in sets:
        raise KeyError(f"the reference {reference!r} must be one of the ablations: {sorted(sets)}")

    development = table[fold != sealed]

    # Checked before anything is fitted. A misaligned index is otherwise discovered after eight
    # cross-validations, and the failure it prevents is silent: each listing's score would be
    # paired with a different listing's grade and the table would still fill in.
    for name, (supplied, _) in (extra_scores or {}).items():
        if not supplied.index.equals(development.index):
            raise ValueError(
                f"{name!r} supplies scores on a different index than this table's development "
                "rows, so the paired difference would be taken across two populations"
            )

    scores: dict[str, pd.Series] = {}
    sizes = {name: len(columns) for name, columns in sets.items()}
    for name, columns in sets.items():
        print(f"\n{name}: {len(columns)} features")
        out_of_fold, _, _, _ = cross_validate(
            table, fold, params=params, seed=seed, features=columns
        )
        scores[name] = out_of_fold.loc[development.index]

    for name, (supplied, n_features) in (extra_scores or {}).items():
        scores[name] = supplied
        sizes[name] = n_features

    # One floor for every ranker: the random reference does not depend on which model is being
    # scored, and recomputing it per call would put bootstrap noise into a constant.
    floor = random_floor(table["grade"], table["query_group"], seed=seed)
    report = comparison_table(
        development["grade"],
        development["query_group"],
        scores,
        reference=reference,
        floor=floor,
        seed=seed,
    ).loc["overall"]
    return report.assign(n_features=pd.Series(sizes))


def flags_out_of_fold(
    table: pd.DataFrame,
    ranked: pd.DataFrame,
    reviews: pd.DataFrame | None = None,
    params: dict[str, object] | None = None,
    seed: int = 0,
    sealed: int = SEALED_FOLD,
) -> tuple[pd.Series, int]:
    """Out-of-fold scores for the flag-encoded matrix, which is a rebuild rather than a subset.

    The other ablations drop columns from the shipped table. This one re-encodes the amenity
    block as binaries over a fitted vocabulary, so it needs the raw ``amenities`` lists and a
    second pass through the assembler.

    Args:
        table: The shipped (bucket-encoded) matrix, used only to check row correspondence.
        ranked: The filtered population ``features/build.prepare_ranked`` produces.
        reviews: Processed reviews, passed through to the assembler.
        params: LightGBM parameters, defaulting to the project defaults.
        seed: Seed for the fits.
        sealed: The held-out fold.

    Returns:
        ``(out-of-fold scores over the development rows, feature count)``.

    Raises:
        ValueError: If the rebuilt matrix does not hold the same listings in the same order.
    """
    from rental_ranking.features.amenities import fit_vocabulary
    from rental_ranking.features.assemble import assemble_feature_table
    from rental_ranking.train.split import assign_folds
    from rental_ranking.train.train import cross_validate

    vocabulary = fit_vocabulary(ranked, k=FLAG_VOCABULARY_SIZE, by=FLAG_VOCABULARY_CRITERION)
    matrix = assemble_feature_table(ranked, reviews, amenity_scheme="flags", vocabulary=vocabulary)

    # Both matrices come from one population sorted the same way, so their rows correspond --
    # checked rather than assumed, because a mismatch would pair each listing's flag-model score
    # with a different listing's grade and still return a perfectly plausible number.
    if not matrix["id"].equals(table["id"]):
        raise ValueError(
            "the flag-encoded matrix does not carry the same listings in the same order as the "
            "shipped one, so its scores cannot be set against those grades row by row"
        )

    fold, _ = assign_folds(matrix)
    columns = feature_columns(matrix)
    print(f"\n{FLAGS_NAME}: {len(columns)} features (matrix rebuilt)")
    out_of_fold, _, _, _ = cross_validate(matrix, fold, params=params, seed=seed, features=columns)
    return out_of_fold.loc[matrix.index[fold != sealed]], len(columns)


def main() -> None:
    """Rebuild the table and write it where notebook 04 reads it from."""
    import argparse

    from rental_ranking.data import paths
    from rental_ranking.features import build as feature_build
    from rental_ranking.train import split

    parser = argparse.ArgumentParser(description="Rebuild the ablation table.")
    parser.add_argument(
        "--no-flags",
        action="store_true",
        help="skip the flag-encoded ablation, which rebuilds the matrix from data/processed/",
    )
    arguments = parser.parse_args()

    table = pd.read_parquet(paths.FEATURE_TABLE_PATH)
    fold, _ = split.assign_folds(table)

    extra = None
    if not arguments.no_flags:
        listings = pd.read_parquet(paths.PROCESSED_DIR / "listings.parquet")
        calendar = pd.read_parquet(paths.PROCESSED_DIR / "calendar.parquet")
        reviews = pd.read_parquet(
            paths.PROCESSED_DIR / "reviews.parquet", columns=["listing_id", "date"]
        )
        ranked = feature_build.prepare_ranked(listings, calendar)
        extra = {FLAGS_NAME: flags_out_of_fold(table, ranked, reviews, seed=0)}

    report = run_ablations(table, fold, extra_scores=extra)
    ordered = report.sort_values("ndcg@10", ascending=False)

    print(f"\n{'=' * 78}\nablations, development out-of-fold\n{'=' * 78}")
    print(ordered.round(4).to_string())

    paths.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    destination = paths.TRAIN_DIR / "ablations.csv"
    ordered.to_csv(destination)
    print(f"\nwritten: {destination}")


if __name__ == "__main__":
    main()
