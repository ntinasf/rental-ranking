"""The two frozen baselines, defined before any model exists.

**A baseline chosen after seeing the model's score is not a baseline.** These definitions are
frozen — `tests/test_baseline.py` pins their measured NDCG@10 against the ranked population, so
"improving" one later fails a test and has to be argued for rather than done quietly.

**Baseline A — rank by ``number_of_reviews`` within group.** Phase 1 measured the label to be
substantially establishment-driven (mean blocked fraction rises monotonically with listing age in
all three cities), so this is expected to be *strong*. That is the point of freezing it: a
LambdaMART that cannot beat "the listing with the most reviews goes first" is the honest headline
of this project, and discovering that after training turns a finding into an embarrassment.

**Baseline B — the price + rating heuristic** BUILD_GUIDE specifies: rank by good rating and low
price, the naive "best value" ordering a product team would ship on day one.

Two construction choices in B, both made to keep the baseline **strong rather than convenient**,
since a weak baseline flatters the model:

* **Within-group percentile ranks, not raw values or z-scores.** Price has a skew of 6.5 and a
  maximum of 9,243 against a median of 120, so a z-score would be dominated by a handful of
  listings; a percentile is robust and already group-local, which is the right frame for a
  metric computed inside a group.
* **``rating_shrunk`` rather than the raw ``review_scores_rating``.** The raw score ties 32.7 %
  of reviewed listings at exactly 5.0 and is null for the 16.3 % never reviewed, so a baseline
  built on it would be crippled by ties and missingness — and beating a crippled baseline proves
  nothing. The shrunk rating resolves both by construction. It is one line of Phase 2 work, not a
  model.

Neither baseline reads the label, and neither is fitted: there is nothing here to overfit, which
is exactly why the numbers can be recorded before a split exists.

Convention, matching the rest of the package: pure ``DataFrame -> Series`` transforms, no I/O.
"""

import pandas as pd

from rental_ranking.data.validate import require_columns

#: Weight on the rating half of baseline B. 0.5 gives rating and cheapness equal say; it is a
#: stated convention, not a fitted value, and it is frozen. Tuning it against NDCG would make the
#: baseline a model — and one selected on the evaluation set at that.
PRICE_RATING_WEIGHT = 0.5


def rank_by_reviews(listings: pd.DataFrame) -> pd.Series:
    """Baseline A: score each listing by its lifetime review count.

    Deliberately the plainest thing that could work. It is expected to be strong because the
    label is establishment-driven; if it is, that is the finding.

    Args:
        listings: Frame carrying ``number_of_reviews``.

    Returns:
        A float Series aligned to ``listings``, named ``baseline_reviews``. Higher ranks first.

    Raises:
        KeyError: If ``number_of_reviews`` is absent.
    """
    require_columns(listings, ("number_of_reviews",), "listings")
    return listings["number_of_reviews"].astype("float64").rename("baseline_reviews")


def rank_by_price_and_rating(
    listings: pd.DataFrame,
    groups: pd.Series,
    weight: float = PRICE_RATING_WEIGHT,
) -> pd.Series:
    """Baseline B: good rating, low price — both as within-group percentiles.

    ``weight * pct(rating_shrunk) - (1 - weight) * pct(price)``, so a listing rated well for its
    group and priced low for its group ranks first.

    Args:
        listings: Frame carrying ``price`` and ``rating_shrunk``.
        groups: Query-group id per row — the percentiles are taken inside the group, because
            that is the set the listing is being ranked against.
        weight: Share given to the rating half. Defaults to :data:`PRICE_RATING_WEIGHT`.

    Returns:
        A float Series aligned to ``listings``, named ``baseline_price_rating``.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If ``price`` or ``rating_shrunk`` is null — both are complete by
            construction after Phase 1 and Phase 2, so a null means the wrong frame was passed.
    """
    require_columns(listings, ("price", "rating_shrunk"), "listings")

    nulls = listings[["price", "rating_shrunk"]].isna().sum()
    if nulls.any():
        raise ValueError(
            f"baseline B needs complete inputs; found nulls {nulls[nulls > 0].to_dict()}. "
            "`price` is imputed in Phase 1 and `rating_shrunk` returns the city prior at n=0, "
            "so a null here means an unfiltered or pre-imputation frame was passed"
        )

    rating_pct = listings.groupby(groups, observed=True)["rating_shrunk"].rank(pct=True)
    price_pct = listings.groupby(groups, observed=True)["price"].rank(pct=True)
    score = weight * rating_pct - (1 - weight) * price_pct
    return score.astype("float64").rename("baseline_price_rating")
