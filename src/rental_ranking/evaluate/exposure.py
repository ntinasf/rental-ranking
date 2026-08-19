"""What changes when the candidate set widens: composition, geographic spread, cohort reach.

This module exists for one experiment — the large-area search design in
``docs/ab_test_design.md`` — and for one methodological rule.

**Never compare NDCG across grouping schemes.** Changing the query-group key changes the candidate
set, the ideal DCG and the random floor all at once, so the metric is a *different quantity* at
each rung of ``groups.GROUP_CASCADE``. The coarsest rung, ``city x room_type``, is the grading
partition itself, which means its within-group grade distribution is fixed by construction — a
model could get "better NDCG" there by doing nothing at all. Any table putting a rung-1 NDCG beside
a rung-3 NDCG is comparing two things that share a name.

So this module measures **composition and exposure** instead: how large the candidate set becomes,
how many neighbourhoods survive into the top *k*, and what share of a cohort reaches it. Those are
defined identically regardless of grouping, which is exactly what makes them comparable.

Two further precisions:

* The re-keyings here are **plain groupbys on a cascade rung, not the cascade's output.**
  ``groups.query_group`` falls back only for groups under the minimum; this asks the counterfactual
  question "what if the key had been this all along", which is the treatment arm the design
  document proposes.
* The random reference for :func:`cohort_reach` is **analytic, not simulated.** Under a uniform
  shuffle the chance a given listing lands in the top *k* of a group of *n* is exactly
  ``min(k, n) / n``, so there is nothing to estimate and no seed to choose.

Pure transforms, as everywhere else in ``evaluate/``; :func:`main` is the only I/O.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from rental_ranking.cloud.score import MAX_LISTINGS
from rental_ranking.data.validate import require_columns
from rental_ranking.evaluate.metrics import DEFAULT_K, RELEVANT_GRADE
from rental_ranking.features import groups

#: Column carrying the geography the concentration metric is computed over. The neighbourhood is
#: the unit a guest picks and the unit the query-group key drops first, so it is the one that
#: matters for a large-area design.
GEO_COLUMN = "neighbourhood_cleansed"


def candidate_set_profile(
    listings: pd.DataFrame,
    cascade: Sequence[tuple[str, list[str]]] = tuple(groups.GROUP_CASCADE),
    cap: int = MAX_LISTINGS,
) -> pd.DataFrame:
    """How large one search's candidate set becomes at each rung of the key.

    The serving question, not the modelling one. ``cap`` is imported from the scoring script rather
    than restated, because the two must not drift: a rung whose largest group exceeds it describes
    a ranking the deployed endpoint would refuse.

    Args:
        listings: The ranked population. ``capacity_tier`` is derived here rather than read, for
            the same reason ``groups.query_group`` derives it — so a profile can never be computed
            on different tier bounds than the groups were built with.
        cascade: Rungs as ``(name, key)``. Defaults to :data:`groups.GROUP_CASCADE`.
        cap: Per-request listing limit. Defaults to the endpoint's own.

    Returns:
        One row per rung, indexed by rung name: ``key``, ``groups``, ``median``, ``p90``, ``max``,
        ``over_cap`` (groups exceeding it) and ``serviceable``.
    """
    needed = {column for _, key in cascade for column in key} - {"capacity_tier"}
    require_columns(listings, (*sorted(needed), "accommodates"), "ranked listings")
    keyed = listings.assign(capacity_tier=groups.capacity_tier(listings))

    rows = []
    for name, key in cascade:
        sizes = keyed.groupby(list(key), observed=True, dropna=False).size()
        rows.append(
            {
                "rung": name,
                "key": " x ".join(key),
                "groups": int(len(sizes)),
                "median": int(sizes.median()),
                "p90": int(sizes.quantile(0.9)),
                "max": int(sizes.max()),
                "over_cap": int((sizes > cap).sum()),
                "serviceable": bool((sizes <= cap).all()),
            }
        )
    return pd.DataFrame(rows).set_index("rung")


def _normalised_entropy(counts: np.ndarray, available: int, slots: int) -> float:
    """Shannon entropy of a composition, scaled by the most it could have been.

    1.0 is perfectly spread across every geography it could have reached; 0.0 is all one. NaN when
    only one geography was available, because there is then nothing to spread and a number would
    imply otherwise.

    **The denominator is what was available, not what appeared.** Normalising by the observed count
    makes a top-k that collapsed onto a single neighbourhood undefined instead of zero — that is,
    it silently erases the exact failure this metric exists to detect. Caught by
    ``tests/test_exposure.py`` before the number reached the document.
    """
    reachable = min(slots, available)
    if reachable <= 1:
        return float("nan")
    share = counts / counts.sum()
    entropy = float(-(share * np.log(share)).sum())
    return entropy / float(np.log(reachable))


def geographic_concentration(
    listings: pd.DataFrame,
    scores: pd.Series,
    group: pd.Series,
    k: int = DEFAULT_K,
    geo_column: str = GEO_COLUMN,
) -> pd.DataFrame:
    """Does a widened ranking still show a guest more than one neighbourhood?

    The failure mode the Airbnb paper names for large-area search: with a big enough candidate set,
    a ranker with no notion of destination collapses the first screen onto wherever its features
    happen to score highest, and the guest never learns the wider area had anything in it.

    Args:
        listings: Candidates, carrying ``geo_column``.
        scores: The ranker's score per row, aligned to ``listings``.
        group: Query-group id per row.
        k: How many slots the first screen has.
        geo_column: Geography to measure spread over.

    Returns:
        One row per group: ``n``, ``geos_available``, ``geos_in_top_k``, ``coverage`` (of the most
        it could have reached, ``min(geos_available, k)``) and ``entropy``. ``coverage`` is 1.0 and
        ``entropy`` NaN for any group confined to a single neighbourhood — which is every group at
        rung 1, by construction.
    """
    require_columns(listings, (geo_column,), "candidates")
    frame = pd.DataFrame(
        {
            "geo": listings[geo_column].to_numpy(),
            "score": scores.to_numpy(),
            "group": group.to_numpy(),
        }
    )

    rows = []
    for name, block in frame.groupby("group", observed=True, sort=False):
        available = block["geo"].nunique(dropna=False)
        top = block.nlargest(k, "score", keep="first")
        counts = top["geo"].value_counts(dropna=False).to_numpy()
        rows.append(
            {
                "group": name,
                "n": int(len(block)),
                "geos_available": int(available),
                "geos_in_top_k": int(counts.size),
                "coverage": counts.size / min(available, k),
                "entropy": _normalised_entropy(counts, available, k),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def cohort_reach(
    scores: pd.Series,
    group: pd.Series,
    cohort: pd.Series,
    k: int = DEFAULT_K,
) -> pd.Series:
    """What share of a cohort reaches the first screen, against what a shuffle would give it.

    Generalises the cold-start finding of 2026-08-18 — deserving never-reviewed listings reach the
    top 10 at 5.8 % against a shuffle's 9.6 % — so the same question can be asked of any cohort
    under any grouping.

    **The reference is exact, not simulated.** Under a uniform shuffle a listing in a group of
    ``n`` reaches the top ``k`` with probability ``min(k, n) / n``, so the cohort's expected reach
    is the mean of that over its members. No draws, no seed, no Monte Carlo error.

    Args:
        scores: The ranker's score per row.
        group: Query-group id per row.
        cohort: Boolean mask selecting the cohort — the listings whose exposure is in question.
        k: First-screen size.

    Returns:
        ``cohort``, ``reached``, ``reach_rate``, ``random_rate``, ``lift`` (rate minus reference).
        All NaN-free; ``reach_rate`` is NaN only if the cohort is empty.
    """
    frame = pd.DataFrame(
        {"score": scores.to_numpy(), "group": group.to_numpy(), "cohort": cohort.to_numpy()}
    )
    frame["rank"] = frame.groupby("group", observed=True)["score"].rank(
        ascending=False, method="first"
    )
    sizes = frame.groupby("group", observed=True)["score"].transform("size")
    frame["reference"] = np.minimum(k, sizes) / sizes

    members = frame[frame["cohort"]]
    if members.empty:
        return pd.Series(
            {"cohort": 0, "reached": 0, "reach_rate": np.nan, "random_rate": np.nan, "lift": np.nan}
        )
    reached = int((members["rank"] <= k).sum())
    reach_rate = reached / len(members)
    random_rate = float(members["reference"].mean())
    return pd.Series(
        {
            "cohort": int(len(members)),
            "reached": reached,
            "reach_rate": reach_rate,
            "random_rate": random_rate,
            "lift": reach_rate - random_rate,
        }
    )


def rung_labels(listings: pd.DataFrame, key: Sequence[str]) -> pd.Series:
    """Group ids from a **plain re-keying** at one cascade rung — no minimum, no fallback.

    ``groups.query_group`` widens the key only for groups below the minimum. This answers the
    counterfactual the design document needs instead: what if the key had been this for everyone?

    Raises:
        KeyError: If a key column is missing.
    """
    keyed = listings.assign(capacity_tier=groups.capacity_tier(listings))
    require_columns(keyed, tuple(key), "candidates")
    return keyed.groupby(list(key), observed=True, dropna=False).ngroup().rename("rung_group")


# --- the script ------------------------------------------------------------------------------


def _sealed_scores(sealed: pd.DataFrame) -> pd.Series:
    """Score the sealed fold with the served booster.

    **This is not a third read of the holdout.** The project declared two performance reads and
    spent both (docs/decisions_log.md, 2026-08-18). Nothing here computes NDCG, a paired
    difference or any comparison against a baseline; the sealed fold is used because it is the only
    population the refit model did not fit, and scoring the development folds with it would make
    every exposure number optimistic. Composition is read, quality is not.
    """
    import lightgbm as lgb

    from rental_ranking.data import paths
    from rental_ranking.train import lambdamart as lm

    booster = lgb.Booster(model_file=str(paths.SERVING_BUNDLE_DIR / "model.lgb"))
    metadata = json.loads((paths.SERVING_BUNDLE_DIR / "serving_metadata.json").read_text())
    return pd.Series(
        booster.predict(lm.design_matrix(sealed, metadata["features"])),
        index=sealed.index,
        name="score",
    )


def main() -> None:
    """Print the exposure tables and write them, in the pattern of the sweep results."""
    from rental_ranking.cloud import demo
    from rental_ranking.data import paths
    from rental_ranking.train import split

    population = demo.ranked_table()
    sealed = demo.sealed_table()
    scores = _sealed_scores(sealed)
    cold = ~sealed["has_reviews"].astype(bool)
    deserving = cold & sealed["grade"].ge(RELEVANT_GRADE)

    profile = candidate_set_profile(population)
    print(
        "\ncandidate set per search, whole ranked population "
        f"({len(population):,} listings, endpoint cap {MAX_LISTINGS:,})\n"
    )
    print(profile.to_string())

    rows = []
    for name, key in groups.GROUP_CASCADE:
        label = rung_labels(sealed, key)
        spread = geographic_concentration(sealed, scores, label)
        reach = cohort_reach(scores, label, deserving)
        rows.append(
            {
                "rung": name,
                "groups": int(label.nunique()),
                "median_set": int(label.value_counts().median()),
                "geos_in_top_10": spread["geos_in_top_k"].mean(),
                "coverage": spread["coverage"].mean(),
                "entropy": spread["entropy"].mean(),
                "cold_reach": reach["reach_rate"],
                "cold_random": reach["random_rate"],
                "cold_lift": reach["lift"],
            }
        )
    widening = pd.DataFrame(rows).set_index("rung")
    print(
        f"\nexposure on the sealed fold ({len(sealed):,} listings, "
        f"{int(deserving.sum())} deserving cold-start), k={DEFAULT_K}\n"
    )
    print(widening.round(4).to_string())

    # --- the two retrieval policies, over the same broad searches -----------------------------
    development = population[population["fold"] != split.SEALED_FOLD]
    prior = demand_prior(development)
    universe = rung_labels(sealed, groups.GROUP_CASCADE[1][1])

    arms = []
    for k_geo in (1, 2, 3, 5):
        table = simulate_arms(sealed, scores, universe, prior, k_geo=k_geo)
        arms.append(table.assign(k_geo=k_geo).reset_index(names="arm"))
    simulation = pd.concat(arms).set_index(["k_geo", "arm"]).sort_index()

    print(
        f"\nfirst screen under each retrieval policy, broad searches "
        f"({int(universe.nunique())} of them), k={DEFAULT_K}\n"
    )
    print(simulation.round(4).to_string())

    paths.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    profile.join(widening, how="outer", rsuffix="_sealed").to_csv(paths.TRAIN_DIR / "exposure.csv")
    simulation.to_csv(paths.TRAIN_DIR / "retrieval_arms.csv")
    print(
        f"\nwritten: {paths.TRAIN_DIR / 'exposure.csv'}, {paths.TRAIN_DIR / 'retrieval_arms.csv'}"
    )


# --- simulating the two retrieval policies -----------------------------------------------------


def demand_prior(
    training: pd.DataFrame,
    geo_column: str = GEO_COLUMN,
    target: str = "grade",
) -> pd.Series:
    """Historical demand per neighbourhood, for choosing where to look before ranking.

    The stand-in for a personalised destination model: with no user signal, the best available
    guess at where a guest should be shown listings is where listings have historically been in
    demand. It is a **prior over geographies**, not a feature — no listing sees it.

    **Fit this on training data only.** It is derived from the target, so a prior fitted on the
    population it will be evaluated against leaks the answer into the arm being measured.

    Args:
        training: Listings whose target may be read — never the evaluation population.
        geo_column: Geography to score.
        target: Column to average. ``grade`` by construction, since it is the ranked quantity.

    Returns:
        Float Series indexed by ``(city, geo)``, plus a ``(city, None)`` entry per city holding the
        city mean, which is the fallback for a geography the training data never saw.
    """
    require_columns(training, ("city", geo_column, target), "training listings")
    by_geo = training.groupby(["city", geo_column], observed=True)[target].mean()
    by_city = training.groupby("city", observed=True)[target].mean()
    fallback = pd.Series(
        by_city.to_numpy(),
        index=pd.MultiIndex.from_product([by_city.index, [None]], names=by_geo.index.names),
    )
    return pd.concat([by_geo, fallback]).rename("demand_prior")


def select_geographies(
    candidates: pd.DataFrame,
    prior: pd.Series,
    k_geo: int,
    geo_column: str = GEO_COLUMN,
) -> pd.Series:
    """Boolean mask keeping only listings in the ``k_geo`` highest-prior neighbourhoods.

    The narrowing step. A geography absent from the prior falls back to its city's mean rather
    than being dropped, so a neighbourhood the training data never saw is treated as average
    instead of being silently excluded from every search.
    """
    require_columns(candidates, ("city", geo_column), "candidates")
    keys = pd.MultiIndex.from_arrays([candidates["city"], candidates[geo_column]])
    scored = prior.reindex(keys)
    city_fallback = prior.reindex(
        pd.MultiIndex.from_arrays([candidates["city"], [None] * len(candidates)])
    )
    scored = pd.Series(
        np.where(scored.isna(), city_fallback.to_numpy(), scored.to_numpy()),
        index=candidates.index,
    )

    ranked_geos = (
        pd.DataFrame({"geo": candidates[geo_column].to_numpy(), "prior": scored.to_numpy()})
        .groupby("geo", observed=True, dropna=False)["prior"]
        .first()
        .nlargest(k_geo)
        .index
    )
    return candidates[geo_column].isin(ranked_geos).rename("selected")


def screen_composition(
    listings: pd.DataFrame,
    scores: pd.Series,
    group: pd.Series,
    k: int = DEFAULT_K,
    geo_column: str = GEO_COLUMN,
) -> pd.DataFrame:
    """What the first screen contains — the one thing that *is* comparable across policies.

    **This is why there is no NDCG here.** NDCG normalises by the candidate set, so it changes
    meaning the moment the set changes and cannot compare two retrieval policies. The first screen
    is *k* listings under either policy, so its composition is unnormalised and directly
    comparable: the same question a guest would ask, which is whether the ten things in front of
    them are any good.

    Returns:
        One row per group: ``shown``, ``mean_grade``, ``relevant_share`` (grade >= 3),
        ``distinct_geos``, ``cold_share``, ``deserving_cold_share``.
    """
    require_columns(listings, ("grade", "has_reviews", geo_column), "candidates")
    frame = pd.DataFrame(
        {
            "grade": listings["grade"].to_numpy(),
            "cold": ~listings["has_reviews"].to_numpy().astype(bool),
            "geo": listings[geo_column].to_numpy(),
            "score": scores.to_numpy(),
            "group": group.to_numpy(),
        }
    )
    rows = []
    for name, block in frame.groupby("group", observed=True, sort=False):
        top = block.nlargest(k, "score", keep="first")
        rows.append(
            {
                "group": name,
                "shown": int(len(top)),
                "mean_grade": float(top["grade"].mean()),
                "relevant_share": float(top["grade"].ge(RELEVANT_GRADE).mean()),
                "distinct_geos": int(top["geo"].nunique(dropna=False)),
                "cold_share": float(top["cold"].mean()),
                "deserving_cold_share": float(
                    (top["cold"] & top["grade"].ge(RELEVANT_GRADE)).mean()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def simulate_arms(
    candidates: pd.DataFrame,
    scores: pd.Series,
    universe: pd.Series,
    prior: pd.Series,
    k_geo: int,
    k: int = DEFAULT_K,
    geo_column: str = GEO_COLUMN,
) -> pd.DataFrame:
    """Both retrieval policies over the same broad searches, ranked by the same model.

    A broad search is one that names a city, a room type and a party size but no neighbourhood.
    ``universe`` is the set that search could return.

    * **Control** ranks the whole universe and shows the top *k*. No geographic narrowing.
    * **Treatment** keeps the ``k_geo`` highest-prior neighbourhoods, ranks those, shows the top
      *k*.

    The **ranker is identical in both arms**; only the candidate set differs. That is what makes
    the comparison attributable to the retrieval policy rather than to the model.

    Returns:
        Two rows, ``control`` and ``treatment``, averaging :func:`screen_composition` over
        searches, plus ``searches`` and ``median_candidates``.
    """
    frame = candidates.assign(_score=scores.to_numpy(), _universe=universe.to_numpy())

    kept = []
    for _, block in frame.groupby("_universe", observed=True, sort=False):
        kept.append(block[select_geographies(block, prior, k_geo, geo_column).to_numpy()])
    narrowed = pd.concat(kept)

    rows = {}
    for arm, subset in (("control", frame), ("treatment", narrowed)):
        composition = screen_composition(
            subset, subset["_score"], subset["_universe"], k=k, geo_column=geo_column
        )
        sizes = subset.groupby("_universe", observed=True).size()
        rows[arm] = {
            "searches": int(len(composition)),
            "median_candidates": int(sizes.median()),
            **composition.mean(numeric_only=True).drop("shown").to_dict(),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    main()
