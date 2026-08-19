"""Turn an endpoint response into something a reader can judge.

The endpoint returns ``{"id": ..., "score": ..., "rank": ...}`` and nothing else, which is the
right contract — a scoring service has no business shipping the ground truth back to its caller.
But a screenshot of that response proves only that an endpoint existed. It shows an ordering with
no way to tell a good one from a shuffle.

This module is the other half: it builds the requests from real held-out listings, and it joins
the response back to the **grades the model never saw** so the ordering can be read against the
truth. What a demonstration has to show is that the served model ranks *the same way the
evaluated model ranks*, and that the ordering carries signal.

Three things are deliberate:

* **The demo queries are chosen by a stated, label-blind rule** — the largest sealed-fold group
  under 30 listings in each city, ties to the lower group id (:data:`DEMO_QUERIES`). Never by
  score. Picking the query after seeing its NDCG is the ranking equivalent of reporting the best
  seed, and the trio this rule returns includes a mediocre one, which is the point.
* **Every demo query comes from the sealed fold.** Fold 0 was held out of training, tuning and
  every model-selection decision in Phase 3, so these orderings are out-of-sample.
* **A single query's NDCG is an anecdote, not an estimate.** One group carries no interval worth
  quoting. The number to cite is the sealed-fold estimate over 72 groups; the per-query number is
  there so a reader can check the ordering against the rows, not to be quoted on its own.

Layout follows ``cloud/sentiment.py``: pure functions, with the I/O and the network call confined
to :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from rental_ranking.cloud import score
from rental_ranking.cloud.score import ID_FIELD as ID_COLUMN
from rental_ranking.data import paths
from rental_ranking.evaluate.metrics import ndcg_at_k
from rental_ranking.evaluate.report import random_floor
from rental_ranking.features import groups
from rental_ranking.train import baseline, split

#: Query groups the demonstration ships, and why each is here. **The rule that chose them is
#: label-blind**: within the sealed fold, the largest group of at most 30 listings in each city,
#: ties broken by the lower group id. 30 is the cap because the response has to be legible in a
#: screenshot, and the cities are covered because the model's per-city quality differs.
DEMO_QUERIES: dict[str, dict[str, Any]] = {
    "athens": {
        "query_group": 79,
        "note": "29 entire homes in Athens — the weakest of the three, and kept for that reason",
    },
    "crete": {
        "query_group": 305,
        "note": "25 private rooms in Crete",
    },
    "thessaloniki": {
        "query_group": 24,
        "note": "23 entire homes in Thessaloniki — the city with the least training data",
    },
}

#: Columns printed beside the ranking so a reader can see what the model was looking at. Not
#: features of the demonstration — a subset chosen to be humanly readable. ``grade`` and
#: ``blocked_fraction_90`` are the **truth**, held locally and never sent to the endpoint.
READABLE_COLUMNS: tuple[str, ...] = (
    "name",
    "grade",
    "blocked_fraction_90",
    "number_of_reviews",
    "rating_shrunk",
    "reviews_per_month",
    "price",
    "host_is_superhost",
    "listing_age_days",
)

#: Truth columns, which must never appear in a request body. Enforced in :func:`build_payload`.
WITHHELD_COLUMNS: tuple[str, ...] = ("grade", "blocked_fraction_90", "query_group", "cluster_id")

#: Environment variables holding the deployed endpoint's address. Written to ``.env`` (gitignored)
#: while the endpoint is up and removed with it — the key is a live credential.
URI_VAR = "AML_ENDPOINT_URI"
KEY_VAR = "AML_ENDPOINT_KEY"

#: Path a managed online endpoint serves scoring on. The Studio endpoint page lists the **Swagger
#: URI** one line below the REST endpoint, and pasting that one is the easy mistake: the scoring
#: container answers GET on ``/swagger.json`` and nothing else, so a POST there returns 405 "The
#: method is not allowed for the requested URL", which the front door re-wraps as an HTTP **424**.
#: Nothing in that chain mentions the URL, so the guard belongs here.
SCORING_PATH = "/score"

DEFAULT_K = 10

#: The demo query the two edge-case requests are derived from. Deriving them from a real query
#: rather than inventing rows keeps the failure realistic: the unknown-level request is a genuine
#: listing with one field wrong, not a two-key toy object.
EDGE_CASE_QUERY = "thessaloniki"


# --- building requests --------------------------------------------------------------------


def _jsonable(value: object) -> object:
    """One cell as JSON. ``NaN``/``NA`` become ``null``; numpy scalars become Python scalars."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_payload(
    listings: pd.DataFrame,
    features: Sequence[str],
    id_column: str = ID_COLUMN,
) -> dict[str, list[dict]]:
    """Turn a frame of candidates into one request body.

    Args:
        listings: Rows to rank — one query's candidate set.
        features: Feature columns to send, in the served model's order.
        id_column: The identifier the response is keyed on.

    Returns:
        ``{"listings": [{"id": ..., <features>}, ...]}``, JSON-serialisable, nulls for missing.

    Raises:
        ValueError: If ``features`` names a withheld column. The truth stays on this machine; a
            request that carried ``grade`` would make the demonstration meaningless without
            failing, which is the worst kind of bug to have in a demo.
    """
    leaked = sorted(set(features) & set(WITHHELD_COLUMNS))
    if leaked:
        raise ValueError(
            f"refusing to send {leaked} to the endpoint: these are the target and its "
            "identifiers, and the demonstration only means something if the model never sees them"
        )
    columns = [id_column, *features]
    frame = listings.reindex(columns=columns)
    return {
        "listings": [
            {key: _jsonable(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]
    }


#: What a listing does not have on its first day online. Everything here is derived from review
#: history, so a host who published this morning supplies none of it. Nulling these columns turns
#: a real listing into an honest cold-start candidate rather than an invented row.
COLD_START_BLANKS: dict[str, Any] = {
    "number_of_reviews": 0,
    "number_of_reviews_ltm": 0,
    "reviews_per_month": None,
    "reviews_same_season_ly": 0,
    "has_reviews": False,
    "days_since_last_review": None,
    "rating_shrunk": None,
    "review_scores_accuracy": None,
    "review_scores_cleanliness": None,
    "review_scores_checkin": None,
    "review_scores_communication": None,
    "review_scores_location": None,
    "review_scores_value": None,
    "listing_age_days": 0.0,
}

#: A categorical level the model has never seen, used to demonstrate the one failure mode that is
#: otherwise silent — see ``lambdamart.restore_dtypes``. Chosen to be plausible: Airbnb really
#: does list houseboats, just not in this snapshot.
UNKNOWN_LEVEL = {"room_type": "Houseboat"}

#: What the counterfactual takes away from a single listing: its review history and nothing else.
#: Every other field — price, capacity, amenities, location, host portfolio — is held, so a rank
#: that moves moved because of these columns.
COUNTERFACTUAL_BLANKS: dict[str, Any] = {
    key: value
    for key, value in COLD_START_BLANKS.items()
    if key
    in {
        "number_of_reviews",
        "number_of_reviews_ltm",
        "reviews_per_month",
        "reviews_same_season_ly",
        "has_reviews",
        "days_since_last_review",
    }
}


def perturb(payload: Mapping[str, Any], listing_id: str, changes: Mapping[str, Any]) -> dict:
    """A copy of ``payload`` with one listing's fields overwritten.

    The counterfactual request. Sending the original and the perturbed body to the same endpoint
    shows whether the ordering moves for a reason — a demonstration that a fixed ranking cannot
    fake.

    Raises:
        KeyError: If ``listing_id`` is not in the payload.
    """
    listings = [dict(row) for row in payload["listings"]]
    hit = next((row for row in listings if row.get(ID_COLUMN) == listing_id), None)
    if hit is None:
        raise KeyError(f"{listing_id!r} is not in this request")
    hit.update({key: _jsonable(value) for key, value in changes.items()})
    return {**payload, "listings": listings}


# --- reading responses --------------------------------------------------------------------


def blank_history(
    payload: Mapping[str, Any], blanks: Mapping[str, Any] = COLD_START_BLANKS
) -> dict:
    """Every listing in the request with its review history removed — the cold-start query.

    This is the request that shows the endpoint's *stated* behaviour on missing features: they
    arrive as null, LightGBM routes them down a learned branch, and a ranking comes back. It is
    also the request that shows the model's worst measured weakness, and the two facts belong in
    the same screenshot: the service handles the case without error, and the ordering it produces
    for cold-start listings is **worse than random** (5.8 % of deserving new listings reach the
    top 10, against 9.6 % under a shuffle, measured 2026-08-18). Working is not the same as right.
    """
    return {
        **payload,
        "listings": [
            {**row, **{key: _jsonable(value) for key, value in blanks.items() if key in row}}
            for row in payload["listings"]
        ],
    }


def truth_frame(
    listings: pd.DataFrame,
    columns: Sequence[str] = READABLE_COLUMNS,
    id_column: str = ID_COLUMN,
) -> pd.DataFrame:
    """The held-back truth for one candidate set, indexed by id."""
    return listings.set_index(id_column).reindex(columns=list(columns))


def rank_of(response: Mapping[str, Any], listing_id: str) -> int:
    """The rank the endpoint gave one listing, 1-based.

    Raises:
        KeyError: If the response does not contain that listing.
    """
    for row in response["ranked"]:
        if row[ID_COLUMN] == listing_id:
            return int(row["rank"])
    raise KeyError(f"{listing_id!r} is not in this response")


def explain(
    response: Mapping[str, Any],
    truth: pd.DataFrame,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """The endpoint's ordering with the truth attached, best-ranked first.

    Args:
        response: The parsed endpoint response.
        truth: Output of :func:`truth_frame`, indexed by the same ids.
        k: Cut-off — rows at or above it are flagged ``in_top_k``, since NDCG@k only sees those.

    Returns:
        A frame indexed by rank: id, model score, then the readable truth columns.

    Raises:
        ValueError: If the response carries an error, or ids the truth frame does not have.
    """
    if "error" in response:
        raise ValueError(f"endpoint returned an error, not a ranking: {response['error']}")

    ranked = pd.DataFrame(response["ranked"]).set_index("rank").sort_index()
    unknown = set(ranked[ID_COLUMN]) - set(truth.index)
    if unknown:
        raise ValueError(
            f"{len(unknown)} ranked id(s) absent from the truth frame, e.g. {sorted(unknown)[:3]}"
        )

    joined = ranked.join(truth, on=ID_COLUMN)
    joined.insert(len(joined.columns), "in_top_k", joined.index <= k)
    return joined


def query_quality(
    response: Mapping[str, Any],
    truth: pd.DataFrame,
    k: int = DEFAULT_K,
    seed: int = 0,
) -> pd.Series:
    """NDCG@k of the endpoint's ordering against the two frozen baselines and the random floor.

    Everything is computed on the **same candidate set**, so the comparison is like-for-like: the
    baselines rank exactly the listings the endpoint was sent, which is the only way a single
    query's numbers can be read against each other at all.

    Returns:
        A Series: ``endpoint``, ``baseline_reviews``, ``baseline_price_rating``, ``random``,
        ``n_listings``, ``n_relevant`` (grade 3 or 4). NaN for the metrics if every grade in the
        group is equal — a degenerate group has no ranking to get right.
    """
    ordering = pd.DataFrame(response["ranked"]).set_index(ID_COLUMN)["score"]
    frame = truth.copy()
    frame["endpoint"] = ordering.reindex(frame.index)

    grades = frame["grade"]
    groups = pd.Series("one", index=frame.index)

    scores = {
        "endpoint": frame["endpoint"],
        "baseline_reviews": baseline.rank_by_reviews(frame),
        "baseline_price_rating": baseline.rank_by_price_and_rating(frame, groups),
    }
    out = {
        name: float(ndcg_at_k(grades, groups, series, k=k).iloc[0])
        for name, series in scores.items()
    }
    out["random"] = float(random_floor(grades, groups, k=k, draws=200, seed=seed).iloc[0])
    out["n_listings"] = float(len(frame))
    out["n_relevant"] = float((grades >= 3).sum())
    return pd.Series(out, name=f"ndcg@{k}")


# --- the network half -----------------------------------------------------------------------


def endpoint_address() -> tuple[str, str]:
    """The deployed endpoint's URI and key, or a message saying how to get them.

    Raises:
        RuntimeError: If either variable is unset.
    """
    load_dotenv()
    uri, key = os.environ.get(URI_VAR), os.environ.get(KEY_VAR)
    if not uri or not key:
        raise RuntimeError(
            f"set {URI_VAR} and {KEY_VAR} in .env (gitignored — never commit the key):\n"
            "  az ml online-endpoint show    --name rental-ranker --query scoring_uri -o tsv\n"
            "  az ml online-endpoint get-credentials --name rental-ranker "
            "--query primaryKey -o tsv\n"
            "Both die with the endpoint; there is nothing to rotate after teardown."
        )

    uri = uri.strip().rstrip("/")
    if not uri.endswith(SCORING_PATH):
        corrected = uri.rsplit("/", 1)[0] + SCORING_PATH if "/" in uri[8:] else uri + SCORING_PATH
        raise RuntimeError(
            f"{URI_VAR} does not end in {SCORING_PATH!r}:\n  {uri}\n"
            f"Try:\n  {corrected}\n"
            "The Studio endpoint page lists the Swagger URI directly below the REST endpoint, and "
            "only the REST one scores. Posting to the Swagger URI returns HTTP 424 wrapping a 405 "
            "'method is not allowed', which names neither the URL nor the mistake.\n"
            "  az ml online-endpoint show --name rental-ranker --query scoring_uri -o tsv"
        )
    return uri, key


def score_locally(payload: Mapping[str, Any]) -> dict:
    """Run the request through the scoring script in this process, with no endpoint involved.

    Same ``init``/``run`` the container calls, same booster, same metadata — so this is the
    reference the cloud response is diffed against. Two uses: the whole rendering path can be
    checked before an instance is provisioned and starts billing, and after the demonstration a
    reader can reproduce the screenshots without an Azure subscription.
    """
    os.environ.setdefault("AZUREML_MODEL_DIR", str(paths.SERVING_BUNDLE_DIR))
    if score._model is None:
        score.init()
    return score.run(dict(payload))


def invoke(uri: str, key: str, payload: Mapping[str, Any], timeout: int = 60) -> dict:
    """POST one request body to the endpoint and return the parsed response.

    ``urllib`` rather than ``requests`` on purpose: ``requests`` is only present here as a
    transitive dependency of the Azure SDK, and a demonstration script should not be the thing
    that pins it.
    """
    request = urllib.request.Request(  # noqa: S310 — https URI from the workspace, not user input
        uri,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:  # noqa: S310
            return json.loads(answer.read())
    except urllib.error.HTTPError as error:
        return {"error": f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}"}


# --- the script ------------------------------------------------------------------------------


#: Columns joined back from the processed layer purely so a human can read the ranking: the
#: listing's public title and its neighbourhood. **Neither is a feature** — they are not in the
#: served model's column list, so :func:`build_payload` cannot send them. ``name`` is a kept
#: column in the data contract (``host_name`` is the one that is stripped); the id stays hashed.
DESCRIPTION_COLUMNS: tuple[str, ...] = ("name", "neighbourhood_cleansed")


#: Set to a directory to run from a **pre-built bundle** instead of the project's data layers —
#: the container path. The bundle holds the sealed fold and a precomputed coverage report, so an
#: image needs neither the processed layer nor a 25-second re-run of the fold assignment at every
#: start. Built by ``python -m rental_ranking.cloud.demo --bundle <dir>``.
BUNDLE_VAR = "RENTAL_RANKING_DEMO_BUNDLE"

#: What a bundle contains.
BUNDLE_TABLE = "demo_table.parquet"
BUNDLE_COVERAGE = "coverage.json"


def bundle_dir() -> Path | None:
    """The demo bundle directory, if this process was pointed at one."""
    value = os.environ.get(BUNDLE_VAR)
    return Path(value) if value else None


@lru_cache(maxsize=1)
def ranked_table() -> pd.DataFrame:
    """The whole ranked population with its fold, its search key and its readable columns.

    Cached: resolving the folds runs connected components over 44,684 rows, and every query in a
    session needs the same answer.
    """
    table = pd.read_parquet(paths.FEATURE_TABLE_PATH)
    fold, _ = split.assign_folds(table)
    table = table.assign(fold=fold.to_numpy())

    described = pd.read_parquet(
        paths.PROCESSED_DIR / "listings.parquet",
        columns=[ID_COLUMN, *DESCRIPTION_COLUMNS],
    ).drop_duplicates(ID_COLUMN)
    table = table.merge(described, on=ID_COLUMN, how="left", validate="one_to_one")

    # capacity_tier is derived rather than read, for the same reason groups.py derives it: the
    # search form must offer the tiers the groups were actually built from.
    return table.assign(capacity_tier=groups.capacity_tier(table).to_numpy())


@lru_cache(maxsize=1)
def sealed_table() -> pd.DataFrame:
    """Just the sealed fold — the only listings whose grades the served model never saw.

    Read straight from the bundle when one is configured. The bundle is the sealed fold *already
    resolved*, which is the whole point: the fold assignment is connected components over 44,684
    rows and a container should not repeat it on every start to reach an answer that cannot
    change.
    """
    bundle = bundle_dir()
    if bundle is not None:
        return pd.read_parquet(bundle / BUNDLE_TABLE)
    table = ranked_table()
    return table[table["fold"] == split.SEALED_FOLD]


def group_listings(query_group: int) -> pd.DataFrame:
    """One query group's candidate set, sealed fold only.

    **Reads the sealed table, not the ranked one.** Everything that reports a number comes through
    here, so this is the single point where a trained-on group is refused — and routing it through
    ``sealed_table`` is also what lets it work from a bundle, where the rest of the population is
    not present at all. Found by the container: the bundle path was wired into ``sealed_table``
    and ``coverage`` while this function still reached past both for the feature table.

    Raises:
        KeyError: If the group is not in the sealed fold.
    """
    sealed = sealed_table()
    listings = sealed[sealed["query_group"] == query_group]
    if not listings.empty:
        return listings

    detail = "it either does not exist or the served model trained on it"
    if bundle_dir() is None:
        table = ranked_table()
        exists = bool((table["query_group"] == query_group).any())
        detail = "the served model trained on it" if exists else "it does not exist"
    raise KeyError(
        f"query group {query_group} is not in the sealed fold ({detail}). Only fold "
        f"{split.SEALED_FOLD} is out-of-sample"
    )


def _sealed_listings(name: str) -> tuple[pd.DataFrame, dict]:
    """The candidate set for one named demo query."""
    if name not in DEMO_QUERIES:
        raise SystemExit(f"unknown query {name!r}; known: {sorted(DEMO_QUERIES)}")
    spec = DEMO_QUERIES[name]
    try:
        return group_listings(spec["query_group"]), spec
    except KeyError as error:
        raise SystemExit(str(error)) from error


#: What a guest picks, and what the query group is keyed on — the same four things. The console's
#: search form is not a skin over the demo: ``features/groups.py`` builds the group from
#: ``city x neighbourhood_cleansed x room_type x capacity_tier`` precisely because that is "what a
#: guest would have typed", so choosing those four *is* choosing the candidate set.
SEARCH_KEY: tuple[str, ...] = ("city", "neighbourhood_cleansed", "room_type", "capacity_tier")


def search_index() -> pd.DataFrame:
    """Every search that can be answered honestly, and the query group it lands in.

    One row per ``(city, neighbourhood, room type, capacity tier)`` **in the sealed fold**,
    carrying the group those listings belong to and how wide that group turned out to be.

    **Sealed only, and the gap is stated rather than filled.** A demonstration that offered the
    whole population would have to show results for groups the model trained on, where the
    ordering is a memory rather than a prediction. Those are not worth looking at, so the picker
    does not offer them; :func:`coverage` reports what that costs and the console prints it.

    **The width columns are observed, not claimed.** ``neighbourhoods`` and ``tiers`` count what
    the resolved group actually contains. A group with 12 neighbourhoods in it is one whose
    original key was too thin to reach the minimum of 5 and was pooled at a coarser rung
    (``groups.GROUP_CASCADE``) — so a guest who picks that neighbourhood is really competing
    city-wide, and the console says so rather than implying the neighbourhood narrowed anything.
    """
    sealed = sealed_table()
    width = sealed.groupby("query_group", observed=True).agg(
        group_size=(ID_COLUMN, "size"),
        neighbourhoods=("neighbourhood_cleansed", "nunique"),
        tiers=("capacity_tier", "nunique"),
    )
    index = (
        sealed.groupby([*SEARCH_KEY, "query_group"], observed=True)
        .agg(matching=(ID_COLUMN, "size"))
        .reset_index()
        .merge(width, on="query_group", how="left")
    )
    return index.sort_values([*SEARCH_KEY]).reset_index(drop=True)


def tier_guest_choices() -> dict[str, list[int]]:
    """Party sizes to offer per capacity tier, derived from the bounds rather than restated.

    A hand-written map would drift the first time ``CAPACITY_TIER_BOUNDS`` changed, and it would
    drift silently: the console would offer a guest count that lands in a different tier than the
    label above it says.
    """
    bounds = groups.CAPACITY_TIER_BOUNDS
    labels = groups.CAPACITY_TIER_LABELS
    choices: dict[str, list[int]] = {}
    for position, label in enumerate(labels):
        low, high = int(bounds[position]) + 1, int(bounds[position + 1])
        # The top tier is open (bound 100 is a sentinel, not a party size anybody books), so it
        # gets a spread rather than every value up to the sentinel.
        span = (
            [low, low + 2, low + 4, low + 8]
            if position == len(labels) - 1
            else range(low, high + 1)
        )
        choices[label] = list(span)
    return choices


def group_key(listings: pd.DataFrame) -> dict:
    """What one query group actually contains, in the terms a guest searched in.

    Observed rather than looked up: a group formed at a fallback rung spans several
    neighbourhoods, and the only honest way to say which is to count them.
    """
    neighbourhoods = listings["neighbourhood_cleansed"].unique()
    tiers = [str(tier) for tier in listings["capacity_tier"].unique()]
    return {
        "city": str(listings["city"].iloc[0]),
        "room_type": str(listings["room_type"].iloc[0]),
        "neighbourhood": str(neighbourhoods[0]) if len(neighbourhoods) == 1 else None,
        "neighbourhoods": int(len(neighbourhoods)),
        "capacity_tier": tiers[0] if len(tiers) == 1 else None,
        "tiers": int(len(tiers)),
    }


def coverage() -> dict:
    """What the sealed-only picker cannot offer, and why.

    The search is restricted to the held-out fold so every result carries an honest metric. The
    cost is real and worth printing rather than hiding: the grouped split moves whole connected
    components, a large neighbourhood *is* a large component, and so the largest neighbourhood in
    each city tends to land in training entirely. Central Thessaloniki — 89 % of that city — is
    the clearest case.

    Returns:
        Counts and the worst example per city, ready to render.
    """
    bundle = bundle_dir()
    if bundle is not None:
        return json.loads((bundle / BUNDLE_COVERAGE).read_text())

    table = ranked_table()
    per = table.groupby(["city", "neighbourhood_cleansed"], observed=True).agg(
        listings=(ID_COLUMN, "size"),
        sealed=("fold", lambda f: int((f == split.SEALED_FOLD).sum())),
    )
    hidden = per[per["sealed"] == 0]
    biggest = (
        hidden.reset_index()
        .sort_values("listings", ascending=False)
        .drop_duplicates("city")
        .set_index("city")["neighbourhood_cleansed"]
    )
    return {
        "neighbourhoods": int(len(per)),
        "searchable": int(len(per) - len(hidden)),
        "hidden": int(len(hidden)),
        "hidden_listings": int(hidden["listings"].sum()),
        "hidden_share": float(hidden["listings"].sum() / len(table)),
        "biggest_hidden": {
            city: {
                "neighbourhood": name,
                "listings": int(hidden.loc[(city, name), "listings"]),
            }
            for city, name in biggest.items()
        },
    }


def guests_to_tier(guests: int) -> str:
    """The capacity tier a party size falls in, using the bounds the groups were built from.

    Raises:
        ValueError: If ``guests`` is outside the tier bounds.
    """
    tier = pd.cut(
        pd.Series([guests]),
        bins=groups.CAPACITY_TIER_BOUNDS,
        labels=groups.CAPACITY_TIER_LABELS,
    ).iloc[0]
    if pd.isna(tier):
        raise ValueError(
            f"{guests} guests is outside the capacity tiers "
            f"{groups.CAPACITY_TIER_LABELS} (bounds {groups.CAPACITY_TIER_BOUNDS})"
        )
    return str(tier)


def resolve_search(city: str, neighbourhood: str, room_type: str, guests: int) -> dict:
    """Turn one search into the query group that answers it.

    Returns:
        ``{"query_group", "matching", "group_size", "neighbourhoods", "tiers", "pooled"}``.
        ``pooled`` is True when the resolved group spans more than one neighbourhood, meaning the
        chosen neighbourhood did not have five sealed listings of its own and the competition is
        wider than the search implies.

    Raises:
        KeyError: If no sealed listing matches.
    """
    index = search_index()
    tier = guests_to_tier(guests)
    hit = index[
        (index["city"] == city)
        & (index["neighbourhood_cleansed"] == neighbourhood)
        & (index["room_type"] == room_type)
        & (index["capacity_tier"] == tier)
    ]
    if hit.empty:
        raise KeyError(
            f"no held-out listing matches {city} / {neighbourhood} / {room_type} / {guests} "
            f"guests (tier {tier})"
        )
    row = hit.iloc[0]
    return {
        "query_group": int(row["query_group"]),
        "matching": int(row["matching"]),
        "group_size": int(row["group_size"]),
        "neighbourhoods": int(row["neighbourhoods"]),
        "tiers": int(row["tiers"]),
        "capacity_tier": tier,
        "pooled": bool(row["neighbourhoods"] > 1),
    }


def _serving_metadata() -> dict:
    return json.loads((paths.SERVING_BUNDLE_DIR / "serving_metadata.json").read_text())


#: Decimal places per printed column. A single float format renders a listing age of 2,820 days
#: as "2820.000", which buries the columns that actually need three decimals.
_DECIMALS: dict[str, int] = {
    "score": 4,
    "blocked_fraction_90": 3,
    "rating_shrunk": 3,
    "reviews_per_month": 2,
    "price": 2,
    "listing_age_days": 0,
}


#: Width the listing title is cut to in the terminal table. The full title is in the JSON and in
#: the console; a 100-character name would push every numeric column off the screen.
_TITLE_WIDTH = 26


def _display(table: pd.DataFrame) -> pd.DataFrame:
    """Round each column to the precision that column is read at, and cut the title to width."""
    shown = table.round({c: d for c, d in _DECIMALS.items() if c in table.columns})
    if "name" in shown.columns:
        shown["name"] = shown["name"].astype("string").str.slice(0, _TITLE_WIDTH)
    for column, decimals in _DECIMALS.items():
        if decimals == 0 and column in shown.columns:
            shown[column] = shown[column].astype("Int64")
    return shown


def _render(name: str, spec: dict, table: pd.DataFrame, quality: pd.Series, k: int) -> str:
    variant = spec.get("variant", "full")
    heading = f"query '{name}' — group {spec['query_group']}: {spec['note']}"
    if variant != "full":
        heading += f"\nvariant: {variant}"
    lines = [
        heading,
        "",
        _display(table).to_string(),
        "",
        f"NDCG@{k} on this one query, all four rankers on the same {int(quality['n_listings'])} "
        f"listings ({int(quality['n_relevant'])} of grade 3-4):",
        f"  endpoint                {quality['endpoint']:.4f}",
        f"  baseline: reviews       {quality['baseline_reviews']:.4f}",
        f"  baseline: price+rating  {quality['baseline_price_rating']:.4f}",
        f"  random floor            {quality['random']:.4f}",
        "",
        "One query is an anecdote. The estimate is the sealed fold: 0.7530 [0.7148, 0.7903] over",
        "72 groups, against 0.6429 for price+rating and a 0.5519 floor.",
    ]
    if variant == "cold-start":
        lines += [
            "",
            "The columns above are what these listings REALLY are. The request carried none of",
            "it — every review field was sent as null. And note what this query is not: it blanks",
            "the history of the whole candidate set, so nobody is at a disadvantage. The measured",
            "cold-start failure is the mixed case, a new listing competing against established",
            "ones, where the model surfaces the deserving new listing 5.8 % of the time against a",
            "shuffle's 9.6 %.",
        ]
    return "\n".join(lines)


def _capture(features: Sequence[str], send, source: str) -> str:
    """Run every demo query and variant and return the whole transcript as Markdown.

    This is what survives the endpoint. Gotcha #6 says the deployment is deleted the same
    session, so the committed evidence has to be written while it is still up — and it has to
    carry the truth beside the ordering, or it proves only that a service answered.
    """
    blocks = [
        "# Endpoint demonstration",
        "",
        f"Generated by `python -m rental_ranking.cloud.demo --capture` against **{source}**.",
        "Every query is a **sealed-fold** group: fold 0 was held out of training, tuning and every",
        "model-selection decision in Phase 3.",
        "",
        "The queries were chosen by a label-blind rule — largest sealed group of at most 30",
        "listings in each city, ties to the lower group id — not by score. One of the three is",
        "poor, and it is kept.",
        "",
    ]
    for name in DEMO_QUERIES:
        listings, spec = _sealed_listings(name)
        truth = truth_frame(listings)
        payload = build_payload(listings, features)

        response = send(payload)
        if "error" in response:
            raise SystemExit(f"{name}: {response['error']}")
        rendered = _render(
            name, spec, explain(response, truth), query_quality(response, truth), DEFAULT_K
        )

        top = response["ranked"][0][ID_COLUMN]
        after = send(perturb(payload, top, COUNTERFACTUAL_BLANKS))
        if "error" in after:
            raise SystemExit(f"{name} counterfactual: {after['error']}")
        rendered += (
            f"\n\ncounterfactual — {top} with its review history stripped, everything else held:\n"
            f"  rank {rank_of(response, top)} -> {rank_of(after, top)} of {len(payload['listings'])}"
            f"   (true grade {int(truth.loc[top, 'grade'])})"
        )

        blocks += [f"## {name}", "", "```", rendered, "```", ""]
        (paths.ENDPOINT_DEMO_DIR / f"response_{name}.json").write_text(
            json.dumps(response, indent=1)
        )

    listings, spec = _sealed_listings(EDGE_CASE_QUERY)
    truth = truth_frame(listings)
    payload = build_payload(listings, features)

    cold = send(blank_history(payload))
    if "error" in cold:
        raise SystemExit(f"cold-start: {cold['error']}")
    blocks += [
        "## cold start — every review field sent as null",
        "",
        "```",
        _render(
            EDGE_CASE_QUERY,
            {**spec, "variant": "cold-start"},
            explain(cold, truth),
            query_quality(cold, truth),
            DEFAULT_K,
        ),
        "```",
        "",
    ]
    (paths.ENDPOINT_DEMO_DIR / "response_cold_start.json").write_text(json.dumps(cold, indent=1))

    bad = send(perturb(payload, payload["listings"][0][ID_COLUMN], UNKNOWN_LEVEL))
    blocks += [
        "## unknown categorical level — must be refused, not scored",
        "",
        "```",
        json.dumps(bad, indent=1),
        "```",
        "",
        "Scoring it would have returned a confident number **0.1083** away from the truth with",
        "nothing in the response to say so. This is the only silent failure mode measured on the",
        "serving path, which is why `restore_dtypes` exists.",
        "",
    ]
    (paths.ENDPOINT_DEMO_DIR / "response_unknown_level.json").write_text(json.dumps(bad, indent=1))
    return "\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> None:
    """Build demo requests, send them, and print the response against the truth."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--query", default="thessaloniki", help=f"one of {sorted(DEMO_QUERIES)}")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--write-requests",
        action="store_true",
        help="write every demo request body to docs/endpoint_demo/ and exit without calling",
    )
    parser.add_argument(
        "--response",
        type=Path,
        help="render a response already saved by `az ml online-endpoint invoke` instead of calling",
    )
    parser.add_argument(
        "--variant",
        choices=("full", "cold-start", "unknown-level"),
        default="full",
        help="full: every feature. cold-start: review history blanked. unknown-level: one bad "
        "categorical value, which must be rejected rather than scored",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="score in this process via the same scoring script, without calling the endpoint",
    )
    parser.add_argument(
        "--counterfactual",
        action="store_true",
        help="also send the top listing with its review history stripped, and report the move",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="write the container bundle here (the sealed fold plus a coverage report) and exit",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="run every query and variant, write the responses and RESULTS.md to "
        "docs/endpoint_demo/, and exit — the evidence that outlives the endpoint",
    )
    parser.add_argument("--out", type=Path, help="write the response JSON here")
    args = parser.parse_args(argv)

    features = _serving_metadata()["features"]

    if args.bundle is not None:
        args.bundle.mkdir(parents=True, exist_ok=True)
        table = sealed_table()
        table.to_parquet(args.bundle / BUNDLE_TABLE, index=False)
        (args.bundle / BUNDLE_COVERAGE).write_text(json.dumps(coverage(), indent=1))
        print(
            f"{args.bundle / BUNDLE_TABLE}  {len(table)} listings, "
            f"{table['query_group'].nunique()} groups\n"
            f"{args.bundle / BUNDLE_COVERAGE}"
        )
        return

    if args.write_requests:
        destination = paths.ENDPOINT_DEMO_DIR
        destination.mkdir(parents=True, exist_ok=True)
        for name in DEMO_QUERIES:
            listings, spec = _sealed_listings(name)
            body = build_payload(listings, features)
            written = {f"request_{name}": body}

            if name == EDGE_CASE_QUERY:
                first = body["listings"][0][ID_COLUMN]
                written["request_unknown_level"] = perturb(body, first, UNKNOWN_LEVEL)
                written["request_cold_start"] = blank_history(body)

            for stem, content in written.items():
                path = destination / f"{stem}.json"
                path.write_text(json.dumps(content, indent=1))
                print(f"{path}  {len(content['listings'])} listings, {len(features)} features")
        return

    if args.capture:
        if args.local:
            sender, source = score_locally, "the scoring script in-process, NOT the endpoint"
        else:
            uri, key = endpoint_address()
            source = "the live managed online endpoint, immediately before teardown"

            def sender(body: Mapping[str, Any]) -> dict:
                return invoke(uri, key, body)

        paths.ENDPOINT_DEMO_DIR.mkdir(parents=True, exist_ok=True)
        transcript = _capture(features, sender, source)
        target = paths.ENDPOINT_DEMO_DIR / "RESULTS.md"
        target.write_text(transcript + "\n")
        print(transcript)
        print(f"\nwritten: {target}")
        return

    listings, spec = _sealed_listings(args.query)
    payload = build_payload(listings, features)
    truth = truth_frame(listings)

    if args.variant == "cold-start":
        payload = blank_history(payload)
    elif args.variant == "unknown-level":
        payload = perturb(payload, payload["listings"][0][ID_COLUMN], UNKNOWN_LEVEL)

    if args.response is not None:
        response = json.loads(args.response.read_text())
    elif args.local:
        response = score_locally(payload)
    else:
        uri, key = endpoint_address()
        response = invoke(uri, key, payload)

    if "error" in response:
        if args.variant == "unknown-level":
            print(
                "the endpoint refused the request, which is the correct answer:\n\n"
                f"  {response['error']}\n\n"
                "Scoring it instead would have returned a confident number 0.1083 away from the "
                "truth,\nwith nothing in the response to say so."
            )
            return
        raise SystemExit(f"endpoint returned an error: {response['error']}")
    if args.out is not None:
        args.out.write_text(json.dumps(response, indent=1))

    ranked = explain(response, truth, k=args.k)
    quality = query_quality(response, truth, k=args.k)
    print(_render(args.query, {**spec, "variant": args.variant}, ranked, quality, args.k))

    if args.counterfactual:
        top = response["ranked"][0][ID_COLUMN]
        counterfactual = perturb(payload, top, COUNTERFACTUAL_BLANKS)
        if args.local or args.response is not None:
            after = score_locally(counterfactual)
        else:
            uri, key = endpoint_address()
            after = invoke(uri, key, counterfactual)
        if "error" in after:
            raise SystemExit(f"counterfactual failed: {after['error']}")
        print(
            f"\ncounterfactual — {top} with its review history stripped, everything else held:\n"
            f"  rank {rank_of(response, top)} -> {rank_of(after, top)} of {len(payload['listings'])}"
            f"   (true grade {int(truth.loc[top, 'grade'])})"
        )


if __name__ == "__main__":
    main()
