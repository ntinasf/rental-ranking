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


def _sealed_listings(name: str) -> tuple[pd.DataFrame, dict]:
    """The candidate set for one demo query, read from the feature table and the fold assignment."""
    if name not in DEMO_QUERIES:
        raise SystemExit(f"unknown query {name!r}; known: {sorted(DEMO_QUERIES)}")
    spec = DEMO_QUERIES[name]

    table = pd.read_parquet(paths.FEATURE_TABLE_PATH)
    fold, _ = split.assign_folds(table)
    table = table.assign(fold=fold.to_numpy())

    listings = table[table["query_group"] == spec["query_group"]]
    if listings.empty:
        raise SystemExit(f"query group {spec['query_group']} is not in the feature table")
    if not (listings["fold"] == split.SEALED_FOLD).all():
        raise SystemExit(
            f"query group {spec['query_group']} is not entirely in the sealed fold; "
            "a demonstration on training data shows nothing"
        )
    return listings, spec


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


def _display(table: pd.DataFrame) -> pd.DataFrame:
    """Round each column to the precision that column is read at."""
    shown = table.round({c: d for c, d in _DECIMALS.items() if c in table.columns})
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
        "--capture",
        action="store_true",
        help="run every query and variant, write the responses and RESULTS.md to "
        "docs/endpoint_demo/, and exit — the evidence that outlives the endpoint",
    )
    parser.add_argument("--out", type=Path, help="write the response JSON here")
    args = parser.parse_args(argv)

    features = _serving_metadata()["features"]

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
