"""The one sanctioned Azure AI Language run: score one query group, cache, and stop.

**This module spends money — or rather, spends the free tier — and it is the only one that
does.** Everything it produces is a demonstration artifact. No column it computes reaches the
feature matrix, and ``features/sentiment.py`` records the measurement that settled that: within
query groups sentiment adds **+0.015** over the rating, among listings at the rating ceiling it
correlates **-0.026** with the label, and Airbnb's own ``review_scores_value`` beats it 2.6x and
is free.

**BUILD_GUIDE gotcha #5 is the whole design.**

* **Never call inside a loop over listings.** Documents are batched ten to a request, which is
  the service's cap — the difference between one call and several hundred.
* **Cache the raw response before any aggregation.** The JSON goes to disk exactly as it
  arrived, so re-aggregating, re-plotting or re-running the notebook never re-bills. Every rerun
  reads the cache and the network is not touched again; ``--refresh`` is the only way past it,
  and it says so.

**Budget, measured rather than assumed — and the obvious reading of the pricing is wrong.** Azure
bills a text record per 1,000 characters, **rounded up, minimum one per document**. The minimum
is what decides the budget: this corpus has a median review of 195 characters, so nearly every
review costs a whole record rather than the 0.2 its length implies. Priced against the real
selection, 393 reviews cost exactly 393 records — so the F0 tier's 5,000/month is worth about
5,000 short reviews, not 19,000. The run is capped by ``--max-records`` regardless, defaulting to
a fifth of the monthly allowance, because a free tier that fails closed is only useful if you
notice before it does.

**Which group, and why a whole one.** The demonstration scores **every listing in one query
group**, not a sample across many. Scoring the top-N most-reviewed listings inside a group would
make "has sentiment" a proxy for review count — the label's dominant driver — and the comparison
would measure missingness rather than sentiment. A whole group keeps the only comparison a
pairwise ranker ever makes intact: listing against listing, inside one search.

Credentials come from the environment (``AZURE_LANGUAGE_ENDPOINT``, ``AZURE_LANGUAGE_KEY``), never
from a file in the repo, and ``load_dotenv()`` is called here because ``.env`` is read by the VS
Code Python extension and not by Python.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rental_ranking.data.paths import (
    FEATURE_TABLE_PATH,
    PROCESSED_DIR,
    SENTIMENT_CACHE_PATH,
    SENTIMENT_DIR,
)
from rental_ranking.features.sentiment import (
    FREE_TIER_RECORDS_PER_MONTH,
    MAX_CHARS_PER_DOCUMENT,
    MAX_DOCUMENTS_PER_REQUEST,
    aggregate_sentiment,
    batches,
    text_records,
)

#: Environment variables holding the resource's endpoint and key.
ENDPOINT_VAR = "AZURE_LANGUAGE_ENDPOINT"
KEY_VAR = "AZURE_LANGUAGE_KEY"

#: Default ceiling on one run, a fifth of the monthly free allowance. A cap that is never
#: approached is the point: the demonstration should not be the thing that exhausts the tier.
DEFAULT_MAX_RECORDS = FREE_TIER_RECORDS_PER_MONTH // 5

#: The query group scored by default. 23 listings, 100 % of them reviewed, ~1,599 reviews
#: available — a whole group, small enough to score completely inside the cap.
DEFAULT_QUERY_GROUP = 24

#: Reviews per listing, most recent first. Caps a single prolific listing from consuming the
#: budget and skewing the per-listing mean toward whoever has been trading longest.
DEFAULT_REVIEWS_PER_LISTING = 20


def credentials() -> tuple[str, str]:
    """Read the endpoint and key from the environment, or say exactly what is missing.

    Raises:
        RuntimeError: If either variable is unset.
    """
    load_dotenv()
    endpoint, key = os.environ.get(ENDPOINT_VAR), os.environ.get(KEY_VAR)
    missing = [name for name, value in ((ENDPOINT_VAR, endpoint), (KEY_VAR, key)) if not value]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not set. Retrieve them with:\n"
            "  az cognitiveservices account show --name nf-rental-language "
            "-g nf-rental-ranking --query properties.endpoint -o tsv\n"
            "  az cognitiveservices account keys list --name nf-rental-language "
            "-g nf-rental-ranking --query key1 -o tsv\n"
            "then put them in .env. They are secrets: never commit them."
        )
    return endpoint, key


def select_documents(
    query_group: int = DEFAULT_QUERY_GROUP,
    reviews_per_listing: int = DEFAULT_REVIEWS_PER_LISTING,
    features_path: Path = FEATURE_TABLE_PATH,
) -> pd.DataFrame:
    """The documents to score: every listing in one query group, its most recent reviews.

    Returns:
        One row per review — ``listing_id``, ``date`` and ``text`` truncated to
        :data:`~rental_ranking.features.sentiment.MAX_CHARS_PER_DOCUMENT`.
    """
    features = pd.read_parquet(features_path, columns=["id", "query_group", "city"])
    listings = features.loc[features["query_group"].eq(query_group), "id"]
    if listings.empty:
        raise ValueError(f"query group {query_group} holds no listings")

    reviews = pd.read_parquet(
        PROCESSED_DIR / "reviews.parquet", columns=["listing_id", "date", "comments"]
    )
    reviews = reviews[reviews["listing_id"].isin(set(listings))].dropna(subset=["comments"])
    reviews = (
        reviews.sort_values("date", ascending=False)
        .groupby("listing_id", observed=True)
        .head(reviews_per_listing)
        .reset_index(drop=True)
    )
    reviews["text"] = reviews["comments"].str.slice(0, MAX_CHARS_PER_DOCUMENT)
    return reviews[["listing_id", "date", "text"]]


def analyse(documents: pd.DataFrame, max_records: int = DEFAULT_MAX_RECORDS) -> list[dict]:
    """Send the documents to Azure AI Language, ten at a time, and return flat per-review rows.

    **The only function in this project that makes a billable call.** It refuses to start if the
    documents would exceed ``max_records``, because a check after the fact is not a budget.

    Raises:
        RuntimeError: If credentials are missing, or the batch would exceed ``max_records``.
    """
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    endpoint, key = credentials()
    texts = documents["text"].tolist()
    cost = text_records(texts)
    if cost > max_records:
        raise RuntimeError(
            f"{len(texts)} documents cost {cost} text records, over the {max_records} cap "
            f"(free tier is {FREE_TIER_RECORDS_PER_MONTH}/month). Lower --reviews-per-listing "
            "or raise --max-records deliberately"
        )
    print(f"{len(texts)} documents, {cost} text records, {len(batches(texts))} requests")

    client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    ids = documents["listing_id"].tolist()
    rows: list[dict] = []
    for number, (chunk, chunk_ids) in enumerate(
        zip(batches(texts), batches(ids), strict=True), start=1
    ):
        results = client.analyze_sentiment(documents=chunk, language="en")
        for listing_id, result in zip(chunk_ids, results, strict=True):
            if result.is_error:
                rows.append({"listing_id": listing_id, "error": result.error.message})
                continue
            rows.append(
                {
                    "listing_id": listing_id,
                    "sentiment": result.sentiment,
                    "positive": result.confidence_scores.positive,
                    "neutral": result.confidence_scores.neutral,
                    "negative": result.confidence_scores.negative,
                }
            )
        print(f"  request {number}/{len(batches(texts))} -> {len(rows)} scored", flush=True)
    return rows


def load_or_analyse(
    documents: pd.DataFrame,
    cache_path: Path = SENTIMENT_CACHE_PATH,
    refresh: bool = False,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> pd.DataFrame:
    """Return the raw per-review responses, from cache unless ``refresh`` is set.

    **The cache is checked before the network, always.** Gotcha #5's second half: aggregating
    again, re-running the notebook, or re-plotting must never re-bill.
    """
    if cache_path.is_file() and not refresh:
        print(f"cache hit -> {cache_path} (no API call; pass --refresh to overwrite)")
        return pd.DataFrame(json.loads(cache_path.read_text()))

    rows = analyse(documents, max_records=max_records)
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows, indent=1))
    print(f"cached {len(rows)} raw responses -> {cache_path}")
    return pd.DataFrame(rows)


def main() -> None:
    """Score one query group, cache the raw responses, and print the per-listing aggregate."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--query-group", type=int, default=DEFAULT_QUERY_GROUP)
    parser.add_argument("--reviews-per-listing", type=int, default=DEFAULT_REVIEWS_PER_LISTING)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the cache and call the API again. This SPENDS free-tier records",
    )
    parser.add_argument("--estimate-only", action="store_true", help="price it, call nothing")
    args = parser.parse_args()

    documents = select_documents(args.query_group, args.reviews_per_listing)
    cost = text_records(documents["text"].tolist())
    print(
        f"query group {args.query_group}: {documents['listing_id'].nunique()} listings, "
        f"{len(documents)} reviews, {cost} text records "
        f"({cost / FREE_TIER_RECORDS_PER_MONTH:.1%} of the monthly free tier), "
        f"{-(-len(documents) // MAX_DOCUMENTS_PER_REQUEST)} requests"
    )
    if args.estimate_only:
        return

    responses = load_or_analyse(documents, refresh=args.refresh, max_records=args.max_records)
    errors = responses["error"].notna().sum() if "error" in responses else 0
    if errors:
        print(f"WARNING: {errors} document(s) returned an error and are excluded")
        responses = responses[responses.get("error").isna()]

    aggregate = aggregate_sentiment(responses)
    print(f"\nper-listing sentiment, {len(aggregate)} listings")
    print(aggregate.round(4).to_string())
    print(
        "\nThis is a demonstration of the Azure AI Language workflow. It is NOT a model feature: "
        "measured within query groups sentiment adds +0.015 over the rating, correlates -0.026 "
        "with the label among listings at the rating ceiling, and is beaten 2.6x by "
        "review_scores_value, which is free."
    )


if __name__ == "__main__":
    main()
