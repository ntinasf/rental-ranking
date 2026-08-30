"""The Azure AI Language demonstration: score one query group, cache, and stop.

**This was never run.** No call was ever made to Azure AI Language, and no sentiment number in
this project comes from it — the demonstration was designed and tested, then descoped.
Everything here works against a resource in a region that offers the sentiment capability;
ours does not.

Nothing calls the API inside a loop over listings — documents are
batched ten to a request, the service's cap — and raw responses are cached before any
aggregation, so re-aggregating or re-running the notebook never re-bills. ``--refresh`` is the
only way past the cache; ``--max-records`` caps a run at a fifth of the free monthly allowance.

Credentials come from the environment (``AZURE_LANGUAGE_ENDPOINT``, ``AZURE_LANGUAGE_KEY``),
never from a file in the repo.

Run it::

    uv run python -m rental_ranking.cloud.sentiment --estimate-only
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

#: Default ceiling on one run, a fifth of the monthly free allowance — the demonstration should
#: not be the thing that exhausts the tier.
DEFAULT_MAX_RECORDS = FREE_TIER_RECORDS_PER_MONTH // 5

#: The query group scored by default: 23 listings, all reviewed, small enough to score
#: completely inside the cap.
DEFAULT_QUERY_GROUP = 24

#: Reviews per listing, most recent first. Stops a prolific listing consuming the budget and
#: skewing the per-listing mean toward whoever has been trading longest.
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

    A whole group rather than a sample across many — selecting the most-reviewed listings would
    make "has sentiment" a proxy for review count, the label's dominant driver.

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


def detect_languages(client: object, texts: list[str]) -> list[str]:
    """Detect each document's language in a separate first pass.

    Only 28 % of these documents are English (the rest is mostly Greek), and sentiment analysis
    has **no auto-detect** — omitting ``language`` silently defaults to English and would return
    confident nonsense for the other 72 %. Detection doubles the spend and buys correctness.

    Returns:
        ISO codes aligned to ``texts``, falling back to ``"en"`` for any document the service
        could not classify.
    """
    codes: list[str] = []
    for chunk in batches(texts):
        for result in client.detect_language(documents=chunk):
            codes.append("en" if result.is_error else result.primary_language.iso6391_name)
    return codes


def analyse(documents: pd.DataFrame, max_records: int = DEFAULT_MAX_RECORDS) -> list[dict]:
    """Detect each document's language, then score its sentiment. Ten documents per request.

    The only function in this project that makes a billable call. It refuses to start if the work
    would exceed ``max_records``, counting **both** passes, since both bill. Documents are grouped
    by detected language before scoring: the sentiment API takes one ``language`` per request.

    Raises:
        RuntimeError: If credentials are missing, or the work would exceed ``max_records``.
    """
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    endpoint, key = credentials()
    texts = documents["text"].tolist()
    cost = 2 * text_records(texts)  # detection pass + sentiment pass
    if cost > max_records:
        raise RuntimeError(
            f"{len(texts)} documents cost {cost} text records across the detection and sentiment "
            f"passes, over the {max_records} cap (free tier is {FREE_TIER_RECORDS_PER_MONTH}"
            "/month). Lower --reviews-per-listing or raise --max-records deliberately"
        )

    client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    print(f"detecting language for {len(texts)} documents ({len(batches(texts))} requests)")
    languages = detect_languages(client, texts)
    mix = pd.Series(languages).value_counts()
    print(f"  languages: {mix.head(6).to_dict()}")

    frame = documents.assign(language=languages)
    rows: list[dict] = []
    for language, block in frame.groupby("language", observed=True):
        chunks = batches(block["text"].tolist())
        ids = batches(block["listing_id"].tolist())
        for chunk, chunk_ids in zip(chunks, ids, strict=True):
            for listing_id, result in zip(
                chunk_ids,
                client.analyze_sentiment(documents=chunk, language=language),
                strict=True,
            ):
                if result.is_error:
                    rows.append(
                        {
                            "listing_id": listing_id,
                            "language": language,
                            "error": result.error.message,
                        }
                    )
                    continue
                rows.append(
                    {
                        "listing_id": listing_id,
                        "language": language,
                        "sentiment": result.sentiment,
                        "positive": result.confidence_scores.positive,
                        "neutral": result.confidence_scores.neutral,
                        "negative": result.confidence_scores.negative,
                    }
                )
        print(f"  {language}: {len(block)} documents scored", flush=True)
    return rows


def load_or_analyse(
    documents: pd.DataFrame,
    cache_path: Path = SENTIMENT_CACHE_PATH,
    refresh: bool = False,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> pd.DataFrame:
    """Return the raw per-review responses, from cache unless ``refresh`` is set.

    The cache is checked before the network, always: re-aggregating or re-plotting must not re-bill.
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
