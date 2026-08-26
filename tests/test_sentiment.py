"""Tests for the sentiment demonstration — the pure half and the guards on the billable half.

Two of these guard money rather than correctness, which is unusual for this suite and is the
point: ``text_records`` is what a run is priced against **before** it calls, and ``batches`` is
the mechanism behind the never-call-inside-a-loop rule. A bug in either produces a run that works and
quietly spends the free tier — 393 documents sent one at a time still returns the right answer.

The **minimum one record per document** rule is tested explicitly because getting it wrong is
what made the first budget estimate off by 5x: a 195-character review looks like 0.2 records and
costs 1.
"""

import pandas as pd
import pytest

from rental_ranking.cloud import sentiment as cloud
from rental_ranking.features import sentiment as feat

# --- pricing ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ([195], 1),  # a short review still costs a whole record
        ([10], 1),
        ([1_000], 1),
        ([1_001], 2),  # rounded up
        ([2_500], 3),
        ([195, 195, 195], 3),  # the case that decides this project's budget
        ([], 0),
    ],
)
def test_text_records_charges_a_minimum_of_one_per_document(
    lengths: list[int], expected: int
) -> None:
    assert feat.text_records(["x" * n for n in lengths]) == expected


def test_the_real_selection_prices_at_one_record_per_review() -> None:
    """The real selection is 393 reviews and 393 records, not 78: the corpus is short enough
    that the per-document minimum dominates."""
    corpus = ["x" * 195] * 393
    assert feat.text_records(corpus) == 393


# --- batching --------------------------------------------------------------------------------


def test_batches_respect_the_service_cap() -> None:
    assert [len(b) for b in feat.batches(["x"] * 25)] == [10, 10, 5]


def test_batching_is_lossless_and_ordered() -> None:
    documents = [str(i) for i in range(23)]
    assert [d for batch in feat.batches(documents) for d in batch] == documents


def test_a_single_document_is_one_batch_not_one_call_each() -> None:
    """The banned shape is a call per listing; one short list must stay one request."""
    assert len(feat.batches(["a", "b", "c"])) == 1


def test_an_impossible_batch_size_raises() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        feat.batches(["a"], size=0)


# --- aggregation -----------------------------------------------------------------------------


def _responses() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": ["a", "a", "b"],
            "sentiment": ["positive", "positive", "negative"],
            "positive": [0.90, 0.70, 0.20],
            "neutral": [0.05, 0.20, 0.30],
            "negative": [0.05, 0.10, 0.50],
        }
    )


def test_the_listing_score_reads_confidence_not_the_argmax_label() -> None:
    """Two listings can be 100 % 'positive' by label and differ a great deal in conviction."""
    faint = pd.DataFrame(
        {
            "listing_id": ["x", "x"],
            "sentiment": ["positive", "positive"],
            "positive": [0.40, 0.45],
            "neutral": [0.35, 0.30],
            "negative": [0.25, 0.25],
        }
    )
    emphatic = faint.assign(positive=[0.95, 0.98], negative=[0.02, 0.01])

    assert feat.aggregate_sentiment(faint).loc["x", "positive_share"] == 1.0
    assert feat.aggregate_sentiment(emphatic).loc["x", "positive_share"] == 1.0
    assert (
        feat.aggregate_sentiment(emphatic).loc["x", "sentiment_score"]
        > feat.aggregate_sentiment(faint).loc["x", "sentiment_score"] + 0.5
    )


def test_aggregate_is_one_row_per_listing_with_the_score_in_range() -> None:
    out = feat.aggregate_sentiment(_responses())
    assert out.index.tolist() == ["a", "b"]
    assert out.index.name == "listing_id"
    assert out.loc["a", "reviews_scored"] == 2
    assert out.loc["a", "sentiment_score"] == pytest.approx(0.725)
    assert out.loc["b", "sentiment_score"] == pytest.approx(-0.30)
    assert out["sentiment_score"].between(-1, 1).all()


def test_aggregate_refuses_an_incomplete_response_frame() -> None:
    with pytest.raises(KeyError):
        feat.aggregate_sentiment(_responses().drop(columns="negative"))


# --- the billable half's guards ----------------------------------------------------------------


def test_missing_credentials_name_themselves_and_say_how_to_get_them(monkeypatch) -> None:
    monkeypatch.setattr(cloud, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv(cloud.ENDPOINT_VAR, raising=False)
    monkeypatch.delenv(cloud.KEY_VAR, raising=False)

    with pytest.raises(RuntimeError) as caught:
        cloud.credentials()
    message = str(caught.value)
    assert cloud.ENDPOINT_VAR in message
    assert cloud.KEY_VAR in message
    assert "az cognitiveservices" in message
    assert "never commit" in message


def test_the_run_refuses_before_calling_when_it_would_exceed_the_cap(monkeypatch) -> None:
    """A budget checked after the call is not a budget. This must raise before any network use."""
    monkeypatch.setenv(cloud.ENDPOINT_VAR, "https://example.invalid/")
    monkeypatch.setenv(cloud.KEY_VAR, "not-a-real-key")
    documents = pd.DataFrame({"listing_id": ["a"] * 50, "text": ["x" * 2_000] * 50})

    with pytest.raises(RuntimeError, match="over the .* cap"):
        cloud.analyse(documents, max_records=10)


def test_the_default_cap_stays_well_inside_the_free_tier() -> None:
    assert cloud.DEFAULT_MAX_RECORDS < feat.FREE_TIER_RECORDS_PER_MONTH
    assert cloud.DEFAULT_MAX_RECORDS == feat.FREE_TIER_RECORDS_PER_MONTH // 5


def test_documents_are_truncated_to_the_service_limit_not_split() -> None:
    """Splitting a long review would let it vote twice in its listing's mean."""
    assert feat.MAX_CHARS_PER_DOCUMENT == 5_120
    long_review = "x" * 9_000
    assert len(long_review[: feat.MAX_CHARS_PER_DOCUMENT]) == feat.MAX_CHARS_PER_DOCUMENT
