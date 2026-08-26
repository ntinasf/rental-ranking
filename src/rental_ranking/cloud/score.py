"""Scoring script for the managed online endpoint: a candidate set in, a ranked list out.

A ranker's score is uncalibrated — it is not a probability and no threshold on it means anything —
so the contract takes a **candidate set**, the listings competing in one search, and returns them
**ordered**, with the scores attached but the ordering already applied.

**Input** (one query)::

    {"listings": [{"id": "abc123", "accommodates": 4, "price": 85.0, ...}, ...]}

**Output**::

    {"ranked": [{"id": "abc123", "score": 1.83, "rank": 1}, ...],
     "n_listings": 23, "n_features_supplied": 61}

Missing features are allowed and become NaN, which LightGBM routes down a learned branch. Unknown
categorical levels are rejected: coercing them to NaN would score them as "missing" rather than
fail, which is a wrong answer wearing the costume of a valid one.

A custom script rather than a no-code MLflow deployment, because no-code returns raw predictions
in input order, gives no place to validate a request, and cannot express the five ``category``
columns (see ``lambdamart.serving_metadata``).

Azure ML calls :func:`init` once per worker and :func:`run` per request.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

# The package is NOT pip-installed in the scoring image. Azure ML copies `code_configuration.code`
# into /var/azureml-app/<dir> and puts only the *script's own* directory on sys.path, so
# `import rental_ranking...` raises ModuleNotFoundError at request time — after init() has
# succeeded and the container reports healthy. Caught by the local Docker smoke test, which is
# why one runs before the deployment that costs money.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from rental_ranking.train.lambdamart import restore_dtypes  # noqa: E402 — needs the path above

#: Ceiling on one request, set above the largest query group (2,088 listings). It bounds the work
#: a single call can demand, not real searches.
MAX_LISTINGS = 5_000

#: Field every listing must carry, so the response can be joined back to the caller's rows. An
#: identifier, never a feature.
ID_FIELD = "id"

_logger = logging.getLogger(__name__)
_model: lgb.Booster | None = None
_metadata: dict = {}


def init() -> None:
    """Load the booster and its serving metadata from the registered model directory."""
    global _model, _metadata

    root = Path(os.environ.get("AZUREML_MODEL_DIR", "."))
    # The registered asset may be mounted at its own directory name; find the booster rather
    # than assuming a layout that differs between local and deployed runs.
    booster = next(root.rglob("model.lgb"), None) or next(root.rglob("booster.txt"), None)
    if booster is None:
        raise FileNotFoundError(f"no model.lgb or booster.txt under {root}")
    metadata = next(root.rglob("serving_metadata.json"), None)
    if metadata is None:
        raise FileNotFoundError(f"no serving_metadata.json under {root}")

    _model = lgb.Booster(model_file=str(booster))
    _metadata = json.loads(metadata.read_text())
    _logger.info(
        "loaded booster (%d trees) and metadata for %d features, %d categorical",
        _model.num_trees(),
        len(_metadata["features"]),
        len(_metadata["categories"]),
    )


def _validate(payload: object) -> list[dict]:
    """Check the request and return its listings.

    Raises:
        ValueError: With a message naming what was wrong and what was expected.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"body must be a JSON object, got {type(payload).__name__}")
    if "listings" not in payload:
        raise ValueError("body must carry a 'listings' key holding the candidate set")

    listings = payload["listings"]
    if not isinstance(listings, list):
        raise ValueError(f"'listings' must be a list, got {type(listings).__name__}")
    if not listings:
        raise ValueError("'listings' is empty; a ranking needs at least one candidate")
    if len(listings) > MAX_LISTINGS:
        raise ValueError(f"{len(listings)} listings exceeds the {MAX_LISTINGS} per-request cap")
    if not all(isinstance(row, dict) for row in listings):
        raise ValueError("every entry in 'listings' must be a JSON object")

    ids = [row.get(ID_FIELD) for row in listings]
    if any(value is None for value in ids):
        missing = sum(value is None for value in ids)
        raise ValueError(f"{missing} listing(s) carry no {ID_FIELD!r}; the response is keyed on it")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{ID_FIELD!r} values must be unique within one request")
    return listings


def run(raw_data: str | bytes | dict) -> dict:
    """Rank one candidate set. Returns ``{"error": ...}`` rather than raising, per the AML contract."""
    if _model is None:
        return {"error": "model not loaded; init() did not run"}

    try:
        payload = raw_data if isinstance(raw_data, dict) else json.loads(raw_data)
    except json.JSONDecodeError as error:
        return {"error": f"body is not valid JSON: {error}"}

    try:
        listings = _validate(payload)
    except ValueError as error:
        return {"error": str(error)}

    frame = pd.DataFrame(listings)
    supplied = [c for c in _metadata["features"] if c in frame.columns]

    try:
        matrix = restore_dtypes(frame, _metadata)
    except ValueError as error:
        return {"error": str(error)}

    scores = _model.predict(matrix)
    ranked = (
        pd.DataFrame({ID_FIELD: frame[ID_FIELD], "score": scores})
        .sort_values("score", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.index + 1

    return {
        "ranked": ranked.to_dict(orient="records"),
        "n_listings": len(ranked),
        "n_features_supplied": len(supplied),
        "n_features_missing": len(_metadata["features"]) - len(supplied),
    }
