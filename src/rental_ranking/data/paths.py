"""Filesystem layout: every path the project reads or writes."""

import os
from pathlib import Path

_ROOT_ENV_VAR = "RENTAL_RANKING_ROOT"
_ROOT_MARKER = "pyproject.toml"


def _project_root() -> Path:
    """Resolve the project root without assuming this file's depth in the tree."""
    override = os.environ.get(_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    for candidate in Path(__file__).resolve().parents:
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT: Path = _project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

#: The assembled feature matrix, registered as an Azure data asset.
FEATURES_DIR: Path = DATA_DIR / "features"
FEATURE_TABLE_PATH: Path = FEATURES_DIR / "feature_table.parquet"

#: Training outputs. The hyperparameter search is expensive (35 configurations x 4 folds), so its
#: results table is written once and read from disk rather than recomputed: gitignored, and
#: rebuilt by one documented command.
TRAIN_DIR: Path = DATA_DIR / "train"
SWEEP_RESULTS_PATH: Path = TRAIN_DIR / "sweep_results.csv"

#: Cached responses from the Azure AI Language demonstration, written exactly as they arrived and
#: before any aggregation, so re-aggregating never re-bills.
SENTIMENT_DIR: Path = DATA_DIR / "sentiment"
SENTIMENT_CACHE_PATH: Path = SENTIMENT_DIR / "language_responses.json"

#: What the endpoint is given: the booster plus its serving metadata, written by
#: ``train.export_serving_bundle`` and uploaded as the model asset. Kept apart from the MLflow run
#: artifacts so a scoring image carries the two files it reads and nothing else.
SERVING_BUNDLE_DIR: Path = TRAIN_DIR / "serving_bundle"

#: Endpoint evidence — request bodies, responses, and the ranking read against the held-out
#: grades. **Committed**: the endpoint is torn down after the demonstration, so these files are
#: all that survives it.
ENDPOINT_DEMO_DIR: Path = PROJECT_ROOT / "docs" / "endpoint_demo"

#: Report figures — the charts and extracted notebook plots that ``docs/report.md`` embeds.
#: **Committed**: the report is a deliverable and cannot depend on a gitignored directory.
FIGURES_DIR: Path = PROJECT_ROOT / "docs" / "figures"
