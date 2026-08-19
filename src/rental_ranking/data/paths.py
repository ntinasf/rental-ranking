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

#: Phase 2 output: the assembled feature matrix, and the second (and last) thing the
#: project registers as an Azure data asset.
FEATURES_DIR: Path = DATA_DIR / "features"
FEATURE_TABLE_PATH: Path = FEATURES_DIR / "feature_table.parquet"

#: Phase 3 outputs. The hyperparameter search is expensive (35 configurations x 4 folds), so its
#: results table is written once and read by notebook 04 rather than recomputed there — the same
#: contract the feature table has: gitignored, and rebuilt by one documented command.
TRAIN_DIR: Path = DATA_DIR / "train"
SWEEP_RESULTS_PATH: Path = TRAIN_DIR / "sweep_results.csv"

#: Cached raw responses from the Azure AI Language demonstration. **Raw, and cached before any
#: aggregation** — BUILD_GUIDE gotcha #5: re-aggregating must never re-bill, so a response goes
#: to disk exactly as it arrived and every rerun reads it from there.
SENTIMENT_DIR: Path = DATA_DIR / "sentiment"
SENTIMENT_CACHE_PATH: Path = SENTIMENT_DIR / "language_responses.json"

#: What the endpoint is given: the booster plus its serving metadata, written by
#: ``train.export_serving_bundle`` and uploaded as the model asset. Kept apart from the MLflow
#: run artifacts because a scoring image should carry the two files it reads and nothing else.
SERVING_BUNDLE_DIR: Path = TRAIN_DIR / "serving_bundle"

#: Endpoint evidence — request bodies, responses, and the ranking read against the held-out
#: grades. **Committed**, unlike everything else here, because it is the demonstration: the
#: endpoint itself is deleted the moment the screenshots are taken (BUILD_GUIDE gotcha #6), so
#: these files are all that survives it.
ENDPOINT_DEMO_DIR: Path = PROJECT_ROOT / "docs" / "endpoint_demo"
