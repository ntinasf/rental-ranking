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
