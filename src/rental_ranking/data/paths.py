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
