"""Download Inside Airbnb snapshot files per city into data/raw/<city>/<snapshot_date>/.

Each snapshot folder gets a manifest.json recording url, size, and SHA-256 per file.
The hashes are integrity fingerprints, not a security measure: they pin down exactly
which bytes every analysis ran against (snapshots rotate off the public site, so the
local copy becomes the reference), they expose corrupted or partial downloads, and the
listings hash doubles as the dataset-version tag logged with every MLflow training run.
Deliberately unsalted — anyone re-downloading the same file must get the same hash.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from rental_ranking.data.paths import RAW_DIR

THESSALONIKI_URLS = {
    "listings": "https://data.insideairbnb.com/greece/central-macedonia/thessaloniki/2026-06-29/data/listings.csv.gz",
    "calendar": "https://data.insideairbnb.com/greece/central-macedonia/thessaloniki/2026-06-29/data/calendar.csv.gz",
    "reviews": "https://data.insideairbnb.com/greece/central-macedonia/thessaloniki/2026-06-29/data/reviews.csv.gz",
    "neighbourhoods": "https://data.insideairbnb.com/greece/central-macedonia/thessaloniki/2026-06-29/visualisations/neighbourhoods.csv",
}

ATHENS_URLS = {
    "listings": "https://data.insideairbnb.com/greece/attica/athens/2026-06-28/data/listings.csv.gz",
    "calendar": "https://data.insideairbnb.com/greece/attica/athens/2026-06-28/data/calendar.csv.gz",
    "reviews": "https://data.insideairbnb.com/greece/attica/athens/2026-06-28/data/reviews.csv.gz",
    "neighbourhoods": "https://data.insideairbnb.com/greece/attica/athens/2026-06-28/visualisations/neighbourhoods.csv",
}

CRETE_URLS = {
    "listings": "https://data.insideairbnb.com/greece/crete/crete/2026-06-29/data/listings.csv.gz",
    "calendar": "https://data.insideairbnb.com/greece/crete/crete/2026-06-29/data/calendar.csv.gz",
    "reviews": "https://data.insideairbnb.com/greece/crete/crete/2026-06-29/data/reviews.csv.gz",
    "neighbourhoods": "https://data.insideairbnb.com/greece/crete/crete/2026-06-29/visualisations/neighbourhoods.csv",
}


SNAPSHOTS = {
    "thessaloniki": {
        "as_of": "2026-06-29",
        "files": THESSALONIKI_URLS,
    },
    "athens": {
        "as_of": "2026-06-28",
        "files": ATHENS_URLS,
    },
    "crete": {
        "as_of": "2026-06-29",
        "files": CRETE_URLS,
    },
}


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_manifest(city: str, snapshot_date: str, files: list[dict[str, str | int]]) -> None:
    manifest = {
        "city": city,
        "snapshot_date": snapshot_date,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "files": files,
    }

    manifest_path = RAW_DIR / city / snapshot_date / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def download_snapshot(city: str, snapshot_date: str, files: dict[str, str]) -> list[dict]:
    """Fetch one city-snapshot's files (skipping ones already present); return manifest entries."""
    entries = []
    for file_type, url in files.items():
        source_name = Path(urlparse(url).path).name
        destination = RAW_DIR / city / snapshot_date / source_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            partial = destination.with_name(destination.name + ".part")
            urlretrieve(url, partial)
            partial.rename(destination)
        entries.append(
            {
                "type": file_type,
                "filename": destination.name,
                "url": url,
                "size_bytes": destination.stat().st_size,
                "sha256": calculate_sha256(destination),
            }
        )
    return entries


def main() -> None:
    for city, snapshot in SNAPSHOTS.items():
        write_manifest(
            city, snapshot["as_of"], download_snapshot(city, snapshot["as_of"], snapshot["files"])
        )


if __name__ == "__main__":
    main()
