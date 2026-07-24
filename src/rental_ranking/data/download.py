"""Download Inside Airbnb snapshot files (listings, calendar, reviews) per city.

Raw files land untouched in data/raw/<city>/ locally and under raw/ in Blob when
on Azure; raw files are never edited. Use the latest public snapshots only — old
snapshots rotate out of availability.
"""

# TODO: define the city list (2-3 cities) and their snapshot URLs.
# TODO: download listings, calendar, and reviews files per city into data/raw/<city>/.
# TODO: record each city's snapshot "as of" date alongside the files — snapshot dates
#       differ per city, and mixing them silently causes temporal misalignment.

THESSALONIKI_URLS = {
    "listings": "https://data.insideairbnb.com/greece/central-macedonia/thessaloniki/2026-06-29/data/listings.csv.gz",
    "calendar": "https://data.insideairbnb.com/greece/central-macedonia/thessaloniki/2026-06-29/data/calendar.csv.gz",
    "reviews": "https://data.insideairbnb.com/greece/central-macedonia/thessaloniki/2026-06-29/data/reviews.csv.gz",
    "neighborhoods": "https://data.insideairbnb.com/greece/central-macedonia/thessaloniki/2026-06-29/visualisations/neighbourhoods.csv",
}

ATHENS_URLS = {
    "listings": "https://data.insideairbnb.com/greece/attica/athens/2026-06-28/data/listings.csv.gz",
    "calendar": "https://data.insideairbnb.com/greece/attica/athens/2026-06-28/data/calendar.csv.gz",
    "reviews": "https://data.insideairbnb.com/greece/attica/athens/2026-06-28/data/reviews.csv.gz",
    "neighborhoods": "https://data.insideairbnb.com/greece/attica/athens/2026-06-28/visualisations/neighbourhoods.csv",
}

CRETE_URLS = {
    "listings": "https://data.insideairbnb.com/greece/crete/crete/2026-06-29/data/listings.csv.gz",
    "calendar": "https://data.insideairbnb.com/greece/crete/crete/2026-06-29/data/calendar.csv.gz",
    "reviews": "https://data.insideairbnb.com/greece/crete/crete/2026-06-29/data/reviews.csv.gz",
    "neighborhoods": "https://data.insideairbnb.com/greece/crete/crete/2026-06-29/visualisations/neighbourhoods.csv",
}
