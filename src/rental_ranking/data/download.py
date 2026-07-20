"""Download Inside Airbnb snapshot files (listings, calendar, reviews) per city.

Raw files land untouched in data/raw/<city>/ locally and under raw/ in Blob when
on Azure; raw files are never edited. Use the latest public snapshots only — old
snapshots rotate out of availability.
"""

# TODO: define the city list (2-3 cities) and their snapshot URLs.
# TODO: download listings, calendar, and reviews files per city into data/raw/<city>/.
# TODO: record each city's snapshot "as of" date alongside the files — snapshot dates
#       differ per city, and mixing them silently causes temporal misalignment.
