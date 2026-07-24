# Data pipeline design — the contract for `src/rental_ranking/data/`

Agreed 2026-07-24. This document is the reference for download, anonymization, cleaning, and
filtering. Column-level definitions and source caveats: [data_assumptions.md](data_assumptions.md).
Decisions and their rationale: [decisions_log.md](decisions_log.md).

## Markets and snapshots

| City | Snapshot date (T) | Notes |
| --- | --- | --- |
| Thessaloniki | 2026-06-29 | Links to prior host-scale project ("one year later" bonus) |
| Athens | 2026-06-28 | Largest market; note T differs from the other two |
| Crete | 2026-06-29 | Region, not city: "neighbourhoods" are municipalities; seasonal market |

Every date computation is relative to the **row's own** `snapshot_date`. Never use a global
reference date (and never `max(last_review)` — that misaligns cities).

## The label window points forward

The calendar file covers T → T+365 only; there is no past calendar in a snapshot. The
occupancy demand proxy is therefore **the fraction of blocked nights in the 90 days after T**
(July–September: peak season, when blocked-because-booked plausibly dominates
blocked-because-closed). Named biases to carry into validation: far-future nights are
less booked simply because they are further out; blocked ≠ booked always (demand proxy
language everywhere). `availability_90` from listings is the label in column form — use it
to cross-check the calendar aggregation, never as a feature.

Temporal design: all listings.csv attributes are known at T; the label is what the calendar
says about after T. Features = as-of-T attributes. Calendar-derived and review-rate-derived
listing columns are **not** features (blocklist below).

## Layered flow

```
download.py     data/raw/<city>/<snapshot_date>/   immutable + manifest (url, sha256, size,
                                                   downloaded_at, snapshot_date)
                        │        (mirrored to Blob raw/ and registered per city/snapshot)
anonymize.py    PII removal + ID hashing            lossless otherwise
clean (typing)  parsing, casting, dedup, bounds     lossless: no row dropped except integrity
                        ▼
data/processed/ ONE parquet per entity (listings, calendar, reviews) concatenated across
                cities with `city` + `snapshot_date` columns; schemas asserted equal first
                        │
filters.py      applied at label-build time, never baked into processed data; identical
                criteria for all cities; per-city counts returned for reporting
                        ▼
feature table   Phase 2 output; registered as a versioned Azure data asset (job input)
```

Rules:

- **Preprocessing is lossless.** Row exclusion is an analytical decision that lives in
  `filters.py`, parameterized and revisable without re-processing. No rank-based outlier
  removal ("top 10 by min_nights") anywhere — thresholds only, identical across cities.
  Price outliers are not removed at all; price treatment belongs to grading/features.
- **Raw is forever.** Snapshots rotate off the public site; the stored copy is the only
  path to reproduction. Raw files are never edited.
- **Registration is separate from download.** `download.py` fetches and manifests — it must
  work offline from Azure with no credentials. Asset registration is a one-time act per
  snapshot done after inspection, via the recorded CLI commands in
  [azure_setup.md](azure_setup.md) (SDK variant optional later, in its own script — never
  inside download.py).
- Group key downstream is **city + neighbourhood_cleansed** (avoids cross-city name
  collisions).

## Anonymization policy

Threat model: the *publication* boundary (repo, notebook outputs, README). Private local
disk and the private Blob container are storage, not publication. Data under `data/` is
never committed; processed data is what notebooks load, so it is PII-free by construction.

- **IDs** (`id`, `host_id`, and `listing_id` in calendar + reviews — consistently, or every
  join breaks): salted SHA-256, truncated to **12 hex chars**. Salt lives in `.env`
  (`ANON_SALT`), never committed. Rationale: 6-hex MD5 gives expected collisions at ~40k+
  listings; unsalted hashes of public enumerable IDs are reversible by dictionary.
- **Dropped outright**: `host_name`, `host_about`, `host_thumbnail_url`, `host_picture_url`,
  `host_url`, `host_profile_id`, `host_profile_url`, `reviewer_id`, `reviewer_name`.
- **Derived then dropped**: `host_location` → `host_is_local` flag; `license` →
  `license_status` (licensed/exempt/unlicensed) + salted-hashed value (duplicate licenses
  across listings = commercial-operator signal).
- **Kept raw**: listing `name` (marketing copy, not host PII — but published notebook cells
  must not render name-bearing rows gratuitously; revisit if a processed *dataset* is ever
  published). Coordinates kept **unrounded**: Airbnb already jitters them 0–150 m at source;
  rounding only degrades spatial features. Keep per-city bounding-box *validation* in
  cleaning.

## Column spec — listings

The implementation should encode this as a literal spec (column → action) that the
inventory notebook asserts against.

**Drop (beyond PII above):** `listing_url`, `scrape_id`, `source`, `picture_url`,
`neighborhood_overview`, `description`, `neighbourhood` (raw; use cleansed),
`host_neighbourhood`, `host_listings_count` (keep the other two counts),
`calendar_updated`, `calendar_last_scraped`, `minimum_minimum_nights`,
`maximum_minimum_nights`, `minimum_maximum_nights`, `maximum_maximum_nights`,
`maximum_nights_avg_ntm` (keep `minimum_nights_avg_ntm` as the one calendar-rule
representative), `hosts_time_as_user_years`, `hosts_time_as_user_months`,
`hosts_time_as_host_months` (redundant with `host_since`; keep `hosts_time_as_host_years`
or derive tenure — implementer's choice, pick one), `host_verifications`,
`host_has_profile_pic` (near-constant; confirm in inventory).

**Keep — Phase 1 core:** `id`, `host_id`, `last_scraped` (→ per-city `snapshot_date`
anchor), `name`, `neighbourhood_cleansed`, `room_type`, `accommodates`, `price`,
`price_quote_checkin_date`, `price_quote_checkout_date`, `price_quote_total_price`,
`price_quote_price_per_night`, `price_quote_raw` (insurance for price nullity — prune after
inventory), `minimum_nights`, `maximum_nights`, `has_availability`, `number_of_reviews`,
`number_of_reviews_ltm`, `number_of_reviews_l30d`, `number_of_reviews_ly`, `first_review`,
`last_review`, `reviews_per_month`.

**Keep — Phase 2/3 features:** `latitude`, `longitude`, `property_type`, `bathrooms`,
`bathrooms_text` (reconcile into numeric + shared flag), `bedrooms`, `beds`, `amenities`,
`review_scores_rating`, `review_scores_accuracy`, `review_scores_cleanliness`,
`review_scores_checkin`, `review_scores_communication`, `review_scores_location`,
`review_scores_value`, `host_since`, `host_response_time`, `host_response_rate`,
`host_acceptance_rate`, `host_is_superhost`, `host_identity_verified`,
`host_total_listings_count`, `calculated_host_listings_count`,
`calculated_host_listings_count_entire_homes`,
`calculated_host_listings_count_private_rooms`,
`calculated_host_listings_count_shared_rooms`, `instant_bookable`.

**Keep but feature-blocklisted** (validation/asserts only; export as a named constant, e.g.
`LABEL_ADJACENT_COLUMNS`, imported by the feature code and checked by a test):
`availability_30`, `availability_60`, `availability_90`, `availability_365`,
`estimated_occupancy_l365d`, `estimated_revenue_l365d`.

**Checklist for `neighbourhood_group_cleansed`:** likely 100% null — confirm per city in
the inventory notebook, then drop if so.

## Column spec — calendar and reviews

- **calendar:** keep `listing_id` (hashed), `date`, `available`, `price` (posted price
  schedule is set by the host at T; whether window-average asking price is a legitimate
  feature is a Phase 2 decision — keep the data, defer the call). Drop `adjusted_price`
  (deprecated), per-date `minimum_nights`/`maximum_nights` (listing-level kept).
- **reviews:** keep `listing_id` (hashed), `id`, `date`, `comments`. Reviewer fields
  dropped (PII).

## Filters (Phase 1, `filters.py` — re-derived, not ported)

Applied at label-build time, identical thresholds per city, per-city removal counts
returned: (1) zero reviews ever AND fully blocked calendar (inactive/personal use);
(2) `first_review` inside the label window (partial exposure); (3) `minimum_nights` > ~30
(de facto long-term rental). Nothing else — no price-based row removal.

## Azure data assets

Register exactly two layers: **raw** per city per snapshot (`uri_folder`, version =
snapshot date) after upload to Blob `raw/`, and the **feature table** (Phase 2) that the
training job consumes. The intermediate processed parquet stays local. PII in raw is fine:
the container is private; anonymization gates publication, not storage.

## Pipeline mechanics

Small pure functions (DataFrame in → DataFrame out) in `src/rental_ranking/data/`, chained
by one thin orchestrator; unit tests per function. **No per-function Azure ML components** —
ceremony without demonstration value at this scale. If the pipeline pattern is wanted on
the CV, one two-step pipeline YAML (prep → train) at Phase 3 makes the point.

## Dependencies settled

Added: `pyarrow` (parquet), `scipy` (direct import in `eda.py`). Rejected: `squarify`
(treemap dropped from EDA), `geopy` (vectorized haversine in Phase 2 instead), `requests`
(stdlib `urllib` suffices for four files per city).
