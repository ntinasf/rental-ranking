# Data pipeline design — the contract for `src/rental_ranking/data/`

Agreed 2026-07-24; amended 2026-07-25 after profiling the downloaded snapshots. This document
is the reference for download, anonymization, cleaning, and filtering. Column-level definitions,
per-city null rates, and source caveats: **[data_dictionary.md](data_dictionary.md)** — the
source of truth this contract is derived from. Decisions and their rationale:
[decisions_log.md](decisions_log.md).

## Markets and snapshots

| City | Release date | Observed `last_scraped` | Listings | Notes |
| --- | --- | --- | --- | --- |
| Thessaloniki | 2026-06-29 | 06-29, 07-02 | 4,965 | Links to prior host-scale project ("one year later" bonus) |
| Athens | 2026-06-28 | 06-28, 06-29, 06-30 | 14,337 | Largest market |
| Crete | 2026-06-29 | 06-29, **06-30**, 07-01, 07-03 | 27,333 | Region, not city: "neighbourhoods" are municipalities; seasonal market |

**Three dates, never conflated** (verified 2026-07-25):

- **Release date** — the folder name and the Azure data-asset version. Not a scrape date.
- **Row scrape date** — `last_scraped`; varies *within* a city. Crete's modal value is
  2026-06-30, not the 06-29 in the folder name.
- **T** — the label anchor, defined below.

Every date computation is relative to the **row's own** dates. Never use a global reference date,
never the release date as if it were T, and never `max(last_review)` — each misaligns cities.

## The label window points forward, anchored per listing

The calendar file covers 365 days forward from each listing's own scrape; there is no past
calendar in a snapshot. The occupancy demand proxy is **the fraction of blocked nights in the
90 days after T**, where

> **T = `min(calendar.date)` for that listing** — equivalently its own first calendar row.

Per-listing, not per-city: scrape dates spread across up to four days inside one city, so a fixed
per-city T would give later-scraped listings a short or shifted window. Verified 2026-07-25: with
the per-listing anchor, the calendar-derived available-night count over the first 90 days
reproduces `availability_90` for **99.96 / 99.99 / 99.97 %** of listings (mean abs diff ≤ 0.03
nights). Every listing has exactly 365 calendar rows, so windows are equal length.

`availability_90` is the label in column form — it is the standing cross-check for `label.py`,
never a feature.

Named biases to carry into validation: far-future nights are less booked simply because they are
further out; blocked ≠ booked always (demand proxy language everywhere). Because T lands at the
end of June, the window is July–September peak season, when blocked-because-booked most plausibly
dominates blocked-because-closed.

Temporal design: all listings.csv attributes are known at T; the label is what the calendar says
about after T. Features = as-of-T attributes. Calendar-derived and review-rate-derived listing
columns are **not** features (blocklist below).

## Layered flow

```text
download.py     data/raw/<city>/<release_date>/    immutable + manifest (url, sha256, size,
                                                   downloaded_at, release_date)
                        │        (mirrored to Blob raw/ and registered per city/release)
anonymize.py    PII removal + ID hashing            lossless otherwise
clean (typing)  parsing, casting, dedup, bounds     lossless: no row dropped except integrity
                        ▼
data/processed/ ONE parquet per entity (listings, calendar, reviews) concatenated across
                cities with `city` + `scrape_date` columns; schemas asserted equal first
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
  collisions) × `room_type` × capacity tier. **Sizing warning:** Thessaloniki has only 7
  neighbourhoods, Shared room has 3 / 30 / 18 rows across the three cities, and Hotel room is
  absent from Thessaloniki — this key will produce singleton groups, which contribute nothing
  to NDCG and waste LambdaMART's gradient. Group construction needs a minimum-size rule and
  probably a collapse of rare `room_type`s; the choice belongs to Phase 1.

## Anonymization policy

Threat model: the *publication* boundary (repo, notebook outputs, README). Private local
disk and the private Blob container are storage, not publication. Data under `data/` is
never committed; processed data is what notebooks load, so it is PII-free by construction.

- **IDs** (`id`, `host_id`, and `listing_id` in calendar + reviews — consistently, or every
  join breaks): salted SHA-256, truncated to **12 hex chars**. Salt lives in `.env`
  (`ANON_SALT`), never committed. Rationale: 6-hex MD5 gives expected collisions at ~40k+
  listings; unsalted hashes of public enumerable IDs are reversible by dictionary.
- **Dropped outright**: `host_name`, `host_thumbnail_url`, `host_picture_url`, `host_url`,
  `host_profile_id`, `host_profile_url`, `reviewer_id`, `reviewer_name`. `host_profile_id` is
  verified 1:1 with `host_id` — dropping it loses nothing, and hashing a second host key would
  only invite an inconsistent join.
- **Derived then dropped**: `host_location` → `host_is_local` (three-way: the column is 34–37 %
  null, so "unknown" is a real category, not an imputation) · `host_about` → `host_has_about`
  presence flag · `license` → `license_status` (registered/exempt/missing) + salted-hashed value.
  The licence hash earns its keep: only 2.8–4.3 % of rows are null (Greece's AMA registry), and
  duplicate licence numbers cover 369 / 2,204 / **10,925** listings — a strong
  commercial-operator signal.
- **Kept raw**: listing `name` (marketing copy, not host PII — but published notebook cells
  must not render name-bearing rows gratuitously; revisit if a processed *dataset* is ever
  published). Coordinates kept **unrounded**: Airbnb already jitters them 0–150 m at source;
  rounding only degrades spatial features. Keep per-city bounding-box *validation* in
  cleaning.

## Column spec — listings

Encoded as a literal spec in [`src/rental_ranking/data/columns.py`](../src/rental_ranking/data/columns.py),
which the inventory notebook asserts against. Per-column evidence and null rates:
[data_dictionary.md §4](data_dictionary.md). The lists below and that module must agree.

**Drop — empty (100% null in all three cities, verified):** `neighborhood_overview`,
`host_since`, `host_response_time`, `host_response_rate`, `host_acceptance_rate`,
`host_thumbnail_url`, `host_neighbourhood`, `host_total_listings_count`, `host_verifications`,
`neighbourhood`, `neighbourhood_group_cleansed`, `calendar_updated`, `instant_bookable`.

**Drop — other:** `listing_url`, `scrape_id`, `source` (no variance in Athens), `picture_url`,
`description`, `host_url`, `host_picture_url`, `calendar_last_scraped` (duplicate of
`last_scraped`), `has_availability` (constant `t`; only its nullity varies, and that tracks the
label), `minimum_minimum_nights`, `maximum_minimum_nights`, `minimum_maximum_nights`,
`maximum_maximum_nights`, `maximum_nights_avg_ntm` (keep `minimum_nights_avg_ntm` as the one
calendar-rule representative), `price_quote_total_price` and `price_quote_price_per_night`
(exact duplicates of `price`), `price_quote_raw` (restates the quote dates in JSON; ~1 KB/row).

**Derive, then drop the raw column:** `last_scraped` → `scrape_date` · `host_location` →
`host_is_local` (**three-way** {local, remote, unknown} — the column is 34–37 % null) ·
`host_about` → `host_has_about` · `license` → `license_status` ∈ {registered, exempt, missing}
plus a salted hash of the number · the four `hosts_time_as_*` → `user_tenure_months` and
`host_tenure_months` (`years*12 + months`; the `_months` fields are 0–11 remainders).

> **Reversal, 2026-07-25.** The `hosts_time_as_*` fields were previously slated for dropping as
> redundant with `host_since`. `host_since` is 100 % null, so they are the *only* tenure signal,
> and user tenure ≠ host tenure (years agree in only 62–77 % of rows) — keep both derivations.
> Likewise `host_listings_count` moves from drop to **keep**, because the two counts it was
> meant to defer to include `host_total_listings_count`, which is empty.

**Keep — Phase 1 core:** `id` (hashed), `host_id` (hashed), `name`, `neighbourhood_cleansed`,
`room_type`, `accommodates`, `price` (parse + impute — see below), `minimum_nights`,
`maximum_nights`, `minimum_nights_avg_ntm`, `number_of_reviews`, `number_of_reviews_ltm`,
`number_of_reviews_l30d`, `number_of_reviews_ly`, `first_review`, `last_review`,
`reviews_per_month`.

`number_of_reviews_ly` is confirmed to be the **calendar-2025** review count (exact match against
`reviews.csv`, all three cities). All of 2025 precedes every T, so it is leakage-free and
`ltm − ly` is a clean momentum feature.

**Keep — Phase 2/3 features:** `latitude`, `longitude`, `property_type`, `bathrooms`,
`bathrooms_text` (the near-complete one — reconcile into numeric + shared flag), `bedrooms`,
`beds`, `amenities`, the seven `review_scores_*`, `host_is_superhost`, `host_identity_verified`,
`host_has_profile_pic`, `host_listings_count`, `calculated_host_listings_count`,
`calculated_host_listings_count_entire_homes`, `calculated_host_listings_count_private_rooms`,
`calculated_host_listings_count_shared_rooms`.

**Keep but feature-blocklisted** (validation/asserts only; exported as `LABEL_ADJACENT_COLUMNS`,
imported by the feature code and checked by a test): `availability_30`, `availability_60`,
`availability_90`, `availability_365`, **`availability_eoy`**, `estimated_occupancy_l365d`,
`estimated_revenue_l365d`, **`price_quote_checkin_date`**, **`price_quote_checkout_date`**.

> **The quote dates are a target leak.** `price_quote_checkin_date` is the listing's first
> available calendar date in 91.5 / 86.5 / 68.5 % of rows; Spearman correlation between quote
> lead and blocked-fraction-90 is 0.56–0.67, and where the quote date is null, mean
> blocked-fraction-90 is 0.98–0.99. Inside Airbnb's scraper walks forward until it finds an
> opening, so the date it lands on *is* the availability signal. They stay in the processed data
> only so notebook 02 can demonstrate the leak. `availability_eoy` joins them: it is a forward
> availability window (0–186 days) that fully contains the label window.

## Price is endogenous

`price` is no longer a standing nightly rate — it equals `price_quote_price_per_night` exactly,
i.e. it is the per-night figure from a quote for a dated stay, and that date is chosen by
availability. Two consequences, of different severity:

- **Missingness is severe.** `price` is null in 5.0 / 1.0 / 6.6 % of rows, and 84 / 15 / 68 % of
  those have `availability_90 == 0`. **Never expose a price-missingness indicator as a feature**,
  and do not pass NaN through to LightGBM — its native missing handling would split directly on
  the leak. **Impute** from the `city × neighbourhood_cleansed × room_type × accommodates`
  median. Rows needing this after the inactive-listing filter: 188 / 47 / 1,231 (3.8 / 0.3 /
  4.5 %). Dropping them instead would systematically delete the highest-demand listings.
- **Value contamination is mild.** Within `room_type × accommodates × neighbourhood` strata,
  median relative price shifts only 0.98–1.05 / 0.95–1.01 / 0.78–1.07 across quote-lead buckets.
  Usable with a stated caveat. Build price tiers within city (and probably room type) so residual
  seasonal drift does not become a cross-market artefact.

## Column spec — calendar and reviews

- **calendar (5 columns, verified):** keep `listing_id` (hashed), `date`, `available`. Drop the
  per-date `minimum_nights`/`maximum_nights` — they fall inside the label window, and the
  listing-level values are kept instead. **There is no `price` or `adjusted_price` column in
  v4.7**, so the earlier "keep the posted price schedule and decide in Phase 2 whether a
  window-average asking price is a legitimate feature" question is void: no per-date price
  exists anywhere in this project.
- **calendar join integrity:** Athens' calendar carries 5 `listing_id`s with no matching listings
  row (Thessaloniki and Crete match exactly). Notebook 01 asserts this; the resolution is
  explicit, not silent. **Reviews are not clean either:** one of those same five
  (`1361558511345756012`) also carries 2 review rows, so the reviews table has 1 orphan
  `listing_id` in Athens and none elsewhere. `tests/test_io.py` pins the exact orphan set per
  city, so a *new* orphan fails rather than being absorbed by a tolerance.
- **reviews:** keep `listing_id` (hashed), `id`, `date`, `comments`. Reviewer fields
  dropped (PII).

## Filters (Phase 1, `filters.py` — re-derived, not ported)

Applied at label-build time, identical thresholds per city, per-city removal counts
returned: (1) zero reviews ever AND fully blocked calendar (inactive/personal use);
(2) `first_review` inside the label window (partial exposure); (3) `minimum_nights` > ~30
(de facto long-term rental). Nothing else — no price-based row removal.

Rule (3) is a safety net rather than a material filter here: `minimum_nights > 30` covers only
0.50 / 0.01 / 0.16 % of rows. Keep it — thresholds identical across cities are worth more than
the rows they remove — but do not expect it to change any distribution.

## Azure data assets

Register exactly two layers: **raw** per city per release (`uri_folder`, version = release
date) after upload to Blob `raw/`, and the **feature table** (Phase 2) that the training job
consumes. The intermediate processed parquet stays local. PII in raw is fine: the container is
private; anonymization gates publication, not storage.

## Pipeline mechanics

Small pure functions (DataFrame in → DataFrame out) in `src/rental_ranking/data/`, chained
by one thin orchestrator; unit tests per function. **No per-function Azure ML components** —
ceremony without demonstration value at this scale. If the pipeline pattern is wanted on
the CV, one two-step pipeline YAML (prep → train) at Phase 3 makes the point.

## Dependencies settled

Added: `pyarrow` (parquet), `scipy` (direct import in `eda.py`). Rejected: `squarify`
(treemap dropped from EDA), `geopy` (vectorized haversine in Phase 2 instead), `requests`
(stdlib `urllib` suffices for four files per city).
