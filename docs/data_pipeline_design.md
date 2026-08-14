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
nights). Each listing's calendar is contiguous — 365 rows for all but 8 listings (2 Athens,
6 Crete) which run 366–367 — so the 90-day window is always fully covered. Do not write code
that assumes exactly 365.

`availability_90` is the label in column form — it is the standing cross-check for `label.py`,
never a feature.

Named biases to carry into validation: far-future nights are less booked simply because they are
further out; blocked ≠ booked always (demand proxy language everywhere). Because T lands at the
end of June, the window is July–September peak season, when blocked-because-booked most plausibly
dominates blocked-because-closed.

> **What the label encodes — a two-stage signal (measured 2026-08-03).** Mean blocked fraction
> rises monotonically with listing age in every city — never reviewed / new this year /
> established = 0.214 / 0.277 / 0.359 (Thessaloniki), 0.243 / 0.323 / 0.416 (Athens),
> 0.418 / 0.561 / 0.638 (Crete) — and ~30 % of each market is new since last summer. Restricting
> to established listings collapses the review-*volume* correlation to 0.034 / 0.151 / 0.065,
> which is why `reviews_per_month`, the one tenure-normalised signal, is the weakest of all of
> them.
>
> Within the established cohort, **review quality orders the label**: by rating tercile (≥ 10
> reviews, so the rating is stable) mean blocked fraction runs 0.275 / 0.371 / 0.442
> (Thessaloniki), 0.383 / 0.429 / 0.470 (Athens), 0.606 / 0.686 / 0.693 (Crete) — and
> Thessaloniki's poorly-rated established listings (0.275) are indistinguishable from its
> brand-new ones (0.277). The city ordering **inverts** between quality (ρ = 0.255 / 0.136 /
> 0.177) and volume (0.084 / 0.172 / 0.128): large liquid markets rank on establishment, small
> ones on quality. Ungated, Crete looks non-monotone only because its top rating band has a
> median of 6 reviews against 37 in the middle band — noise in the instrument, not the signal.
>
> **Claim: forward availability tracks listing establishment, and within the established cohort
> it is ordered by review quality.** Not: that the label measures demand among
> otherwise-comparable listings. Three standing limits — Airbnb ratings are ceiling-compressed
> (95th percentile = 5.0 everywhere), rating is 100 % null for never-reviewed listings, and
> "less blocked" cannot be separated from "the host left it open because it does not sell"
> without bookings or impressions. Survivorship works the other way: the worst low-rated
> listings delisted, so the measured gradient is attenuated.
>
> **Validation instrument.** Prefer a *same-season-last-year* review window — reviews in
> `[T − 365, T − 365 + 90)`, anchored per listing. It is leakage-free and seasonally matched to
> the label window, unlike a trailing window from T, which lands in shoulder season. It raises
> Thessaloniki's correlation 50 % over a trailing 40-day window. At equal 90-day width, where
> season is the only difference, it gives 0.124 / 0.225 / 0.194 against the trailing window's
> 0.061 / 0.194 / 0.215 — and it does so with *more* empty windows in two of three cities
> (44.8 / 48.7 % against 38.2 / 42.5 %), so the gain is seasonal alignment, not extra data.
> Extending it to absorb review-posting lag (104/110/120 days) was tested
> and does not help. Never pool across cities: the between-city gradient runs opposite to the
> within-city one (Crete is the most blocked market *and* the thinnest-reviewed), so a pooled
> correlation understates the signal.

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
  to NDCG and waste LambdaMART's gradient.

  **Phase split, settled 2026-08-05** (the earlier "the choice belongs to Phase 1" conflicted
  with BUILD_GUIDE, which defines query groups in Phase 2): a capacity **tier** is a listing
  *attribute*, a query **group** is an *assembly*. Phase 1 owns the attribute, because price
  imputation consumes it as a cascade rung; Phase 2 assembles groups from it and sets the
  minimum-size rule. Both live in `features/groups.py`.

  Tier bounds are **1–2 / 3–4 / 5–7 / 8+**. `accommodates` takes 16 distinct values (1–16) with
  87 % of listings at ≤ 6, so leaving it raw fragments the key badly — measured on the ranked
  population: raw gives **1,126 groups with 237 singletons** and a median group of 6, against
  **512 groups, 66 singletons** and a median of 15 under these bounds. A five-tier split
  (7–8 / 9+) costs 62 more groups and 13 more singletons for no gain; a three-tier split
  (5+) reaches 432 / 44 but lumps 5- and 16-guest properties into one search intent. The
  5–6 / 7+ cut measures identically (517 / 64, same cascade fills); 5–7 / 8+ wins on semantics,
  because capacity clusters on even values (5 = 9.1 %, 6 = 11.7 %, **7 = 2.7 %**, 8 = 4.3 %) so
  `8+` starts the top tier on a real party size rather than a near-empty one.

  **Group construction (Phase 2): minimum size 5, falling back by dropping the *neighbourhood*
  dimension** — undersized groups rank against `city × room_type × capacity_tier`. Measured:
  512 groups / 66 singletons with no minimum, against **394 groups / 5 singletons**, median
  size 29 and 99.9 % of listings in a usable group (size > 1 and more than one grade) with the
  fallback; 284 listings (0.6 %) use it. **No threshold on `neighbourhood_cleansed` itself**, and
  no spatial merging of sparse neighbourhoods — thresholding one factor cannot fix a product.
  Of those 284 listings, only 43 sit in a neighbourhood below 50 listings while 186 are in
  neighbourhoods of 100+ (Χανίων holds 6,009). Merging was rejected on measurement too: sub-50
  neighbourhoods sit 0.6 km apart in Athens but **16.6 km median, 39.8 km max in Crete**, so a
  merge would invent exactly the incoherent group the fallback avoids.

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
  median. Rows needing this after the filters: **193 / 121 / 1,503** = 4.0 / 0.9 / 5.6 %
  (re-measured 2026-08-05 against the two-rule `filters.py`; the earlier 188 / 47 / 1,231 was
  computed under a filter definition that no longer exists). Dropping them instead would
  systematically delete the highest-demand listings.

  **The cascade is a contract term, not defensive coding**, because 14 rows have no priced peer
  in their own stratum. Rungs, with rows resolved at each (measured):

  | rung | fills | still open |
  | --- | ---: | ---: |
  | `city × neighbourhood × room_type × accommodates` | 1,803 | 14 |
  | `city × neighbourhood × room_type × capacity_tier` | 4 | 10 |
  | `city × neighbourhood × room_type` | 4 | 6 |
  | `city × room_type` | 6 | 0 |
  | `city` | 0 | 0 |

  The `city` rung never fires on these snapshots; it stays as the guaranteed terminator. Note
  the capacity-tier rung is the reason Phase 1 owns the tier definition. **The key is structural
  throughout** — never cohort, listing age or `rating_shrunk`; see the decisions log for why.
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

Applied at label-build time, identical thresholds per city, per-city **per-rule** removal counts
returned. **Two rules, not three** (amended 2026-08-01 — see below):

| rule | Thessaloniki | Athens | Crete | total |
| --- | ---: | ---: | ---: | ---: |
| (1) zero reviews ever **and** fully blocked calendar (inactive / personal use) | 75 | 101 | 523 | 699 |
| (3) `minimum_nights` > 30 (de facto long-term rental) | 25 | 1 | 43 | 69 |
| **removed** (the two rules overlap on 2 rows) | 99 | 102 | 565 | **766** |

Measured 2026-08-01 against `data/processed/`: 766 of 46,635 rows, 1.6 %; 45,869 survive.
Nothing else — no price-based row removal.

Rule (3) is a safety net rather than a material filter here: `minimum_nights > 30` covers only
0.50 / 0.01 / 0.16 % of rows. Keep it — thresholds identical across cities are worth more than
the rows they remove — but do not expect it to change any distribution.

> **Rule (2) — `first_review` inside the label window — is void and has been dropped.**
> Measured: it removes **0 rows in all three cities**, and it cannot do otherwise. The reviews
> file ends at the scrape date and T ≈ the scrape date, so `first_review > T` is impossible by
> construction under a **forward** window; the rule was inherited from the trailing-window design
> abandoned on 2026-07-24. The thin-history population it was reaching for does exist — 59 / 202
> / 518 listings had their first review within 30 days of T — and is deliberately **kept**: that
> is the cold-start cohort, which Phase 2 flags (`has_reviews`) and Phase 5 studies. Filtering it
> would delete the population later phases are about.

**Ordering constraint.** Filters run *before* price imputation, price tiering and grading.
Quantile boundaries must be computed on the population that will actually be ranked. Note also
that rule (1) removes exactly the (zero reviews, blocked fraction = 1.0) corner, so it
mechanically raises the label's correlation with review signals — label validation must report
that correlation both **before and after** filtering, or the validation is partly circular.

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
