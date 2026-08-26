# Inside Airbnb data dictionary — v4.7 as actually delivered

**This file is the single source of truth for what the raw data contains.** It supersedes and
replaces the two earlier research reports (`data_assumptions.md`, describing schema v4.3, and
`data_assumptions_2.md`, inferring the v4.7 additions), both deleted on 2026-07-25 — git history
retains them. Where those reports disagreed with the files on disk, the files win, and the
disagreement is recorded in [§2](#2-what-v47-actually-changed) rather than quietly overwritten.

How it was built: the two reports supplied Inside Airbnb's official definitions and methodology;
every claim that could be checked was then run against our own three snapshots
(46,635 listings, 17.0 M calendar rows, profiled 2026-07-25).

Evidence tags used throughout:

- **[official]** — Inside Airbnb's published data dictionary (v4.3 tab) or Data Assumptions page.
- **[inferred]** — reasoned from naming conventions or platform behaviour; unverified.
- **[verified]** — measured in our own files. The number is quoted so it can be re-derived.

The v4.7 dictionary tab exists in Inside Airbnb's Google Sheet but is an **empty placeholder** —
no field descriptions are published for anything added after v4.3. That is why 14 of the columns
below have no official definition and had to be established empirically.

Companion documents: [data_pipeline_design.md](data_pipeline_design.md) is the contract for
`src/rental_ranking/data/` and takes its column dispositions from here;
[decisions_log.md](decisions_log.md) records why.

---

## 1. File inventory

Per city, four files under `data/raw/<city>/<release_date>/`:

| File | Columns | Thessaloniki | Athens | Crete |
| --- | --- | --- | --- | --- |
| `listings.csv.gz` (detailed) | **90** | 4,965 | 14,337 | 27,333 |
| `calendar.csv.gz` | **5** | 1,812,225 | 5,234,832 | 9,976,554 |
| `reviews.csv.gz` (detailed) | **6** | — | — | — |
| `neighbourhoods.csv` | **2** | 7 rows | 44 rows | 24 rows |

**[verified]** The 90-column listings header is byte-identical across all three cities, in the
same order. The pre-concat schema assertion in the contract should still run — this is a
property of one download, not a guarantee.

We do not use the `visualisations/` summary listings file (a strict subset of the detailed one).

### Three different dates, do not conflate them

| Name | Where it lives | Values |
| --- | --- | --- |
| **Release date** | folder name, Azure asset version | Thessaloniki 2026-06-29 · Athens 2026-06-28 · Crete 2026-06-29 |
| **Row scrape date** | `last_scraped` (= `calendar_last_scraped`) | Thessaloniki {06-29, 07-02} · Athens {06-28, 06-29, 06-30} · Crete {06-29, **06-30**, 07-01, 07-03} |
| **T (label anchor)** | `min(calendar.date)` per listing | equals that listing's scrape date |

**[verified]** A city's release date is *not* the scrape date of most of its rows — Crete's modal
`last_scraped` is 2026-06-30, and 1,369 Crete listings were scraped on 07-03. Every listing has
exactly 365 calendar rows starting at its own scrape date, which is why the label anchor is
per-listing and never per-city.

---

## 2. What v4.7 actually changed

Corrections to the superseded reports. Each row is a claim we would have implemented had the
data not been checked first.

| # | Claim in the old reports | What the files show |
| --- | --- | --- |
| 1 | `calendar.csv` has 7 columns including `price` and `adjusted_price` | **[verified]** 5 columns. There is **no price in the calendar at all** — no per-date price schedule exists in v4.7. |
| 2 | `price` is the host's standing nightly asking rate | **[verified]** `price == price_quote_price_per_night` exactly, in 100 % of rows with both present, all three cities. `price` is a **dated quote**, not a standing rate. |
| 3 | `hosts_time_as_*` are redundant with `host_since`; drop them | **[verified]** `host_since` is **100 % NULL**. The four `hosts_time_as_*` columns are the only host-tenure signal in the file. |
| 4 | `host_profile_id` may be identical to `host_id`; drop if so | **[verified]** Always differs (int64, ~19 digits), but maps **1:1** onto `host_id` in all three cities. Carries no information; drop as an identifier. |
| 5 | `availability_eoy` ≈ days available through 31 December *(inferred)* | **[verified] Confirmed.** Range 0–186 in every city = 2026-06-29 → 2026-12-31 inclusive. |
| 6 | `number_of_reviews_ly` ≈ previous calendar year *(inferred)* | **[verified] Confirmed exactly.** 100.0 % row-for-row match against `reviews.csv` counts for 2025-01-01…2025-12-31, all three cities. Not a rolling window. |
| 7 | `license` is "frequently 100 % null" | **[verified]** 2.8 / 3.6 / 4.3 % null. Greece's AMA registry makes this a populated, useful column. |
| 8 | `neighbourhood_group_cleansed` "often entirely null" | **[verified]** 100 % null in all three cities → drop. |
| 9 | `source` values are `neighbourhood search` / `previous scrape` **[official, v4.3]** | **[verified]** The first value is now spelled **`city scrape`**. Athens is 100 % `city scrape` (no variance). |
| 10 | Prefer `price_quote_price_per_night` over `price` | **[verified]** Moot — they are the same number. |
| 11 | `estimated_revenue_l365d` = occupancy × price | **[verified]** Consistent: its null rate (5.0 / 1.0 / 6.6 %) matches `price`'s exactly, while `estimated_occupancy_l365d` is 0 % null. |

### Columns that are 100 % NULL in all three cities

`calendar_updated`, `host_acceptance_rate`, `host_neighbourhood`, `host_response_rate`,
`host_response_time`, `host_since`, `host_thumbnail_url`, `host_total_listings_count`,
`host_verifications`, `instant_bookable`, `neighborhood_overview`, `neighbourhood`,
`neighbourhood_group_cleansed`.

Thirteen columns. Six of them (`host_since`, the two response fields, `host_acceptance_rate`,
`host_total_listings_count`, `instant_bookable`) were on the planned feature list. The host-quality
block is therefore much thinner than the v4.3 documentation implies: what survives is
`host_is_superhost`, `host_identity_verified`, `host_has_profile_pic`, `host_listings_count`,
the four `calculated_host_listings_count*` fields, and the tenure pair derived from
`hosts_time_as_*`.

`has_availability` is a fourteenth casualty: **[verified]** `nunique == 1` (`t`) with 0.2–0.3 %
null. Its only variation is its nullity, which tracks the label — see §5.

---

## 3. The target leak

**This is the most important thing in this document.** `price_quote_checkin_date` is, in most
rows, the listing's **first available calendar date** — a direct function of the label.

| | Thessaloniki | Athens | Crete |
| --- | --- | --- | --- |
| `quote_checkin == first_available_date` | **91.5 %** | **86.5 %** | **68.5 %** |
| Spearman(quote lead days, blocked-fraction-90) | 0.557 | 0.564 | 0.667 |
| blocked-fraction-90 where quote date is null | 0.981 | *(n=1)* | 0.990 |

Blocked-fraction over the 90-day label window rises monotonically with quote lead, e.g.
Thessaloniki: lead 0 → **0.17**, 1–3 d → 0.25, 4–7 d → 0.31, 8–30 d → 0.43, 31–90 d → 0.64,
>90 d → **0.99**. Athens and Crete show the same shape.

The mechanism is plain once seen: Inside Airbnb's scraper asks Airbnb for a real quote, and a
quote only succeeds on dates the listing is actually free — so it walks forward until it finds
an opening. The date it lands on *is* the availability signal we are trying to predict.

Consequences, all of which are enforced in the contract:

1. `price_quote_checkin_date` and `price_quote_checkout_date` are **label-adjacent** — blocklisted
   alongside `availability_*` and `estimated_*`. Kept in the processed data only so notebook 02
   can demonstrate the leak.
2. `price_quote_raw` is **dropped** — it restates both dates inside its JSON payload, and at
   ~1 KB/row it is the largest column in the file.
3. `availability_eoy` joins the blocklist: it is a forward-calendar availability window
   (0–186 days) that fully contains the 90-day label window.
4. **Price missingness must never become a feature.** See §5.

---

## 4. Listings — column-by-column

Null percentages are Thessaloniki / Athens / Crete. Disposition vocabulary:
**KEEP** (into the processed table) · **DROP** · **HASH** (salted SHA-256, 12 hex) ·
**DERIVE** (compute a feature, then drop the raw column) · **BLOCK** (kept but on
`LABEL_ADJACENT_COLUMNS`; validation and asserts only, never a model input).

### 4a. Identifiers and scrape metadata

| # | Column | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 1 | `id` | 0/0/0 | **HASH** | **[official]** Primary key; joins to `listing_id` in calendar and reviews. Must use the same salt everywhere or joins break. |
| 2 | `listing_url` | 0/0/0 | DROP | **[official — calculated]** Reconstructible from `id`; re-identifies the listing. |
| 3 | `scrape_id` | 0/0/0 | DROP | **[verified]** `nunique == 1` per city. |
| 4 | `last_scraped` | 0/0/0 | **DERIVE** | **[official]** → `scrape_date`. **[verified]** 2–4 distinct values per city; not the release date. |
| 5 | `source` | 0/0/0 | DROP | **[official]** `city scrape` / `previous scrape`. **[verified]** No variance in Athens; a staleness flag too weak to carry. |
| 6 | `name` | 0/0/0 | **KEEP** raw | Marketing copy, not host PII. Published notebook cells should not render name-bearing rows gratuitously. |
| 7 | `description` | 3.4/2.1/2.3 | DROP | **[official]** Dropped; a candidate to revisit if text features are ever added. |
| 8 | `neighborhood_overview` | **100** | DROP | Empty. |
| 9 | `picture_url` | 0/0/0 | DROP | Not used. |

### 4b. Host attributes

| # | Column | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 10 | `host_id` | 0/0/0 | **HASH** | **[official]** 2,140 / 5,505 / 11,333 distinct hosts. |
| 11 | `host_url` | 0/0/0 | DROP | PII. |
| 12 | `host_profile_id` | 0/0/0 | DROP | **[verified]** 1:1 with `host_id`; no information, and an identifier. |
| 13 | `host_profile_url` | 0/0/0 | DROP | PII. |
| 14 | `host_name` | 0/0/0 | DROP | PII. |
| 15 | `host_since` | **100** | DROP | Empty — see §2. |
| 16 | `hosts_time_as_user_years` | 0/0/0 | **DERIVE** | **[verified]** Range 0–16; pairs with `_months` (0–11 remainder) → `user_tenure_months = years*12 + months`. |
| 17 | `hosts_time_as_user_months` | 0/0/0 | **DERIVE** | As above. |
| 18 | `hosts_time_as_host_years` | 0/0/0 | **DERIVE** | → `host_tenure_months`. Range 0–15. **[verified]** Distinct from user tenure: years agree in only 62 / 69 / 77 % of rows, months in 48 / 49 / 59 %. Hosting experience ≠ account age. |
| 19 | `hosts_time_as_host_months` | 0/0/0 | **DERIVE** | As above. |
| 20 | `host_location` | 36.5/33.6/33.9 | **DERIVE** | **[official]** Noisy free text → `host_is_local`. High nullity means a three-way {local, remote, unknown}, not a boolean. |
| 21 | `host_about` | 55.7/48.0/51.8 | **DERIVE** | → `host_has_about` presence flag, then drop the free text (may contain PII). |
| 22 | `host_response_time` | **100** | DROP | Empty. |
| 23 | `host_response_rate` | **100** | DROP | Empty. |
| 24 | `host_acceptance_rate` | **100** | DROP | Empty. |
| 25 | `host_is_superhost` | 0/0/0 | **KEEP** | **[official]** `t`/`f` → bool. **[verified]** Well balanced (40 / 49 / 44 % superhost) — the strongest surviving host-quality signal. Correlated with review count and scores. |
| 26 | `host_thumbnail_url` | **100** | DROP | Empty + PII. |
| 27 | `host_picture_url` | 0/0/0 | DROP | PII. |
| 28 | `host_neighbourhood` | **100** | DROP | Empty. |
| 29 | `host_listings_count` | 0/0/0 | **KEEP** | **[official — Airbnb's own opaque count]** Kept **because `host_total_listings_count` is empty**, reversing the earlier plan. **[verified]** median 6 / 7 / 4, max 1,284 / 1,284 / 3,062. |
| 30 | `host_total_listings_count` | **100** | DROP | Empty. |
| 31 | `host_verifications` | **100** | DROP | Empty. |
| 32 | `host_has_profile_pic` | 0/0/0 | **KEEP** | **[verified]** 5.4 / 3.4 / 4.3 % false — imbalanced but not constant; retained provisionally, drop if it earns nothing. |
| 33 | `host_identity_verified` | 0/0/0 | **KEEP** | **[verified]** 0.5 / 0.4 / 1.2 % false — very near-constant; same caveat. |

### 4c. Location

| # | Column | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 34 | `neighbourhood` | **100** | DROP | Empty (and **[official]** unreliable even when present). |
| 35 | `neighbourhood_cleansed` | 0/0/0 | **KEEP** | **[official — calculated]** Geocoded against public shapefiles. Half the group key. **[verified]** 7 / 44 / 24 distinct values — see §7 on group sizing. |
| 36 | `neighbourhood_group_cleansed` | **100** | DROP | Empty. |
| 37 | `latitude` | 0/0/0 | **KEEP** unrounded | **[official]** Airbnb already jitters 0–150 m at source; rounding only degrades spatial features. Per-city bounding-box *validation* (warn, never drop) belongs in cleaning. |
| 38 | `longitude` | 0/0/0 | **KEEP** unrounded | As above. |

### 4d. Property characteristics

| # | Column | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 39 | `property_type` | 0/0/0 | **KEEP** | **[verified]** 30 / 42 / 76 values — collapse into buckets for modelling. |
| 40 | `room_type` | 0/0/0 | **KEEP** | **[official]** Half the group key. **[verified]** Severely skewed: Entire home/apt 97 / 93 / 91 %; Shared room only 3 / 30 / 18 rows; Hotel room absent in Thessaloniki. |
| 41 | `accommodates` | 0/0/0 | **KEEP** | Capacity-tier source. **[verified]** Median 4 everywhere; p75 = 4 / 4 / 6. |
| 42 | `bathrooms` | 10.9/5.0/11.8 | **KEEP** | **[official]** Numeric, but partly superseded by the text field. |
| 43 | `bathrooms_text` | 0.0/0.0/0.1 | **KEEP** → derive | **[official]** Effectively complete, so it is the **primary** source: parse to numeric count + shared/private flag, reconcile against `bathrooms`. |
| 44 | `bedrooms` | 9.7/8.3/6.6 | **KEEP** | Studios may be 0 or null. |
| 45 | `beds` | 7.9/2.5/9.0 | **KEEP** | Overlaps conceptually with `accommodates`. |
| 46 | `amenities` | 0/0/0 | **KEEP** | **[official]** JSON array → list; multi-hot later. Very high cardinality, inconsistent naming across scrapes. |

### 4e. Price — read §5 before using any of these

| # | Column | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 47 | `price` | 5.0/1.0/6.6 | **KEEP** (parse, impute) | `$`-prefixed string → float. **[verified]** Identical to `price_quote_price_per_night`; it is a dated quote. Endogenous — §5. |
| 48 | `price_quote_checkin_date` | 4.3/0.0/4.8 | **BLOCK** | **[verified]** ≈ first available date. §3. |
| 49 | `price_quote_checkout_date` | 4.3/0.0/4.8 | **BLOCK** | **[verified]** Stay length = checkout − checkin equals `minimum_nights` in 83 / 87 / 66 % of rows; ≥28-night monthly quotes are only 0.7 / 0.4 / 0.3 %. |
| 50 | `price_quote_total_price` | 5.0/1.0/6.6 | DROP | **[verified]** = `price` × quote nights. No independent content; its "insurance against price nullity" rationale is void since it is null exactly when `price` is. |
| 51 | `price_quote_price_per_night` | 5.0/1.0/6.6 | DROP | **[verified]** Exact duplicate of `price`. |
| 52 | `price_quote_raw` | 4.3/0.0/4.8 | DROP | JSON payload restating both quote dates plus discount line items. **[verified]** `taxes`, `service_fee`, `cleaning_fee` were null in every sample inspected, so the fee-inclusive-pricing motivation from the old report does not apply here. |

### 4f. Stay-length rules

| # | Column | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 53 | `minimum_nights` | 0/0/0 | **KEEP** | **[official]** Filter input. **[verified]** `> 30` for only 0.50 / 0.01 / 0.16 % of rows. |
| 54 | `maximum_nights` | 0/0/0 | **KEEP** | Often a default (365 / 1125). |
| 55–58 | `minimum_minimum_nights`, `maximum_minimum_nights`, `minimum_maximum_nights`, `maximum_maximum_nights` | 0/0/0 | DROP | **[official — calculated]** Collinear with the base fields. |
| 59 | `minimum_nights_avg_ntm` | 0/0/0 | **KEEP** | The one calendar-rule representative. Computed over the forward year, but it describes **host policy**, not booked/blocked outcomes — not label-adjacent. Revisit if it ever behaves like one. |
| 60 | `maximum_nights_avg_ntm` | 0/0/0 | DROP | Redundant. |

### 4g. Availability — the label in column form

| # | Column | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 61 | `calendar_updated` | **100** | DROP | Empty. |
| 62 | `has_availability` | 0.3/0.2/0.2 | DROP | **[verified]** Constant (`t`); only its nullity varies, and that tracks the label. |
| 63 | `availability_30` | 0/0/0 | **BLOCK** | **[official — calculated]** |
| 64 | `availability_60` | 0/0/0 | **BLOCK** | |
| 65 | `availability_90` | 0/0/0 | **BLOCK** | **The label cross-check.** **[verified]** Equals the calendar-derived available-night count over each listing's own first 90 days for **99.96 / 99.99 / 99.97 %** of listings (mean abs diff ≤ 0.03). Use it to validate `label.py`; never as a feature. |
| 66 | `availability_365` | 0/0/0 | **BLOCK** | |
| 67 | `calendar_last_scraped` | 0/0/0 | DROP | Duplicate of `last_scraped`. |
| 71 | `availability_eoy` | 0/0/0 | **BLOCK** | **[verified]** 0–186 = scrape date → 31 Dec. Forward availability window containing the label window. Also not comparable across scrape months, so useless even without the leak. |

**[official]** Standing caveat on all of these: "a listing may not be available because it has
been booked by a guest or blocked by the host." Low availability means high demand **or** a host
taking the listing offline. This is exactly why the target is a *demand proxy*.

### 4h. Reviews — counts and dates

| # | Column | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 68 | `number_of_reviews` | 0/0/0 | **KEEP** | **[official]** Lifetime; heavily right-skewed. |
| 69 | `number_of_reviews_ltm` | 0/0/0 | **KEEP** | **[verified]** Matches a trailing-365-day count from `reviews.csv` at 95.5 / 99.5 / 98.7 % (residual explained by scrape timing and review lag). |
| 70 | `number_of_reviews_l30d` | 0/0/0 | **KEEP** | Very sparse. |
| 72 | `number_of_reviews_ly` | 0/0/0 | **KEEP** | **[verified]** Exactly the calendar-2025 review count (100.0 %, all cities). **Entirely pre-T**, so it is leakage-free, and `ltm − ly` gives a clean momentum feature. |
| 75 | `first_review` | 12.4/13.5/20.0 | **KEEP** | **[official]** Null ⇔ no reviews. Also filter (2) input. |
| 76 | `last_review` | 12.4/13.5/20.0 | **KEEP** | |
| 90 | `reviews_per_month` | 12.4/13.5/20.0 | **KEEP** | **[official — calculated]** Lifetime average, not recent rate. The San Francisco Model's demand proxy — expect it to correlate with our label; that is the validation, not a leak. |

### 4i. Review scores

| # | Columns | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 77–83 | `review_scores_rating`, `_accuracy`, `_cleanliness`, `_checkin`, `_communication`, `_location`, `_value` | 12.4/13.5/20.0 | **KEEP** | **[official — scraped]** 0–5 scale in current data. **[verified]** All seven share exactly the null pattern of `first_review`/`last_review`/`reviews_per_month` — one no-reviews block. Left-skewed and ceiling-inflated; highly inter-correlated. **Do not impute zeros**: missingness is correlated with low review count, i.e. with popularity, so naive imputation leaks a demand signal. Carry an explicit `has_reviews` flag and let LightGBM handle the NaNs. |

### 4j. Regulatory and host-portfolio

| # | Column | Null % | Disposition | Note |
| --- | --- | --- | --- | --- |
| 84 | `license` | 2.8/3.6/4.3 | **DERIVE** | **[official]** Greek AMA registry, so unusually well populated. → `license_status` ∈ {registered, exempt, missing} (`Exempt` appears 40 / 146 / 327 times) **plus** a salted hash of the number. **[verified]** Duplicate licence numbers cover 369 / 2,204 / **10,925** listings — a strong commercial-operator signal, and the reason the value is hashed rather than discarded. |
| 85 | `instant_bookable` | **100** | DROP | Empty — **[official]** called "an indicator of a commercial listing", now unavailable. |
| 86 | `calculated_host_listings_count` | 0/0/0 | **KEEP** | **[official — calculated]** Inside Airbnb's own within-region count. Key commercialisation feature. |
| 87–89 | `..._entire_homes`, `..._private_rooms`, `..._shared_rooms` | 0/0/0 | **KEEP** | As above, split by room type. |
| 73 | `estimated_occupancy_l365d` | 0/0/0 | **BLOCK** | **[official — methodology]** San Francisco Model output, derived from review counts. Would leak into a demand target. Admissible only as a caveated comparison in notebook 02. |
| 74 | `estimated_revenue_l365d` | 5.0/1.0/6.6 | **BLOCK** | ≈ occupancy × price, so it inherits both the model's assumptions and `price`'s endogeneity. |

---

## 5. Price is endogenous — the rule that follows

`price` is not a standing rate. It is the per-night figure from a quote for a specific dated stay,
and that date is chosen by availability (§3). Two distinct problems, with different severities:

**Missingness: severe.** `price` is null in 5.0 / 1.0 / 6.6 % of rows, and those rows are heavily
enriched for fully-blocked listings — **[verified]** 84.2 / 14.5 / 67.5 % of price-null rows have
`availability_90 == 0`; where the quote *date* is null, mean blocked-fraction-90 is 0.98–0.99.
A "price is missing" indicator would be close to a free look at the top of the label distribution.

Therefore: **impute** `price` from the `city × neighbourhood_cleansed × room_type × accommodates`
median, and **never expose a missingness flag as a feature**. Passing NaN through to LightGBM is
*not* neutral here — its native missing-value handling would learn a split on exactly the leak.
Rows still needing imputation after the inactive-listing filter: **188 / 47 / 1,231**
(3.8 / 0.3 / 4.5 %). Dropping them instead would systematically delete the highest-demand
listings, which is worse.

**Value contamination: mild.** Quotes for different listings fall on different dates, and quote
date correlates with the label — so in principle price is measured at label-dependent points in
the season. Measured within `room_type × accommodates × neighbourhood` strata, median price
relative to the stratum median moves only 0.98–1.05 (Thessaloniki), 0.95–1.01 (Athens),
0.78–1.07 (Crete) across quote-lead buckets. Raw medians look far worse in Crete (€132 → €223)
but that is listing composition, not the quote date.

So the price *level* is usable with a stated caveat; the price *date* and price *missingness* are
not usable at all. Price tiers for grading should be built within city (and probably room type)
so the residual seasonal drift does not become a cross-market artefact.

---

## 6. Calendar, reviews, neighbourhoods

### `calendar.csv.gz` — 5 columns

| Column | Disposition | Note |
| --- | --- | --- |
| `listing_id` | **HASH** | Same salt and function as `listings.id`, or every join breaks. |
| `date` | **KEEP** | **[verified]** Exactly 365 rows per listing, starting at that listing's scrape date. `min(date)` per listing is **T**. |
| `available` | **KEEP** | `t`/`f` → bool. The label's only input. **[official]** Conflates booked with host-blocked. |
| `minimum_nights` | DROP | Per-date; falls inside the label window. Listing-level `minimum_nights` is kept instead. |
| `maximum_nights` | DROP | As above. |

There is **no `price` or `adjusted_price` column**. The contract's earlier "keep calendar price,
decide later whether a window-average asking price is a legitimate feature" question is void
— the data does not exist. Nothing in this project can use a per-date price.

**[verified] Join integrity:** Athens' calendar contains **5 `listing_id`s with no matching
listings row**; Thessaloniki and Crete match exactly. Notebook 01 must assert this and the
resolution must be explicit (almost certainly: drop the orphans).

### `reviews.csv.gz` — 6 columns

| Column | Disposition | Note |
| --- | --- | --- |
| `listing_id` | **HASH** | |
| `id` | **KEEP** | Review identifier. |
| `date` | **KEEP** | **[official]** Reviews lag stays by up to 14 days (double-blind window: both parties have 14 days after checkout, and reviews post once both submit or the window closes). |
| `reviewer_id` | DROP | PII. |
| `reviewer_name` | DROP | PII. |
| `comments` | **KEEP** | Multilingual free text; the sentiment demonstration reads it. May contain automated cancellation notices. |

**[official]** Only public reviews are published, and roughly 50 % of stays leave one — the basis
of the occupancy model below.

### `neighbourhoods.csv` — 2 columns

`neighbourhood_group` (empty in all three cities) and `neighbourhood`: a lookup list of the
categories used by `neighbourhood_cleansed`. Thessaloniki's are Latin transliterations; Athens'
and Crete's are Greek, Athens' in uppercase — watch encoding and casing when joining or plotting.
The matching `neighbourhoods.geojson` was not downloaded; it would be needed for point-in-polygon
work or choropleths.

---

## 7. Inside Airbnb methodology worth carrying forward

**[official]** Verbatim from the Data Assumptions page, retained because these constraints shape
the whole project:

- **Snapshot, not history.** "The data presented here is a snapshot of listings available at a
  particular time." Listings can be deleted; snapshots rotate off the site. Our stored copy plus
  its SHA-256 manifest is the only route to reproduction.
- **Location is pre-anonymised.** "the location for a listing on the map, or in the data will be
  from 0-450 feet (150 metres) of the actual address." Listings in one building are jittered
  individually and may look scattered.
- **Availability is understated.** "The Airbnb calendar for a listing does not differentiate
  between a booked night vs an unavailable night… This serves to understate the Availability
  metric because popular listings will be 'booked' rather than being 'blacked out' by a host."
- **Neighbourhoods are recomputed** by Inside Airbnb from coordinates against public shapefiles,
  because Airbnb's own names are "not used because of their inaccuracies".
- **The San Francisco Model** behind `estimated_occupancy_l365d` / `estimated_revenue_l365d`:
  "A Review Rate of 50% is used to convert reviews to estimated bookings"; average length of stay
  is city-configured with a 3-night default; "If a listing has a higher minimum nights value than
  the average length of stay, the minimum nights value was used instead"; and "The occupancy rate
  was capped at 70%." These are modelled estimates, disputed by Airbnb, and must never be
  presented as measured bookings or income.
- **Licence:** the data is CC BY 4.0 — attribution is required in the README.

### Biases to name in any write-up

1. **Far-future availability bias.** Nights further from T are unbooked partly because they are
   further out. Our 90-day window keeps this bounded but does not remove it.
2. **Blocked ≠ booked.** The label is a *demand proxy*. Never call it bookings.
3. **Peak-season framing.** T ≈ end of June, so the window is July–September — the season where
   blocked-because-booked most plausibly dominates blocked-because-closed. Crete, a strongly
   seasonal market, should look different from the two cities; that is a validation signal, not
   a bug.
4. **Asking-price bias.** Scraped prices are what hosts ask, not what guests paid.
5. **Review-score missingness is popularity-correlated** — see §4i.

### Query-group sizing, a problem the group key has to solve

**[verified]** The intended group key (`city × neighbourhood_cleansed × room_type × capacity
tier`) will fragment badly: Thessaloniki has only **7** neighbourhoods, and Shared room has
**3 / 30 / 18** rows across the three cities, with Hotel room absent from Thessaloniki entirely.
Cross that with capacity tiers and many groups will hold one listing — singleton groups
contribute nothing to NDCG and waste LambdaMART's gradient. Group construction needs a minimum
size rule and probably a collapse of rare `room_type`s. The choice belongs with the group key; the
constraint is recorded here so it is not discovered late.

---

## 8. Reproducing these numbers

Every percentage and match rate above was measured against
`data/raw/<city>/<release_date>/` on 2026-07-25 with the snapshots recorded in
`src/rental_ranking/data/download.py` and fingerprinted in each folder's `manifest.json`. If a
newer snapshot is ever downloaded, the numbers here describe the old one — re-profile before
trusting them, because Inside Airbnb changes schema between monthly scrapes without notice
(v4.3 → v4.7 removed `weekly_price`, `monthly_price`, `security_deposit`, `cleaning_fee`,
`guests_included`, `extra_people`, and, as we found, the entire calendar price schedule).
