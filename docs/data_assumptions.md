# Inside Airbnb Data Dictionary: A Complete, Column-by-Column Reference for Recommender-System Feature Engineering

## TL;DR
- Inside Airbnb publishes six file types per city/region — `listings.csv` (detailed, ~75 columns), `listings.csv` (summary, 18 columns), `calendar.csv` (7 columns, one row per listing-day), `reviews.csv` (detailed, 6 columns) and its summary variant, plus `neighbourhoods.csv`/`neighbourhoods.geojson` — and the authoritative field definitions live in the official "Inside Airbnb Data Dictionary" Google Sheet (docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4) linked as "View the Data Dictionary" from the Data Assumptions page.
- The single most important provenance fact: some columns are scraped verbatim from Airbnb (price, room_type, amenities, review scores, host flags) while others are **calculated by Inside Airbnb** (marked "y" in the dictionary's "Calculated" column: neighbourhood_cleansed, all availability_x, minimum/maximum-nights aggregates, calculated_host_listings_count, reviews_per_month, and the estimated occupancy/revenue fields), and these calculated fields carry documented modeling assumptions you must understand before using them as features.
- For ML you should treat `price` as heavily right-skewed (log-transform), the review-score fields as left-skewed/ceiling-inflated and frequently null for listings with few reviews, availability fields as understated (a booked night and a host-blocked night are indistinguishable), and `neighbourhood_cleansed`/`latitude`/`longitude` as the trustworthy location features rather than the raw `neighbourhood` field.

## Key Findings

**Provenance model.** The official data dictionary is a Google Sheet referenced from insideairbnb.com/data-assumptions. It contains dated, versioned tabs: `listings.csv detail v4.7`, `v4.3` (introduced August 2022), `v4`, `v3`, `listings.csv summary v2`/`v1`, `reviews.csv v1`, and `calendar.csv v2`/`v1`. The dictionary has four columns per field: **Field, Type, Calculated (y = derived by Inside Airbnb, blank = scraped from Airbnb), Description, Reference.** Everything below preserves that provenance: I explicitly mark each field as **[Official — scraped]**, **[Official — calculated by Inside Airbnb]**, or **[Industry/inferred]** when the definition comes from Airbnb's Help Center or general vacation-rental usage rather than Inside Airbnb's own dictionary.

**Data-generation caveats that affect every file (verbatim from the official Data Assumptions page).**
1. Data is a snapshot at scrape time: "The data presented here is a snapshot of listings available at a particular time." Listings can be deleted.
2. Location anonymization: "the location for a listing on the map, or in the data will be from 0-450 feet (150 metres) of the actual address. Listings in the same building are anonymized by Airbnb individually, and therefore may appear 'scattered' in the area surrounding the actual address."
3. Availability understatement: "The Airbnb calendar for a listing does not differentiate between a booked night vs an unavailable night... This serves to understate the Availability metric because popular listings will be 'booked' rather than being 'blacked out' by a host."
4. Neighbourhood names are recomputed by Inside Airbnb from geographic coordinates against public shapefiles, not taken from Airbnb (which are "not used because of their inaccuracies").
5. Occupancy/revenue estimates use the "San Francisco Model": "A Review Rate of 50% is used to convert reviews to estimated bookings... 50% was chosen as it sits almost exactly between 72% and 30.5%"; average length of stay is city-configured ("Where no public statements were made about average stays, a value of 3 nights per booking was used"; e.g., "Airbnb reported 5.5 nights as the average length of stay for guests using Airbnb in San Francisco"; if minimum_nights is higher than the average stay, minimum_nights is used); and "The occupancy rate was capped at 70% - a relatively high, but reasonable number for a highly occupied 'hotel'." The alternative 30.5% review rate is "based on comparing public data of reviews to the The New York Attorney General's report on Airbnb released in October 2014."

## Details

### 1. Listings Data — `listings.csv` (detailed file, v4.3/v4.7)

This is the primary file for a recommender system. Columns are grouped logically below.

#### 1a. Identifiers, scrape metadata, and free text
- **id** — Airbnb's unique identifier for the listing. Type: integer. **[Official — scraped]**. Primary key; joins to `listing_id` in calendar and reviews. Caveat: an academic study (Alsudais 2020) found that in some scrapes an "experience" and a "place" could share the same numeric id, contaminating review joins — verify one-to-one joins.
- **listing_url** — The Airbnb web page for the listing. Type: text (URL). **[Official — calculated]** (constructed from id).
- **scrape_id** — The Inside Airbnb "scrape" batch this row belongs to. Type: bigint. **[Official — calculated]**.
- **last_scraped** — UTC date/time the listing was scraped. Type: datetime. **[Official — calculated]**.
- **source** — One of "neighbourhood search" (found by searching the city) or "previous scrape" (seen in a scrape within the last 65 days and confirmed still live on the Airbnb site). Type: text. **[Official — scraped/derived]**. A recently added field (per the dictionary's change control).
- **name** — Listing title. Type: text. **[Official — scraped]**. Useful for NLP features.
- **description** — Detailed listing description. Type: text. **[Official — scraped]**.
- **neighborhood_overview** — Host's free-text description of the neighbourhood. Type: text. **[Official — scraped]**. Frequently null.
- **picture_url** — URL to the Airbnb-hosted image. Type: text (URL). **[Official — scraped]**.

#### 1b. Host attributes
- **host_id** — Airbnb's unique host/user identifier. Type: integer. **[Official — scraped]**. Use to group listings by host.
- **host_url** — Airbnb page for the host. Type: text (URL). **[Official — calculated]**.
- **host_name** — Host's name, usually first name(s) only. Type: text. **[Official — scraped]**.
- **host_since** — Date the host/user account was created (for hosts who were first guests, this may be their guest signup date). Type: date. **[Official — scraped]**. Good for a "host tenure" feature.
- **host_location** — Host's self-reported location. Type: text. **[Official — scraped]**. Noisy free text; frequently null; useful for a "host is local vs. remote" flag.
- **host_about** — Host's self-description. Type: text. **[Official — scraped]**.
- **host_response_time** — Categorical bucket of how quickly the host responds. Type: categorical text. **[Official — scraped]**. Values are Airbnb's buckets: "within an hour", "within a few hours", "within a day", "a few days or more" (and null). **[Industry — Airbnb Help Center article 430]**: response time is the average time to respond to new messages over the past 30 days; response *rate* (not time) is what most affects Superhost status and search placement. Frequently null for listings with little inquiry activity.
- **host_response_rate** — Percentage of new inquiries/requests the host responded to within 24 hours. Type: percentage stored as a string with "%" (e.g., "100%"), or "N/A". **[Official — scraped]**. **[Industry — Airbnb Help Center article 430]**: based on the past 30 days (or the 10 most recent threads over 90 days if fewer than 10); the Superhost calculation instead uses the past 365 days. Must be parsed to numeric; high missingness.
- **host_acceptance_rate** — "The rate at which a host accepts booking requests." Type: percentage stored as a string with "%". **[Official — scraped]**. Distinct from response rate (accepting vs. merely replying).
- **host_is_superhost** — Whether the host holds Superhost status. Type: boolean `t`/`f`. **[Official — scraped]**. **[Industry — Airbnb Superhost program]**: requires an overall rating of 4.8+, a response rate of 90%+ (within 24 hours), a cancellation rate below 1%, and a minimum of 10 completed stays (or 3 reservations totaling at least 100 nights), evaluated quarterly (Jan 1, Apr 1, Jul 1, Oct 1) over a trailing 365-day window. Strong quality signal but correlated with review scores and review count.
- **host_thumbnail_url**, **host_picture_url** — URLs to host profile images. Type: text (URL). **[Official — scraped]**.
- **host_neighbourhood** — Host's self-reported neighbourhood. Type: text. **[Official — scraped]**. Noisy; differs from the listing's cleansed neighbourhood.
- **host_listings_count** — Number of listings the host has, "per Airbnb unknown calculations." Type: reported as text/integer. **[Official — scraped]**. Airbnb's own count; may include listings outside the current city and non-home listings.
- **host_total_listings_count** — Also "the number of listings the host has (per Airbnb unknown calculations)." Type: text/integer. **[Official — scraped]**. The precise difference from host_listings_count is undocumented; the two are near-duplicates and highly correlated — do not use both.
- **host_verifications** — List of verification methods (e.g., email, phone). Type: array/JSON-like string. **[Official — scraped]**. Needs parsing.
- **host_has_profile_pic** — Boolean `t`/`f`. **[Official — scraped]**.
- **host_identity_verified** — Boolean `t`/`f`. **[Official — scraped]**.

#### 1c. Location
- **neighbourhood** — The raw neighbourhood text (tied to the host's `neighborhood_overview`). Type: text. **[Official — scraped]**. Inside Airbnb explicitly warns Airbnb neighbourhood names are inaccurate — prefer the cleansed field. Frequently null.
- **neighbourhood_cleansed** — Neighbourhood assigned by geocoding the listing's latitude/longitude against open/public digital shapefiles. Type: text. **[Official — calculated]**. This is the reliable geographic categorical for modeling.
- **neighbourhood_group_cleansed** — The larger neighbourhood group from the same geocoding. Type: text. **[Official — calculated]**. Often entirely null (only populated where a city defines groups, e.g., NYC boroughs) — for many cities this column is 100% missing.
- **latitude**, **longitude** — WGS84 coordinates. Type: numeric. **[Official — scraped, but anonymized by Airbnb]**. Accurate only to within 0–150 m (0–450 ft) of the true location; fine for neighbourhood-level and distance-to-landmark features, unreliable for exact-address work.

#### 1d. Property characteristics
- **property_type** — Self-selected property type (e.g., "Entire rental unit", "Private room in home", "Boutique hotel"). Type: text. **[Official — scraped]**. High-cardinality; hotels and B&Bs are self-described here. Usually collapsed into fewer categories for modeling.
- **room_type** — One of `Entire home/apt`, `Private room`, `Shared room`, `Hotel room`. Type: categorical text. **[Official — scraped; definitions from Airbnb Help article 5]**. Entire place = whole space to yourself; Private room = own room, some shared spaces; Shared room = sleeping space shared with others; Hotel room = hotel/serviced inventory. One of the strongest price and preference predictors; distributions differ so much by room_type that price outliers should be examined *within* room_type.
- **accommodates** — Maximum guest capacity. Type: integer. **[Official — scraped]**. Strong price driver.
- **bathrooms** — Number of bathrooms. Type: numeric. **[Official — scraped]**. Often null in recent scrapes because Airbnb migrated this to a textual field (see below).
- **bathrooms_text** — Textual bathroom description (e.g., "1 bath", "1.5 shared baths", "Half-bath"). Type: string. **[Official — scraped]**. The dictionary notes the field "evolved from a number to a textual description"; for older scrapes `bathrooms` is used. **[Industry — Airbnb Help article 3424]**: bathrooms may be "private and attached", "dedicated" (private but accessed via a shared space), or "shared"; a half-bath = toilet + sink, no shower/tub. Requires regex parsing to recover a numeric count plus a shared/private flag.
- **bedrooms** — Number of bedrooms. Type: integer. **[Official — scraped]**. Can be null (studios are sometimes coded 0 or null).
- **beds** — Number of beds. Type: integer. **[Official — scraped]**. Frequently null; overlaps conceptually with accommodates.
- **amenities** — JSON array of amenity strings (Wifi, Kitchen, Air conditioning, etc.). Type: JSON. **[Official — scraped]**. Requires multi-hot encoding; cardinality is very high and amenity naming is inconsistent across scrapes. Rich source of binary features.

#### 1e. Price
- **price** — Daily price in the local currency. Type: stored as a string with a "$" sign and thousands separators (e.g., "$1,250.00"). **[Official — scraped]**. The dictionary note: "the $ sign is a technical artifact of the export, please ignore it." **Must be cleaned** (strip $ and commas, cast to float). Statistically it is heavily right-skewed with extreme outliers and a floor at zero; the mode is around $100/night in many US cities; log-transformation is standard before linear modeling and outlier trimming (e.g., top ~1–10%) is common. Represents the host's *asking* price, not realized/booked price — a well-known upward bias versus actual transaction prices. In recent Inside Airbnb data, weekly_price, monthly_price, security_deposit, cleaning_fee, guests_included, and extra_people have been **removed** (they existed in older schemas), so do not expect them.

#### 1f. Minimum/maximum nights
- **minimum_nights** — Minimum night stay for the listing. Type: integer. **[Official — scraped]**. Dictionary note: "minimum number of night stay for the listing (calendar rules may be different)." A high value (e.g., 30+) often signals a de facto long-term rental; feeds into the occupancy model. Can contain unusually large values.
- **maximum_nights** — Maximum night stay. Type: integer. **[Official — scraped]**. Often a default like 365 or 1125.
- **minimum_minimum_nights** / **maximum_minimum_nights** — Smallest / largest minimum-night value seen in the next 365 calendar days. Type: integer. **[Official — calculated]**.
- **minimum_maximum_nights** / **maximum_maximum_nights** — Smallest / largest maximum-night value over the next 365 days. Type: integer. **[Official — calculated]**.
- **minimum_nights_avg_ntm** / **maximum_nights_avg_ntm** — Average minimum / maximum night value over the next 365 days ("ntm" = next twelve months). Type: numeric. **[Official — calculated]**. These six aggregate fields are collinear with the base min/max night fields; typically keep only one representative.

#### 1g. Availability
- **calendar_updated** — When the host last updated the calendar. Type: date. **[Official — scraped]**. In modern extracts this is almost always null/deprecated.
- **has_availability** — Whether the listing has any availability. Type: boolean `t`/`f`. **[Official — scraped/derived]**.
- **availability_30 / _60 / _90 / _365** — Number of days available for booking in the next 30/60/90/365 days per the calendar. Type: integer. **[Official — calculated]**. Dictionary note: "a listing may not be available because it has been booked by a guest or blocked by the host" — so low availability can mean high demand OR a host taking the listing offline; the two are indistinguishable. This is a key modeling caveat: availability is a **noisy, understated** proxy for occupancy. The four windows are nested and highly correlated.
- **calendar_last_scraped** — Date the calendar was scraped. Type: date. **[Official — calculated]**.

#### 1h. Reviews — counts and dates
- **number_of_reviews** — Total lifetime reviews. Type: integer. **[Official — scraped]**. Often used as a demand/booking proxy (Inside Airbnb assumes ~50% of stays leave a review). Heavily right-skewed (many listings with 0).
- **number_of_reviews_ltm** — Reviews in the last 12 months. Type: integer. **[Official — calculated]**. Better recency signal than lifetime count.
- **number_of_reviews_l30d** — Reviews in the last 30 days. Type: integer. **[Official — calculated]**. Very sparse.
- **first_review** — Date of the oldest review. Type: date. **[Official — calculated]**. Null when there are no reviews.
- **last_review** — Date of the newest review. Type: date. **[Official — calculated]**. Null when there are no reviews.

#### 1i. Review scores
All seven are scraped from Airbnb. **Important scale note:** the v4.3 dictionary tab leaves the Type/Description **blank** for these fields, so the scale is *not stated in Inside Airbnb's own dictionary*. In **current** data all seven are on Airbnb's **0–5 star scale** (e.g., 4.85), confirmed by Airbnb's own Help Center (guests rate an overall experience plus six categories on a 1–5 scale). In **older** Inside Airbnb data (pre-2019), `review_scores_rating` was on a **0–100** scale while the six sub-scores were on a **0–10** scale (e.g., Statista cites Hong Kong listings averaging "92 points out of 100" as of July 2020) — a critical discontinuity if you concatenate historical scrapes.
- **review_scores_rating** — Overall rating. Type: numeric. **[Official — scraped; scale from Airbnb Help Center]**. **[Industry — Airbnb]**: the overall star is an *independent* rating, NOT the average of the six subcategories. Left-skewed/ceiling-inflated — the large majority of listings sit above 4.5 stars, so variance is low and the field behaves almost like a "penalty" signal. Null unless the listing has at least the minimum reviews Airbnb requires to display a score (3).
- **review_scores_accuracy** — How accurately the listing matched reality. Type: numeric (0–5). **[Official — scraped]**.
- **review_scores_cleanliness** — Cleanliness. Type: numeric (0–5). **[Official — scraped]**. The subscore guests weigh most heavily.
- **review_scores_checkin** — Check-in experience. Type: numeric (0–5). **[Official — scraped]**.
- **review_scores_communication** — Host communication. Type: numeric (0–5). **[Official — scraped]**.
- **review_scores_location** — Location. Type: numeric (0–5). **[Official — scraped]**.
- **review_scores_value** — Value for money. Type: numeric (0–5). **[Official — scraped]**. Typically the lowest-scoring category on average. All six subscores are highly inter-correlated (multicollinearity risk) and share the same missingness pattern (null when the listing has no/few reviews) — this null-correlated-with-review-count structure means naive imputation can leak the target in a popularity model.

#### 1j. Regulatory, booking, and host-portfolio fields
- **license** — The licence/permit/registration number. Type: text. **[Official — scraped]**. Very frequently null (in many cities 100% missing); where a regulatory regime exists it may be populated or contain values like "Exempt". Useful as a binary "has license" feature more than as a value.
- **instant_bookable** — Whether a guest can book without host approval. Type: boolean `t`/`f`. **[Official — scraped]**. Dictionary calls it "an indicator of a commercial listing."
- **calculated_host_listings_count** — Number of listings this host has **in the current scrape, within the city/region geography**. Type: integer. **[Official — calculated]**. This is Inside Airbnb's own count, and it differs from host_listings_count (which is Airbnb's opaque, possibly global count). Key commercialization feature.
- **calculated_host_listings_count_entire_homes** — Same, restricted to Entire home/apt. Type: integer. **[Official — calculated]**.
- **calculated_host_listings_count_private_rooms** — Same, Private rooms. Type: integer. **[Official — calculated]**.
- **calculated_host_listings_count_shared_rooms** — Same, Shared rooms. Type: integer. **[Official — calculated]**.
- **reviews_per_month** — Average reviews per month over the listing's lifetime. Type: numeric. **[Official — calculated]**. Dictionary pseudocode: if (scrape_date − first_review) ≤ 30 then number_of_reviews, else number_of_reviews / ((scrape_date − first_review + 1) / (365/12)). Null when there are no reviews. Widely used as the demand/occupancy proxy in the San Francisco Model.

#### 1k. Newer estimated fields (v4.7)
Recent listings dictionary versions add Inside-Airbnb-calculated estimates not present in v4.3:
- **estimated_occupancy_l365d** — Estimated number of booked/occupied nights over the last 365 days. Type: integer. **[Official — calculated]**. Derived via the San Francisco Model (reviews × 50% review rate → bookings × average length of stay, capped at 70% occupancy).
- **estimated_revenue_l365d** — Estimated revenue over the last 365 days (estimated occupied nights × price). Type: numeric. **[Official — calculated]**. Both inherit every assumption and bias of the occupancy model (asking-price bias, 50% review-rate assumption, 3-night default stay, 70% cap) and should be treated as coarse estimates, not ground truth. (These definitions are inferred from Inside Airbnb's Data Assumptions methodology; the exact v4.7 dictionary wording could not be programmatically extracted — see Caveats.)

### 2. Summary Listings Data — `listings.csv` (visualisations/summary file, 18 columns)

A slimmed file (in the `visualisations/` folder) meant for maps and quick analysis. All columns are subsets of the detailed file: **id, name, host_id, host_name, neighbourhood_group, neighbourhood, latitude, longitude, room_type, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, calculated_host_listings_count, availability_365, number_of_reviews_ltm, license.** Definitions match the detailed file. Note `neighbourhood_group` is frequently entirely null, and `license` is frequently entirely null. If you already load the detailed file, the summary file is redundant. (In this summary file, `price` is often already cast as an integer rather than a "$" string, so check types.)

### 3. Calendar Data — `calendar.csv` (v2, 7 columns)

One row per listing per future date, covering 365 days forward from the scrape date. Columns:
- **listing_id** — Foreign key to listings.id. Type: integer.
- **date** — The calendar date. Type: date (YYYY-MM-DD).
- **available** — Whether the date is available to book. Type: boolean `t`/`f`. Caveat (verbatim from Data Assumptions): "unavailable" conflates booked nights and host-blocked nights.
- **price** — Daily price for that date in local currency. Type: string with "$".
- **adjusted_price** — Price for that date after calendar/length-of-stay adjustments. Type: string with "$". Caveat: often identical to `price`, and in recent extracts this column is frequently empty/deprecated.
- **minimum_nights** — Minimum stay applying to that specific date. Type: integer.
- **maximum_nights** — Maximum stay applying to that specific date. Type: integer.

All are **[Official]**; `listing_id`, `date`, `available`, `price` are scraped, the rest derived from calendar rules. Because the calendar reflects only the snapshot moment, far-future dates (e.g., next summer scraped in December) look "available" simply because they are too far out to have been booked — a major bias when estimating occupancy from calendar data.

### 4. Reviews Data — `reviews.csv`

**Detailed file (6 columns):**
- **listing_id** — Foreign key to listings.id. Type: integer.
- **id** — Unique review identifier. Type: bigint.
- **date** — Date the review was posted. Type: date.
- **reviewer_id** — Airbnb's unique reviewer/user id. Type: integer.
- **reviewer_name** — Reviewer's first name. Type: text.
- **comments** — Free-text review body. Type: text. Multilingual; the richest field for NLP/sentiment features. May contain automated/cancellation-notice text.

**Summary file** contains only the first five columns (no `comments`) and is used to plot review volume over time. All **[Official — scraped]**, except that Inside Airbnb only publishes public reviews (≈50% of stays leave one — the basis of the occupancy model). Per Airbnb's Help Center, reviews are double-blind: "Both parties will have 14 days after checkout to submit a review. Reviews are only posted after both parties have submitted their reviews, or once the 14-day period has ended—whichever comes first." So review dates lag stays by up to two weeks.

### 5. Neighbourhoods Data — `neighbourhoods.csv` and `neighbourhoods.geojson`

- **`neighbourhoods.csv`** — Two columns: **neighbourhood_group** (often blank) and **neighbourhood**. Type: text. A simple lookup list of the geographic categories used in `neighbourhood_cleansed`. **[Official — calculated]**.
- **`neighbourhoods.geojson`** — A GeoJSON FeatureCollection of neighbourhood boundary polygons/multipolygons. Each Feature has properties **neighbourhood** and **neighbourhood_group** plus a geometry (Polygon/MultiPolygon in WGS84). **[Official — calculated]** from public shapefiles. Use it to compute spatial features (distance to city center, point-in-polygon neighbourhood assignment, choropleths) and to join to `neighbourhood_cleansed`.

## Recommendations

**Stage 1 — Ingestion and typing.** Load the *detailed* `listings.csv`, `calendar.csv`, and detailed `reviews.csv`; skip the summary/visualisation files (redundant). Immediately clean `price` (strip `$` and `,`; cast to float), parse `host_response_rate`/`host_acceptance_rate` from "%" strings to floats, cast all `t`/`f` booleans (host_is_superhost, instant_bookable, host_has_profile_pic, host_identity_verified, has_availability, calendar.available) to 0/1, and parse `amenities` and `host_verifications` from JSON.

**Stage 2 — Handle missingness and provenance deliberately.** Expect the review-score block, `reviews_per_month`, `first_review`, `last_review` to be null together for listings with no reviews — add an explicit `has_reviews` flag rather than imputing zeros into scores. Expect `license`, `neighbourhood_group(_cleansed)`, `bathrooms`, `beds`, `bedrooms`, and `calendar_updated` to be heavily/entirely null in many cities; drop columns that are 100% null for your city. Keep the *calculated* fields (availability_x, calculated_host_listings_count, reviews_per_month, estimated_*) but remember they embed Inside Airbnb's assumptions.

**Stage 3 — Feature engineering choices.** Use `neighbourhood_cleansed` + `latitude`/`longitude` for location (never the raw `neighbourhood`). Log-transform `price` and consider modeling/trimming outliers *within* `room_type`. Collapse `property_type` to a handful of buckets. Multi-hot encode `amenities`. Engineer host-tenure from `host_since` and a "multi-listing/commercial host" feature from `calculated_host_listings_count` and `instant_bookable`. Derive numeric bathroom count + shared/private flag from `bathrooms_text`.

**Stage 4 — Avoid redundancy and leakage.** Drop near-duplicate columns: keep one of {host_listings_count, host_total_listings_count}; keep one representative of the six min/max-night aggregates; keep one or two availability windows rather than all four; the six review subscores are collinear (consider a single composite or PCA). Do not feed `estimated_occupancy_l365d`/`estimated_revenue_l365d` into a model whose target is occupancy or demand — they are derived from `number_of_reviews` and would leak.

**Benchmarks that change the plan.** If your city's `license` or `neighbourhood_group` column is >95% null, drop it. If concatenating multiple scrape dates, check whether any predate ~2019 (review_scores_rating on a 0–100 scale) and rescale before merging. If more than ~30% of rows lack review scores, prefer tree models with native missing-value handling over linear models requiring imputation.

## Caveats
- **Definition provenance:** Column names, types, and "Calculated" flags for the detailed `listings.csv` come verbatim from the official Inside Airbnb Data Dictionary (v4.3 tab, introduced August 2022). The review-score fields, `host_response_time`, `host_response_rate`, and `host_verifications` are left **blank** in that tab, so their scale/format definitions here are sourced from Airbnb's own Help Center and observed data, and are flagged **[Industry/inferred]** accordingly.
- **Version drift:** Inside Airbnb's schema changes over time (v3 → v4 → v4.3 → v4.7). Older files include now-removed columns (weekly_price, monthly_price, security_deposit, cleaning_fee, guests_included, extra_people, is_location_exact, street, city, state, zipcode, market, smart_location, bed_type); newer files add estimated_occupancy_l365d / estimated_revenue_l365d. Always check the header of your specific download against the dictionary tab matching its date.
- **The occupancy/revenue model is an estimate, not measured data.** The 50% review rate, 3-night default stay, and 70% occupancy cap are documented assumptions; academic critiques (e.g., Alsudais 2020; and commentary from David Wachsmuth of McGill, and Gurran & Phibbs of the University of Sydney) note both systemic scraping/collection limitations and that the data nonetheless provides "a useful basis for examining and monitoring Airbnb practices." Treat all derived demand/income figures as directional.
- **Airbnb's own position:** Airbnb disputes the accuracy of scraped data (inactive listings, duplicate listings, and its preference for median over mean income). Inside Airbnb data is best for market/neighbourhood-level structure, not exact per-listing ground truth.
- The exact verbatim wording of the v4.7 tab's new-column descriptions and the calendar/reviews tabs could not be extracted programmatically (Google Sheets serves only the default v4.3 tab without JavaScript); those definitions are corroborated from the Data Assumptions page and multiple independent analyses of the files rather than quoted directly from those dictionary tabs. To capture the exact official wording, open the Google Sheet's v4.7, `reviews.csv v1`, and `calendar.csv v2` tabs in a JavaScript-enabled browser.