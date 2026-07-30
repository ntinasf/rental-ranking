"""Column spec for the Inside Airbnb v4.7 schema — constants only, no logic.

This is the executable form of the column dispositions in docs/data_dictionary.md §4,
which the pipeline contract (docs/data_pipeline_design.md) derives from. Every one of the
90 listings columns appears in exactly one disposition set; ``test_columns.py`` enforces
that, so a schema change in a future snapshot fails loudly instead of silently passing an
unknown column through.

The set that matters most is ``LABEL_ADJACENT_COLUMNS``: those columns are kept in the
processed data for validation, and must never reach a model. Two of them are not obvious —
``price_quote_checkin_date``/``_checkout_date`` are the listing's first *available* calendar
date, i.e. a direct read of the label (see docs/data_dictionary.md §3).
"""

# --- listings -------------------------------------------------------------------------

#: Verified 100% null in Thessaloniki, Athens and Crete (2026-07-25 snapshots).
ALL_NULL_COLUMNS: frozenset[str] = frozenset(
    {
        "calendar_updated",
        "host_acceptance_rate",
        "host_neighbourhood",
        "host_response_rate",
        "host_response_time",
        "host_since",
        "host_thumbnail_url",
        "host_total_listings_count",
        "host_verifications",
        "instant_bookable",
        "neighborhood_overview",
        "neighbourhood",
        "neighbourhood_group_cleansed",
    }
)

#: Direct or indirect personal identifiers, dropped rather than hashed.
PII_DROP_COLUMNS: frozenset[str] = frozenset(
    {
        "host_name",
        "host_picture_url",
        "host_profile_id",  # verified 1:1 with host_id — no information lost
        "host_profile_url",
        "host_url",
        "listing_url",
    }
)

#: Salted SHA-256 (12 hex). Same salt and helper everywhere, or joins break.
HASH_COLUMNS: frozenset[str] = frozenset({"id", "host_id"})

#: Raw columns consumed by a derivation, then dropped. Value = what they become.
DERIVED_FROM: dict[str, str] = {
    "last_scraped": "scrape_date",
    "host_location": "host_is_local",
    "host_about": "host_has_about",
    "license": "license_status + license_hash",
    "hosts_time_as_user_years": "user_tenure_months",
    "hosts_time_as_user_months": "user_tenure_months",
    "hosts_time_as_host_years": "host_tenure_months",
    "hosts_time_as_host_months": "host_tenure_months",
}

#: Dropped for reasons other than emptiness or PII (redundancy, constancy, cost).
REDUNDANT_DROP_COLUMNS: frozenset[str] = frozenset(
    {
        "calendar_last_scraped",  # duplicate of last_scraped
        "description",  # deferred to Phase 4 text features
        "has_availability",  # constant 't'; only its nullity varies, and that tracks the label
        "maximum_maximum_nights",
        "maximum_minimum_nights",
        "maximum_nights_avg_ntm",
        "minimum_maximum_nights",
        "minimum_minimum_nights",
        "picture_url",
        "price_quote_price_per_night",  # exact duplicate of price
        "price_quote_raw",  # restates the quote dates as JSON, ~1 KB/row
        "price_quote_total_price",  # = price x quote nights
        "scrape_id",  # constant per city
        "source",  # no variance in Athens
    }
)

#: Kept in the processed table but banned as model inputs; validation and asserts only.
LABEL_ADJACENT_COLUMNS: frozenset[str] = frozenset(
    {
        "availability_30",
        "availability_60",
        "availability_90",  # the label cross-check: matches the calendar at 99.96-99.99%
        "availability_365",
        "availability_eoy",  # forward window to 31 Dec, contains the label window
        "estimated_occupancy_l365d",
        "estimated_revenue_l365d",
        "price_quote_checkin_date",  # ~= first available date; leaks the label
        "price_quote_checkout_date",
    }
)

#: Kept and usable as features (after typing, and after price imputation — see the contract).
KEEP_COLUMNS: frozenset[str] = frozenset(
    {
        "accommodates",
        "amenities",
        "bathrooms",
        "bathrooms_text",
        "bedrooms",
        "beds",
        "calculated_host_listings_count",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "first_review",
        "host_has_profile_pic",
        "host_id",
        "host_identity_verified",
        "host_is_superhost",
        "host_listings_count",
        "id",
        "last_review",
        "latitude",
        "longitude",
        "maximum_nights",
        "minimum_nights",
        "minimum_nights_avg_ntm",
        "name",
        "neighbourhood_cleansed",
        "number_of_reviews",
        "number_of_reviews_l30d",
        "number_of_reviews_ltm",
        "number_of_reviews_ly",  # verified = calendar-2025 count, entirely pre-T
        "price",
        "property_type",
        "review_scores_accuracy",
        "review_scores_checkin",
        "review_scores_cleanliness",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_rating",
        "review_scores_value",
        "reviews_per_month",
        "room_type",
    }
)

#: The full v4.7 listings header, in file order. Concatenation asserts against this.
LISTINGS_COLUMNS: tuple[str, ...] = (
    "id",
    "listing_url",
    "scrape_id",
    "last_scraped",
    "source",
    "name",
    "description",
    "neighborhood_overview",
    "picture_url",
    "host_id",
    "host_url",
    "host_profile_id",
    "host_profile_url",
    "host_name",
    "host_since",
    "hosts_time_as_user_years",
    "hosts_time_as_user_months",
    "hosts_time_as_host_years",
    "hosts_time_as_host_months",
    "host_location",
    "host_about",
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_thumbnail_url",
    "host_picture_url",
    "host_neighbourhood",
    "host_listings_count",
    "host_total_listings_count",
    "host_verifications",
    "host_has_profile_pic",
    "host_identity_verified",
    "neighbourhood",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "price",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
    "price_quote_total_price",
    "price_quote_price_per_night",
    "price_quote_raw",
    "minimum_nights",
    "maximum_nights",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "calendar_updated",
    "has_availability",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "calendar_last_scraped",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "availability_eoy",
    "number_of_reviews_ly",
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d",
    "first_review",
    "last_review",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "license",
    "instant_bookable",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "reviews_per_month",
)

# --- calendar and reviews -------------------------------------------------------------

#: v4.7 calendar header. Note: no `price`, no `adjusted_price` — they no longer exist.
CALENDAR_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "date",
    "available",
    "minimum_nights",
    "maximum_nights",
)

#: Per-date min/max nights fall inside the label window; listing-level values are kept instead.
CALENDAR_KEEP: frozenset[str] = frozenset({"listing_id", "date", "available"})

REVIEWS_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "id",
    "date",
    "reviewer_id",
    "reviewer_name",
    "comments",
)

REVIEWS_KEEP: frozenset[str] = frozenset({"listing_id", "id", "date", "comments"})

#: Reviewer identity, dropped from reviews.
REVIEWS_PII_DROP: frozenset[str] = frozenset({"reviewer_id", "reviewer_name"})
