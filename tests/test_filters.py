"""Skeleton tests for rental_ranking.data.filters — fleshed out once the logic exists."""

import pytest

pytestmark = pytest.mark.skip(reason="filter logic not implemented yet")


def test_zero_reviews_and_fully_blocked_listing_is_removed():
    """Zero-reviews-ever plus 100% blocked calendar means inactive/personal use."""


def test_listing_with_first_review_inside_label_window_is_removed():
    """Partial exposure during the label window makes the label untrustworthy."""


def test_listing_with_minimum_nights_above_threshold_is_removed():
    """minimum_nights > ~30 indicates long-term rental, out of scope."""


def test_active_listing_with_reviews_is_kept():
    """A plausible short-term listing passes all filters unchanged."""


def test_filter_counts_are_reported_per_city_and_per_rule():
    """Filters return removal counts per city for the reporting requirement."""
