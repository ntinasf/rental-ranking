"""Skeleton tests for rental_ranking.features.label — fleshed out once the logic exists."""

import pytest

pytestmark = pytest.mark.skip(reason="label logic not implemented yet")


def test_occupancy_fraction_is_bounded_between_zero_and_one():
    """Trailing-90d occupancy must lie in [0, 1] for every listing."""


def test_occupancy_window_is_trailing_from_each_citys_snapshot_date():
    """The 90-day window ends at the city's own snapshot date, not a global date."""


def test_grades_are_assigned_within_price_tier_quantiles():
    """A given occupancy can map to different grades in different price tiers."""


def test_grades_cover_zero_through_four():
    """Bucketing produces integer grades 0-4 and nothing outside that range."""


def test_features_never_use_data_from_the_label_window():
    """Temporal split: feature inputs end at day T; the label window starts after T."""
