"""Skeleton tests for rental_ranking.features.aggregates — fleshed out once the logic exists."""

import pytest

pytestmark = pytest.mark.skip(reason="aggregate logic not implemented yet")


def test_neighbourhood_median_excludes_the_listing_itself():
    """Leave-one-out: the listing's own value never enters its neighbourhood aggregate."""


def test_single_listing_neighbourhood_has_no_self_referential_aggregate():
    """With one listing in the neighbourhood, leave-one-out must yield NaN/sentinel, not its own value."""


def test_aggregate_changes_when_own_value_changes_only_for_neighbours():
    """Perturbing listing A's price changes neighbours' aggregates but not A's own."""
