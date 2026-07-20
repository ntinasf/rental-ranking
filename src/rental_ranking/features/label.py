"""Trailing-90-day occupancy demand proxy, bucketed into graded relevance 0-4.

The label is a demand proxy, never "bookings": the calendar is forward-looking
availability, and blocked days include personal use, maintenance, and seasonality.
Features must only use data available before the label window starts.
"""

# TODO: compute raw occupancy — fraction of blocked days over the trailing 90 days
#       from each city's own snapshot date.
# TODO: validate against independent demand signals (reviews_per_month, recent review
#       counts) — correlation must be clearly positive.
# TODO: decide explicitly how to treat the spikes at 0 and 1 in the distribution.
# TODO: bucket into relevance grades 0-4 using quantiles within price tier.
# TODO: encode the temporal split — features from data up to day T, label window after T.
