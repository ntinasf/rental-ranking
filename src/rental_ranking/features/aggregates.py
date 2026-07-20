"""Leave-one-out neighbourhood aggregates (median price, occupancy, etc.).

Aggregates that include the listing itself leak the label — the most common
silent leak in this design. Always exclude the listing (leave-one-out) or
compute on the training window only.
"""

# TODO: leave-one-out neighbourhood median price per listing.
# TODO: leave-one-out neighbourhood occupancy statistics per listing.
# TODO: guard against tiny neighbourhoods where leave-one-out is undefined or noisy.
