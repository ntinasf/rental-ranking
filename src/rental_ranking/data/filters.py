"""Dead/implausible-listing removal, applied before the label is trusted.

Thresholds are re-derived against the new label definition (not copied from the
Thessaloniki project); identical criteria apply to every city, and filter counts
per city are reported.
"""

# TODO: filter listings with zero reviews ever AND 100% blocked calendar (inactive/personal use).
# TODO: filter listings whose first review falls inside the label window (partial exposure).
# TODO: filter minimum_nights above ~30 (long-term rentals, out of scope).
# TODO: return per-city counts of rows removed by each rule, for reporting.
