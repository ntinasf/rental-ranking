"""NDCG@k, Recall@k, and bootstrap confidence intervals over query groups.

A point estimate without variance is not a result: bootstrap over groups, report
CIs, and break results out per city and by group size.
"""

# TODO: NDCG@k on graded relevance, per query group.
# TODO: Recall@k against relevant = grade >= 3 within group (also used by V2 retrieval).
# TODO: bootstrap over query groups for confidence intervals.
# TODO: result breakdowns: overall, per city, by group size.
