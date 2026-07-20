"""Price+rating heuristic ranker — the frozen baseline.

Computed per query group before any model is trained; NDCG@10 and Recall@10 are
frozen before training anything. A strong baseline is a finding, not a failure.
"""

# TODO: rank listings within each query group by a simple price+rating heuristic.
# TODO: score the baseline with the shared evaluation module (evaluate.metrics).
# TODO: freeze and persist the baseline results for the headline comparison.
