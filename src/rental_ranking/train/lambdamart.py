"""LightGBM LambdaMART training script, run as an Azure ML command job.

objective=lambdarank, metric=ndcg, MLflow autologging. The group-size array's
order must exactly match row order — misalignment fails silently; always assert
sum(groups) == n_rows and hand-check one group.
"""

# TODO: load the versioned feature table; log the dataset version as an MLflow tag.
# TODO: build group arrays from query groups; assert sum(groups) == n_rows.
# TODO: train LGBMRanker (lambdarank, ndcg) on the temporal training window.
# TODO: count zero-variance groups; many flat groups means revisiting grading quantiles.
# TODO: entry point usable both locally and via pipelines/train_job.yml.
