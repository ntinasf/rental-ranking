# Endpoint demonstration

The managed online endpoint was deployed, exercised and deleted in one session, because an
instance bills by the hour whether or not a request arrives. So this directory *is* the
demonstration, and it has to answer the question a screenshot of a live service cannot: not that
an endpoint existed, but that the thing it served ranks.

| file | what it is |
|---|---|
| `request_athens.json`, `request_crete.json`, `request_thessaloniki.json` | one query each, the full candidate set, 61 features per listing |
| `request_cold_start.json` | the Thessaloniki query with every review field sent as `null` |
| `request_unknown_level.json` | the Thessaloniki query with one listing's `room_type` set to `Houseboat` |
| `response_*.json` | exactly what came back |
| `RESULTS.md` | each response joined to the **held-out grades**, so the ordering sits beside the truth |
| `endpoint.json` | the endpoint descriptor, for the record |

## Why these queries

The rule was fixed before anything was scored: **within the sealed fold, the largest query group
of at most 30 listings in each city, ties to the lower group id.** Thirty because the response has
to fit on a screen, per city because the model's quality differs by city.

Picking a query after seeing its NDCG would be the ranking equivalent of reporting the best seed.
The rule returns one query the model handles poorly, Athens at **0.5976**, and that one is kept.

Every listing involved is in fold 0, held out of training, of the hyperparameter sweep and of
every model-selection decision.

The **counterfactual** is the part a static ranking cannot fake. The top-ranked listing is sent a
second time with its review history stripped and everything else held. In Thessaloniki it falls
from rank 1 to rank 15 of 23; in Crete it does not move. That contrast is the honest picture of
what the model leans on, and it agrees with what notebook 04 measured.

## Running it

Everything here runs against the local scoring script, with no Azure account and no cost:

```bash
uv run python -m rental_ranking.cloud.demo --query thessaloniki --local --counterfactual
uv run python -m rental_ranking.cloud.demo --query athens --variant cold-start --local
```

`--local` calls the same `init()` and `run()` the container calls, on the same bundle. It is the
reference the cloud responses were diffed against.

A local console searches on the same four fields a query group is built from, then lays the
returned order against the held-out grades:

```bash
uv run python -m rental_ranking.cloud.console --local
```

**Both need the data layer built first.** They read `data/train/serving_bundle/` and the feature
table, which are gitignored derivatives, so a bare clone has nothing to score. `README.md` at the
repository root has the build sequence, and it needs the raw Inside Airbnb snapshots.

`--capture` rewrites `RESULTS.md`. Against a live endpoint that is the intended use. With
`--local` it refuses, because the committed transcript was captured against an endpoint that no
longer exists and could not be regenerated.

## What the console does not cover

The picker offers **held-out listings only**, so every score it shows is out of sample: 112
searches across the 74 sealed groups. The cost of that is stated on the page rather than hidden.
**17 of 75 neighbourhoods are unsearchable, 34.8 % of listings**, including the largest in every
city. Central Thessaloniki alone is 4,162 listings, 89 % of that city. The grouped split moves
whole connected components, and a large neighbourhood *is* a large component, so it lands in
training entire. Offering those groups anyway would mean scoring data the model fitted, where the
ordering is a memory rather than a prediction.

**41 % of searches land in a pooled group**, because fewer than five sealed listings shared the
exact key and the cascade re-keyed them at a coarser rung. The banner says so, names how many
neighbourhoods the group really spans, and highlights the listings that matched the literal
search.

Two further constraints are deliberate. The console **edits a real listing rather than composing
one**, since an invented row carries no grade and its ranking cannot be checked. And `city` and
`room_type` are shown but not editable: they are the query-group key, constant inside a group, so
changing either builds a candidate the search never contained.

## The container image

`docker/Dockerfile` packages the console with the booster, the sealed fold already resolved and a
precomputed coverage report, so it starts without the processed data layer. **This is not the
scoring image.** Azure ML builds its own from `pipelines/score_environment.yml`, which carries
`azureml-inference-server-http` and nothing else. Two images, because they answer to different
callers.

Versions are pinned to the environment that produced every number in the write-up, and the
container reproduces group 24 at **0.809461900603779**, digit for digit.

Building it needs `data/train/serving_bundle/` and `data/train/demo_bundle/`, both gitignored, so
the image is reproducible from this repository plus the raw snapshots rather than from a bare
clone. It redistributes Inside Airbnb data under CC BY 4.0 and carries the attribution in the
page footer.

## Elsewhere

- Screenshots from the live session, including a search the model loses to chance, are in
  [`../screenshots/`](../screenshots/) and discussed in [the report](../report.md).
- The redeploy runbook is [`../azure_setup.md`](../azure_setup.md), step 6; the teardown
  discipline is step 8.
