# Endpoint demonstration — what is here, and how to reproduce it

The managed online endpoint is deleted the moment the demonstration ends (BUILD_GUIDE gotcha #6:
an instance bills per hour whether or not a request arrives). So this directory *is* the
demonstration. It has to answer the question a screenshot of a live endpoint cannot: **not that a
service existed, but that the thing it served ranks.**

| file | what it is |
|---|---|
| `request_athens.json`, `request_crete.json`, `request_thessaloniki.json` | one query each — the full candidate set, 61 features per listing |
| `request_cold_start.json` | the Thessaloniki query with every review field sent as `null` |
| `request_unknown_level.json` | the Thessaloniki query with one listing's `room_type` set to `Houseboat` |
| `response_*.json` | exactly what came back |
| `RESULTS.md` | each response joined to the **held-out grades** — the ordering with the truth beside it |
| `endpoint.json` | the endpoint descriptor, for the record |

## Why these queries

Chosen by a rule fixed before anything was scored: **within the sealed fold, the largest query
group of at most 30 listings in each city, ties to the lower group id.** 30 because the response
has to fit in a screenshot; per city because the model's quality differs by city.

Picking the query after seeing its NDCG would be the ranking equivalent of reporting the best
seed. The rule returns one query the model handles poorly (Athens, 0.5976) and it is kept.

Every one of them is in **fold 0**, held out of training, of the hyperparameter sweep, and of
every model-selection decision in Phase 3.

## What to read in `RESULTS.md`

Each block is the endpoint's ordering with the truth attached, then four NDCG@10 numbers computed
on the *same* candidate set: the endpoint, the two frozen baselines, and a random floor. **A
single query's NDCG is an anecdote.** The estimate is the sealed fold — 0.7530 [0.7148, 0.7903]
over 72 groups against 0.6429 for price+rating — and the per-query number is there so the
ordering can be checked against the rows, not to be quoted.

The **counterfactual** is the part a static ranking cannot fake: the top-ranked listing is sent a
second time with its review history stripped and everything else held, and the rank it lands on
is reported. In Thessaloniki it falls 1 → 15 of 23; in Crete it does not move. That contrast is
the honest picture of what the model leans on, and it agrees with what notebook 04 measured.

## The console

`python -m rental_ranking.cloud.console` serves a local page that works the way a guest would:

```bash
uv run python -m rental_ranking.cloud.console --local   # no endpoint, no cost
uv run python -m rental_ranking.cloud.console           # proxies to AML_ENDPOINT_URI
```

**Search** picks a city, a neighbourhood, a room type and a party size. That is not a skin over
the demo — it is *literally the query-group key*. `features/groups.py` builds the group from
`city × neighbourhood_cleansed × room_type × capacity_tier` precisely because that is "what a
guest would have typed", so choosing those four selects the candidate set the model ranks. 112
searches resolve across 74 sealed groups.

**41 % of those searches land in a pooled group** — fewer than five sealed listings shared the
exact key, so the cascade re-keyed them at a coarser rung. The console says so in the banner and
highlights the listings that actually matched your search, because a guest who picks a thin
neighbourhood is really competing city-wide, and hiding that would misrepresent the group.

**Edit** any listing in the result: dropdowns carrying the real training levels for the five
categoricals, boxes for the numerics, one-click presets. Re-rank and the table redraws with the
listing's new position, the movement, and the metric before and after.

Two constraints are deliberate. It **edits a real listing rather than composing one** — an
invented row has no grade, so its ranking cannot be checked, and the most such a demo can show is
that the service responds. And `city`/`room_type` are **shown but not editable**: they are the
query-group key, constant inside a group, so changing one builds a candidate the search never
contained.

Re-ranking without touching anything is a true no-op — an untouched box submits the value at full
precision rather than the rounded one it displays — so any movement is attributable to the edit.

It is a **proxy, not a nicety**: a browser cannot call a managed endpoint directly (no CORS
headers on the preflight), and putting the auth key in a page would hand a live credential to
anything the browser loads. The key stays in the Python process. Standard library only.

## Reproducing it

Everything below runs against **the local scoring script**, no Azure account and no cost:

```bash
uv run python -m rental_ranking.cloud.demo --query thessaloniki --local --counterfactual
uv run python -m rental_ranking.cloud.demo --query athens --variant cold-start --local
uv run python -m rental_ranking.cloud.demo --capture --local     # rewrites RESULTS.md
```

`--local` calls the same `init()`/`run()` the container calls, on the same bundle. It is the
reference the cloud responses were diffed against.

## Redeploying, if the demonstration has to be run live again

Full runbook with timings and the meter discipline: `docs/azure_setup.md`, step 8.

```bash
# 1. deploy — the endpoint itself is free; the deployment provisions the billed instance (~15 min)
az ml model create -f pipelines/model_asset.yml   # only if rental-ranker:1 is not registered
az ml online-endpoint  create -f pipelines/endpoint.yml
az ml online-deployment create -f pipelines/deployment.yml --all-traffic

# 2. address -> .env (gitignored; the key dies with the endpoint, so there is nothing to rotate)
az ml online-endpoint show --name rental-ranker --query scoring_uri -o tsv
az ml online-endpoint get-credentials --name rental-ranker --query primaryKey -o tsv

# 3. demonstrate
uv run python -m rental_ranking.cloud.demo --query thessaloniki --counterfactual
uv run python -m rental_ranking.cloud.demo --capture          # writes this directory

# 4. TEAR DOWN — issue it, then confirm. Deletion takes ~8 minutes.
az ml online-endpoint delete --name rental-ranker --yes
az ml online-endpoint list -o table                            # must be empty
```

Left running, `Standard_DS2_v2` at $0.1360/hr is about **$98/month**.
