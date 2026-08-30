# Rental Ranking

**A search ranking system for short-term rental listings, built on 44,684 listings from public
[Inside Airbnb](https://insideairbnb.com/) data for Athens, Thessaloniki and Crete, trained
locally and deployed on Azure ML.**

When someone searches for a place to stay, hundreds of listings match. This repository builds a
LambdaMART model that decides which ones appear first, measures how well it works, and states
plainly what those measurements can support.

---

## The result

The model orders listings within a search. Scored on a sealed fifth of the data it never saw
during training:

| ranker | NDCG@10 | |
|---|---|---|
| **this model** | **0.753** | 95 % interval [0.715, 0.790] |
| rank by good rating and low price | 0.643 | frozen before the model existed |
| rank by number of reviews | 0.639 | frozen before the model existed |
| **random shuffle** | **0.552** | the floor |

NDCG@10 rewards putting the best listings near the top and cares much less about the bottom of the
list. A random shuffle already scores 0.552, because with only ten slots and a decent share of good
listings even a shuffle gets some of them right. Read every number here against that floor: **the
usable range is 0.55 to 1.00**.

Paired search by search against the review-count rule, the stronger of the two baselines over the
whole population: **+0.114 [0.080, 0.153]**.

---

## What the model actually predicts

There are no clicks, no bookings and no search logs in this data, so "which listings guests like
best" is a question it cannot answer.

What it does carry is each listing's **calendar**: which future nights are open and which are
blocked. So the target is **the fraction of the next 90 nights that are blocked**. A blocked night
might be booked, or the host might be using the place themselves, doing repairs, or holding it off
the market for the season, and the data cannot tell those apart.

That makes the target a **demand proxy**, and it is the ceiling on the whole project.

---

## Start here

| if you want | go to |
|---|---|
| the full write-up, and how every number was arrived at | **[`docs/report.md`](docs/report.md)** |
| the evidence cell by cell | [`notebooks/`](notebooks/): data inventory, target validation, feature analysis, evaluation |
| the A/B test designed for this system and never run | [`docs/ab_test_design.md`](docs/ab_test_design.md) |
| how the raw snapshots become the feature table | [`docs/data_pipeline_design.md`](docs/data_pipeline_design.md) |
| how to run all of this on Azure yourself | [`docs/azure_setup.md`](docs/azure_setup.md) |
| what the live endpoint was sent and what it returned | [`docs/endpoint_demo/`](docs/endpoint_demo/) |
| the code, which holds all of the logic | [`src/rental_ranking/`](src/rental_ranking/) |

Read the report if you read one thing.

---

## How it works

Three Greek cities, one snapshot each, in three tables: listings, calendars and reviews. Host and
reviewer identifying information is stripped or hashed before anything is written to disk.

![From the processed parquets to the feature matrix: label, filters, price imputation, grading and grouping, then four feature blocks assembled into the table the ranker trains on](docs/figures/pipeline_features.png)

- **What counts as one search.** A ranker orders listings *within* a search, so a search has to be
  defined before anything can be measured. Here it is **city × neighbourhood × room type × party
  size**, which yields **393 searches**. Crossings too thin to rank fall down a two-rung ladder
  instead of being deleted, and 99.4 % of listings never leave the full four-part key.
- **The model.** **LambdaMART** (LightGBM), gradient-boosted trees built for ranking. It learns
  from *pairs*, asking which of these two listings should come first, and reads **61 features**
  covering size and capacity, price relative to the neighbourhood, review counts and scores, host
  portfolio, 19 categories of amenity, and geography. This is the system's one learned stage, with
  candidate generation done by a filter.
- **Data leakage, avoided.** Every feature is measured **before** the 90-day window the target
  comes from. Sixteen columns that peek at it, Airbnb's own availability counts and the price-quote
  dates among them, sit on a blocklist that a test enforces. Two further features were measured and
  dropped before shipping: review sentiment, which loses **2.6×** to a column Airbnb already
  publishes, and imputing `bedrooms`, whose missingness carries no signal.
- **Judged fairly.** One fifth of the data is sealed away, split so that near-duplicate listings
  and listings competing in the same search always stay on the same side. **That sealed fifth was
  opened exactly twice.**

---

## The part that does not work

**The model is worse than random at surfacing good new listings.** Of the never-reviewed listings
that the target itself grades 3 or above, the model puts **5.8 %** into a top ten. A random shuffle
would put **9.6 %** of them there.

Within the cohort it still orders new listings against each other sensibly. What it does is apply a
penalty to the whole cohort, and on average that penalty is correct, since 48.4 % of never-reviewed
listings really do sit in the bottom two grades. It is doing exactly what it was trained to do.

It is still a problem, because a marketplace that buries new listings keeps them new. And the
headline score cannot see it at all: the score *improves* in precisely the searches where the
burial is worst. The fix is a product decision about exploration versus exploitation.

The same blindness was then tested on host size cohorts. Listings from large operators sit **0.113
of a group lower than their grades warrant**, and the offset that would correct it turns out to be
zero, because correcting it makes the ranking worse. NDCG@10 reports the same number either way.
Both measurements are in [report §10](docs/report.md#10-where-it-fails).

---

## Run it yourself

A local console searches on the same four fields a search is built from, then lays the returned
order against the held-out grades:

```bash
uv run python -m rental_ranking.cloud.console --local     # no cloud account needed
```

![the console](docs/screenshots/console_win_kalamaria.png)

Screenshots from the live session are in [`docs/screenshots/`](docs/screenshots/), including one
where the model does well and one where it loses to a coin flip. Both are there on purpose. The
endpoint itself was deployed, exercised and deleted; what survives is in
[`docs/endpoint_demo/`](docs/endpoint_demo/), where every request, every response and the grades
they should have matched can be checked.

To build everything from the raw snapshots:

```bash
uv sync                                                   # environment
uv run python -m rental_ranking.data.build                # raw -> processed
uv run python -m rental_ranking.train.train               # split, train, evaluate
uv run pytest tests/ -q                                   # 784 tests
```

Every number in this repository is produced by code in `src/`. The notebooks import from it and
hold no logic of their own.

---

## In the cloud

Development and training run locally, because the data fits in memory and iteration is faster.
Azure ML is there to demonstrate the production workflow:

- four **versioned data assets**, so a training run records which data it saw
- a two-step **pipeline job**, raw snapshots → processed layer → feature table, which rebuilt the
  training data in the cloud and reproduced the local feature table byte for byte
- one **training job** on a scale-to-zero cluster, reproducing the local result exactly
- a **managed endpoint**, deployed, demonstrated and torn down, returning floats bit-identical to
  the local model's

Inference is fast: **13 ms** per request on the smallest instance offered, and the whole catalogue
of 44,684 listings scores in **87 ms**. Commands on how to replicate this are in
[`docs/azure_setup.md`](docs/azure_setup.md).

---

## An experiment designed but not run

[`docs/ab_test_design.md`](docs/ab_test_design.md) specifies an A/B test for a real product
question: on a search covering a whole city, should the system narrow to a few neighbourhoods
before ranking, or rank the whole city at once?

**No test was run.** There is no live product and no traffic. The document is a design, plus the
offline analysis motivating it, with every quantity marked as measured, derived or assumed.

---

## Limitations

1. **The target is availability, not demand.** A ceiling that no modelling choice moves, described
   [above](#what-the-model-actually-predicts).
2. **The grades are constructed.** They are quartiles of that target rather than human relevance
   judgements, which puts this closer to a rank correlation than to the NDCG in a learning-to-rank
   paper. Treat 0.753 as internal to this setup.
3. **It buries good new listings**, and the headline score is blind to it, as
   [above](#the-part-that-does-not-work).
4. **There is no information about the guest.** No dates, no party history, no previous searches. A
   "search" here is a market segment, and the model emits one fixed order per segment.
5. **The split is structural, and does not cross time.** One snapshot per city means nothing tests
   training on an earlier snapshot and ranking a later one, which is what deployment would need.
6. **Nothing here is causal.** The model orders listings by a demand proxy. It can say which
   listings do well, and it stays silent on what makes them do well.

---

## Repository map

| path | contents |
|---|---|
| `src/rental_ranking/data/` | acquisition, anonymisation, typing, raw → processed |
| `src/rental_ranking/features/` | target construction, filters, query groups, features |
| `src/rental_ranking/train/` | split, baselines, model, hyperparameter search, ablations |
| `src/rental_ranking/evaluate/` | metrics, reporting, exposure, comparability, fairness |
| `src/rental_ranking/cloud/` | scoring script, endpoint demo, console |
| `pipelines/` | Azure ML job, environment and endpoint definitions |
| `docker/` | the console as a standalone image |
| `tests/` | 784 tests |

---

## Licence and attribution

The code is MIT. The listing data comes from [Inside Airbnb](https://insideairbnb.com/) and is
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), so the figures, tables and
captured responses derived from it carry that licence and require attribution. Both are set out
in [LICENSE](LICENSE).

Host and reviewer identifying information is stripped or hashed by
`src/rental_ranking/data/anonymize.py` before anything is committed or published. Listing ids are
hashed; listing titles are public and are shown as-is.
