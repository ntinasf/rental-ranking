# Rental Ranking

**A search ranking system for short-term rental listings, built on 44,684 listings from public
[Inside Airbnb](https://insideairbnb.com/) data for Athens, Thessaloniki and Crete, trained
locally and deployed on Azure ML.**

When someone searches for a place to stay, hundreds of listings match. Something has to decide
which ten appear first. This repository builds that something, measures how well it works, and is
careful about what the measurement does and does not mean.

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
list. The floor sits at 0.552 rather than 0, because with only ten slots and a decent share of good
listings even a shuffle gets some of them right, so **the usable range is 0.55 → 1.00, not
0 → 1**. A ranking score means nothing without its floor, which is why the floor is in the table
rather than a footnote.

Paired search by search against the review-count rule, the strongest of the two baselines over the
whole population: **+0.114 [0.080, 0.153]**.

---

## Start here

| if you want | go to |
|---|---|
| the full write-up, and how every number was arrived at | **[`docs/report.md`](docs/report.md)** |
| the evidence cell by cell | [`notebooks/`](notebooks/): data inventory, target validation, feature analysis, evaluation |
| the A/B test designed for this system and never run | [`docs/ab_test_design.md`](docs/ab_test_design.md) |
| every non-trivial choice, what was rejected and why | [`docs/decisions_log.md`](docs/decisions_log.md) |
| the cloud commands, costs and teardown timestamps | [`docs/azure_setup.md`](docs/azure_setup.md) |
| what the live endpoint was sent and what it returned | [`docs/endpoint_demo/`](docs/endpoint_demo/) |
| the code, which holds all of the logic | [`src/rental_ranking/`](src/rental_ranking/) |

Read the report if you read one thing. Everything below is its summary.

---

## What the model actually predicts

Not "which listings guests like best". Nobody publishes that.

What *is* public is each listing's **calendar**: which future nights are open and which are
blocked. So the target is **the fraction of the next 90 nights that are blocked**. A blocked night
might be booked, or the host might be visiting family, doing repairs, or holding the place off the
market for the season, and the data cannot tell those apart.

That makes the target a **demand proxy**. It is never called "bookings" anywhere in this
repository, and no amount of modelling fixes it. It is the ceiling on the whole project.

There is a second, subtler issue. A listing's calendar is full partly *because Airbnb ranked it
well and sent it traffic*, so "this model predicts demand" and "this model partly reproduces
Airbnb's ranking, seen through its effect on occupancy" cannot be fully separated from a single
public snapshot. The strong version of that worry was tested: a model stripped of every feature
about a listing's track record, review counts, tenure and age, still scores **0.703**. Most of the
signal is not the feedback loop. That bounds the problem without removing it.

---

## How it works

![From the processed parquets to the feature matrix: label, filters, price imputation, grading and grouping, then four feature blocks assembled into the table the ranker trains on](docs/figures/pipeline_features.png)

- **The data.** Three Greek cities, one snapshot each. Three tables: listings, calendars, reviews.
  Host and reviewer identifying information is stripped or hashed before anything is written to
  disk.
- **What counts as one search.** A ranker orders listings *within* a search, so a search has to be
  defined before anything can be measured. Here it is **city × neighbourhood × room type × party
  size**, the four things a guest actually picks, which yields **393 searches**. Crossings too thin
  to rank fall down a two-rung ladder instead of being deleted, and 99.4 % of listings never leave
  the full four-part key.
- **The target, made rankable.** The raw target is a fraction between 0 and 1. Ranking models want
  **grades**, so it is cut into five: grade 0 for listings with no blocked nights at all, grades
  1–4 by quartile above that, cut *inside* each city and room type so that a beach villa is graded
  against beach villas.
- **The model.** **LambdaMART** (LightGBM), gradient-boosted trees built for ranking. Instead of
  predicting each listing's demand separately it learns from *pairs*: which of these two should
  come first? It reads **61 features**, covering size and capacity, price relative to the
  neighbourhood, review counts and scores, host portfolio, 19 categories of amenity, and geography.
  This is the system's one learned stage; candidate generation is a filter, not a second model.
- **The classic mistake, avoided.** Every feature is measured **before** the 90-day window the
  target comes from. Sixteen columns that peek at it, Airbnb's own availability counts and the
  price-quote dates among them, sit on a blocklist that a test enforces. Two more features were
  measured and dropped before any of them shipped: review sentiment, which loses **2.6×** to a
  column Airbnb already publishes, and imputing `bedrooms`, whose missingness carries no signal.
- **Judged fairly.** One fifth of the data is sealed away, split so that near-duplicate listings
  and listings competing in the same search always stay on the same side. **That sealed fifth was
  opened exactly twice**, both times announced in advance, and it is now closed.

---

## The part that does not work

**The model is worse than random at surfacing good new listings.** Of the never-reviewed listings
that the target itself grades 3 or above, the model puts **5.8 %** into a top ten. A random shuffle
would put **9.6 %** of them there.

The model is not confused about them, and within the cohort it still orders new listings against
each other sensibly. What it does is apply a penalty to the whole cohort, and on average that
penalty is *correct*, since 48.4 % of never-reviewed listings really do sit in the bottom two
grades. It is doing exactly what it was trained to do.

It is still a problem, because a marketplace that buries new listings keeps them new. And the
headline score cannot see it at all: the score *improves* in precisely the searches where the
burial is worst. The fix is a product decision, an exploration boost or a reserved slot, rather
than feature engineering.

The same blindness was then tested on a cohort where nobody knew the answer in advance. Listings
from large operators sit **0.113 of a group lower than their grades warrant**, and the offset that
would correct it turns out to be zero, because correcting it makes the ranking worse. NDCG@10
reports the same number either way. Both measurements are in
[report §10](docs/report.md#10-where-it-fails).

---

## See it running

A local console searches on the same four fields a search is built from, then lays the returned
order against the held-out grades:

```bash
uv run python -m rental_ranking.cloud.console --local     # no cloud account needed
```

![the console](docs/screenshots/console_win_kalamaria.png)

Two screenshots from the live session are kept in [`docs/screenshots/`](docs/screenshots/), one
where the model does well and one where it loses to a coin flip. Both are there on purpose. The
endpoint itself was deployed, exercised and deleted in the same session; what survives is in
[`docs/endpoint_demo/`](docs/endpoint_demo/), where every request, every response and the grades
they should have matched can be checked.

---

## Running it

```bash
uv sync                                                   # environment
uv run python -m rental_ranking.data.build                # raw -> processed
uv run python -m rental_ranking.train.train               # split, train, evaluate
uv run pytest tests/ -q                                   # 777 tests
```

Every number in this repository is produced by code in `src/`. The notebooks import from it and
hold no logic of their own; each runs cleanly from a fresh kernel.

---

## In the cloud

Development and training run locally, because the data fits in memory and iteration is faster.
Azure ML demonstrates the production workflow rather than being needed by the work:

- four **versioned data assets**, so a training run records which data it saw
- a two-step **pipeline job**, raw snapshots → processed layer → feature table, which rebuilt the
  training data in the cloud and reproduced the local feature table byte for byte
- one **training job** on a scale-to-zero cluster, reproducing the local result exactly
- a **managed endpoint**, deployed, demonstrated and torn down in one session, returning floats
  bit-identical to the local model's

![The preprocessing pipeline on Azure ML](docs/screenshots/preprocess_pipeline_dag.png)

Inference is cheap: **13 ms** per request on the smallest instance offered, and the whole catalogue
of 44,684 listings scores in **87 ms**. The cost of running this is instance-hours, not
computation. Commands, costs and teardown timestamps are in
[`docs/azure_setup.md`](docs/azure_setup.md).

**This demonstrates the workflow, not production behaviour.** Nothing here tests how the model
would perform as a live product.

---

## An experiment designed but not run

[`docs/ab_test_design.md`](docs/ab_test_design.md) specifies an A/B test for a real product
question: on a search covering a whole city, should the system narrow to a few neighbourhoods
before ranking, or rank the whole city at once?

**No test was run.** There is no live product and no traffic. The document is a design, plus the
offline analysis motivating it, with every quantity marked as measured, derived or assumed.
Simulating both arms on held-out data shows what narrowing *costs*. What no offline dataset can
show is whether the neighbourhoods it keeps are the ones the guest wanted, which is exactly the
question that needs real users.

---

## Limitations

1. **The target is availability, not demand.** A ceiling no modelling choice moves.
2. **The grades are constructed, not judged by people.** They are quartiles of the target, not
   human relevance ratings. Do not benchmark 0.753 against published ranking papers; it is a
   different quantity wearing the same name.
3. **It buries good new listings**, worse than chance, and the headline score is blind to it.
4. **There is nothing about the guest.** No dates, no party history, no previous searches. The
   "search" is a market segment, and the model emits one fixed order per segment.
5. **The split is structural, not across time.** Nothing tests "train on last month, rank next
   month", which is what deployment needs. One snapshot per city means this can be stated, not
   fixed.
6. **Nothing here is causal.** The model orders listings by expected demand. It does not identify
   what *causes* a listing to do well, and no sentence in this repository claims it does.

Two more, on what 393 searches can and cannot support and on how evenly the sealed fifth covers the
map, are in [report §12](docs/report.md#12-limitations) with the measurement behind each.

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
| `tests/` | 777 tests |

---

## Attribution

Data from [Inside Airbnb](https://insideairbnb.com/), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Host and reviewer identifying
information is stripped or hashed by `src/rental_ranking/data/anonymize.py` before anything is
committed or published. Listing ids are hashed; listing titles are public and are shown as-is.
