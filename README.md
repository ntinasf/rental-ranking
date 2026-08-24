# Rental Ranking

A search ranking system for short-term rental listings, built end to end on public
[Inside Airbnb](https://insideairbnb.com/) data for Athens, Thessaloniki and Crete — 44,684
listings — and deployed on Azure ML.

When someone searches for a place to stay, hundreds of listings match. Something has to decide
which ten show up first. This project builds that something, measures how well it works, and is
careful about what the measurement does and does not mean.

---

## The result

The model puts listings in order within a search. Scored on listings it never saw during
training:

| ranker | score | |
|---|---|---|
| **this model** | **0.753** | 95 % interval [0.715, 0.790] |
| rank by number of reviews | 0.639 | the strongest simple rule |
| rank by good rating and low price | 0.643 | what a product team ships on day one |
| **shuffle the results at random** | **0.552** | the floor any ranker has to beat |

The score is **NDCG@10**, a standard ranking measure between 0 and 1 that rewards putting the
best listings near the top and cares much less about the bottom of the list. The random floor
sits at 0.552 rather than 0, because with only ten slots and a decent share of good listings,
even a shuffle gets some of them right. **A ranking score means nothing without its floor**,
which is why it is in the table rather than a footnote.

Against the review-count rule, on the same searches, the improvement is **+0.114 [0.080, 0.153]**.

---

## What the model actually predicts

The honest answer is not "which listings guests like best". Nobody publishes that.

What *is* public is each listing's **calendar**: which future nights are open and which are
blocked. So the target is **the fraction of the next 90 nights that are blocked**. A blocked
night might be booked — or the host might be visiting family, doing repairs, or have taken the
place off the market for the season. The data cannot tell those apart.

That makes the target a **demand proxy**. It is never "bookings", it is not called bookings
anywhere in this repository, and no amount of modelling fixes it. It is the ceiling on the whole
project.

There is a second, subtler issue. A listing's calendar is full partly *because Airbnb ranked it
well and sent it traffic*. So "this model predicts demand" and "this model partly reproduces
Airbnb's existing ranking, seen through its effect on occupancy" cannot be fully separated using
a single public snapshot. Testing the strong version of that worry: a model stripped of every
feature about a listing's track record — review counts, tenure, age — still scores **0.703**.
Most of the signal is not the feedback loop. That bounds the problem without removing it.

---

## How it works, briefly

- **The data.** Three Greek cities, one snapshot each. Three tables: listings, calendars,
  reviews. Host and reviewer identifying information is stripped or hashed before anything is
  written to disk.
- **What counts as one search.** A ranker orders listings *within* a search, so searches have to
  be defined. Here one search is **city × neighbourhood × room type × party size** — the four
  things a guest actually picks. That gives 393 searches, with a fallback that widens the
  definition when a neighbourhood is too thin to stand alone.
- **The target, made rankable.** The raw target is a fraction between 0 and 1. Ranking models
  want **grades**, so it is cut into five: grade 0 for listings with no blocked nights at all,
  grades 1–4 for the rest by quartile — cut *inside* each city and room type, so a beach villa
  is graded against beach villas.
- **The model.** **LambdaMART** (LightGBM), a gradient-boosted tree model built for ranking.
  Instead of predicting each listing's demand separately, it learns from *pairs*: which of these
  two should come first? It reads **61 features** — size and capacity, price relative to the
  neighbourhood, review counts and scores, host history, 19 categories of amenity, and distance
  to the nearest landmark.
- **Avoiding the classic mistake.** Every feature is measured **before** the 90-day window the
  target comes from. Columns that peek — Airbnb's own availability counts, its occupancy
  estimates, the price-quote dates — are on a blocklist, and a test fails if any of them reach
  the model. The guarantee is counted, not assumed: **26 listings out of 44,684 (0.06 %)** had
  attributes scraped a day or two after their window opened. None of them gained a review in
  that gap. They are kept and named rather than quietly dropped.
- **Judging it fairly.** One fifth of the data is sealed away, split so that near-duplicate
  listings and listings competing in the same search always stay on the same side. **That sealed
  fifth was opened exactly twice**, both times announced in advance, and it is now closed.

**The full write-up, with all the evidence, is in [`docs/report.md`](docs/report.md).** It is the
document to read if you want to know *how* the numbers were arrived at rather than what they
are.

---

## The part that does not work

**The model is worse than random at surfacing good new listings.** A listing with no reviews
yet, which the data says deserves to be near the top, reaches the top ten **5.8 %** of the time.
A random shuffle would put it there **9.6 %** of the time.

The model is not confused about them — it can still order new listings against each other
sensibly. It applies a penalty to the whole cohort, and on average that penalty is *correct*,
since most never-reviewed listings really do sit in the bottom grades. It is doing exactly what
it was trained to do.

It is still a problem, because a marketplace that buries new listings ensures they stay new. And
the headline score cannot see it at all — the score actually *improves* in precisely the
searches where this is worst. Marketplaces solve this with an explicit boost or an exploration
policy; this repository measures the problem and states the target for a fix.

---

## See it running

The endpoint was deployed to Azure ML, exercised, and deleted the same session. What survives is
in [`docs/endpoint_demo/`](docs/endpoint_demo/): the requests sent, the responses returned, and
each ranking joined back to the correct answers so it can be checked.

A local console lets you search, pick a listing, change its details and watch where it lands:

```bash
uv run python -m rental_ranking.cloud.console --local     # no cloud account needed
```

![the console](docs/screenshots/console_win_kalamaria.png)

Two screenshots from the live session — one where the model does well, one where it loses to a
coin flip — are in [`docs/screenshots/`](docs/screenshots/). Both are kept on purpose.

---

## Running it

```bash
uv sync                                                   # environment
uv run python -m rental_ranking.data.build                # raw -> processed
uv run python -m rental_ranking.train.train               # split, train, evaluate
uv run pytest tests/ -q                                   # 774 tests
```

Every number above is produced by code in `src/`. Notebooks import from it and hold no logic of
their own; each runs cleanly from a fresh kernel.

---

## In the cloud

Development and training run locally — the data fits in memory and iteration is faster. Azure ML
is used to demonstrate the production workflow rather than because the work needs it:

- four **versioned data assets**, so a training run records which data it saw
- a two-step **pipeline job**, raw snapshots → processed layer → feature table, which rebuilt the
  training data in the cloud and reproduced the local feature table exactly
- one **training job** on a scale-to-zero cluster, reproducing the local result exactly
- a **managed endpoint**, deployed, demonstrated and torn down in the same session

![The preprocessing pipeline on Azure ML](docs/screenshots/preprocess_pipeline_dag.png)

Inference is cheap: **13 ms** per request on the smallest instance offered, and the entire
catalogue of 44,684 listings scores in **87 ms**. The cost of running this is instance-hours,
not computation. Commands, costs and teardown timestamps are in
[`docs/azure_setup.md`](docs/azure_setup.md).

---

## What it cannot do

1. **The target is availability, not demand.** Covered above. It is a ceiling nothing else
   moves.
2. **The grades are constructed, not judged by people.** They are quartiles of the target, not
   human relevance ratings. Do not compare 0.753 against published ranking papers — it is a
   different quantity wearing the same name.
3. **It buries good new listings**, worse than chance, and the headline score is blind to it.
4. **There is nothing about the guest.** No dates, no party history, no previous searches. The
   "search" is a market segment, and the model produces one fixed order per segment. It is a
   listing-quality score being used as a ranker — the ranking a system falls back on when it
   knows nothing about who is asking.
5. **The split is structural, not across time.** Features come before the target window, but
   nothing here tests "train on last month, rank next month", which is what deployment needs.
   One snapshot per city means this can be stated, not fixed.
6. **393 searches is a small sample** and the headline rests on 72 of them. The overall effect
   survives that; no per-city claim does — Thessaloniki least of all, with four searches in the
   sealed fifth.
7. **The test set does not cover geography evenly**, though this is measured rather than feared:
   17 of 75 neighbourhoods have no listings in the sealed fifth, and the model scores those
   neighbourhoods **−0.008 [−0.044, +0.030]** against the rest — no detectable bias.
8. **The target sits downstream of Airbnb's own ranking.** Covered above; bounded, not
   eliminated.
9. **Nothing here is causal.** The model orders listings by expected demand. It does not
   identify what *causes* a listing to do well, and no sentence in this repository claims it
   does.

---

## An experiment that was designed but not run

[`docs/ab_test_design.md`](docs/ab_test_design.md) specifies an A/B test for a real product
question: on a search covering a whole city, should the system narrow to a few neighbourhoods
before ranking, or rank the whole city at once?

**No test was run.** There is no live product and no traffic. The document is a design, plus the
offline analysis that motivates it, with every number marked as measured, derived or assumed.
Simulating both options on held-out data shows what narrowing *costs*; what it cannot show is
whether the neighbourhoods kept are the ones the guest wanted — which is exactly the question
that needs real users, and the reason to run the test.

One other planned piece, sentiment analysis of review text, was **designed and priced but never
run**. It sits in the repository unexecuted and is labelled as such.

---

## Repository map

| path | contents |
|---|---|
| [`docs/report.md`](docs/report.md) | **the full write-up** — how every number was arrived at |
| [`notebooks/`](notebooks/) | four narrative notebooks: data inventory, target validation, feature analysis, evaluation |
| `src/rental_ranking/data/` | acquisition, anonymisation, typing, raw → processed |
| `src/rental_ranking/features/` | target construction, filters, query groups, features |
| `src/rental_ranking/train/` | split, baselines, model, hyperparameter search |
| `src/rental_ranking/evaluate/` | metrics, reporting, exposure, comparability, fairness |
| `src/rental_ranking/cloud/` | scoring script, endpoint demo, console |
| `pipelines/` | Azure ML job, environment and endpoint definitions |
| `docker/` | the console as a standalone image |
| [`docs/`](docs/) | data dictionary, pipeline contract, decisions log, Azure setup, A/B design |
| `tests/` | 774 tests |

[`docs/decisions_log.md`](docs/decisions_log.md) records every non-trivial choice with what was
rejected and why.

---

## Attribution

Data from [Inside Airbnb](https://insideairbnb.com/), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Host and reviewer identifying
information is stripped or hashed by `src/rental_ranking/data/anonymize.py` before anything is
committed or published. Listing ids are hashed; listing titles are public and are shown as-is.
