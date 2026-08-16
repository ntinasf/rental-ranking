# Rental Ranking — Two-Stage Search Ranking on Azure ML

A portfolio data science project: a two-stage search ranking system (retrieval → ranking) for
short-term rental listings, built on [Inside Airbnb](https://insideairbnb.com/) data. Listings
are ranked within query groups by a graded demand proxy derived from **forward**-90-day calendar
availability, anchored per listing, with a price+rating baseline and a LightGBM LambdaMART ranker.
Development and training run locally; Azure ML carries the cloud workflow (versioned data assets,
a command job, an endpoint demo). This README grows with the project — see
`docs/BUILD_GUIDE.md` for the roadmap and `NEXT_STEPS.md` for current status.

## Structure

- `notebooks/` — numbered narrative notebooks (data inventory, label validation, feature analysis, evaluation); they import from `src/` and never duplicate logic
- `src/rental_ranking/data/` — acquisition, anonymization, typing and the raw → processed build (complete)
- `src/rental_ranking/` — label and feature construction, training, evaluation (Phase 1 onward)
- `pipelines/` — Azure ML job and pipeline YAML
- `tests/` — unit tests per module; 322 passing
- `docs/` — build guide, data dictionary, data-pipeline contract, decisions log, A/B test design
- `data/` — local data, gitignored

## Temporal design — how leakage is prevented

The ranking target is a **forward** demand proxy: the fraction of blocked nights in the 90 days
after **T**, where `T = min(calendar.date)` for that listing — its own first calendar row, never a
per-city date, because scrape dates spread across up to four days inside a single market.

The split follows from that anchor:

- **Features** are as-of-T listing attributes. Nothing computed from the label window is eligible,
  and the columns that read the window — `availability_30/60/90/365`, `availability_eoy`,
  `estimated_occupancy_l365d`, `estimated_revenue_l365d`, and the price-quote dates, which are the
  listing's *first available calendar date* — are blocklisted rather than merely unused.
- **The label** is what the calendar says about T onward.
- **Review windows** used as features are anchored at the same per-listing T and look backward
  (`[T−365, T−365+90)` for the same-season window), so a validation window and a training feature
  are computed by one function with one anchor.

This is a guarantee about **feature provenance**, and it is counted rather than assumed:
notebook 02 §7 reports the only breach — **26 listings of 44,684 (0.06 %)** whose attributes were
scraped one or two days after their window opened. None of them gained a review in that gap, so
the one attribute class that could have moved did not. They are kept and named; row exclusion in
this project belongs to `filters.py` with a stated threshold and a reported count.

**What it is not.** One snapshot cannot supply a train/test split *in time*, and this is not one.
The Phase 3 holdout is a separate decision — a **grouped** split on `features/groups.py::cluster_id`,
so near-twin listings (same host, same point, same capacity) land wholly in train or wholly in
test rather than straddling the boundary. Reporting feature provenance as if it were temporal
validation would be an overclaim; the two are stated separately on purpose.

## Data attribution

This project uses data from [Inside Airbnb](https://insideairbnb.com/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Host and reviewer PII is
stripped/hashed (`src/rental_ranking/data/anonymize.py`) before anything is committed or published.
