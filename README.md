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
- `tests/` — unit tests per module; 257 passing
- `docs/` — build guide, data dictionary, data-pipeline contract, decisions log, A/B test design
- `data/` — local data, gitignored

## Data attribution

This project uses data from [Inside Airbnb](https://insideairbnb.com/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Host and reviewer PII is
stripped/hashed (`src/rental_ranking/data/anonymize.py`) before anything is committed or published.
