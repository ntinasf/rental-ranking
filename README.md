# Rental Ranking — Two-Stage Search Ranking on Azure ML

A portfolio data science project: a two-stage search ranking system (retrieval → ranking) for
short-term rental listings, built on [Inside Airbnb](https://insideairbnb.com/) data. Listings
are ranked within query groups by a graded demand proxy derived from trailing-90-day calendar
occupancy, with a price+rating baseline and a LightGBM LambdaMART ranker, trained and evaluated
on Azure ML. This README grows with the project — see `docs/BUILD_GUIDE.md` for the roadmap.

## Structure

- `notebooks/` — numbered narrative notebooks (data inventory, label validation, feature analysis, evaluation); they import from `src/` and never duplicate logic
- `src/rental_ranking/` — data acquisition, anonymization, filtering, label and feature construction, training, evaluation
- `pipelines/` — Azure ML job and pipeline YAML
- `tests/` — unit tests for label and feature logic
- `docs/` — build guide, decisions log, A/B test design
- `data/` — local data, gitignored

## Data attribution

This project uses data from [Inside Airbnb](https://insideairbnb.com/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Host and reviewer PII is
stripped/hashed (`src/rental_ranking/data/anonymize.py`) before anything is committed or published.
