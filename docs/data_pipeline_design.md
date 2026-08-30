# Data pipeline

The contract for `src/rental_ranking/data/` and `src/rental_ranking/features/`. Every one of the
90 raw listings columns has a disposition, encoded literally in
[`data/columns.py`](../src/rental_ranking/data/columns.py) and enforced by `tests/test_columns.py`.
The two diagrams below are the flow; this text is only what they cannot show.

## Raw → processed

![Three raw city snapshots through io, anonymize and clean inside the data/build.py orchestrator, past three cross-frame gates, into the three processed parquets](figures/pipeline_processed.png)

```sh
uv run python -m rental_ranking.data.build      # needs ANON_SALT
```

**Crete is a region, not a city**: its "neighbourhoods" are municipalities, and it is a seasonal
market roughly twice the size of the other two combined.

**Three dates, never conflated.** Every date computation is relative to the **row's own** dates —
never a global reference date, never the release date as if it were the label anchor, never
`max(last_review)`. Each of those misaligns the cities.

| date | what it is |
|---|---|
| release date | the folder name and the Azure data-asset version. **Not** a scrape date |
| `scrape_date` | from `last_scraped`, and it varies *within* a city. Athens spans three days, Crete four; Crete's modal value is 06-30, not the 06-29 in its folder name |
| T | the label anchor, `min(calendar.date)` per listing |

Each listing's calendar is contiguous and 365 rows, except 8 listings (2 Athens, 6 Crete) running
366–367. Do not write code that assumes exactly 365.

**Raw is immutable.** Snapshots rotate off the public site, so the stored copy is the only path to
reproduction, and raw files are never edited. `download.py` fetches and manifests and must work
with no Azure credentials at all; asset registration is separate, and its commands are in
[azure_setup.md](azure_setup.md).

**Preprocessing is lossless.** No row is dropped except on integrity, and deduplication on `id` is
the single exception. Row exclusion belongs to `filters.py`, at label-build time, with identical
thresholds across cities and per-rule counts returned. Checks report rather than delete: they warn
once with a count and keep the row.

## Anonymisation

The boundary is **publication** — the repo, notebook outputs, the README. Local disk and the
private Blob container are storage, not publication. `data/` is never committed, and notebooks load
the processed layer, so what a reader sees is PII-free by construction.

- **IDs** — `id`, `host_id`, and `listing_id` in both calendar and reviews, consistently or every
  join breaks — are **salted SHA-256 truncated to 12 hex characters**. The salt is `ANON_SALT` in
  `.env` and is never committed. A different salt yields a structurally identical dataset with
  entirely different ids.
- **Dropped outright:** `host_name`, `host_thumbnail_url`, `host_picture_url`, `host_url`,
  `host_profile_id`, `host_profile_url`, `reviewer_id`, `reviewer_name`.
- **Derived, then the raw column dropped:** `host_location` → `host_is_local`
  {local, remote, unknown} · `host_about` → `host_has_about` · `license` → `license_status`
  {registered, exempt, missing} plus a salted hash of the number.
- **Kept raw:** listing `name`, which is marketing copy rather than host PII. Coordinates stay
  **unrounded** — Airbnb already jitters them 0–150 m at source, so rounding only degrades spatial
  features while adding no privacy.

## Processed → feature table

![The features/build.py orchestrator chains label, filters, price imputation, grading and grouping, fans out into four feature blocks, and assembles the feature table](figures/pipeline_features.png)

```sh
uv run python -m rental_ranking.features.build
```

Every module in both layers is a pure `DataFrame → DataFrame` transform with no I/O. The two
`build.py` orchestrators are the only writers, and the only entry points. Checks needing more than
one frame — schema equality across cities, join integrity after hashing — live in the orchestrator
rather than in an entity transform.

**Feature-blocklisted columns** are kept in the processed layer for validation and asserts, and are
never model inputs. They are exported as `LABEL_ADJACENT_COLUMNS`, imported by the feature code and
checked by a test: the four `availability_*` windows, `availability_eoy`,
`estimated_occupancy_l365d`, `estimated_revenue_l365d`, and `price_quote_checkin_date` /
`price_quote_checkout_date`. Three are not obvious: `availability_eoy` is a forward window to
31 December that contains the label window, and the two quote dates are the listing's first
*available* calendar date. [report.md §2](report.md) has the measurement.

`price` is the project's **sole imputed column**, filled from the median of a structural cascade
— `city × neighbourhood × room_type × accommodates`, then four progressively coarser structural
rungs down to `city` alone, which is the guaranteed terminator. Everything else passes through as
NaN for LightGBM to learn a split direction on.

**The trainer takes its input list from `feature_columns()`, never from `table.columns`** — that
one line is the difference between training on the features and training on the answer.
