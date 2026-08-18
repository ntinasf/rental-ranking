# Azure setup — reproducible record

One-time provisioning is done through the Azure portal or CLI; either way, the equivalent
CLI commands are recorded here so the setup is reproducible. Fill in real names/regions as
resources are created. Day-one items per the build guide: submit the compute-quota request
immediately (approval takes days) and create the budget alert before the first training job.

## Prerequisites (local)

```sh
brew install azure-cli
az extension add -n ml
az login
az configure --defaults group=nf-rental-ranking workspace=nf-rental-ranking-ws
```

The defaults line is not optional decoration. Without it every `az ml` command needs both
`-g nf-rental-ranking` and `-w nf-rental-ranking-ws`, and supplying only one produces a
**misleading** error — `(ResourceGroupNotFound) Resource group 'None' could not be found`, which
names the resource group when the missing argument is the workspace. Verify with
`az configure --list-defaults -o table`; empty output means they were never set in this shell's
profile. Every command below is written with both flags explicit, so it works either way.

## Resource group + workspace

Created 2026-07-20 via portal. Region: **italynorth**. CLI equivalent:

```sh
# az group create --name nf-rental-ranking --location italynorth
# az ml workspace create --name nf-rental-ranking-ws --resource-group nf-rental-ranking
```

Region check (verified 2026-07-20): AI Language (TextAnalytics) is available in italynorth,
including the free F0 tier. AI Search availability to be confirmed if/when Phase 4 (V2
vector index) starts; neither service has to be co-located with the workspace.

## Quota (verified 2026-07-20 — no increase request needed)

```sh
az ml compute list-usage -o table
```

Subscription limits in italynorth: 16 total dedicated cores; 6 vCPUs each for DSv2/DSv3/Dv2
families; 16 for FSv2; 10 for DASv4. Zero GPU quota (irrelevant — CPU-only project).
Training (one 4-vCPU node) and the endpoint demo (2-vCPU instance, ×1.2 reservation) each
fit comfortably; they never run simultaneously per the teardown rule.

## Compute cluster (scale-to-zero)

```sh
# TODO when first training job approaches (Phase 3):
# az ml compute create --name cpu-cluster --type AmlCompute \
#   --size Standard_F4s_v2 --min-instances 0 --max-instances 1
# (F4s_v2: 4 vCPU compute-optimized, fits the 16-core FSv2 quota; Standard_DS3_v2 also fine)
```

Min instances 0 — an idle cluster costs nothing.

## Budget alert

Created via portal (Cost Management → Budgets): €25–30/month, alerts at 50% and 80%.

## Data assets

Two layers get registered (contract: docs/data_pipeline_design.md): raw per city/snapshot,
and the assembled feature table (Phase 2). Registration is a one-time act per snapshot, done
from here after inspecting the downloads — deliberately not part of download.py.

**Specified in YAML, not in flags.** The four assets live in `pipelines/data/*.yml`, matching
how the environment and training job are already specified. The deciding reason is that
`az ml data create` **has no `--tags` argument** in the installed extension (confirm with
`az ml data create --help`) — tags are expressible only in the YAML file, and the tags are the
entire traceability scheme below. A second reason: `path` inside a data YAML resolves relative
to the **YAML file**, not the working directory, so `pipelines/data/raw_athens.yml` can be run
from anywhere in the repo and still point at the right folder.

```sh
# Raw, once per city after download + inspection. Immutable: these YAMLs are written once.
az ml data create -f pipelines/data/raw_thessaloniki.yml \
  -g nf-rental-ranking -w nf-rental-ranking-ws
az ml data create -f pipelines/data/raw_athens.yml \
  -g nf-rental-ranking -w nf-rental-ranking-ws
az ml data create -f pipelines/data/raw_crete.yml \
  -g nf-rental-ranking -w nf-rental-ranking-ws

# Feature table (the training job's input). Rebuild, then bump `version` in the YAML to today
# and re-register; --set injects the commit that built it.
uv run python -m rental_ranking.features.build
az ml data create -f pipelines/data/feature_table.yml \
  -g nf-rental-ranking -w nf-rental-ranking-ws \
  --set tags.git=$(git rev-parse --short HEAD)
```

Registering an existing name+version fails rather than overwriting — assets are immutable, which
is the property that makes a run tag worth trusting. To re-register after a change, bump the
version.

**Registered 2026-08-17**, all four, confirmed with `az ml data list -o table`:

| Asset | Version | Type |
| --- | --- | --- |
| `raw-thessaloniki` | 2026.06.29 | `uri_folder` |
| `raw-athens` | 2026.06.28 | `uri_folder` |
| `raw-crete` | 2026.06.29 | `uri_folder` |
| `features` | 2026.08.17 | `uri_file` |

`pipelines/train_job.yml` takes `azureml:features:2026.08.17` as its input — a named version, not
`@latest`, so a re-registration cannot silently change what a recorded run was trained on.

**Why only these two.** The processed parquets stay local and are deliberately *not* registered:
they are a reproducible intermediate — raw plus `data/build.py` reproduces them exactly — and
nothing in the cloud consumes them, because preprocessing runs locally and the training job takes
the feature table. Registering them would add a third thing to version and keep in step for no
demonstration value.

**Versioning, by layer.** Raw is versioned by **snapshot date**, because that is the only thing
that distinguishes one raw pull from another and it never changes once downloaded. The feature
table is versioned by **build date**, because it moves for two independent reasons — a new
snapshot *or* a feature change — and the date is the one string that increments under both. The
tags carry what the date cannot: `raw_versions` names the three assets it derives from and
`git` names the commit that built it. Every training run logs the table version as an MLflow tag
(see CLAUDE.md), so a model traces to a table, a table traces to a commit and three snapshots,
and the chain closes.

`git` is kept out of `feature_table.yml` and injected with `--set` on purpose: a hardcoded SHA
that someone forgets to bump is a *wrong* tag, whereas a forgotten `--set` leaves a *missing*
tag. Only one of those breaks the chain silently.

A local `path` uploads to the workspace's default Blob container (`workspaceblobstore`) and
registers in one step — ~300 MB for the three raw folders (32/132/136 MB), 3 MB for the feature
table. The container is private; PII in raw is storage, not publication.

## Standing rule

Managed endpoints: deploy → demo → screenshot → **tear down**. Never left running.

## Phase 3 — the training command job (2026-08-18)

Measured prices for **italynorth**, from the Azure retail price API rather than from memory:

| item | price | note |
|---|---|---|
| `Standard_F4s_v2` dedicated, Linux | **$0.1940/hr** | one training job ≈ 30 min ≈ **$0.10** |
| `Standard_F4s_v2` low-priority | $0.0388/hr | 80 % cheaper, preemptible — not worth it for a one-shot demo |
| **ACR Basic** | **$0.1666/day = $5.00/month** | created by the first environment build, and it **persists** |

**The compute is not the cost; the registry is.** Running the *entire* pipeline end to end —
data build, features, training and the whole 35-configuration sweep — is roughly 3-4 hours of
F4s_v2, i.e. **under $1**. The ACR that the first custom-environment build creates costs 50× the
job that created it, every month, whether or not anything else is ever submitted. It is the item
that belongs in the teardown, not the cluster: `--min-instances 0` genuinely costs nothing idle.

```bash
# Cluster (idle costs nothing at min-instances 0)
az ml compute create --name cpu-cluster --type amlcompute --size Standard_F4s_v2 \
  --min-instances 0 --max-instances 1 --idle-time-before-scale-down 120 \
  -g nf-rental-ranking -w nf-rental-ranking-ws

# Re-pin the environment from uv.lock BEFORE submitting (an unpinned cloud env is how
# "it trained differently up there" happens), then submit exactly one job.
az ml job create -f pipelines/train_job.yml -g nf-rental-ranking -w nf-rental-ranking-ws
```

**Two things the job needed that local running did not.**

1. **`matplotlib` in `environment.yml`.** `train.py` logs the per-fold learning curves as a run
   artifact, so the cloud image needs it. The old comment said "no plotting libraries"; that
   stopped being true when the figure became evidence rather than decoration.
2. **`--dataset-version` passed explicitly.** The job snapshot is `src/` alone, so
   `PROJECT_ROOT` resolves to the working directory and `pipelines/data/feature_table.yml` is
   absent — `dataset_version()` would silently return `"unregistered"` and break the one
   traceability claim this job exists to demonstrate.

**On cloud-versus-local numbers — the caveat was declared in advance and turned out to be
unnecessary.** Before the first run I recorded that LightGBM's multithreaded histogram
construction is not order-guaranteed across a 10-thread macOS ARM build and a 4-vCPU Linux x86
one, so a divergence from the local 0.7530 should be read as a platform artifact.

**No divergence occurred.** The cloud reproduced the development folds exactly —
`0.7026 / 0.7632 / 0.7078 / 0.7119`, stopping at `514 / 718 / 158 / 211`, median 362 — and the
run's `dataset_digest` (`8c74b2840854`) matched the local one, proving the same bytes were read.
Determinism held across platform, architecture and thread count. The caveat is kept here rather
than deleted because declaring it before the fact is what made the check meaningful, but it
should not be repeated as though it were a live risk.

### The first submission failed, and the reason is worth keeping

`clever_wheel_m2qlw8szhf` failed in the image build:

```
ERROR: Could not find a version that satisfies the requirement numpy==2.5.1
ERROR: Ignored the following versions that require a different python version:
       2.5.1 Requires-Python >=3.12
CondaEnvException: Pip failed
```

**"Pin from uv.lock" is not a text operation.** `requires-python = ">=3.11"`, so uv.lock carries
a resolution for *every* supported interpreter — it lists both

```
{ name = "numpy", version = "2.4.6", marker = "python_full_version < '3.12'" }
{ name = "numpy", version = "2.5.1", marker = "python_full_version >= '3.12'" }
```

Scraping the first version string per package silently selected the 3.12 branch and pinned it
against `python=3.11`. Two packages were wrong the same way: `numpy` (2.5.1 → **2.4.6**) and
`scipy` (1.18.0 → **1.17.1**).

The correct source is the **resolved** environment — the lock as this interpreter actually
solved it — which is what the regenerate command in `environment.yml` now reads. Local is
Python 3.11.15, so the conda pin of `python=3.11` is load-bearing, not cosmetic.

**A failed build still creates the registry.** The ACR (`Basic`) existed after this failure even
though no job ever ran, so the $5/month meter starts at the first *attempt*, not the first
success. Budget from submission, not from completion.

### Three real incompatibilities, found by three failed jobs

The demonstration is one job; getting there took four submissions and about $0.40 of compute.
Each failure was a genuine incompatibility rather than a typo, and each is worth keeping.

1. **`clever_wheel_m2qlw8szhf` — the marker-blind pin** (see above). `numpy==2.5.1` against
   `python=3.11`. Build failed.
2. **`green_pen_d7rtlkmmwd` — `azureml-mlflow` requires `mlflow-skinny<=3.13.0`.** We pinned
   `mlflow==3.14.0`, and MLflow 3.14 passes `tracking_uri` into `get_artifact_repository`, which
   the plugin's builder does not accept: `TypeError: azureml_artifacts_builder() got an
   unexpected keyword argument 'tracking_uri'`. **Training had already finished** when it fired.
   Fixed by holding the cloud at `mlflow==3.13.0` — the one package allowed to differ from local,
   because it is the tracking client and takes no part in the computation.
3. **`strong_tiger_9myv0v3105` — Azure ML does not serve MLflow 3's LoggedModel API.**
   `mlflow.lightgbm.log_model` posts to `/api/2.0/mlflow/logged-models`, which returns **404**
   against the workspace. Again the run had completed and its tags had landed; only the last line
   failed. Fixed by attempting the flavoured model and falling back to a raw `booster.txt`
   artifact with an explanatory tag.

**Consequence for step 8:** deploying an MLflow-flavoured model to a managed endpoint was the
clean path and may not be available. Expect the scoring script to load the booster directly.

## The sentiment demonstration resource (step 6)

```bash
az cognitiveservices account create --name nf-rental-language --kind TextAnalytics --sku F0 \
  --location italynorth -g nf-rental-ranking --custom-domain nf-rental-language --yes

# Endpoint and key -> .env (gitignored). Never commit them.
az cognitiveservices account show --name nf-rental-language -g nf-rental-ranking \
  --query properties.endpoint -o tsv
az cognitiveservices account keys list --name nf-rental-language -g nf-rental-ranking \
  --query key1 -o tsv
```

**F0 is free and stays free**, but the budget is smaller than the pricing page suggests. Azure
bills a text record per 1,000 characters **rounded up, minimum one per document**. This corpus
has a median review of 195 characters, so nearly every review costs a whole record — 5,000
records/month is worth about **5,000 short reviews, not 19,000**. The planned run is query group
24: 23 listings, 393 reviews, **393 records (7.9 % of the month), 40 requests**.

Run it with `uv run python -m rental_ranking.cloud.sentiment` (add `--estimate-only` to price it
without calling). The response is cached to `data/sentiment/`, and every rerun reads the cache —
`--refresh` is the only way to bill again, and it says so.

### Teardown

- [ ] `az ml compute delete --name cpu-cluster` — optional; min-instances 0 already bills nothing
- [ ] **`az acr delete`** — this is the $5/month one
- [ ] `az cognitiveservices account delete --name nf-rental-language -g nf-rental-ranking` —
      F0 is free, but one free instance per subscription is a resource worth releasing
- [ ] `az group delete --name nf-rental-ranking` when the project wraps — removes everything
