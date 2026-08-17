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
