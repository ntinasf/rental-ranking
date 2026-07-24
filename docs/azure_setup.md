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
and later the assembled feature table (Phase 2). Registration is a one-time act per
snapshot, done from here after inspecting the downloads — deliberately not part of
download.py.

```sh
# Raw, once per city after download + inspection (version = snapshot date):
# az ml data create --name raw-thessaloniki --version 2026.06.29 --type uri_folder \
#   --path data/raw/thessaloniki/2026-06-29
# az ml data create --name raw-athens --version 2026.06.28 --type uri_folder \
#   --path data/raw/athens/2026-06-28
# az ml data create --name raw-crete --version 2026.06.29 --type uri_folder \
#   --path data/raw/crete/2026-06-29

# Feature table, Phase 2 (the training job's input):
# az ml data create --name features --version <v> --type uri_file \
#   --path data/processed/feature_table.parquet
```

`az ml data create` with a local `--path` uploads to the workspace's Blob storage and
registers in one step. The container is private — PII in raw is storage, not publication.

## Standing rule

Managed endpoints: deploy → demo → screenshot → **tear down**. Never left running.
