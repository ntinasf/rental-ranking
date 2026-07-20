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
az configure --defaults group=<RESOURCE_GROUP> workspace=<WORKSPACE_NAME>
```

## Resource group + workspace

```sh
# TODO: fill in actual names and region once created
# az group create --name <RESOURCE_GROUP> --location <REGION>
# az ml workspace create --name <WORKSPACE_NAME> --resource-group <RESOURCE_GROUP>
```

Region note: confirm Azure AI Search and AI Language availability in-region *before* choosing.

## Compute cluster (scale-to-zero)

```sh
# TODO: adjust size after quota approval
# az ml compute create --name cpu-cluster --type AmlCompute \
#   --size Standard_DS3_v2 --min-instances 0 --max-instances 2
```

Min instances 0 — an idle cluster costs nothing.

## Budget alert

Created via portal (Cost Management → Budgets): €25–30/month, alerts at 50% and 80%.

## Data assets

Registered from the CLI/SDK once raw files are downloaded (see BUILD_GUIDE Phase 0):

```sh
# TODO per city/snapshot:
# az ml data create --name raw-<city> --version <snapshot-date> --type uri_folder --path data/raw/<city>
```

## Standing rule

Managed endpoints: deploy → demo → screenshot → **tear down**. Never left running.
