# Running this project on Azure ML

Development, training and evaluation all run locally. Azure ML demonstrates the cloud workflow:
versioned data assets, a command job, a two-step pipeline job, and a managed endpoint that is
deployed, exercised and deleted in one session. **The local build stays canonical**, and nothing
here is required to reproduce the model or any number in the report.

Every command runs against your own subscription. Set the names once — they are used explicitly
throughout, so this works whether or not you configure CLI defaults:

```sh
export RG=<your-resource-group>
export WS=<your-workspace>
export LOC=italynorth        # any region with AI Language F0, if you want the sentiment demo
```

---

## 1. Prerequisites

```sh
brew install azure-cli          # or the platform equivalent
az extension add -n ml
az login
az configure --defaults group="$RG" workspace="$WS"
```

Every `az ml` command needs both `-g` and `-w`. Supplying only one reports
`(ResourceGroupNotFound) Resource group 'None' could not be found`, which names the resource group
when the missing argument is the workspace. Verify with `az configure --list-defaults -o table`;
empty output means they were never set in this shell.

---

## 2. Provision

```sh
az group create --name "$RG" --location "$LOC"
az ml workspace create --name "$WS" --resource-group "$RG"
```

**Check quota first** — increase requests take days to approve:

```sh
az ml compute list-usage -g "$RG" -w "$WS" -o table
```

The project is CPU-only, so GPU quota is irrelevant. Training takes one 4-vCPU node and the
endpoint demo a 2-vCPU instance with a ×1.2 reservation, and they never run at the same time.
16 dedicated cores in the FSv2 family is comfortable.

```sh
az ml compute create --name cpu-cluster --type amlcompute --size Standard_F4s_v2 \
  --min-instances 0 --max-instances 1 --idle-time-before-scale-down 120 \
  -g "$RG" -w "$WS"
```

`Standard_F4s_v2` is 4 vCPU compute-optimised and fits the FSv2 quota; `Standard_DS3_v2` also
works. Minimum instances 0 means an idle cluster consumes nothing.

For the sentiment demonstration, check that AI Language offers **sentiment analysis** in your
region, not merely that the resource kind creates there. A `Microsoft.CognitiveServices` account
provisions successfully in regions that do not serve sentiment, and the gap surfaces only in
Language Studio or at call time. The resource does not have to sit in the workspace's region.

---

## 3. Register the data assets

Two layers are registered: **raw**, one asset per city per snapshot, and the **feature table** the
training job consumes. The processed parquets in between are not registered — raw plus
`data/build.py` reproduces them exactly, and the preprocessing pipeline's outputs are pipeline
outputs rather than new assets.

The four assets are declared in `pipelines/data/*.yml` rather than as CLI flags, because
`az ml data create` has **no `--tags` argument** and tags carry the traceability below. `path`
inside a data YAML resolves relative to the **YAML file**, so these run from anywhere in the repo.

```sh
# Raw, once per city after download and inspection.
for city in thessaloniki athens crete; do
  az ml data create -f "pipelines/data/raw_${city}.yml" -g "$RG" -w "$WS"
done

# Feature table. Build it, bump `version` in the YAML, then register.
uv run python -m rental_ranking.features.build
az ml data create -f pipelines/data/feature_table.yml -g "$RG" -w "$WS" \
  --set tags.git=$(git rev-parse --short HEAD)
```

Registering an existing name and version **fails rather than overwriting**. To re-register after a
change, bump the version.

Raw is versioned by **snapshot date**; the feature table by **build date**, since it moves for a
new snapshot *or* a feature change. Tags carry the rest: `raw_versions` names the three assets it
derives from, `git` names the commit that built it. Every training run logs the table version as an
MLflow tag, so a model traces to a table, and a table to a commit and three snapshots.

`git` is injected with `--set` rather than written into the YAML: a stale hardcoded SHA is a
*wrong* tag, while a forgotten `--set` is merely a *missing* one.

`pipelines/train_job.yml` names a specific version rather than `@latest`, so a re-registration
cannot silently change what a recorded run was trained on. Point it at the version you register.

A local `path` uploads to the workspace's default Blob container and registers in one step: about
300 MB for the three raw folders, 3 MB for the feature table. That container is private, which is
the relevant boundary for PII in raw. The anonymisation policy is in
[data_pipeline_design.md](data_pipeline_design.md).

---

## 4. The training job

Re-pin the environment from `uv.lock` **before** submitting — the regenerate command is in
`pipelines/environment.yml`. An unpinned cloud environment is how "it trained differently up there"
happens.

```sh
az ml job create -f pipelines/train_job.yml -g "$RG" -w "$WS"
```

Two things the job needs that local running does not:

1. **`matplotlib` in `environment.yml`.** `train.py` logs the per-fold learning curves as a run
   artifact, so the image needs it even though nothing is displayed.
2. **`--dataset-version` passed explicitly.** The job snapshot is `src/` alone, so `PROJECT_ROOT`
   resolves to the working directory and `pipelines/data/feature_table.yml` is absent.
   `dataset_version()` would otherwise return `"unregistered"` and break the traceability the job
   exists to demonstrate.

---

## 5. The preprocessing pipeline job

The training job shows a versioned asset going in and an MLflow run coming out. It cannot show a
**multi-step pipeline** — two components with an intermediate flowing between them — which is the
only reason `pipelines/preprocess_pipeline.yml` exists. It reuses the training environment, so no
image is built.

### The salt

Hashing is `sha256(f"{salt}:{value}")[:16]`, so **a different salt produces entirely different
listing ids**: a processed layer that looks healthy and matches nothing, neither the registered
feature asset nor any committed evidence, with no downstream failure. Handle it deliberately.

`ANON_SALT: ""` is declared **empty on purpose** in the pipeline YAML. `anonymize._resolve_salt`
raises on a blank salt, so a forgotten `--set` fails in the first second rather than building a
whole dataset under the wrong one.

Step 1 prints a one-way fingerprint before doing any work:

```
ANON_SALT length 32, sha256 prefix <first 16 hex of sha256(salt)>
```

and refuses to continue on an empty value **or** on an unsubstituted `${{...}}` placeholder. The
placeholder is the dangerous case: it is non-empty, so without the check it would build a complete
dataset under a salt that is literally the placeholder text.

### Storing the salt in the workspace Key Vault

Every AML workspace is created with a Key Vault attached, so this adds no resource.

**1. Find the vault.** Portal → your workspace → **Overview** → **Essentials** panel → the
**Key vault** link. The name is auto-generated at workspace creation.

**2. Check the permission model.** Vault → **Settings → Access configuration**, which reads either
*Azure role-based access control* or *Vault access policy*. Doing the wrong one of the two steps
below appears to succeed and still leaves you unable to write the secret.

**3a. RBAC vaults.** Vault → **Access control (IAM)** → **+ Add** → **Add role assignment**. Take
**Key Vault Secrets Officer**, not *Secrets User*, which can only read. Assign it to your own
account; propagation takes a minute or two.

**3b. Access-policy vaults.** Vault → **Access policies** → **+ Create**, tick **Get**, **List**
and **Set** under Secret permissions, select your own account. Leave existing policies alone — the
workspace's managed identity has one, and removing it breaks the workspace.

**4. Create the secret** named `anon-salt`, value only.

> **No `ANON_SALT=` prefix and no surrounding quotes.** A stray quote or newline is a *different*
> salt, and produces a full dataset whose listing ids match nothing, with no error anywhere.

<details>
<summary>CLI equivalent</summary>

```sh
KV=$(az ml workspace show -n "$WS" -g "$RG" --query key_vault -o tsv | awk -F/ '{print $NF}')

az role assignment create --role "Key Vault Secrets Officer" \
  --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --scope "$(az keyvault show -n "$KV" --query id -o tsv)"

SALT=$(grep -m1 '^ANON_SALT=' .env | cut -d= -f2- | tr -d "\"'")
az keyvault secret set --vault-name "$KV" --name anon-salt --value "$SALT" --output none
unset SALT
```

</details>

The secret is injected as a job input, so it **is visible in the job's properties** to anyone with
workspace access. The stricter alternative is to give the compute cluster a managed identity, grant
that identity `get` on the secret, and read the vault at runtime via `azure-identity` — so the
secret never appears in job metadata.

### Submitting

The salt is scoped to **step 1 only**; step 2 reads ids that are already hashed.

```sh
# `read -rs` does not echo, and the value never enters shell history.
read -rs ANON_SALT

# Confirm the paste WITHOUT printing it. Compare against your own recorded fingerprint.
printf '%s' "$ANON_SALT" | shasum -a 256 | cut -c1-16     # sha256sum on Linux

az ml job create -f pipelines/preprocess_pipeline.yml \
  --set inputs.anon_salt="$ANON_SALT" -g "$RG" -w "$WS"

unset ANON_SALT
```

> **The salt travels as a pipeline-level input, not an environment-variable override.** `--set
> jobs.build_processed.environment_variables.ANON_SALT=...` is accepted and reports no error, but
> never reaches the container. `inputs.anon_salt` is the parameterisation Azure ML supports.

To pull it from the vault instead of pasting:

```sh
KV=$(az ml workspace show -n "$WS" -g "$RG" --query key_vault -o tsv | awk -F/ '{print $NF}')
ANON_SALT=$(az keyvault secret show --vault-name "$KV" --name anon-salt --query value -o tsv)
```

---

## 6. The managed endpoint

⚠ **The only resource here that runs while you are not looking.** An endpoint with no deployment
is inert; the moment a deployment provisions an instance, that instance stays up regardless of
traffic. Deploy → invoke → capture → **delete**, same session.

```sh
# 1. Local Docker smoke test FIRST.
az ml online-deployment create --local -f pipelines/deployment.yml
az ml online-endpoint invoke --local --name rental-ranker --request-file request.json
az ml online-deployment delete --local --name blue --endpoint-name rental-ranker --yes

# 2. Then the real one
az ml online-endpoint create -f pipelines/endpoint.yml -g "$RG" -w "$WS"
az ml online-deployment create -f pipelines/deployment.yml --all-traffic -g "$RG" -w "$WS"

# 3. TEAR DOWN, same session
az ml online-endpoint delete --name rental-ranker --yes -g "$RG" -w "$WS"
```

The deployment uses a **custom scoring script rather than a no-code MLflow deployment**: no-code
returns raw predictions in input order rather than a ranked list keyed on `id`, offers nowhere to
validate a request, and the MLflow signature cannot express the five `category` columns.

Two constraints worth knowing before you iterate:

- **`az ml online-deployment update --local` fails** with `conda: error: unrecognized arguments:
  --root`. Delete and recreate the local deployment instead of updating it.
- **Model asset versions must be positive integers**, unlike the date strings used for data assets
  and environments. The local deployment accepts a date string; only the cloud rejects it.

**Take the scoring URI from the CLI, not the portal.** The Studio endpoint page lists the
**Swagger URI** directly below the **REST endpoint**, and only the second one scores. Posting to
`/swagger.json` returns `HTTP 424: The method is not allowed for the requested URL` with nothing in
the chain naming the mistake, and the endpoint healthy throughout.

```sh
az ml online-endpoint show --name rental-ranker --query scoring_uri -o tsv -g "$RG" -w "$WS"
```

### Driving it

`cloud/demo.py` builds requests from held-out listings, joins each response back to the held-out
grades locally, and scores the endpoint against both frozen baselines and a random floor on the
same candidate set. Ranking against no truth reads the same whether the model is good or shuffled,
which is why the join happens on the caller rather than in the container.

```sh
# no Azure account needed: same init()/run() the container calls
uv run python -m rental_ranking.cloud.demo --query thessaloniki --local --counterfactual

# against a live endpoint (AML_ENDPOINT_URI / AML_ENDPOINT_KEY in .env)
uv run python -m rental_ranking.cloud.demo --query athens --counterfactual
uv run python -m rental_ranking.cloud.demo --query thessaloniki --variant cold-start
uv run python -m rental_ranking.cloud.demo --capture      # writes docs/endpoint_demo/
```

---

## 7. The sentiment demonstration (optional)

```sh
az cognitiveservices account create --name <language-resource> --kind TextAnalytics --sku F0 \
  --location "$LOC" -g "$RG" --custom-domain <language-resource> --yes

# Endpoint and key go to .env, which is gitignored. Never commit them.
az cognitiveservices account show --name <language-resource> -g "$RG" \
  --query properties.endpoint -o tsv
az cognitiveservices account keys list --name <language-resource> -g "$RG" --query key1 -o tsv
```

The F0 tier has a monthly record quota, and Azure counts a text record per 1,000 characters
**rounded up, with a minimum of one per document**. This corpus has a median review of 195
characters, so nearly every review consumes a whole record. Size it before calling:

```sh
uv run python -m rental_ranking.cloud.sentiment --estimate-only
uv run python -m rental_ranking.cloud.sentiment
```

Responses are cached to `data/sentiment/` and every rerun reads the cache; `--refresh` is the only
path that calls the service again, and it says so. Sentiment is a workflow demonstration only — it
is not a model feature, and notebook 03 §9 explains why it was measured and rejected.

---

## 8. Teardown

**The standing rule: deploy → demo → capture → tear down.** Managed endpoints are never left
running.

```sh
az ml online-endpoint delete --name rental-ranker --yes -g "$RG" -w "$WS"
az cognitiveservices account delete --name <language-resource> -g "$RG"
az acr delete --name <registry> -g "$RG"
az ml compute delete --name cpu-cluster -g "$RG" -w "$WS"   # optional; idle clusters are inert
az group delete --name "$RG"                                # removes everything
```

The container registry is created by the first custom-environment build — including a build that
*fails* — and persists after the jobs that needed it. Delete it explicitly.

**One wrinkle the Key Vault adds.** Vaults have **soft delete** enabled and it cannot be turned
off, so deleting the resource group does not remove the vault outright: it enters a soft-deleted
state for the retention period, 90 days by default. The **name stays reserved**, so recreating a
workspace later can collide with it. To clear it immediately: Portal → **Key vaults** → **Manage
deleted vaults** → select the region → select the vault → **Purge**. Purge is blocked if *purge
protection* was enabled; AML enables soft delete but generally leaves purge protection off.
