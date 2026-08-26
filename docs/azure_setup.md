# Running this project on Azure ML

Development, training and evaluation all run locally. Azure ML is here to demonstrate the cloud
workflow: versioned data assets, a command job, a two-step pipeline job, and a managed endpoint
that is deployed, exercised and deleted in one session. **The local build stays canonical**, and
nothing in this document is required to reproduce the model or any number in the report.

This is a working guide rather than a log: every command below runs against your own subscription,
and the names are yours to choose. Two things to know before starting.

**You cannot reproduce this project's hashes.** Listing ids are salted before anything is written
to disk, and the salt is private. A reproducer building from the same raw snapshots gets a
structurally identical dataset with different ids, so the SHA-256 values quoted here are internal
consistency checks for one salt, not targets to hit.

**Set the names once.** Every command below uses them explicitly, so the guide works whether or
not you configure CLI defaults:

```sh
export RG=<your-resource-group>
export WS=<your-workspace>
export LOC=italynorth        # any region with AI Language F0, if you want the sentiment demo
```

---

## What it costs, and where the cost actually is

Measured for `italynorth` from the Azure retail price API. Prices are region-specific; re-check
yours rather than trusting this table.

| item | price | note |
|---|---|---|
| `Standard_F4s_v2` dedicated, Linux | **$0.1940/hr** | one training job ≈ 30 min ≈ **$0.10** |
| `Standard_F4s_v2` low-priority | $0.0388/hr | 80 % cheaper, preemptible, not worth it for a one-shot demo |
| `Standard_DS2_v2` (endpoint instance) | $0.1360/hr | ≈ **$98/month** if you forget to delete it |
| **ACR Basic** | **$0.1666/day = $5.00/month** | created by the first environment build, and it **persists** |

**The compute is not the cost; the registry is.** Running the entire pipeline end to end, meaning
the data build, features, training and the whole 35-configuration sweep, is roughly three to four
hours of F4s_v2, which is **under $1**. The container registry that the first custom-environment
build creates costs 50× the job that created it, every month, whether or not anything else is ever
submitted.

Two consequences worth internalising before you submit anything:

- **A failed image build still creates the registry.** The meter starts at the first *attempt*, not
  the first success. Budget from submission.
- **`--min-instances 0` genuinely costs nothing idle**, so the cluster is not the thing to tear
  down. The ACR is.

Set a budget alert before the first job: Cost Management → Budgets, €25–30/month, alerts at 50 %
and 80 %.

---

## 1. Prerequisites

```sh
brew install azure-cli          # or the platform equivalent
az extension add -n ml
az login
az configure --defaults group="$RG" workspace="$WS"
```

The defaults line is not optional decoration. Without it every `az ml` command needs both `-g` and
`-w`, and supplying only one produces a **misleading** error:

```
(ResourceGroupNotFound) Resource group 'None' could not be found
```

which names the resource group when the missing argument is the workspace. Verify with
`az configure --list-defaults -o table`; empty output means they were never set in this shell.

---

## 2. Provision

```sh
az group create --name "$RG" --location "$LOC"
az ml workspace create --name "$WS" --resource-group "$RG"
```

**Check quota before anything else**, because increase requests take days to approve:

```sh
az ml compute list-usage -g "$RG" -w "$WS" -o table
```

This project needs very little. It is CPU-only, so zero GPU quota is irrelevant. Training takes one
4-vCPU node and the endpoint demo a 2-vCPU instance with a ×1.2 reservation, and they never run at
the same time. A subscription with 16 dedicated cores in the FSv2 family is comfortable.

```sh
az ml compute create --name cpu-cluster --type amlcompute --size Standard_F4s_v2 \
  --min-instances 0 --max-instances 1 --idle-time-before-scale-down 120 \
  -g "$RG" -w "$WS"
```

`Standard_F4s_v2` is 4 vCPU compute-optimised and fits the FSv2 quota; `Standard_DS3_v2` also
works. Minimum instances 0 means an idle cluster costs nothing.

If you want the sentiment demonstration, confirm AI Language (TextAnalytics) is available in your
region, including the free F0 tier. It does not have to be co-located with the workspace.

---

## 3. Register the data assets

Two layers are registered: **raw**, one asset per city per snapshot, and the **feature table** the
training job consumes. The processed parquets in between are deliberately not registered. They are
derivable, since raw plus `data/build.py` reproduces them exactly, and a derivable artifact earns a
version only when something needs to *name* it. The preprocessing pipeline's outputs are pipeline
outputs, not new assets.

Registration is a one-time act per snapshot, done after inspecting the downloads. It is deliberately
not part of `download.py`, which must work with no Azure credentials at all.

**Specified in YAML, not in flags.** The four assets live in `pipelines/data/*.yml`. The deciding
reason is that `az ml data create` **has no `--tags` argument** in the ML extension, and tags are
the entire traceability scheme below. A second reason: `path` inside a data YAML resolves relative
to the **YAML file**, not the working directory, so the file can be run from anywhere in the repo.

```sh
# Raw, once per city after download and inspection. These YAMLs are written once.
for city in thessaloniki athens crete; do
  az ml data create -f "pipelines/data/raw_${city}.yml" -g "$RG" -w "$WS"
done

# Feature table. Build it, bump `version` in the YAML, then register.
uv run python -m rental_ranking.features.build
az ml data create -f pipelines/data/feature_table.yml -g "$RG" -w "$WS" \
  --set tags.git=$(git rev-parse --short HEAD)
```

Registering an existing name and version **fails rather than overwriting**. Assets are immutable,
which is the property that makes a run tag worth trusting; to re-register after a change, bump the
version.

**Versioning, by layer.** Raw is versioned by **snapshot date**, the only thing that distinguishes
one raw pull from another and something that never changes once downloaded. The feature table is
versioned by **build date**, because it moves for two independent reasons, a new snapshot *or* a
feature change, and the date is the one string that increments under both. Tags carry what the date
cannot: `raw_versions` names the three assets it derives from, and `git` names the commit that
built it. Every training run logs the table version as an MLflow tag, so a model traces to a table,
a table traces to a commit and three snapshots, and the chain closes.

`git` is kept out of `feature_table.yml` and injected with `--set` on purpose. A hardcoded SHA that
someone forgets to bump is a *wrong* tag; a forgotten `--set` leaves a *missing* tag. Only one of
those breaks the chain silently.

`pipelines/train_job.yml` names a specific version rather than `@latest`, so a re-registration
cannot silently change what a recorded run was trained on. Point it at whatever version you
register.

A local `path` uploads to the workspace's default Blob container and registers in one step: about
300 MB for the three raw folders, 3 MB for the feature table. The container is private, which is
the relevant boundary: PII in raw is storage, not publication. The anonymisation policy is in
[data_pipeline_design.md](data_pipeline_design.md).

---

## 4. The training job

```sh
# Re-pin the environment from uv.lock BEFORE submitting. An unpinned cloud env is how
# "it trained differently up there" happens.
az ml job create -f pipelines/train_job.yml -g "$RG" -w "$WS"
```

**Two things the job needs that local running does not.**

1. **`matplotlib` in `environment.yml`.** `train.py` logs the per-fold learning curves as a run
   artifact, so the cloud image needs it even though nothing is displayed.
2. **`--dataset-version` passed explicitly.** The job snapshot is `src/` alone, so `PROJECT_ROOT`
   resolves to the working directory and `pipelines/data/feature_table.yml` is absent.
   `dataset_version()` would silently return `"unregistered"` and break the one traceability claim
   the job exists to demonstrate.

**On cloud-versus-local numbers.** LightGBM's multithreaded histogram construction is not
order-guaranteed across a 10-thread macOS ARM build and a 4-vCPU Linux x86 one, so a divergence
from the local score would be a platform artifact rather than a bug. That caveat was recorded
before the first run and **no divergence occurred**: the cloud reproduced the development folds
exactly (0.7026 / 0.7632 / 0.7078 / 0.7119, stopping at 514 / 718 / 158 / 211, median 362), and the
run's dataset digest matched the local one, proving the same bytes were read. Determinism held
across platform, architecture and thread count.

**One known logging gap.** The MLflow tag reads `git_commit : unknown` while Azure's own Git field
carries the real SHA. The job runs from a code snapshot with no `.git`, so the tag has nothing to
read; passing the commit as a job parameter at submit time is the fix.

---

## 5. The preprocessing pipeline job

The single-step training job demonstrates a versioned asset going in and an MLflow run coming out.
What it cannot show is a **multi-step pipeline**: two components with an intermediate flowing
between them, which is what most people mean by "an Azure ML pipeline". That is the only reason
`pipelines/preprocess_pipeline.yml` exists.

It **reuses the training environment**, so no image is built. The build path imports exactly
`pandas`, `numpy`, `scipy`, `pyarrow` and `dotenv`, traced by importing both build modules and
diffing `sys.modules`. The first four are pinned directly; `python-dotenv` arrives transitively
through `mlflow`, verified against PyPI rather than assumed.

Measured peak memory, so the instance size is a decision rather than a hope: **step 1 4.4 GiB in
14 s, step 2 2.1 GiB in 3 s**, against `Standard_F4s_v2`'s 8 GiB.

### The salt is the one input that can break the run

Hashing is `sha256(f"{salt}:{value}")[:16]`, so **a different salt produces entirely different
listing ids**: a processed layer that looks perfectly healthy and matches nothing, not the
registered feature asset and not any committed evidence. Nothing downstream fails loudly. This is
why the salt is handled deliberately rather than pasted into the YAML.

`ANON_SALT: ""` is declared **empty on purpose** in the pipeline YAML. `anonymize._resolve_salt`
raises on a blank salt, so a forgotten `--set` fails in the first second instead of building a
whole dataset under the wrong one.

Step 1 prints a **one-way fingerprint** of the salt before doing any work:

```
ANON_SALT length 32, sha256 prefix <first 16 hex of sha256(salt)>
```

and refuses to continue on an empty value **or** on an unsubstituted `${{...}}` placeholder. The
second is the dangerous one: it is non-empty, so without the check it would build a complete
dataset under a salt that is literally the placeholder text. The fingerprint is a hash, so it is
safe in a job log, and it confirms the *right* salt arrived rather than merely some salt.

### Storing the salt in the workspace Key Vault

Every AML workspace is created with a Key Vault attached, so this adds no resource. Storing the
salt there makes the vault the system of record instead of a `.env` file on one laptop.

**1. Find the vault.** Portal → your workspace → **Overview** → in the **Essentials** panel, click
the **Key vault** link. The name is auto-generated at workspace creation, which is why it is worth
following the link rather than guessing.

**2. Check the permission model.** Vault → **Settings → Access configuration**. It reads either
*Azure role-based access control* or *Vault access policy*. Doing the wrong one of the two steps
below appears to succeed and still leaves you unable to write the secret.

**3a. RBAC vaults.** Vault → **Access control (IAM)** → **+ Add** → **Add role assignment**. Take
**Key Vault Secrets Officer**, not *Secrets User*, which can only read. Assign it to your own
account. Role assignments take a minute or two to propagate; a permissions error at step 4 usually
means waiting, not failure.

**3b. Access-policy vaults.** Vault → **Access policies** → **+ Create**, tick **Get**, **List**
and **Set** under Secret permissions, select your own account. Leave any existing policy
alone: the workspace's managed identity has one, and removing it breaks the workspace.

**4. Create the secret** named `anon-salt`, value only.

> **No `ANON_SALT=` prefix and no surrounding quotes.** A stray quote or newline produces a
> *different* salt, and a different salt produces a full dataset whose listing ids match nothing,
> with no error anywhere. The fingerprint check below catches it in one second.

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

> **The salt travels as a pipeline-level input, not an environment-variable override.** The first
> attempt used `--set jobs.build_processed.environment_variables.ANON_SALT=...`; `--set` accepts
> nested `jobs.*` properties and reports no error, but the override never reached the container.
> `inputs.anon_salt` is the parameterisation Azure ML actually supports: the service substitutes
> it into the step's command rather than the CLI patching a nested object.

If you would rather pull it from the vault than paste it:

```sh
KV=$(az ml workspace show -n "$WS" -g "$RG" --query key_vault -o tsv | awk -F/ '{print $NF}')
ANON_SALT=$(az keyvault secret show --vault-name "$KV" --name anon-salt --query value -o tsv)
```

**Be precise about what the vault achieves.** It becomes the system of record and the secret stays
out of the repo and the job YAML, both real gains. But the value is injected as a job input, so it
**is visible in the job's properties** to anyone with workspace access. For a single-owner
workspace that is torn down afterwards, that is proportionate.

The stricter alternative, worth knowing as the production answer: give the compute cluster a
**managed identity**, grant *that* identity `get` on the secret, and have the job read the vault at
runtime via `azure-identity` and `azure-keyvault-secrets`, so the secret never leaves Azure and
never appears in job metadata. It costs an identity, an RBAC grant, two packages in the environment
and a shim that sets `os.environ["ANON_SALT"]` before `main()`, because `build.py` reads the
variable rather than a vault. Not taken here: real setup for a job that runs once before the
workspace is deleted, removing an exposure to the workspace's only user.

### What the run establishes

Both steps print a SHA-256 of what they wrote. In this project's run, **three of the four artifacts
were byte-identical to the local build, including the feature table**; `listings.parquet` differed.

`listings` is the only one of the three processed files carrying a nested column (`amenities`, a
`list<string>`); calendar and reviews are flat primitives and both matched. Arrow's physical
encoding of nested types can differ between an ARM macOS build and an x86 Linux one while the
logical content is unchanged.

The byte-identical feature table is the load-bearing part: it is built from `listings` through
label construction, filters, price imputation, grading, grouping and assembly, so a content
difference in any column feeding a feature would have propagated into it. The discrepancy is
confined to physical encoding, or to columns no feature reads.

**What that does not establish**, stated so nobody reads more into it: nobody compared the two
`listings.parquet` files column by column, so "identical content" is an inference from the
downstream hash rather than a measurement. The honest claim is *the feature table the model trains
on reproduces exactly; the intermediate listings file differs in physical encoding*, not "the
pipeline reproduces byte-for-byte".

---

## 6. The managed endpoint

⚠ **The only resource here with a meter that runs while you are not looking.** An endpoint with no
deployment costs nothing; the moment a deployment provisions an instance it bills per hour
regardless of traffic. Deploy → invoke → capture → **delete**, same session.

```sh
# 1. Local Docker smoke test FIRST. This is not ceremony; see the findings below.
az ml online-deployment create --local -f pipelines/deployment.yml
az ml online-endpoint invoke --local --name rental-ranker --request-file request.json
az ml online-deployment delete --local --name blue --endpoint-name rental-ranker --yes

# 2. Then the real one
az ml online-endpoint create -f pipelines/endpoint.yml -g "$RG" -w "$WS"
az ml online-deployment create -f pipelines/deployment.yml --all-traffic -g "$RG" -w "$WS"

# 3. TEAR DOWN, same session
az ml online-endpoint delete --name rental-ranker --yes -g "$RG" -w "$WS"
```

**A custom scoring script, not a no-code MLflow deployment**, for three reasons: no-code returns
raw predictions in input order rather than a ranked list keyed on `id`, it offers nowhere to
validate a request, and the MLflow signature cannot express the five `category` columns.

### What the local smoke test caught

1. **`ModuleNotFoundError: No module named 'rental_ranking'`.** Azure copies
   `code_configuration.code` to `/var/azureml-app/<dir>` and puts **only the script's own
   directory** on `sys.path`; the package is not pip-installed. `init()` succeeded and the container
   reported healthy, then it failed at *request* time. Fixed by inserting the package root into
   `sys.path` inside the scoring script.
2. **`az ml online-deployment update --local` breaks** with `conda: error: unrecognized arguments:
   --root`. Delete and recreate the local deployment instead of updating it.
3. **Model asset versions must be positive integers.** A date string, the convention used for data
   assets and environments, is rejected for *models*. **The local deployment accepted it; only the
   cloud validates.** The build date moved to a tag.

### The categorical contract, measured rather than assumed

Two of three guesses about categorical handling at inference turned out to be wrong:

| claim | reality |
|---|---|
| Category *code order* shifts → silent wrong answers | **False.** The booster stores training levels in `pandas_categorical` and re-maps *by label*. A request covering 2 of 3 levels scored identically, to 0.000000 |
| JSON strings score fine | **False, but loud.** `object` dtype raises `train and valid dataset categorical_feature do not match` |
| (not guessed) | **The real silent bug is an unseen level.** `room_type="Houseboat"` predicts with no error and **0.1083** away from truth, because `set_categories` turns it into NaN and the model scores it as *missing* |

So `serving_metadata.json` ships beside the booster and `restore_dtypes` rejects unseen levels.
That is the one failure that would otherwise reach a caller as a confident number.

### The shape of the cost

One captured session: **16 minutes to provision, 29 seconds of use, 8 minutes to delete**, about
9 minutes of billed instance life on `Standard_DS2_v2`, roughly **$0.02**. The demo is cheap; what
would not be cheap is forgetting the delete. The endpoint itself is free until a deployment
attaches an instance.

Cloud scores matched the local booster to the last digit. The captured request and response pairs
are in [`endpoint_demo/`](endpoint_demo/), and notebook 04 §10 recomputes the comparison rather
than asserting it.

**Take the scoring URI from the CLI, not the portal.** The Studio endpoint page lists the
**Swagger URI** directly below the **REST endpoint**, and only the second one scores. Posting to
`/swagger.json` returns `HTTP 424: The method is not allowed for the requested URL`, a 424 from
the front door wrapping a 405 from the inference server, with nothing in the chain naming the URL
or the mistake, and the endpoint perfectly healthy throughout.

```sh
az ml online-endpoint show --name rental-ranker --query scoring_uri -o tsv -g "$RG" -w "$WS"
```

### Driving it

`cloud/demo.py` builds requests from held-out listings, joins each response back to the held-out
grades locally, and scores the endpoint against both frozen baselines and a random floor on the
same candidate set. Ranking against no truth reads the same whether the model is good or shuffled,
which is why the join happens on the caller rather than in the container.

```sh
# no Azure account, no cost: same init()/run() the container calls
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

**F0 is free and stays free**, but the budget is smaller than the pricing page suggests. Azure
bills a text record per 1,000 characters **rounded up, with a minimum of one per document**. This
corpus has a median review of 195 characters, so nearly every review costs a whole record: 5,000
records a month is worth about **5,000 short reviews, not 19,000**.

```sh
uv run python -m rental_ranking.cloud.sentiment --estimate-only   # price it without calling
uv run python -m rental_ranking.cloud.sentiment
```

Responses are cached to `data/sentiment/` and every rerun reads the cache. `--refresh` is the only
way to bill again, and it says so. Sentiment is a workflow demonstration only; it is not a model
feature, and notebook 03 §9 explains why it was measured and rejected.

---

## 8. Teardown

**The standing rule: deploy → demo → capture → tear down.** Managed endpoints are never left
running.

```sh
az ml online-endpoint delete --name rental-ranker --yes -g "$RG" -w "$WS"
az cognitiveservices account delete --name <language-resource> -g "$RG"
az acr delete --name <registry> -g "$RG"          # the $5/month one
az ml compute delete --name cpu-cluster -g "$RG" -w "$WS"   # optional, bills nothing idle
az group delete --name "$RG"                      # removes everything
```

**One wrinkle the Key Vault adds.** Vaults have **soft delete** enabled and it cannot be turned
off, so deleting the resource group does not remove the vault outright: it goes into a soft-deleted
state for the retention period, 90 days by default. That costs nothing, but the **name stays
reserved**, so recreating a workspace later can collide with it. To clear it immediately: Portal →
**Key vaults** → **Manage deleted vaults** → select the region → select the vault → **Purge**.
Purge is blocked if *purge protection* was enabled; AML enables soft delete but generally leaves
purge protection off, so this normally works.

---

## Appendix: failures worth keeping

The demonstration is one training job. Getting there took four submissions and about $0.40 of
compute. Each failure was a genuine incompatibility rather than a typo.

**1. "Pin from uv.lock" is not a text operation.** The image build failed with:

```
ERROR: Could not find a version that satisfies the requirement numpy==2.5.1
ERROR: Ignored the following versions that require a different python version:
       2.5.1 Requires-Python >=3.12
```

`requires-python = ">=3.11"`, so `uv.lock` carries a resolution for *every* supported interpreter:

```
{ name = "numpy", version = "2.4.6", marker = "python_full_version < '3.12'" }
{ name = "numpy", version = "2.5.1", marker = "python_full_version >= '3.12'" }
```

Scraping the first version string per package silently selects the 3.12 branch and pins it against
`python=3.11`. Two packages were wrong the same way: `numpy` and `scipy`. The correct source is the
**resolved** environment, the lock as this interpreter actually solved it, which is what the
regenerate command in `environment.yml` now reads. The conda pin of `python=3.11` is load-bearing,
not cosmetic.

**2. `azureml-mlflow` requires `mlflow-skinny<=3.13.0`.** With `mlflow==3.14.0` pinned, MLflow 3.14
passes `tracking_uri` into `get_artifact_repository`, which the plugin's builder does not accept:
`TypeError: azureml_artifacts_builder() got an unexpected keyword argument 'tracking_uri'`.
**Training had already finished** when it fired. Fixed by holding the cloud at `mlflow==3.13.0`,
the one package allowed to differ from local, because it is the tracking client and takes no part
in the computation.

**3. Azure ML does not serve MLflow 3's LoggedModel API.** `mlflow.lightgbm.log_model` posts to
`/api/2.0/mlflow/logged-models`, which returns **404** against the workspace. Again the run had
completed and its tags had landed; only the last line failed. Fixed by attempting the flavoured
model and falling back to a raw `booster.txt` artifact with an explanatory tag. The consequence
reached the endpoint step: deploying an MLflow-flavoured model was the clean path and was not
available, which is why the scoring script loads the booster directly.

**4. One bug the smoke test did not catch, and the demo did.** A feature column present and `null`
for every listing arrives from `json.loads` as `object`, and LightGBM rejects the frame with
`pandas dtypes must be int, float or bool`, an unhandled exception, so a 500 with no message. An
*absent* column was always fine, because `reindex` gives it `float64` NaN, which is why a "sparse"
request with 59 of 61 features missing passed while a cold-start request crashed. `restore_dtypes`
now coerces and names the offending column.
