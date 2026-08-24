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

**Why only these two.** The processed parquets are deliberately *not* registered: they are a
reproducible intermediate — raw plus `data/build.py` reproduces them exactly, verified
byte-for-byte on 2026-08-22 — so registering them would add a third thing to version and keep in
step for no traceability gain.

*Amended 2026-08-22.* This paragraph used to give a second reason: that "nothing in the cloud
consumes them, because preprocessing runs locally". That half is now **false** —
`pipelines/preprocess_pipeline.yml` consumes the raw assets and produces both layers in the
cloud. The conclusion is unchanged and the surviving reason is the one that was always doing the
work: the processed layer is derivable, and a derivable artifact earns a version only if
something needs to *name* it. The pipeline's outputs are therefore pipeline outputs, not new
assets. Preprocessing still runs locally as the real path; the pipeline is a workflow
demonstration, and the local build remains canonical.

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

## The preprocessing pipeline job (2026-08-22)

The single-step `train_job.yml` demonstrates a versioned asset going in and an MLflow run coming
out. What it cannot show is a **multi-step pipeline**: two components with an intermediate
flowing between them, which is what most people mean by "an Azure ML pipeline". That is the only
reason `pipelines/preprocess_pipeline.yml` exists. **The local build remains the real path.**

**It reuses `rental-ranking-train` — no new environment.** The build path imports exactly
`pandas`, `numpy`, `scipy`, `pyarrow` and `dotenv` (traced by importing both build modules and
diffing `sys.modules`). The first four are pinned directly; `python-dotenv` arrives through
`mlflow==3.13.0 -> mlflow-skinny==3.13.0 -> python-dotenv<2,>=0.19.0`, verified against PyPI
rather than assumed. So the already-built image is sufficient and the pipeline starts without an
image build.

**Unlike the training job, this one is expected to reproduce its local counterpart.** Verified
locally 2026-08-22 by rebuilding into an isolated root and comparing SHA-256 — all four artifacts
byte-identical, twice. The post-run check is therefore `shasum`, not eyeballing:

| artifact | sha256 |
| --- | --- |
| `processed/listings.parquet` | `870af111eb0e260c3f628284c3566f9d2f5c5f5fc265a1bc9c50c4a80b2d7033` |
| `processed/calendar.parquet` | `e4aa159e18aad04184d8d12fd144ae2c27f006aadde876ba37051a5a0caf29cc` |
| `processed/reviews.parquet` | `23149de36435a5c3a48c28decf692e9b1b337302db6d54c0ee724a076086543e` |
| `features/feature_table.parquet` | `8c74b28408544aee20ba7c47bc1485a916889321c20c8dcb4d13a0e3c70db7e0` |

### Result of the run — 2026-08-22

![The preprocessing pipeline graph](screenshots/preprocess_pipeline_dag.png)

**Job `preprocess-and-features` completed.** The salt arrived intact: step 1's first log line read
`ANON_SALT length 32, sha256 prefix [redacted]`, matching the local `.env` exactly. That
also closes the earlier fingerprint mismatch — it was the `read -rs` capture in the terminal, not
the vault and not the paste.

**Three of the four artifacts are byte-identical to the local build**, including
`feature_table.parquet`. **`listings.parquet` differs.**

The likely cause, and the reason it was not chased further: `listings` is the **only** one of the
three processed files carrying a nested column — `amenities`, a `list<string>`. Calendar and
reviews are flat primitives and both matched. Arrow's physical encoding of nested types (offset
buffers, child-array chunking) can differ between the local ARM macOS build and the cloud x86
Linux one while the logical content is unchanged.

**What the byte-identical feature table establishes.** It is built from `listings` through label
construction, filters, price imputation, grading, grouping and assembly. A content difference in
any column feeding a feature would have propagated into it. So the discrepancy is confined to
physical encoding and/or to columns no feature reads — not to the data the project uses.

**What that does NOT establish**, stated so no one reads more into it than is there: nobody has
compared the two `listings.parquet` files column by column, so "identical content" is an
inference from the downstream hash, not a measurement. The honest claim is **"the feature table
the model trains on reproduces exactly; the intermediate listings file differs in physical
encoding"** — not "the pipeline reproduces byte-for-byte". Settling it would take one download
and a `DataFrame.equals`, and is left undone deliberately.

### A note on the salt's visibility

Confirmed in practice: the pipeline's `anon_salt` input **is visible in the job's inputs** in
Studio. That is the exposure this document predicted for the environment-variable approach and it
applies equally to a pipeline parameter — the value is job metadata either way.

**Do not rotate the salt in response.** Hashing is `sha256(f"{salt}:{value}")[:16]`, so a new salt
changes every listing id and would invalidate the registered `features` asset and the committed
`docs/endpoint_demo/` evidence. The exposure is bounded by workspace access, the salt protects
linkage to ids that are public anyway, and deleting the workspace removes it. Rotating would cost
real artifacts to fix a risk that teardown already closes.

Measured peak memory, so the instance size is a decision rather than a hope: **step 1 4.4 GiB /
14 s, step 2 2.1 GiB / 3 s** against `Standard_F4s_v2`'s 8 GiB. Comfortable, and the reason the
existing cluster needs no resizing.

### The salt, and why it is the one input that can break the run

Hashing is `sha256(f"{salt}:{value}")[:16]`, so **a different salt produces entirely different
listing ids** — a processed layer that looks perfectly healthy and matches nothing: not the
registered `features:2026.08.17` asset, not the endpoint demo captures, not the hashes above.
Nothing downstream would fail loudly. This is why the salt is handled deliberately rather than
pasted into the YAML.

`ANON_SALT: ""` is declared **empty on purpose** in the pipeline YAML. `anonymize._resolve_salt`
raises on a blank salt, so a forgotten `--set` fails in the first second instead of building a
whole dataset under the wrong one.

### Put the salt in the workspace Key Vault — portal

Every AML workspace is created with a Key Vault attached; this uses that one rather than adding a
resource. Storing the salt there makes the vault the system of record instead of a `.env` file
that exists on exactly one laptop.

**1 — Find the vault.** Portal → search `nf-rental-ranking-ws` → the workspace's **Overview**
page → in the **Essentials** panel, click the **Key vault** link. The name is auto-generated at
workspace creation (something like `nfrentalrankingws0123456789`), which is why it is worth
following the link rather than guessing it.

**2 — Check which permission model the vault uses.** In the vault's left nav:
**Settings → Access configuration**. It reads either *Azure role-based access control* or
*Vault access policy*. Do whichever of 3a / 3b matches — doing the wrong one appears to succeed
and still leaves you unable to write the secret.

**3a — RBAC vaults.** Vault → **Access control (IAM)** → **+ Add** → **Add role assignment**.

- *Role* tab: search **Key Vault Secrets Officer**. Take that one, not *Key Vault Secrets User* —
  User can only read, and you need to write.
- *Members* tab: **Assign access to** = *User, group, or service principal* →
  **+ Select members** → your own account.
- **Review + assign**.

Role assignments take a minute or two to propagate. If step 4 shows a permissions error, wait and
refresh rather than assuming the assignment failed.

**3b — Access-policy vaults.** Vault → **Access policies** → **+ Create**.

- *Permissions* tab: under **Secret permissions** tick **Get**, **List**, **Set**.
- *Principal* tab: search and select your own account.
- *Application* tab: skip it.
- **Review + create**.

Leave any existing policy alone — the workspace's own managed identity has one, and removing it
breaks the workspace.

**4 — Create the secret.** Vault → **Objects → Secrets** → **+ Generate/Import**.

| field | value |
| --- | --- |
| Upload options | **Manual** |
| Name | `anon-salt` |
| Secret value | the value from `.env`, everything after `ANON_SALT=` |
| Content type | optional — `anonymisation salt (hex)` is a useful label |
| Activation / expiration date | leave unset |
| Enabled | Yes |

Then **Create**.

> **Paste the value only — no `ANON_SALT=` prefix and no surrounding quotes.** Checked on
> 2026-08-22: this repo's `.env` stores the salt **unquoted**, 32 characters, so the value after
> the `=` is exactly what belongs in the box. A stray quote or newline produces a *different*
> salt, and a different salt produces a full dataset whose listing ids match nothing — with no
> error anywhere. Verify before submitting using the fingerprint check below.

**5 — Verify.** Secrets → `anon-salt` → click the current version → **Show Secret Value**. It
should match `.env` exactly.

<details>
<summary>CLI equivalent, if you prefer it later</summary>

```sh
KV=$(az ml workspace show -n nf-rental-ranking-ws -g nf-rental-ranking \
       --query key_vault -o tsv | awk -F/ '{print $NF}')

az role assignment create --role "Key Vault Secrets Officer" \
  --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --scope "$(az keyvault show -n "$KV" --query id -o tsv)"

SALT=$(grep -m1 '^ANON_SALT=' .env | cut -d= -f2- | tr -d "\"'")
az keyvault secret set --vault-name "$KV" --name anon-salt --value "$SALT" --output none
unset SALT
```

</details>

### Submit the pipeline

The salt is injected as an environment variable scoped to **step 1 only** — step 2 reads ids that
are already hashed and never needs it.

Copy the value from the portal (Secrets → `anon-salt` → **Show Secret Value** → copy), then:

```sh
# `read -rs` does not echo, and the value never enters shell history.
read -rs ANON_SALT
# ...paste, press Enter...

# Confirm you pasted the right thing WITHOUT printing it. Expect: [redacted]
printf '%s' "$ANON_SALT" | shasum -a 256 | cut -c1-16

az ml job create -f pipelines/preprocess_pipeline.yml \
  --set inputs.anon_salt="$ANON_SALT" \
  -g nf-rental-ranking -w nf-rental-ranking-ws

unset ANON_SALT
```

> **The first attempt used `--set jobs.build_processed.environment_variables.ANON_SALT=...`
> and the value never reached the container** (2026-08-22). `--set` is accepted on nested
> `jobs.*` properties and reports no error, but the override did not survive into the step's
> runtime environment. The salt now travels as a **pipeline-level input** (`inputs.anon_salt`),
> which is the parameterisation Azure ML actually supports — the service substitutes it into the
> step's command rather than the CLI patching a nested object.
>
> Step 1 now also prints, before doing any work:
>
> ```
> ANON_SALT length 32, sha256 prefix [redacted]
> ```
>
> and refuses to continue on an empty value **or** on an unsubstituted `${{...}}` placeholder —
> the second being the dangerous one, since it is non-empty and would otherwise build a complete
> dataset under a salt that is literally the placeholder text. The fingerprint is a one-way hash,
> so it is safe in a job log and lets you confirm the *right* salt arrived, not merely some salt.

The fingerprint line is the guard that makes the whole arrangement safe: a mistyped or
quote-wrapped paste is caught in one second, instead of surfacing as a dataset whose ids match
nothing and whose failure mode is silence. On Linux the command is `sha256sum` rather than
`shasum -a 256`.

If you would rather pull it from the vault than copy it by hand:

```sh
KV=$(az ml workspace show -n nf-rental-ranking-ws -g nf-rental-ranking \
       --query key_vault -o tsv | awk -F/ '{print $NF}')
ANON_SALT=$(az keyvault secret show --vault-name "$KV" --name anon-salt --query value -o tsv)
```

**Be precise about what this achieves.** The vault becomes the system of record and the secret
stays out of the repo and the job YAML — both real. But the value is injected as a job
environment variable, so it is visible to anyone with workspace access in the job's properties.
For a single-owner workspace that is torn down afterwards, that is proportionate.

The stricter alternative, worth knowing as the production answer: give the compute cluster a
**managed identity**, grant *that* identity `get` on the secret, and have the job read the vault
at runtime via `azure-identity` + `azure-keyvault-secrets` — so the secret never leaves Azure and
never appears in job metadata. It costs an identity, an RBAC grant, two extra packages in the
environment, and a shim that sets `os.environ["ANON_SALT"]` before calling `main()`, because
`build.py` reads the variable rather than a vault. Not taken here: that is real setup for a job
that runs once before the workspace is deleted, and the exposure it removes is exposure to the
workspace's only user.

### After the run

```sh
# The DAG screenshot is the deliverable; the hash check is the evidence.
az ml job show -n <run-name> -g nf-rental-ranking -w nf-rental-ranking-ws --query status -o tsv
```

Both steps print `sha256sum` of what they wrote — compare against the table above. The
intermediate and the feature table land in the workspace blob store as pipeline outputs and are
deliberately **not** registered as new data assets: `features:2026.08.17` already exists and was
built from the identical bytes, and registering a second copy would add a third thing to version
and keep in step for no demonstration value.

Teardown is unchanged and still applies — the cluster idles at zero, but see the standing rule.

**One teardown wrinkle the Key Vault adds.** Vaults have **soft delete** enabled and it cannot be
turned off. Deleting the resource group therefore does not remove the vault outright: it goes into
a soft-deleted state for the retention period (90 days by default). That costs nothing, but the
**name stays reserved**, so recreating a workspace later can collide with it. To clear it
immediately: Portal → **Key vaults** → **Manage deleted vaults** → select the region
(*italynorth*) → select the vault → **Purge**. Purge is blocked if *purge protection* was enabled
on the vault; AML enables soft delete but generally leaves purge protection off, so this normally
works. Nothing else in this project needs the vault to survive — the salt's system of record
returns to `.env` once the workspace is gone.

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

## Step 8 — the managed online endpoint (2026-08-18)

⚠ **The only resource in this project with a meter that runs while you are not looking.** An
endpoint with no deployment costs nothing; the moment a deployment provisions an instance, it
bills per hour regardless of traffic. Deploy → invoke → screenshot → **delete**, same session.

```bash
# 1. Local Docker smoke test FIRST — this is not ceremony, see the findings below
az ml online-deployment create --local -f pipelines/deployment.yml
az ml online-endpoint invoke --local --name rental-ranker --request-file request.json
az ml online-deployment delete --local --name blue --endpoint-name rental-ranker --yes

# 2. Then the real one
az ml online-endpoint create -f pipelines/endpoint.yml
az ml online-deployment create -f pipelines/deployment.yml --all-traffic

# 3. TEAR DOWN, same session
az ml online-endpoint delete --name rental-ranker --yes
```

**A custom scoring script, not a no-code MLflow deployment**, for three reasons: no-code returns
raw predictions in input order rather than a ranked list keyed on `id`; it offers nowhere to
validate a request; and the MLflow signature cannot express the five `category` columns.

### What the local smoke test caught, which the roadmap put there for exactly this reason

1. **`ModuleNotFoundError: No module named 'rental_ranking'`.** Azure copies
   `code_configuration.code` to `/var/azureml-app/<dir>` and puts **only the script's own
   directory** on `sys.path` — the package is not pip-installed. `init()` succeeded and the
   container reported healthy; it failed at *request* time. Fixed by inserting the package root
   into `sys.path` inside the scoring script.
2. **`az ml online-deployment update --local` broke with `conda: error: unrecognized arguments:
   --root`.** Delete and recreate the local deployment instead of updating it.
3. **Model asset versions must be positive integers.** `version: "2026.08.18"` — the convention
   used for the data assets and the environment — is rejected for *models* with
   `Model version must be a positive integer`. **The local deployment accepted it; only the
   cloud validates.** The build date moved to a tag.

### The categorical contract, measured rather than assumed

An earlier note here claimed category *code order* shifts silently at inference. **That was
wrong**, and the measurement is worth keeping because two of three guesses failed:

| claim | reality |
|---|---|
| Category order shifts codes → silent wrong answers | **False.** The booster stores training levels in `pandas_categorical` and re-maps *by label*. A request covering 2 of 3 levels scored identically, to 0.000000 |
| JSON strings score fine | **False, but loud.** `object` dtype raises `train and valid dataset categorical_feature do not match` |
| — | **The real silent bug: an unseen level.** `room_type="Houseboat"` predicts with no error, **0.1083** away from truth, because `set_categories` turns it into NaN and the model scores it as *missing* |

So `serving_metadata.json` ships beside the booster and `restore_dtypes` rejects unseen levels.
That is the one failure that would otherwise reach a caller as a confident number.

### Demo captured and torn down — 2026-08-18

Request/response pairs are in `docs/endpoint_demo/`: a full query group (23 listings, 61
features), an invalid-level rejection, and a sparse request with 59 of 61 features absent.
**Cloud scores matched local to the last digit** (rank 1 = 0.44751797704526974 in the local
container and on the endpoint).

| event | UTC |
|---|---|
| endpoint created (no deployment — **no charge**) | 18:46:34 |
| deployment provisioned — **meter starts** | 19:02:46 |
| invoked (3 requests) | 19:03:22 |
| `az ml online-endpoint delete` issued | 19:03:51 |
| **teardown confirmed, 0 endpoints remaining** | **19:11:44** |

**Instance live ~9 minutes on `Standard_DS2_v2` at $0.1360/hr ≈ $0.02.** Note the shape of that
timeline: **16 minutes to provision, 29 seconds of use, 8 minutes to delete.** The demo is
cheap; what would not be cheap is forgetting the delete, at ~$98/month for one idle DS2_v2.
The endpoint itself is free until a deployment attaches an instance.

### Teardown

- [x] `az ml online-endpoint delete --name rental-ranker --yes` — **done 2026-08-18T19:11:44Z**
- [ ] `az ml compute delete --name cpu-cluster` — optional; min-instances 0 already bills nothing
- [ ] **`az acr delete`** — this is the $5/month one
- [x] `az cognitiveservices account delete/purge --name nf-rental-language` — **done 2026-08-18**
- [ ] `az group delete --name nf-rental-ranking` when the project wraps — removes everything

### Step 8, second pass (2026-08-19) — a demonstration that shows the model, not the endpoint

The first deployment proved a service existed and answered. It did not show that what it served
ranks: `{"id","score","rank"}` with no truth beside it reads the same whether the model is good
or shuffled. `src/rental_ranking/cloud/demo.py` is the missing half — it builds requests from
sealed-fold listings, joins the response back to the held-out grades on this machine, and scores
the endpoint against both frozen baselines and a random floor on the same candidate set.

```bash
# no Azure account, no cost — same init()/run() the container calls
uv run python -m rental_ranking.cloud.demo --query thessaloniki --local --counterfactual

# against the live endpoint (AML_ENDPOINT_URI / AML_ENDPOINT_KEY in .env)
uv run python -m rental_ranking.cloud.demo --query athens --counterfactual
uv run python -m rental_ranking.cloud.demo --query thessaloniki --variant cold-start
uv run python -m rental_ranking.cloud.demo --query thessaloniki --variant unknown-level
uv run python -m rental_ranking.cloud.demo --capture      # writes docs/endpoint_demo/
```

Reference numbers, measured locally on the sealed fold before redeploying — the cloud must
reproduce them exactly, as the training job did:

| query | n | endpoint | reviews | price+rating | random |
|---|---|---|---|---|---|
| athens (group 79) | 29 | 0.5976 | 0.4063 | 0.4384 | 0.3852 |
| crete (group 305) | 25 | 0.7962 | 0.7494 | 0.7550 | 0.5587 |
| thessaloniki (group 24) | 23 | 0.8095 | 0.6081 | 0.4128 | 0.5743 |

Counterfactual, top listing with review history stripped: Thessaloniki **1 → 15** of 23,
Crete **1 → 1**. Cold-start variant (whole group blanked): 0.7230, still above its 0.5743 floor.

**One bug found, and it was found by the demo rather than by the smoke test.** The cold-start
request crashed the scoring path: a feature column present and `null` for every listing arrives
from `json.loads` as `object`, and LightGBM rejects the frame with `pandas dtypes must be int,
float or bool` — an unhandled exception, so a 500 with no message. An *absent* column was always
fine, because `reindex` gives it `float64` NaN, which is why the first pass's "sparse" request
passed. `restore_dtypes` now coerces, and names the column when a value genuinely is not a
number. Fix and three regression tests in `train/lambdamart.py` and `tests/test_demo.py`.

**Endpoint URI gotcha (2026-08-19).** The Studio endpoint page lists the **Swagger URI** directly
below the **REST endpoint**, and only the second one scores. Posting to `/swagger.json` returns:

```
HTTP 424: {"message": "The method is not allowed for the requested URL."}
```

424 from the front door wrapping a 405 from the inference server, which answers GET on that path
and nothing else. Nothing in the chain names the URL or the mistake, and the endpoint is perfectly
healthy throughout. `demo.endpoint_address` now refuses a URI that does not end in `/score` and
prints the corrected one. Take it from the CLI rather than the portal:

```bash
az ml online-endpoint show --name rental-ranker --query scoring_uri -o tsv
```

### The live run (2026-08-19) — evidence and what it measured

Second deployment, to exercise the rebuilt demonstration against a real endpoint. Captured in
`docs/endpoint_demo/RESULTS.md` and `docs/screenshots/`.

**Cloud reproduced local to the last bit.** Not "close": `max |score difference| = 0.0` across all
three query groups, identical ordering. Notebook 04 §10 recomputes this rather than asserting it.
Third independent confirmation, after the training job and the first invocation — the pre-declared
platform-drift caveat was never needed.

| query | endpoint | reviews | price+rating | random |
|---|---|---|---|---|
| athens (79, n=29) | 0.5976 | 0.4063 | 0.4384 | 0.3852 |
| crete (305, n=25) | 0.7962 | 0.7494 | 0.7550 | 0.5587 |
| thessaloniki (24, n=23) | 0.8095 | 0.6081 | 0.4128 | 0.5743 |

**Inference latency, from the container's own log:** `POST /score 200` at **12.9–14.5 ms** on
`Standard_DS2_v2`, the smallest instance offered — for candidate sets of 23 to 29 listings. Nothing
about this model is expensive to serve. The log also shows the platform's `kube-probe` GETs every
10 s, which is what a healthy deployment's idle traffic looks like.

**One client-side failure, no code wrong:** the Studio Swagger URI pasted in place of the REST
endpoint. See the gotcha above; `demo.endpoint_address` now refuses it.

Screenshots kept in `docs/screenshots/`:

| file | what it shows |
|---|---|
| `console_win_kalamaria.png` | 43 listings, **0.8785** against a 0.6163 floor, top three all grade 4 |
| `console_loss_kypseli.png` | 15 listings, **0.6040** against a **0.6563** floor — the model loses |
| `console_chania_hotel.png` | 18 listings, 0.7847, a mid case |
| `endpoint_container_logs.png` | `POST /score 200` with latencies, from the deployment log |
| `train_job_overview.png` | the training run's Overview tab — the reproducibility claim in one frame: data asset `features:2026.08.17`, registered environment `rental-ranking-train:2026.08.18`, git branch and commit, and the `protocol: sealed fold 0 of 5` tag |
| `train_job_metrics.png` | the same run's logged metrics |
| `preprocess_pipeline_dag.png` | the preprocessing pipeline's graph: three raw data assets fanning into `build_processed`, the `processed` intermediate flowing into `build_features`, both outputs registered. The one thing the single-step training job cannot show |

The Overview shot also exposes a small logging gap worth fixing before the next cloud run: the
MLflow tag reads `git_commit : unknown` while Azure's own Git field carries the real SHA. The job
runs from a code snapshot with no `.git`, so the tag has nothing to read; passing the commit as a
job parameter at submit time is the fix.

**Teardown.** `az ml online-endpoint delete` issued at the end of the session; verified absent
afterwards — `az ml online-deployment list` returns `ParentResourceNotFound` and
`az ml online-endpoint list` is empty. The ACR (~$5/month) and the min-instances-0 compute cluster
are all that remain, and the cluster bills nothing idle.
