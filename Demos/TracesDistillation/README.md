# Traces → SFT Distillation

This demo fine-tunes a student model to reproduce a larger hosted agent's tool-using behavior, using **real production traces** from your Foundry agent as training data. No labels required.

## What it shows

1. **Pull** real conversation traces from a deployed Foundry agent via the traces recipe of the Data Generation API
2. **Transform** the raw export into Azure-FT-ready JSONL (6 fixes the Foundry export currently requires for tool-using agents — all applied inline in the notebook)
3. **Score** the base student on a held-out test set using **structural tool-call comparison**
4. **Submit** one fine-tuning job (3 epochs, lr multiplier 1.0)
5. **Monitor** training to completion
6. **Deploy** the fine-tuned model
7. **Evaluate** it on the same test set and report the lift

Evaluation is driven by the **Foundry evaluations SDK** (`azure-ai-evaluation`) with a custom tool-call structural evaluator.

## Results

Every run below pulls ~720 hours of real traces from a deployed Zava-style Post-Purchase Resolution Desk agent (`gpt-4.1-mini` behind it) and distills them into a student model.

**Verified runs — `gpt-4.1-mini` student** (each: 100 samples → 99 rows → 79 train / 9 val / 11 test, 3 epochs, lr=1.0):

| Run | Baseline | Fine-tuned | Lift |
|-----|----------|------------|------|
| 1 | 6.82 @ 36.4% | **9.82 @ 100%** | **+44.0%** |
| 2 | 7.73 @ 54.5% | **10.00 @ 100%** | **+29.4%** |
| 3 | 7.27 @ 45.5% | **10.00 @ 100%** | **+37.6%** |

All three runs reach a 100% pass rate on their held-out split. The lift differs because the *baseline* differs — see the run-to-run variance note below.

**Originally published — `gpt-4.1-nano` student:**

| Model | Combined | Pass Rate | Lift |
|-------|----------|-----------|------|
| Baseline `gpt-4.1-nano` | 7.38 | 60% | — |
| Fine-tuned `gpt-4.1-nano` (3ep, lr=1.0) | 8.60 | 100% | +16.5% |

> **What the lift actually buys you.** With a `gpt-4.1-nano` student the win is model size — same tool-selection accuracy at roughly 10× lower cost per token. With a **`gpt-4.1-mini` student the model size is unchanged**, so the saving is different in kind: the fine-tuned model reproduces the agent's tool selection in a single call, without the agent's large system prompt, tool catalog, and orchestration round-trips. Measure prompt-token savings, not parameter count. Pick your student accordingly, and say which one you mean when you present numbers to a customer.

> **Student model choice.** The notebook defaults to `STUDENT_MODEL = "gpt-4.1-mini"`. `gpt-4.1-nano` (version `2025-04-14`) is in `Deprecating` status and Azure **blocks new deployments** of it, so the original nano result is no longer reproducible on a fresh tenant. Don't read the nano/mini lift numbers as a comparison between the two students — they were measured on different test sets, and the nano run's smaller lift (+16.5%) reflects its higher baseline, not a weaker result.

> **Test sets differ between runs.** Each datagen pull samples a different slice of trace history, so the baseline moves run to run (we observed 6.82, 7.27 and 7.73 across three pulls of the same agent). Always compare baseline and fine-tuned **within the same run** — the notebook does this by scoring both on one held-out split. Quoting a lift number without its baseline is meaningless: run 2 above has the *better* fine-tuned score but the *smaller* lift, purely because it drew an easier test set.

## Running this in a customer tenant

Everything below has to exist in the customer's subscription before the notebook will run. This is the checklist to send them ahead of a POC.

### Azure resources

| Resource | Why | Notes |
|---|---|---|
| Azure AI Foundry (AIServices) account + project | Hosts the teacher agent and its traces | This is the `AZURE_AI_PROJECT_ENDPOINT` |
| A **deployed hosted agent** with trace history | The teacher. Any hosted Foundry agent emits traces automatically | Use `fixtures/push_prompts.py` to populate history if it's new |
| Application Insights, **workspace-based**, connected to the project | Where traces land | Must be workspace-based (`IngestionMode: LogAnalytics`); the datagen job reads `AppDependencies` / `AppGenAIContent` |
| An AIServices/OpenAI account **in a region that supports fine-tuning** | Runs the training job and hosts the deployment | This is the `OPENAI_BASE_URL` |
| A student model deployment (`gpt-4.1-mini`) | Needed for the **baseline** eval before training | Fine-tuning itself doesn't need a pre-existing deployment |

> **Expect two resources, in two regions.** The agent's region often does not support fine-tuning — in our run `eastus2` reported `fineTune=None` for the entire `gpt-4.1` family, so the agent lived in `eastus2` and training ran in `northcentralus`. This is normal and the notebook is built for it: `AZURE_AI_PROJECT_ENDPOINT` and `OPENAI_BASE_URL` are independent.
>
> A direct consequence worth warning the customer about: **fine-tuning jobs only appear in the Foundry portal for the project on the resource they were submitted to.** If they're looking at the agent's project and don't see the job, they're looking at the wrong resource — not at a failure. Check `az cognitiveservices account list -g <rg> -o table` and open the project on the resource that matches `OPENAI_BASE_URL`.

### Permissions

The configuration this demo was verified against — **both roles on both accounts** (the agent account and the fine-tuning account):

| Role | Scope | Why |
|---|---|---|
| `Foundry Owner` | Both AIServices accounts | Invoke the agent, submit datagen jobs, create the deployment |
| `Cognitive Services OpenAI User` | Both AIServices accounts | Data-plane calls against `/openai/v1` |

If the customer won't grant `Foundry Owner`, the least-privilege equivalent is roughly:

| Role | Scope | Why |
|---|---|---|
| `Azure AI Developer` | Foundry project | Invoke the agent, submit datagen jobs |
| `Cognitive Services OpenAI Contributor` | Fine-tuning account | Upload files, submit and monitor FT jobs |
| `Cognitive Services Contributor` (or a custom role with `Microsoft.CognitiveServices/accounts/deployments/write`) | Fine-tuning account | Create the fine-tuned model deployment |
| `Monitoring Reader` | Application Insights | Only if you want to inspect traces by hand |

We did not test the least-privilege set end to end — treat it as a starting point for a security review, not a verified configuration.

Two **managed identities** also need `Monitoring Metrics Publisher` on the Application Insights resource, or **no traces are emitted at all** and the datagen job returns zero samples:

- the Foundry **project's** managed identity
- the **agent instance's** managed identity

These are distinct principals. Verify both:

```bash
az role assignment list --assignee <mi-object-id> \
  --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Insights/components/<appinsights> \
  --query "[].roleDefinitionName" -o tsv
```

### Quota

| Quota | Needed | Check |
|---|---|---|
| Teacher agent's model TPM | **≥ 300K** to push 500 prompts without throttling | At 50K TPM the fixture script throttles badly; 300K completed 500/500 in ~3 min |
| `OpenAI.GlobalStandard.<student>-finetune` | ≥ 100 (the notebook deploys at capacity 100) | `az cognitiveservices usage list -l <region>` |
| `OpenAI.FineTuned.Deployments` | ≥ 1 (limit is typically 10) | Same command |

### Policy blockers to check first

- **`disableLocalAuth=true`** on either account — common in enterprise tenants. The notebook already handles this: it authenticates with AAD via `DefaultAzureCredential` and passes the bearer token as `api_key`. No API keys are used anywhere.
- **Private endpoints / disabled public network access** — the notebook runs from a workstation and needs data-plane reachability to both accounts.
- **Azure Policy denying new Cognitive Services deployments** or restricting SKUs — this blocks the deploy step at the very end, after training has already been paid for. Check before starting.
- **Region restrictions** — if policy pins resources to a region without fine-tuning support, the two-resource split above is not possible and the demo can't run as written.

### Cost

Training is billed per token (our run: 278K tokens). The **deployment is billed continuously for as long as it exists** — it does not stop when the notebook finishes. Delete it when the POC is done (see [Cleanup](#cleanup)).

## Prerequisites

- Azure CLI (`az login`) — all authentication is AAD; no API keys required
- Python 3.11+ with:

```bash
pip install "openai>=2.0" "azure-ai-projects>=2.2.0" "azure-identity>=1.21" "azure-ai-evaluation>=1.0"
```

> Quote the specifiers. Unquoted, `>=` is parsed by the shell as a redirect and silently truncates the install.

## Files in this folder

| File | Purpose |
|------|---------|
| `notebook.ipynb` | End-to-end runnable walkthrough — **fully self-contained**, no external scripts required |
| `fixtures/push_prompts.py` | Optional standalone script that pushes diverse retail prompts through any hosted agent (use this before the notebook if your agent has no trace history yet) |
| `fixtures/zava_system_prompt.md` | Sample system prompt for the Zava resolution-desk agent (replace with your own) |
| `fixtures/zava_tools.json` | Sample tool catalog (OpenAI chat-completions format) — replace with your own |

## Run it

```bash
export AZURE_AI_PROJECT_ENDPOINT="https://<agent-resource>.services.ai.azure.com/api/projects/<project>"
export OPENAI_BASE_URL="https://<finetuning-resource>.openai.azure.com/openai/v1"
export AZURE_SUBSCRIPTION_ID="<subscription-id>"
export AZURE_RESOURCE_GROUP="<resource-group>"

az login   # AAD is the only auth path; no AZURE_OPENAI_API_KEY needed

# (Optional) populate trace history first if your agent has none:
python fixtures/push_prompts.py \
    --agent-name <your-hosted-agent> \
    --agent-version <version> \
    --num-prompts 500 \
    --project-endpoint $AZURE_AI_PROJECT_ENDPOINT

# Wait ~90 seconds for traces to land in App Insights, then:
jupyter notebook notebook.ipynb
```

Full run is ~60 minutes: datagen ~5 min, training ~50 min (queue-dependent), deployment ~4 min.

> The two endpoints are **usually different resources** — see [Running in a customer tenant](#running-this-in-a-customer-tenant). `OPENAI_BASE_URL` must point at the account whose region supports fine-tuning.

> `/openai/v1` paths reject the `api-version` query parameter. Use `openai.OpenAI(...)`, not `AzureOpenAI(...)`. The notebook already does.

### Re-running on a resource you've already used

Azure **will not repoint an existing fine-tuned deployment at a different model**. A second run reaches the deployment cell and fails with:

```
400 Bad Request
{"error":{"code":"ModelUpgradeNotSupported",
          "message":"Model updates are not supported for finetuned model deployments."}}
```

The notebook handles this: the deployment cell checks what `DEPLOY_NAME` currently points at, deletes it if it's a different model, and waits for the delete to finish before creating. Nothing to do manually — but budget the extra ~2 minutes, and don't be surprised to see a delete in the logs on run 2+.

If you'd rather keep both runs' models live for comparison, change `DEPLOY_NAME` per run instead. Watch the `OpenAI.FineTuned.Deployments` quota (default 10) and remember each live deployment bills continuously.

**Don't gate on inferenceability alone.** While a replacement deployment provisions, the data plane keeps serving the *previous* model under the same deployment name. A readiness check that just calls the endpoint and waits for a 200 therefore passes instantly — against the old model — and the comparison cell then scores the wrong thing without any error. We hit exactly this: the endpoint answered at 0s while ARM still reported `Creating`, and only reached `Succeeded` three minutes *after* the evaluation had already run. The notebook now polls `provisioningState` until `Succeeded` before it probes the endpoint. If you write your own deployment code, do the same — this failure is completely silent, and on a re-run it produces a plausible-looking number for a model you didn't train.

## Why the transform step exists

The Foundry traces export currently emits chat JSONL with several issues that Azure FT preprocessing rejects. Note that these are **stricter than the repo's own `validate_sft.py`** — data can pass local validation and still fail Azure preprocessing with `contains invalid schema`, which reports only line numbers and no reason.

1. **Overlapping snapshots** — Each LangGraph node invocation gets its own span; the worker stitches them into one messages list, so one row contains N overlapping snapshots of the same conversation
2. **Fragment rows** — When the customer simulator ends right after the agent asks a clarifying question, you get 2-msg rows with no assistant tool_calls (no SFT signal)
3. **content="null"** — Assistant tool-call rows have `content` as the literal string `"null"` instead of JSON null. Azure FT rejects rows where content is present alongside tool_calls — the field must be **omitted** entirely
4. **Consecutive assistant tool_call turns** — Sometimes two adjacent assistant spans each issue one tool call. Azure FT requires tool replies immediately after each assistant turn; the fix is to merge consecutive `asst(tc)` turns into one assistant message with parallel tool_calls
5. **System message handling** — The export now emits its own system message. Prepending the agent's system prompt unconditionally produces two consecutive system turns, which Azure FT rejects. The transform strips exported system turns and keeps exactly one, and attaches the row-level `tools` array that FT preprocessing needs for tool-using models
6. **Empty `tool_call_id`** — Every `tool` message comes back with `tool_call_id: ""` while the assistant turn carries the real `call_...` id, so Azure can't pair the reply to its request and rejects the row. The transform re-links them in order, synthesising ids when the assistant's `tool_calls[].id` is also empty. **Only rows containing tool replies are affected**, so a dataset can fail on 2 of 79 rows and look fine everywhere else

The notebook's inline transform functions apply all six fixes and are idempotent, so they're safe to keep as the export matures.

## Notes on the agent invocation API

Hosted agents are addressed via an `agent_reference` object in the request body:

```python
responses.create(input=prompt, store=False,
                 extra_body={"agent_reference": {"type": "agent_reference",
                                                 "name": AGENT_NAME, "version": AGENT_VERSION}})
```

The older form — passing `"<agent>:<version>"` as `model` — is no longer routed and returns `404 DeploymentNotFound`. Note that `type` is required, and `model` must be **omitted entirely** rather than set to `None`. `fixtures/push_prompts.py` tries the new shape first and falls back to the legacy one for older service versions.

## Choosing hyperparameters

The notebook trains with **3 epochs, learning-rate multiplier 1.0, batch size 1**. In our run training and validation loss both reached ~5e-05 with validation token accuracy at 1.000 **by step 20 of 237** — the task converges almost immediately because tool-call outputs are highly structured.

If you're cost-sensitive, **1 epoch will likely match this result** at a third of the training spend. Check the validation curve before assuming you need three.

## Cleanup

The deployment bills continuously until deleted.

```python
req = urllib.request.Request(deploy_url, method="DELETE", headers={"Authorization": f"Bearer {token}"})
urllib.request.urlopen(req)
client.files.delete(train_file.id)
client.files.delete(val_file.id)
```

The fine-tuned model itself is retained after the deployment is deleted and can be redeployed later from the job id.

## Bring your own agent

Replace `AGENT_NAME` / `AGENT_VERSION` and the two fixture files with your agent's system prompt + tool catalog.
