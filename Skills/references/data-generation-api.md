# Foundry Data Generation API

Reference for the Data Generation API exposed under `project_client.beta.datasets` in the Azure AI Projects SDK (`azure-ai-projects` ≥ `2.2.0`). This service turns existing sources — agent traces, documents, deployed agents, or uploaded files — into JSONL files or Foundry datasets ready for fine-tuning or evaluation.

> **Status:** Productionised behind the `Foundry-Features: DataGenerationJobs=V1Preview` capability header. The SDK adds the header automatically when calling `beta.datasets` operations.

For task-by-task walkthroughs see:
- `workflows/traces-to-dataset.md` — distill a deployed agent's real conversations
- `workflows/synthetic-datagen.md` — generate Q&A or tool-calling data from documents / agent specs

> **Verified live (2026-05-28)** against `cont-learning-faos`, `foundrysdk-eastus-resource`, and `REDACTED-FOUNDRY-PROJECT` using `azure-ai-projects 2.2.0a20260528004` and direct REST calls:
> - SDK + REST end-to-end with `SimpleQnA` + `File` source + `SUPERVISED_FINETUNING` (14 samples in 70s / 50s respectively)
> - SDK with `SimpleQnA` + `Prompt` source + `EVALUATION` (15 samples, Dataset output)
> - SDK with `SimpleQnA` + `Agent` source + `EVALUATION` (15 samples, Dataset output)
> - SDK with `Traces` + `traces` recipe + `SUPERVISED_FINETUNING` from `demo1-retail-agent-langraph-responses`
> - SDK with `ToolUseFineTuning` + `File` source (`.json` tool catalog) + `SUPERVISED_FINETUNING` — runs but is slow (20–40 min)
> - Service rejections (all confirmed): `File`+`EVAL`, RFT scenario with non-Traces options, oversized output names, undersized file content, `ToolUseFineTuning` without a `.json` file source
>
> **Documented but not yet verified in this validation run:** `REINFORCEMENT_FINETUNING` + `Traces` (needs a trace stream with ground-truth assistant answers); `Agent` source + `ToolUseFineTuning` against an agent whose tool definitions are registered server-side (every hosted agent we tried surfaced tools at runtime only and was rejected — see constraints below).

## Mental model: one job, three layers

Every job is built by picking one value from each of three layers:

```
DataGenerationJob(inputs=DataGenerationJobInputs(
    scenario      = <SUPERVISED_FINETUNING | REINFORCEMENT_FINETUNING | EVALUATION>,  # what for
    sources       = [<one Source class>],                                              # raw material
    options       = <one Options class>,                                               # recipe
    output_options= DataGenerationJobOutputOptions(name=<output prefix, ≤50 chars>),   # naming
))
```

The same source can be paired with multiple recipes, and the same recipe can run against multiple sources. **Not every combination is valid** — see [the compatibility matrix](#what-pairs-with-what).

## Layer 1 — Scenario

```python
from azure.ai.projects.models import DataGenerationJobScenario
```

| Value | Output shape | Output type |
|-------|-------------|-------------|
| `SUPERVISED_FINETUNING` | Chat-completion JSONL (`{"messages": [...]}`, with `tools` array when applicable) | **File** — drop into `fine_tuning.jobs.create(training_file=...)` |
| `REINFORCEMENT_FINETUNING` | RFT JSONL (prompt + ground-truth `answer`) | **File** |
| `EVALUATION` | Evaluation JSONL (`{"query": [...], "response": [...]}`) | **Dataset** — named & versioned, drop into Foundry evaluations |

Note: SFT/RFT produce **files** (via the Azure OpenAI Files API). EVALUATION produces a **dataset** (registered in the project's dataset catalogue). The code paths to consume them differ — see [Reading the result](#reading-the-result).

## Layer 2 — Sources (where the raw material comes from)

Pick exactly one. Each class takes a different required field.

| Source class | What it pulls | Required field | Optional fields |
|---|---|---|---|
| `TracesDataGenerationJobSource` | Spans from your deployed agent in Application Insights | `agent_name` | `agent_version`, `start_time`, `end_time`, `description` |
| `PromptDataGenerationJobSource` | Inline text (≤10,000 chars) — paste a policy doc, agent description, etc. | `prompt` | `description` |
| `AgentDataGenerationJobSource` | A deployed agent's instructions and tool definitions — **no traffic required** | `agent_name` | `agent_version`, `description` |
| `FileDataGenerationJobSource` | An Azure OpenAI file you've already uploaded (`purpose="user_data"`) | `id` | `description` |

> A 5th class, `DatasetDataGenerationJobSource`, appears in some draft docs but is **not in the SDK** as of `2.2.0a20260528004`. Don't import it.

### TracesDataGenerationJobSource

```python
from datetime import datetime, timedelta, timezone
from azure.ai.projects.models import TracesDataGenerationJobSource

TracesDataGenerationJobSource(
    agent_name="retail-agent-langgraph",
    agent_version="3",                                  # strongly recommended
    start_time=datetime.now(timezone.utc) - timedelta(hours=1),
    end_time=datetime.now(timezone.utc),
    description="Last hour of production traffic",      # optional
)
```

**How spans become rows.** Each conversation in App Insights is reconstructed turn-by-turn. The output contains one row per assistant response, with the prior conversation as growing context. Tool calls appear as `tool_call` parts in the assistant message; tool results appear as `role: "tool"` messages.

**Gotchas:**
- App Insights ingestion lag is **30–90 seconds**. Sleep ≥90s between capturing traffic and submitting the job, otherwise the window is empty and you get zero samples.
- Without `agent_version`, the job mixes spans across every active version — useful for "all traffic, however it looks" but harmful when you're iterating on prompts.
- Requires Application Insights attached to the project. Configure under **Project settings → Telemetry** in the portal.

### PromptDataGenerationJobSource

```python
from azure.ai.projects.models import PromptDataGenerationJobSource

PromptDataGenerationJobSource(
    prompt=open("refund_policy.md").read(),    # <10,000 chars
    description="Company refund policy v3 (2026 Q1)",
)
```

Treats `prompt` as inline source text. Best paired with `SimpleQnADataGenerationJobOptions`.

**Size cap: 10,000 chars.** For larger corpora upload as a file (`purpose="user_data"`) and use `FileDataGenerationJobSource` (for SFT) — or split your corpus into multiple sub-10k prompts.

### AgentDataGenerationJobSource

```python
from azure.ai.projects.models import AgentDataGenerationJobSource

AgentDataGenerationJobSource(
    agent_name="retail-agent-langgraph",
    agent_version="3",
    description="Agent definition (no traffic required)",
)
```

Reads the agent's *instructions* and *tool definitions* — no traces required. Use for cold-start: an agent that's been deployed but doesn't have meaningful traffic yet. Pair with `ToolUseFineTuningDataGenerationJobOptions` for tool-calling SFT or `SimpleQnADataGenerationJobOptions` for general Q&A.

### FileDataGenerationJobSource

```python
import io
from azure.ai.projects.models import FileDataGenerationJobSource

# Upload via the Azure OpenAI Files API first
aoai = project_client.get_openai_client()
seed = aoai.files.create(
    file=("handbook.pdf", open("handbook.pdf", "rb")),
    purpose="user_data",                       # must be "user_data"
)
# Wait until status == "processed"
while seed.status not in ("processed", "error"):
    seed = aoai.files.retrieve(file_id=seed.id)
    time.sleep(2)

FileDataGenerationJobSource(id=seed.id)
```

**Content requirements:**
- File must be ≥ 1 KB.
- Content must be **substantive and standalone** — references to external sources, sparse outlines, or near-empty files fail with `"File content lacks sufficient context to generate quality questions. Ensure it is comprehensive and standalone."`
- For SFT against an LLM-judged target, ~5 KB+ of dense prose tends to be the sweet spot for ~15 samples.

## Layer 3 — Options (the recipe)

Pick exactly one. The class determines the *generation algorithm*.

| Options class | Recipe | Use when |
|---|---|---|
| `TracesDataGenerationJobOptions` | Reconstruct multi-turn conversations from spans | Source is `TracesDataGenerationJobSource` |
| `SimpleQnADataGenerationJobOptions` | Single-turn Q&A pairs from a corpus | Source is a document or agent definition |
| `ToolUseFineTuningDataGenerationJobOptions` | Tool-calling conversations (SFT only) | Source has tool definitions (`AgentDataGenerationJobSource`, traces with tools) |

All three share these fields:

| Field | Type | Description |
|-------|------|-------------|
| `max_samples` | `int` | **15–1000.** Ceiling, not a guarantee. Always check `job.result.generated_samples`. |
| `train_split` | `float` (0.0–1.0) | When set, outputs two files: training (index 0) and validation (index 1). When omitted, you get a single combined file. **EVAL scenario ignores `train_split`.** |
| `model_options` | `DataGenerationModelOptions` | Selects the teacher model. **Required** for `SimpleQnA` and `ToolUseFineTuning`. Optional for `Traces` (traces don't need a teacher since they're real conversations). |

### TracesDataGenerationJobOptions

```python
from azure.ai.projects.models import TracesDataGenerationJobOptions

TracesDataGenerationJobOptions(
    max_samples=200,
    train_split=0.8,
    # model_options optional for traces
)
```

The underlying `trace_filtering` flow applies hard filters (PII, schema), heuristic quality scoring, optional LLM scoring, and top-K selection.

### SimpleQnADataGenerationJobOptions

```python
from azure.ai.projects.models import (
    SimpleQnADataGenerationJobOptions,
    DataGenerationModelOptions,
)

SimpleQnADataGenerationJobOptions(
    max_samples=100,
    train_split=0.9,                                      # ignored for EVAL
    model_options=DataGenerationModelOptions(model="gpt-4.1-mini"),  # REQUIRED
)
```

**`model_options` is required** — the service uses this model to synthesise the QnA pairs.

For `EVALUATION`, the model must support the Azure OpenAI **Responses API**. For `SUPERVISED_FINETUNING`, any chat-completion model works.

### ToolUseFineTuningDataGenerationJobOptions

```python
from azure.ai.projects.models import ToolUseFineTuningDataGenerationJobOptions

ToolUseFineTuningDataGenerationJobOptions(
    max_samples=50,
    train_split=0.8,
    model_options=DataGenerationModelOptions(model="gpt-4.1-mini"),
)
```

SFT-only — the API rejects this options class when `scenario=EVALUATION` or `REINFORCEMENT_FINETUNING`.

**Source requirement:** Tool-use jobs MUST have a source containing tool definitions. The exact format depends on the source type:

- `FileDataGenerationJobSource` pointing at a `.json` file containing an **OpenAPI 3.0.x or 3.1.x specification** for the tools. This is the canonical cold-start path. The file MUST validate as OpenAPI 3.x — not the OpenAI chat-completions tool format (`[{"type":"function","function":{...}}]`). If the file isn't OpenAPI, the job fails in-flight with `OpenAPI specification: Invalid or unsupported OpenAPI version. Supported versions: 3.0.x and 3.1.x`.
- `TracesDataGenerationJobSource` — when the traced conversations include tool calls. The tool catalog is inferred from the trace stream.
- `AgentDataGenerationJobSource` — only when the registered agent has tool definitions on the server side. Most hosted agents (`HostedAgentDefinition`) surface tools at runtime only and are rejected with `Tool use data generation requires exactly one .json file`.

**If you only have OpenAI tool-spec JSON**, you must convert it to an OpenAPI 3.0 spec before upload. A minimal conversion: each tool becomes a `POST /<operationId>` with `requestBody.schema = function.parameters`. See `scripts/generate_dataset.py --tools-to-openapi` for a built-in converter.

**Wall-clock:** Tool-use jobs can be quick (~2-3 min) on idle backends, but during queue spikes have been observed taking 20-40 min (long queue wait, then 5-10 min generating). Plan timeouts for the worst case if running unattended.

## What pairs with what

Not every (source × recipe × scenario) tuple is valid. The service-verified matrix:

| Source | Recipe | SFT | RFT | EVAL |
|--------|--------|-----|-----|------|
| `Traces` | `traces` | ✅ | ✅ (intended use) | ✅ |
| `Prompt` | `qna` | ✅ | ❌ (fails in-flight) | ✅ |
| `File` | `qna` | ✅ | ❌ (fails in-flight) | ❌ (use `Prompt` or `Agent` for EVAL) |
| `Agent` | `qna` | ✅ | ❌ (fails in-flight) | ✅ |
| `File` (OpenAPI 3.x spec) | `tool-use` | ✅ (~2-25 min) | ❌ rejected | ❌ rejected |
| `Traces` | `tool-use` | ✅ | ❌ rejected | ❌ rejected |
| `Agent` | `tool-use` | ❌ rejected ("requires exactly one .json file") | ❌ rejected | ❌ rejected |
| `Prompt` | `tool-use` | ❌ rejected (same) | ❌ rejected | ❌ rejected |
| `File` (OpenAI tool-spec JSON) | `tool-use` | ❌ fails in-flight (`Invalid or unsupported OpenAPI version`) | ❌ rejected | ❌ rejected |

Legend: ✅ verified live · ❌ rejected by API or fails in-flight.

**Key constraints discovered live:**
- `File` source + `SimpleQnA` + `EVALUATION` → `Invalid payload: At least Prompt or prompt agent_name is required for SimpleQnA Evaluation output format`. For eval datasets, use `Prompt` (≤10k chars) or `Agent` source.
- RFT scenario submits successfully with non-traces options but the in-flight generation fails with `"Something went wrong"`. RFT semantically requires a ground-truth answer to grade against; only the `Traces` recipe (where the production assistant response becomes the answer) produces useful RFT data.
- Tool-use options + non-SFT scenario → rejected at submit time.
- **Tool-use options REQUIRE a `.json` file source containing an OpenAPI 3.0/3.1 spec.** Submitting with only `Agent` or `Prompt` sources is rejected at submit with `Tool use data generation requires exactly one .json file`. Submitting an OpenAI tool-spec JSON (`[{"type":"function","function":{...}}]`) instead of an OpenAPI spec is accepted at submit but fails in-flight with `OpenAPI specification: Invalid or unsupported OpenAPI version. Supported versions: 3.0.x and 3.1.x`.
- **Tool-use jobs vary widely in wall-clock.** Observed range: ~2.5 min (idle backend) to ~25 min (queue spike). Plan timeouts for the worst case if running unattended.

## Output naming

```python
from azure.ai.projects.models import DataGenerationJobOutputOptions

DataGenerationJobOutputOptions(name="my-policy-qna-v1")    # ≤50 chars
```

**Required.** The service uses this as a prefix:
- SFT/RFT: outputs are `{name}_train_dg.jsonl` and `{name}_valid_dg.jsonl` (when `train_split` is set), else `{name}_dg.jsonl`.
- EVAL: output is a dataset named `{name}` with `version` "1.0" (or next available).

Names >50 chars are rejected. Auto-generated names should include a unique run id (e.g. timestamp + random suffix) since names collide across runs.

## End-to-end client setup

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project_client = AIProjectClient(
    endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>",
    credential=DefaultAzureCredential(),
)
```

> Older drafts of these docs claimed `allow_preview=True` was required. **It's not** — the SDK sends `Foundry-Features: DataGenerationJobs=V1Preview` automatically for `beta.datasets` operations. `allow_preview=True` is only needed to opt into other preview features outside the data-generation surface.

## Submitting and polling

```python
import time
from azure.ai.projects.models import JobStatus

job = project_client.beta.datasets.create_generation_job(job=job_request)
print(f"Submitted {job.id}")

TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
while job.status not in TERMINAL:
    time.sleep(10)
    job = project_client.beta.datasets.get_generation_job(job_id=job.id)

if job.status != JobStatus.SUCCEEDED:
    raise RuntimeError(f"Job ended in {job.status}: {job.error.message if job.error else '<no error>'}")
```

**Status values:** `QUEUED`, `IN_PROGRESS`, `SUCCEEDED`, `FAILED`, `CANCELLED`. (No `RUNNING` or `COMPLETED`.)

**Observed runtimes** (cont-learning-faos, gpt-4.1-mini teacher):
- SimpleQnA SFT, ~2 KB prompt, 15 samples: **60–70 s**
- SimpleQnA EVAL, ~2 KB prompt, 15 samples: **~250 s** (~4× slower than SFT)
- Trace jobs scale with conversation count — typically 1–5 minutes for hundreds of samples.

## Reading the result

`job.result.outputs` is a heterogeneous list — use `isinstance` to discriminate:

```python
from azure.ai.projects.models import (
    DatasetDataGenerationJobOutput,
    FileDataGenerationJobOutput,
)

print(f"Generated {job.result.generated_samples} samples")

for o in job.result.outputs:
    if isinstance(o, FileDataGenerationJobOutput):
        info = aoai.files.retrieve(file_id=o.id)
        print(f"file_id={o.id}  filename={info.filename}  bytes={info.bytes}")
    elif isinstance(o, DatasetDataGenerationJobOutput):
        print(f"dataset name={o.name}  version={o.version}")
```

**Always check `generated_samples`** — `max_samples` is a ceiling. Empty results commonly mean the App Insights window had no usable traffic, the document was too thin, or hard filters dropped everything.

**File ordering with `train_split`:**
- Index `0` → **training** file (`*_train_dg.jsonl`)
- Index `1` → **validation** file (`*_valid_dg.jsonl`)

Without `train_split`, you get exactly one combined file.

## Downloading file outputs

File outputs are Azure OpenAI file ids; download via the OpenAI client:

```python
aoai = project_client.get_openai_client()

for o in job.result.outputs:
    if isinstance(o, FileDataGenerationJobOutput):
        info = aoai.files.retrieve(file_id=o.id)
        with open(info.filename, "wb") as f:
            f.write(aoai.files.content(o.id).content)
```

Or skip the download and pipe directly to a fine-tuning job:

```python
ft_outputs = [o for o in job.result.outputs if isinstance(o, FileDataGenerationJobOutput)]
ft = aoai.fine_tuning.jobs.create(
    training_file=ft_outputs[0].id,
    validation_file=ft_outputs[1].id,
    model="gpt-4.1-nano",
    suffix="distilled-from-traces",
)
```

## Consuming a dataset output (EVAL scenario)

```python
out = next(o for o in job.result.outputs if isinstance(o, DatasetDataGenerationJobOutput))
dataset = project_client.datasets.get(name=out.name, version=out.version)
print(dataset.id)
# Pass dataset.id to your evaluation pipeline
```

## Managing jobs

```python
from azure.ai.projects.models import DataGenerationJobScenario

for j in project_client.beta.datasets.list_generation_jobs(
    limit=20,
    order="desc",
    scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
):
    print(f"{j.id}  {j.status:<12}  {j.inputs.name}")

# Cancel a running job
project_client.beta.datasets.cancel_generation_job(job_id="datagen-...")

# Delete the job record (does NOT delete the produced output files/datasets)
project_client.beta.datasets.delete_generation_job(job_id="datagen-...")
```

`delete_generation_job` removes the job record only. Output files persist in the project's file store; output datasets persist in the dataset catalogue. Clean those up separately:

```python
# Files: aoai.files.delete(file_id=...)
# Datasets: project_client.datasets.delete(name=..., version=...)
```

## REST equivalent

For teams that can't take the SDK dependency:

```
POST   {endpoint}/data_generation_jobs?api-version=v1
GET    {endpoint}/data_generation_jobs/{job_id}?api-version=v1
GET    {endpoint}/data_generation_jobs?api-version=v1
POST   {endpoint}/data_generation_jobs/{job_id}/cancel?api-version=v1
DELETE {endpoint}/data_generation_jobs/{job_id}?api-version=v1
```

**Auth:** AAD bearer token. Scope: `https://ai.azure.com/.default` (note: **not** `cognitiveservices.azure.com`).

**Required header:** `Foundry-Features: DataGenerationJobs=V1Preview`

**Request body matches the SDK model 1:1.** Source and options classes use snake_case discriminators:

```json
{
  "inputs": {
    "name": "my-policy-qna-v1",
    "scenario": "supervised_finetuning",
    "sources": [{"type": "file", "id": "file-abc123"}],
    "options": {
      "type": "simple_qna",
      "max_samples": 15,
      "train_split": 0.8,
      "model_options": {"model": "gpt-4.1-mini"}
    },
    "output_options": {"name": "my-policy-qna-v1"}
  }
}
```

Source `type` values: `traces`, `prompt`, `agent`, `file`.
Options `type` values: `traces`, `simple_qna`, `tool_use_fine_tuning`.
Scenario values: `supervised_finetuning`, `reinforcement_finetuning`, `evaluation`.

Successful POST returns **201 Created** with the full job object.

## Errors and edge cases

| Symptom | Cause | Fix |
|---------|-------|-----|
| `404` from `create_generation_job` (REST) | Project doesn't support DataGenerationJobs=V1Preview, or wrong path | Confirm `Foundry-Features` header is set; confirm path is `/data_generation_jobs` (not `/datasets/generation-jobs`). |
| `UnsupportedApiVersionValue` | Project uses a different api-version surface | Use `api-version=v1`, not date-based. |
| `Unauthorized: audience is incorrect` | Wrong AAD scope | Use scope `https://ai.azure.com/.default`. |
| `Invalid payload: At least Prompt or prompt agent_name is required for SimpleQnA Evaluation` | Used `File` source with EVAL + SimpleQnA | Switch to `Prompt` (≤10k chars) or `Agent` source. |
| `File content is too small to generate QnA. It must be at least 1KB.` | File source content <1KB | Upload a richer document. |
| `File content lacks sufficient context to generate quality questions.` | File content too thin / referential | Make content comprehensive and self-contained; avoid bullet points without context. |
| `Something went wrong during data generation. Please try again.` (SUCCESS → FAILED in ~15s) | Generic, often the teacher model can't serve the request (wrong model, missing permissions, FT-only deployment) | Try a different `--teacher` (e.g. `gpt-4.1-mini`, `gpt-4o`). Confirm the model deployment exists and is healthy. |
| Trace job fails immediately | Application Insights not attached, or `agent_name` doesn't match | Verify telemetry in **Project settings → Telemetry**; confirm exact `agent_name` (case-sensitive). |
| `ToolUseFineTuning` rejected at submit | Used with `scenario=EVALUATION` or `REINFORCEMENT_FINETUNING` | Tool-use is SFT-only. |
| `Tool use data generation requires exactly one .json file.` | Used `ToolUseFineTuning` options with `Agent`/`Prompt`-only sources (no `.json` file) | Provide exactly one `FileDataGenerationJobSource` pointing at an OpenAPI 3.0/3.1 spec uploaded via `aoai.files.create(purpose="user_data")`. |
| `OpenAPI specification: Invalid or unsupported OpenAPI version. Supported versions: 3.0.x and 3.1.x.` (job FAILED in-flight) | Uploaded an OpenAI-format tool catalog JSON (`[{"type":"function",...}]`) instead of an OpenAPI 3.x spec | Convert your tool list to OpenAPI 3.0 (each function → `POST /<name>` with `requestBody.schema = parameters`). See `scripts/_tools_to_openapi.py` for an example. |
| Tool-use job stays QUEUED for 10+ minutes | Normal during backend queue spikes | Wait. Observed range 2.5-25 min; plan ≥45-min timeouts for unattended runs. |
| `'name' is not in the correct format.` | Output name has invalid characters (e.g. `+`) | Use `[a-z0-9-]` only; keep under 50 chars. |
| `max_samples` rejected | Outside [15, 1000] | Clamp to range. |
| `output name exceeds limit` | Output name >50 chars | Shorten or include a shorter unique suffix. |

## Best practices

- **Always pin `agent_version`** on trace jobs unless you specifically want the union across versions.
- **Wait 90s** after the last agent invocation before submitting a trace job.
- **Check `generated_samples`** after every job — don't assume `max_samples` was hit.
- **Use `train_split`** when you'll fine-tune — the service does the random split for you and the output files plug straight into `fine_tuning.jobs.create(...)`.
- **Use `File` source for SFT with rich corpora; `Prompt` for ≤10k-char snippets; `Agent` for EVAL.**
- **For tool-calling SFT from traces**, remember to add the system prompt and `tools` array to each row before submitting to fine-tuning — the trace output contains `messages` but not the agent's system prompt or tool catalog. The `Traces` recipe does not stitch these in.
- **For cold-start tool-use SFT**, upload your tool catalog as a `.json` file (`[{"type":"function","function":{...}},…]`) and use `FileDataGenerationJobSource` with `ToolUseFineTuningDataGenerationJobOptions`. The job will synthesise queries that exercise the tools.
- **Delete files/datasets separately** — `delete_generation_job` doesn't remove output artifacts.
- **Pick a teacher one tier above the student.** Distilling from `gpt-4.1-mini` into `gpt-4.1-nano` works well; distilling from `gpt-4.1-nano` into anything is pointless.

## Related

- `workflows/traces-to-dataset.md` — end-to-end traces → SFT (with RFT and eval variants)
- `workflows/synthetic-datagen.md` — corpus → Q&A; agent spec → tool-use
- `workflows/dataset-creation.md` — overall decision tree across all data-creation approaches
- `scripts/generate_dataset.py` — CLI wrapper over `create_generation_job` with both SDK and REST modes
- `references/dataset-formats.md` — JSONL shapes for SFT, DPO, RFT, and evaluation
