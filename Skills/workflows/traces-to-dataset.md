# Traces → Dataset Workflow

Turn real conversations from a deployed Foundry agent into a training, evaluation, or RFT dataset using the **Data Generation API** (`project_client.beta.datasets`). This is the canonical "distill production traffic into a smaller, cheaper model" pipeline.

> **API spec:** see `references/data-generation-api.md` for the full surface (sources, options, scenarios, constraints, error matrix).
> **No traces yet?** Use `workflows/synthetic-datagen.md` instead.

## When to use this workflow

| You want to… | Read this section |
|---|---|
| Train a smaller model on a bigger model's behaviour | [Distill traces → SFT](#distill-traces--sft) |
| Train an RFT model using traces as seeds | [Traces → RFT](#traces--rft) |
| Build an eval set from real conversations | [Traces → eval set](#traces--eval-set) |
| Cold-start an agent with no traffic | Use `workflows/synthetic-datagen.md` instead |

## Prerequisites

- A Microsoft Foundry project endpoint: `https://<resource>.services.ai.azure.com/api/projects/<project>`
- **Application Insights attached** to the project. Configure in the portal under **Project settings → Telemetry**.
- A **deployed agent** that emits traces. Foundry-hosted agents emit traces automatically; for custom agents (LangGraph, Semantic Kernel, etc.), instrument with OpenTelemetry exporters that target your project's App Insights.
- SDK: `pip install "azure-ai-projects>=2.2.0" azure-identity openai`
- Azure AI Project Contributor role or higher on the project

## Distill traces → SFT

The full pattern: capture traffic against the deployed agent, generate a dataset from those traces, then fine-tune a smaller model on the output.

### Step 1 — Open the client

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project_client = AIProjectClient(
    endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>",
    credential=DefaultAzureCredential(),
)
```

The SDK automatically sends the `Foundry-Features: DataGenerationJobs=V1Preview` header for `beta.datasets` calls — no `allow_preview` flag needed.

### Step 2 — Capture a known traffic window

You need a known time range that contains the conversations you want to distill. The simplest pattern is to drive the agent yourself, recording timestamps around the calls:

```python
import time
from datetime import datetime, timedelta, timezone

AGENT_NAME = "retail-agent-langgraph"
AGENT_VERSION = "3"                    # always pin

window_start = datetime.now(timezone.utc) - timedelta(seconds=5)

# Run conversations against your deployed agent here.
# (your harness, replayed real prompts, evals, whatever)

window_end = datetime.now(timezone.utc) + timedelta(seconds=5)

# Wait for App Insights to ingest the spans (30–90s).
time.sleep(90)
```

**The 90-second wait is non-negotiable.** App Insights ingestion lag is the #1 cause of "job succeeded, generated 0 samples".

For a window that's already passed (e.g. last week's traffic), skip the sleep:

```python
window_start = datetime.now(timezone.utc) - timedelta(days=1)
window_end   = datetime.now(timezone.utc) - timedelta(hours=1)
```

### Step 3 — Submit the job

```python
import uuid
from azure.ai.projects.models import (
    DataGenerationJob,
    DataGenerationJobInputs,
    DataGenerationJobOutputOptions,
    DataGenerationJobScenario,
    JobStatus,
    TracesDataGenerationJobOptions,
    TracesDataGenerationJobSource,
)

# Output name must be ≤50 chars; include a unique suffix per run.
run_id = f"{datetime.now(timezone.utc).strftime('%y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
output_name = f"retail-sft-{run_id}"   # 26 chars, safely under 50

job_request = DataGenerationJob(inputs=DataGenerationJobInputs(
    name=output_name,
    scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
    sources=[TracesDataGenerationJobSource(
        agent_name=AGENT_NAME,
        agent_version=AGENT_VERSION,
        start_time=window_start,
        end_time=window_end,
        description="Last hour of production traffic",
    )],
    options=TracesDataGenerationJobOptions(
        max_samples=200,           # 15–1000 service limit; ceiling not guarantee
        train_split=0.8,           # 80/20 → 2 output files
    ),
    output_options=DataGenerationJobOutputOptions(name=output_name),
))

job = project_client.beta.datasets.create_generation_job(job=job_request)
print(f"Submitted {job.id}")
```

### Step 4 — Poll until done

```python
TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
while job.status not in TERMINAL:
    time.sleep(10)
    job = project_client.beta.datasets.get_generation_job(job_id=job.id)
    print(f"  status: {job.status}")

if job.status != JobStatus.SUCCEEDED:
    err = job.error.message if job.error else "<no error>"
    raise RuntimeError(f"Job ended in {job.status}: {err}")

print(f"Generated {job.result.generated_samples} samples")
```

**Always read `generated_samples`.** If it's `0` or much lower than `max_samples`, see [Troubleshooting](#troubleshooting).

### Step 5 — Pipe straight to fine-tuning (or download)

Outputs are typed; use `isinstance`:

```python
from azure.ai.projects.models import FileDataGenerationJobOutput

aoai = project_client.get_openai_client()
file_outputs = [o for o in job.result.outputs if isinstance(o, FileDataGenerationJobOutput)]

# Option A: just fine-tune — no local download needed
ft = aoai.fine_tuning.jobs.create(
    training_file=file_outputs[0].id,    # index 0 = train (because train_split was set)
    validation_file=file_outputs[1].id,  # index 1 = validation
    model="gpt-4.1-nano",                # the student
    suffix="distilled-from-traces",
)
print(f"Fine-tuning job: {ft.id}")
```

Or download for local inspection:

```python
for o in file_outputs:
    info = aoai.files.retrieve(file_id=o.id)
    with open(info.filename, "wb") as f:
        f.write(aoai.files.content(o.id).content)
    print(f"  saved {info.filename}  ({info.bytes} bytes)")
```

Files come back named `{output_name}_train_dg.jsonl` and `{output_name}_valid_dg.jsonl`.

### Step 6 — Validate before training (recommended)

The Data Generation API produces well-formed JSONL but it never hurts to sanity-check:

```bash
python scripts/validate/validate_sft.py retail-sft-<run_id>_train_dg.jsonl
python scripts/validate/data_stats.py retail-sft-<run_id>_train_dg.jsonl
```

Things to spot-check on a sample of 20 rows: assistant outputs are non-empty, no PII leaked past the filter, tool definitions present when expected.

## Traces → RFT

RFT is its own scenario — set `scenario=DataGenerationJobScenario.REINFORCEMENT_FINETUNING`. The service emits RFT-format JSONL (prompt + ground-truth answer) directly; no post-processing required.

```python
job_request = DataGenerationJob(inputs=DataGenerationJobInputs(
    name=f"retail-rft-{run_id}",
    scenario=DataGenerationJobScenario.REINFORCEMENT_FINETUNING,
    sources=[TracesDataGenerationJobSource(
        agent_name=AGENT_NAME, agent_version=AGENT_VERSION,
        start_time=window_start, end_time=window_end,
    )],
    options=TracesDataGenerationJobOptions(max_samples=300, train_split=0.8),
    output_options=DataGenerationJobOutputOptions(name=f"retail-rft-{run_id}"),
))
```

> **When RFT makes sense.** If your task has a verifiable answer (tool-calling correctness, math, code execution), pair this with a grader. See `references/grader-design.md` and `references/training-types.md`.

## Traces → eval set

Flip the scenario to EVALUATION:

```python
job_request = DataGenerationJob(inputs=DataGenerationJobInputs(
    name=f"retail-eval-{run_id}",
    scenario=DataGenerationJobScenario.EVALUATION,
    sources=[TracesDataGenerationJobSource(
        agent_name=AGENT_NAME, agent_version=AGENT_VERSION,
        start_time=window_start, end_time=window_end,
    )],
    options=TracesDataGenerationJobOptions(
        max_samples=1000,        # eval sets tolerate larger sizes
        # train_split ignored for EVAL
    ),
    output_options=DataGenerationJobOutputOptions(name=f"retail-eval-{run_id}"),
))
```

EVAL output is a **registered Dataset**, not a file:

```python
from azure.ai.projects.models import DatasetDataGenerationJobOutput

ds_out = next(o for o in job.result.outputs if isinstance(o, DatasetDataGenerationJobOutput))
print(f"Eval dataset: name={ds_out.name}  version={ds_out.version}")

# Pull the underlying DatasetVersion for use in eval pipelines:
dataset = project_client.datasets.get(name=ds_out.name, version=ds_out.version)
print(dataset.id)
```

**Tip:** Keep eval sets sourced from a *separate* time window than training data, so you're not measuring memorisation. Common pattern: train on Monday–Friday traffic, evaluate on Saturday's.

## Tool-calling traces

When the agent emits tool calls, those appear in the output as `tool_call` parts in the assistant message and `role: "tool"` messages with results. Two important caveats for fine-tuning:

1. **System prompt is not in the rows.** Trace output gives you `messages`, but not the agent's actual system prompt or `tools` array. You must inject these before submitting to a fine-tuning job, or the student won't learn to use the right system prompt/tools at inference time.
2. **Tool catalog must match.** The `tools` array you inject must match the tools that were actually available during the trace, or the trained model will hallucinate tool names.

If you'd rather have this stitched together automatically, use the **Agent source + ToolUseFineTuning recipe** instead — see `workflows/synthetic-datagen.md`. It pulls the agent's actual instructions and tool catalog, then generates plausible scenarios.

To patch trace-generated rows manually:

```python
import json

SYSTEM_PROMPT = "You are a retail customer service agent..."   # from agent definition
TOOLS = [...]                                                  # from agent definition

with open("retail-sft-..._train_dg.jsonl") as fin, open("train_patched.jsonl", "w") as fout:
    for line in fin:
        row = json.loads(line)
        row["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + row["messages"]
        row["tools"] = TOOLS
        fout.write(json.dumps(row) + "\n")
```

## Using the CLI script

The same end-to-end flow is wrapped in `scripts/generate_dataset.py`:

```powershell
# SFT from traces, last 24h
$env:AZURE_AI_PROJECT_ENDPOINT = "https://<r>.services.ai.azure.com/api/projects/<p>"

python scripts/generate_dataset.py `
    --source traces --agent-name retail-agent-langgraph --agent-version 3 `
    --recipe traces --scenario sft `
    --max-samples 200 --train-split 0.8 `
    --hours 24 --download

# Eval set
python scripts/generate_dataset.py `
    --source traces --agent-name retail-agent-langgraph --agent-version 3 `
    --recipe traces --scenario eval `
    --max-samples 1000 --hours 24

# RFT seeds
python scripts/generate_dataset.py `
    --source traces --agent-name retail-agent-langgraph --agent-version 3 `
    --recipe traces --scenario rft `
    --max-samples 300 --train-split 0.8 --hours 24 --download
```

The script auto-generates a ≤50-char output name. Override with `--output-name` for stable names across runs. Use `--use-rest` for the REST API path (no preview SDK required, but still needs AAD).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `generated_samples == 0` | App Insights window empty | Confirm agent was invoked during window; confirm submit-time ≥ window_end + 90s; widen window. |
| `generated_samples << max_samples` | Quality filter dropped most rows | Widen the time window; inspect dropped rows with App Insights queries (KQL on `traces` table filtered by `customDimensions.agent_name`). |
| Job fails immediately | Missing telemetry config or wrong agent name | Verify App Insights is attached under Project settings → Telemetry. Confirm `agent_name` matches a deployed agent exactly (case-sensitive). |
| `Something went wrong during data generation` (FAILED in 15s) | Generic — often a service-side issue | Try again; pin `agent_version`; reduce `max_samples`; contact service team if persistent. |
| Mixed/weird system prompts in output | `agent_version` wasn't pinned | Add `agent_version="N"` and re-run. |
| Tool-calling rows missing tools array | Expected — trace output doesn't include the catalog | Patch with the snippet above, or switch to the Agent source + ToolUseFineTuning recipe. |

## What to do next

1. **Pick a student model.** `gpt-4.1-nano` or `gpt-4.1-mini` are typical distillation targets. See `references/training-types.md` and `references/hyperparameters.md`.
2. **Submit training.** Use `scripts/submit_training.py` or pass the file ids straight to `aoai.fine_tuning.jobs.create(...)` (shown above).
3. **Track training and pick a checkpoint.** See `workflows/iterative-training.md` and `references/training-curve-analysis.md`.
4. **Evaluate the student.** Use the eval set from [Traces → eval set](#traces--eval-set), or `scripts/evaluate_model.py`.

## Related

- `references/data-generation-api.md` — full API surface
- `workflows/synthetic-datagen.md` — cold-start: generate data without traffic
- `workflows/dataset-creation.md` — overall decision tree across all data-creation approaches
- `references/dataset-formats.md` — JSONL shapes (SFT, DPO, RFT, eval)
- `references/training-types.md` — choosing SFT vs DPO vs RFT
