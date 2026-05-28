# Synthetic Data Generation Workflow

Generate training or eval data without needing live agent traffic, using the **Data Generation API** (`project_client.beta.datasets`). This is the right workflow when you have a corpus (docs, knowledge base) or a deployed agent definition, but not enough real conversations yet.

> **API spec:** see `references/data-generation-api.md` for the full surface, constraints, and error matrix.
> **Have traces?** Use `workflows/traces-to-dataset.md` instead — real traffic beats synthetic.

## Pick a (source × recipe × scenario) combo

Not every combination is valid. The service-verified matrix:

| Source | Recipe | SFT | RFT | EVAL |
|--------|--------|-----|-----|------|
| `Prompt` (inline, ≤10k chars) | `qna` | ✅ | ❌ fails in-flight | ✅ |
| `File` (uploaded corpus) | `qna` | ✅ | ❌ fails in-flight | ❌ — use `Prompt` or `Agent` instead |
| `Agent` (deployed agent definition) | `qna` | ✅ | ❌ fails in-flight | ✅ |
| `File` (OpenAPI 3.x spec) | `tool-use` | ✅ (~2-25 min) | ❌ rejected | ❌ rejected |
| `Agent` only | `tool-use` | ❌ rejected — see below | ❌ rejected | ❌ rejected |

Legend: ✅ verified live · ❌ rejected by API or fails in-flight (`"Something went wrong"`). For RFT use the **Traces** source instead — see `workflows/traces-to-dataset.md`.

> **Tool-use needs a `.json` file source.** Even if you have an agent, the service requires the tool catalog as an uploaded `.json` file (`Tool use data generation requires exactly one .json file`). Dump your agent's tool definitions to JSON and use `FileDataGenerationJobSource`.

Decision tree:

```
What do you have?

├── A document, policy, or knowledge base
│   └── Use SimpleQnA recipe
│       - For SFT: File source (best for >10k chars) or Prompt (≤10k chars)
│       - For EVAL: Prompt or Agent source ONLY (File rejected)
│
├── A tool catalog (or an agent whose tools you can export as an OpenAPI 3.0 spec)
│   └── ToolUseFineTuning recipe + File source pointing at an OpenAPI 3.0.x/3.1.x spec
│       (SFT only; ~2-25 min wall-clock — varies with backend queue)
│
└── Just a task description (no corpus, no agent)
    └── SimpleQnA + Prompt source (lower grounding — manually review every row)
```

## Prerequisites

- Foundry project endpoint: `https://<resource>.services.ai.azure.com/api/projects/<project>`
- SDK: `pip install "azure-ai-projects>=2.2.0" azure-identity openai`
- Azure AI Project Contributor role
- For SimpleQnA / ToolUseFineTuning: a deployed chat-capable model to use as **teacher** (mandatory)
- For the **Agent source**: a deployed agent (the API reads its instructions; tools must be supplied via a separate `.json` file source for tool-use)
- For SimpleQnA + EVAL: the teacher model must support the Azure OpenAI **Responses API**

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project_client = AIProjectClient(
    endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>",
    credential=DefaultAzureCredential(),
)
```

The SDK automatically sends the `Foundry-Features: DataGenerationJobs=V1Preview` header — no `allow_preview` flag needed.

## Recipe 1 — Q&A from a document (SFT)

Bootstrap training data from a policy document, manual, knowledge base, or any text corpus.

### Small inline doc (≤10,000 chars) → Prompt source

```python
import time, uuid
from datetime import datetime, timezone
from azure.ai.projects.models import (
    DataGenerationJob, DataGenerationJobInputs, DataGenerationJobOutputOptions,
    DataGenerationJobScenario, DataGenerationModelOptions, JobStatus,
    FileDataGenerationJobOutput,
    PromptDataGenerationJobSource, SimpleQnADataGenerationJobOptions,
)

run_id = f"{datetime.now(timezone.utc).strftime('%y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
output_name = f"refund-qna-{run_id}"

doc = open("refund_policy.md").read()
assert len(doc) <= 10_000, "Use FileDataGenerationJobSource for prompts >10k chars"

job = project_client.beta.datasets.create_generation_job(
    job=DataGenerationJob(inputs=DataGenerationJobInputs(
        name=output_name,
        scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
        sources=[PromptDataGenerationJobSource(
            prompt=doc,
            description="Company refund policy v3 (2026 Q1)",
        )],
        options=SimpleQnADataGenerationJobOptions(
            max_samples=100,              # 15–1000 service limit
            train_split=0.9,
            model_options=DataGenerationModelOptions(model="gpt-4.1-mini"),  # REQUIRED
        ),
        output_options=DataGenerationJobOutputOptions(name=output_name),
    )),
)

TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
while job.status not in TERMINAL:
    time.sleep(10)
    job = project_client.beta.datasets.get_generation_job(job_id=job.id)

if job.status != JobStatus.SUCCEEDED:
    raise RuntimeError(job.error.message if job.error else "<no error>")

print(f"Generated {job.result.generated_samples} samples")
for o in job.result.outputs:
    if isinstance(o, FileDataGenerationJobOutput):
        print(f"  file_id={o.id}")
```

Output is chat-format JSONL ready for fine-tuning:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant. You will be presented with a question, please provide a clear and accurate answer."}, {"role": "user", "content": "How long do I have to return an item?"}, {"role": "assistant", "content": "Standard items can be returned within 30 days of delivery..."}]}
```

**Observed throughput** (cont-learning-faos, gpt-4.1-mini teacher, 2 KB prompt): **15 samples in ~70s**.

### Large corpus / PDF → upload first, then File source

For anything bigger than ~10k chars, upload via the Files API and reference by id. The service requires content to be **≥ 1 KB** and **substantive** — sparse outlines or referential text will fail with `"File content lacks sufficient context to generate quality questions."`.

```python
import io
from azure.ai.projects.models import FileDataGenerationJobSource

aoai = project_client.get_openai_client()

# Upload as user_data (the only purpose accepted for datagen sources)
with open("employee_handbook.pdf", "rb") as f:
    seed = aoai.files.create(file=("employee_handbook.pdf", f), purpose="user_data")

# Wait for processing
while seed.status not in ("processed", "error"):
    time.sleep(2)
    seed = aoai.files.retrieve(file_id=seed.id)
if seed.status != "processed":
    raise RuntimeError(f"upload failed: {seed.status}")

job = project_client.beta.datasets.create_generation_job(
    job=DataGenerationJob(inputs=DataGenerationJobInputs(
        name=f"handbook-qna-{run_id}",
        scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
        sources=[FileDataGenerationJobSource(id=seed.id,
            description="Employee handbook 2026")],
        options=SimpleQnADataGenerationJobOptions(
            max_samples=300,
            train_split=0.9,
            model_options=DataGenerationModelOptions(model="gpt-4.1-mini"),
        ),
        output_options=DataGenerationJobOutputOptions(name=f"handbook-qna-{run_id}"),
    )),
)
```

**Rule of thumb:** `max_samples ≈ corpus_token_count / 500` for non-redundant Q&A. A 50,000-token handbook supports ~100 distinct pairs without near-duplicates.

### Eval set from a doc

For EVAL scenario, **use `Prompt` or `Agent` source — `File` is rejected**. Cap your doc at 10k chars or pre-summarise it:

```python
job = project_client.beta.datasets.create_generation_job(
    job=DataGenerationJob(inputs=DataGenerationJobInputs(
        name=f"refund-eval-{run_id}",
        scenario=DataGenerationJobScenario.EVALUATION,
        sources=[PromptDataGenerationJobSource(prompt=open("refund_policy.md").read())],
        options=SimpleQnADataGenerationJobOptions(
            max_samples=50,
            model_options=DataGenerationModelOptions(model="gpt-4.1-mini"),
            # train_split ignored for EVAL
        ),
        output_options=DataGenerationJobOutputOptions(name=f"refund-eval-{run_id}"),
    )),
)
```

EVAL output is a **registered Dataset**, not a file:

```python
from azure.ai.projects.models import DatasetDataGenerationJobOutput

ds_out = next(o for o in job.result.outputs if isinstance(o, DatasetDataGenerationJobOutput))
dataset = project_client.datasets.get(name=ds_out.name, version=ds_out.version)
print(f"Eval dataset id: {dataset.id}")
```

**Observed throughput** (cont-learning-faos, gpt-4.1-mini, 2 KB prompt): **15 samples in ~250s** — EVAL is significantly slower than SFT because the service also produces a candidate response for each query.

## Recipe 2 — Tool-calling SFT from an OpenAPI 3.0 spec

Cold-start tool-calling training data. The service needs **exactly one `.json` file** containing a valid **OpenAPI 3.0.x or 3.1.x specification** for your tool catalog — NOT the OpenAI chat-completions tool format. Each tool maps to a `POST /<operationId>` operation whose `requestBody.schema` is the function parameters.

```json
{
  "openapi": "3.0.3",
  "info": {"title": "Retail API", "version": "1.0.0"},
  "paths": {
    "/get_order": {
      "post": {
        "operationId": "get_order",
        "summary": "Get an order by ID",
        "requestBody": {
          "required": true,
          "content": {"application/json": {"schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
          }}}
        },
        "responses": {"200": {"description": "OK", "content": {"application/json": {"schema": {"type": "object"}}}}}
      }
    }
  }
}
```

**If you only have OpenAI tool-spec JSON** (`[{"type":"function","function":{...}}]`), convert it with the helper baked into the CLI script:

```powershell
python scripts/generate_dataset.py `
    --tools-from openai_tools.json `
    --tools-to-openapi-out openapi.json
```

The converter is conservative: tools with no parameters get no `requestBody` (an empty schema is rejected by the OpenAPI validator).

Upload the OpenAPI spec as `user_data` and submit:

```python
import io, json

with open("openapi.json", "rb") as fh:
    seed_file = aoai.files.create(
        file=("openapi.json", fh),
        purpose="user_data",
    )
while aoai.files.retrieve(file_id=seed_file.id).status != "processed":
    time.sleep(2)
```

```python
from azure.ai.projects.models import (
    FileDataGenerationJobSource, ToolUseFineTuningDataGenerationJobOptions,
)

job = project_client.beta.datasets.create_generation_job(
    job=DataGenerationJob(inputs=DataGenerationJobInputs(
        name=f"retail-tools-{run_id}",
        scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
        sources=[FileDataGenerationJobSource(
            id=seed_file.id,
            description="Retail OpenAPI 3.0 spec",
        )],
        options=ToolUseFineTuningDataGenerationJobOptions(
            max_samples=50,
            train_split=0.8,
            model_options=DataGenerationModelOptions(model="gpt-4.1-mini"),
        ),
        output_options=DataGenerationJobOutputOptions(name=f"retail-tools-{run_id}"),
    )),
)
```

Output rows include the system prompt and `tools` array (in OpenAI chat-completions format, ready for fine-tuning without patching):

```python
aoai = project_client.get_openai_client()
file_outs = [o for o in job.result.outputs if isinstance(o, FileDataGenerationJobOutput)]
ft = aoai.fine_tuning.jobs.create(
    training_file=file_outs[0].id,
    validation_file=file_outs[1].id,
    model="gpt-4.1-mini",
    suffix="retail-tools-v1",
)
```

> **Wall-clock.** Tool-use jobs vary widely — observed range **~2 min (idle backend) to ~25 min (queue spike)**. Make sure your polling loop has a sufficient timeout (45+ minutes is safe for unattended runs).

> **Tool-use is SFT-only.** `ToolUseFineTuningDataGenerationJobOptions` is rejected when `scenario=EVALUATION` or `REINFORCEMENT_FINETUNING`. For tool-calling evals, generate with SFT scenario, then run live against the agent.

> **`Agent` source alone is rejected.** The service responds with `Tool use data generation requires exactly one .json file`. Even if you have an agent registration, you must supply the tool catalog as a `.json` OpenAPI 3.x upload.

> **OpenAI tool-spec JSON is also rejected** — submitting `[{"type":"function","function":{...}}]` is accepted at job-submit but the job fails in-flight with `Invalid or unsupported OpenAPI version. Supported versions: 3.0.x and 3.1.x`. Convert first.

**Coverage tip.** The recipe biases toward exercising each tool at least once. If your agent has 15 tools, set `max_samples ≥ 30` so most tools see multiple examples.

## Recipe 3 — Q&A from a task description (no corpus)

When you have neither traces nor a corpus — just a task description — paste the description as the `prompt`. This is the lowest-grounding mode and the riskiest for hallucinations, so use it only to bootstrap a few seed examples for human review:

```python
job = project_client.beta.datasets.create_generation_job(
    job=DataGenerationJob(inputs=DataGenerationJobInputs(
        name=f"seed-{run_id}",
        scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
        sources=[PromptDataGenerationJobSource(
            prompt=(
                "You are designing training data for a healthcare triage assistant. "
                "The assistant helps patients understand symptoms and decide whether "
                "to see a doctor. Generate realistic user questions and expert responses. "
                "Cover at least: chest pain, headaches, fever, rashes, GI issues, "
                "mental health, and pediatric concerns."
            ),
            description="Healthcare triage seed prompt",
        )],
        options=SimpleQnADataGenerationJobOptions(
            max_samples=15,        # service minimum; hand-review every row
            train_split=0.8,
            model_options=DataGenerationModelOptions(model="gpt-4.1-mini"),
        ),
        output_options=DataGenerationJobOutputOptions(name=f"seed-{run_id}"),
    )),
)
```

Plan to manually review every row before training on it. Once you have a curated corpus, switch to Recipe 1 for higher-volume, grounded generation.

## Combining sources

To pool several documents into one job, pass multiple sources of the **same type**:

```python
sources=[
    FileDataGenerationJobSource(id="file-policy-2025-q4"),
    FileDataGenerationJobSource(id="file-policy-2026-q1"),
    FileDataGenerationJobSource(id="file-faq-2026-q1"),
]
```

Mixing source *types* in a single job (e.g. `File` + `Prompt`) is not supported. Run separate jobs and concatenate the resulting JSONLs.

## Picking a teacher model (`model_options`)

The teacher is the model that produces the synthetic Q&A or candidate responses. **It should be at least as capable as the student**, otherwise you're distilling from a weaker source.

| Student you'll fine-tune | Recommended teacher |
|---------|---------------------|
| `gpt-4.1-nano` | `gpt-4.1-mini` or `gpt-4.1` |
| `gpt-4.1-mini` | `gpt-4.1` or `gpt-5.x` |
| `gpt-4.1` | `gpt-5.x` |
| OSS (Llama / Qwen / Ministral) | `gpt-4.1` or stronger |

```python
model_options=DataGenerationModelOptions(model="gpt-4.1")    # deployment name, not model family
```

The teacher value is a **deployment name** in your project, not the model family. If `gpt-4.1` isn't deployed in your project, deploy it first or pick what is available (`project_client.deployments.list()`).

For SimpleQnA + EVAL the teacher must support the **Responses API** — check the supported model list at [Responses API model support](https://learn.microsoft.com/azure/foundry/openai/how-to/responses).

## Using the CLI script

The same flows are wrapped in `scripts/generate_dataset.py`:

```powershell
$env:AZURE_AI_PROJECT_ENDPOINT = "https://<r>.services.ai.azure.com/api/projects/<p>"

# Q&A from a doc (auto-uploads as a File for richer service errors)
python scripts/generate_dataset.py `
    --source prompt-file --prompt-file refund_policy.md `
    --recipe qna --scenario sft --teacher gpt-4.1-mini `
    --max-samples 100 --train-split 0.9 --download

# Q&A from already-uploaded file
python scripts/generate_dataset.py `
    --source file --file-id file-abc123 `
    --recipe qna --scenario sft --teacher gpt-4.1-mini `
    --max-samples 100 --train-split 0.9 --download

# Tool-use from an OpenAPI 3.0 spec (convert OpenAI tool format first if needed)
python scripts/generate_dataset.py `
    --tools-from openai_tools.json --tools-to-openapi-out openapi.json
# Upload openapi.json via aoai.files.create(purpose="user_data"), then:
python scripts/generate_dataset.py `
    --source file --file-id file-openapiabc123 `
    --recipe tool-use --scenario sft --teacher gpt-4.1-mini `
    --max-samples 50 --train-split 0.8 --download
# NOTE: tool-use varies 2-25 min wall-clock; bump --max-polls accordingly for unattended runs.

# Eval set from inline prompt
$prompt = Get-Content refund_policy.md -Raw
python scripts/generate_dataset.py `
    --source prompt-inline --prompt $prompt `
    --recipe qna --scenario eval --teacher gpt-4.1-mini `
    --max-samples 50
```

Source choices:
- `--source prompt-inline` → `PromptDataGenerationJobSource` (≤10k chars)
- `--source prompt-file` → uploads the file as `user_data`, then uses `FileDataGenerationJobSource` (richer corpora, better service errors)
- `--source file --file-id ...` → existing OpenAI file
- `--source agent --agent-name ...` → `AgentDataGenerationJobSource`
- `--source traces --agent-name ...` → `TracesDataGenerationJobSource` (covered in `traces-to-dataset.md`)

## Quality checks before training

Synthetic data is faster than curating by hand, but quality varies more. Before submitting a fine-tuning job:

1. **Spot-check 20 random rows.** Are the answers grounded in the source? Any obvious hallucinations?
2. **Run the validators.**
   ```bash
   python scripts/validate/validate_sft.py train.jsonl
   python scripts/validate/data_stats.py train.jsonl
   ```
3. **Check distribution.** For Q&A: are questions diverse, or all variations of the same 3 topics? For tool-use: is each tool exercised?
4. **(Optional) Score with an LLM judge** before training — `scripts/score_dataset.py` flags low-quality rows.

A small, high-quality dataset (300 examples, manually reviewed) typically out-performs a larger, noisy one. See `workflows/dataset-creation.md` for the evidence.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Invalid payload: At least Prompt or prompt agent_name is required for SimpleQnA Evaluation output format` | `File` source used with `EVALUATION` scenario | Switch to `Prompt` (≤10k chars) or `Agent` source. |
| `File content is too small to generate QnA. It must be at least 1KB.` | File source has <1 KB content | Upload a richer document. |
| `File content lacks sufficient context to generate quality questions.` | File content too thin / referential | Make content comprehensive and self-contained; avoid bullets without surrounding context. |
| `Something went wrong during data generation. Please try again.` (FAILED in ~15s) | Generic — often a teacher-model issue (FT-only deployment, wrong API surface, etc.) | Try a different `--teacher`; confirm the model deployment exists and is healthy. |
| `--teacher required for --recipe qna` | SimpleQnA / ToolUseFineTuning require `model_options` | Pass a teacher deployment name. |
| `max_samples` rejected | Outside [15, 1000] | Clamp to range. |
| `--recipe tool-use is SFT-only` | Tool-use options with EVAL or RFT scenario | Use scenario `sft`. |
| Q&A pairs look near-duplicate | `max_samples` exceeds corpus capacity | Reduce `max_samples`, or pool more diverse source documents. |
| Tool-use output skips some tools | `max_samples` too low to cover the catalog | Increase `max_samples` to ≥ 2× number of tools. |

## Related

- `references/data-generation-api.md` — full API surface
- `workflows/traces-to-dataset.md` — distill from real production traffic
- `workflows/dataset-creation.md` — overall decision tree across all data-creation approaches
- `references/dataset-formats.md` — JSONL shapes (SFT, DPO, RFT, eval)
- `scripts/score_dataset.py` — LLM-judge quality scoring on generated data
