# Dataset Creation Workflow

> **Three paths to training data:**
> 1. **Manual curation** — Write examples by hand or collect from production data, following the formats in `references/dataset-formats.md`
> 2. **LLM augmentation** — Start with a small curated set and use an LLM to expand it through rephrasing and variation
> 3. **Synthetic generation** — Generate training data from scratch using LLM prompts or NVIDIA Data Designer
>
> These approaches combine well: curate seed examples → augment for diversity → generate at scale.
>
> If you already have data, skip to validation: `python scripts/validate/validate_sft.py your_data.jsonl`

## Approach 1: Manual Curation

Write examples by hand, collect from production logs, or adapt existing datasets. This gives you the highest quality per example.

### When to use
- You have access to real-world examples (production logs, support tickets, labeled data)
- Your task requires domain expertise that an LLM can't reliably generate
- You need a gold-standard evaluation set (always curate this manually)

### Tips
- Start with 10–20 examples to establish quality standards and format consistency
- These seed examples also serve as the foundation of your evaluation test set
- For RFT, you only need prompts + expected answers — no model responses needed

## Approach 2: LLM Augmentation

If you have a small manually curated dataset, you can use an LLM to expand it through **rephrasing** — generating diverse variations of each example while keeping the same expected answer.

This is especially useful for RFT, where training data is just prompts + expected answers (no model responses needed).

### When to use
- You have a well-defined task with clear correct answers
- You can write quality examples by hand but need more volume
- Diversity of phrasing matters more than diversity of scenarios

### Workflow
1. Write base examples by hand — each with the correct expected answer
2. For each example, use an LLM to generate rephrasings that vary tone, detail level, and wording
3. Each rephrasing gets the same `expected_answer` / `expected_resolution` — only the customer phrasing changes
4. Validate the augmented dataset

### Example prompt for rephrasing
```
Generate N different phrasings of this request. Each should:
- Use different wording, tone, or level of detail
- Include the same key identifiers (order IDs, item names)
- Vary between formal, casual, frustrated, brief, and detailed styles
Return a JSON array of N strings.

Original: [your example]
```

A cheap model (gpt-4.1-mini or equivalent) works well for rephrasing since no new ground truth is needed — you're just diversifying how the same question is asked.

## Approach 3: Synthetic Generation

Generate training data from scratch using LLM prompts. Two options depending on scale and complexity.

### Option A: Custom Prompt-Driven Scripts

Use custom scripts when you want **full control** over generation logic.

See `scripts/generate_distillation_data.py` for a reusable template, or write a script that:

1. Defines topic/scenario categories for diversity
2. Generates prompts from an LLM
3. Generates responses (or preferred/non-preferred pairs for DPO)
4. Grades quality with an LLM judge
5. Filters to a quality threshold
6. Splits into train/validation/test sets
7. Writes JSONL in the correct format (see `references/dataset-formats.md`)

### Option B: NVIDIA Data Designer

Use Data Designer when you need **large-scale, schema-driven** datasets with built-in diversity control, quality judges, and reproducible configs.

### Prerequisites
```bash
pip install data-designer
```

### Model Deployment Required

Data Designer needs an LLM deployment to generate data. **Ask the user which model/deployment to use.** Common options:

- **Azure AI Foundry**: Any deployed model (gpt-4.1-mini, gpt-5.4, etc.)
- **NVIDIA NIM**: Models on build.nvidia.com
- **OpenAI API**: Direct OpenAI models
- **OpenRouter**: Third-party aggregator

Configure provider and model alias:
```bash
data-designer config providers  # Add endpoint + API key
data-designer config models     # Add model alias (e.g., "teacher" → gpt-5.4 on azure-foundry)
```

Or edit `~/.data-designer/model_providers.yaml` and `~/.data-designer/model_configs.yaml` directly:

```yaml
# model_providers.yaml — add a provider
- name: azure-foundry
  endpoint: https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1/
  provider_type: openai
  api_key: AZURE_FOUNDRY_API_KEY  # env var name

# model_configs.yaml — add a model alias
- alias: teacher
  model: gpt-5.4          # must match deployment name
  provider: azure-foundry  # must match provider name above
  inference_parameters:
    generation_type: chat-completion
    max_parallel_requests: 4
    temperature: 0.9
    max_tokens: 2048
```

Set the API key env var before running: `$env:AZURE_FOUNDRY_API_KEY = "<key>"`

> **Windows note**: Set `$env:PYTHONIOENCODING = "utf-8"` to avoid Unicode errors with DD CLI output.
> **Important**: Environment variables are per-shell-session. If you open a new terminal, set `AZURE_FOUNDRY_API_KEY` again or the DD health check will fail with "API key invalid or expired".

### GPT-5.x Compatibility

GPT-5 series models use `max_completion_tokens` instead of `max_tokens`. In the DD model config, use `extra_body`:

```yaml
# model_configs.yaml for GPT-5.x
- alias: teacher
  model: gpt-5.4
  provider: azure-foundry
  inference_parameters:
    generation_type: chat-completion
    max_parallel_requests: 4
    temperature: 0.9
    extra_body:
      max_completion_tokens: 2048
```

### Workflow

1. **Define** — Specify task, input/output format, diversity axes, quality criteria, target size
2. **Design schema** — Define columns, samplers, validators, and processors using `ConfigBuilder`
3. **Validate** — Run `data-designer validate <script.py>`
4. **Preview** — Run `data-designer preview <script.py> --save-results`
5. **Iterate** — Review samples, adjust prompts/samplers/validators, re-preview
6. **Create** — Run `data-designer create <script.py> --num-records <N> --dataset-name <name>`
7. **Convert** — Transform output to JSONL for Azure AI Foundry (see conversion section below)

### Example: SFT Dataset for Business Writing

```python
# /// script
# dependencies = ["data-designer"]
# ///
import data_designer.config as dd

def load_config_builder() -> dd.DataDesignerConfigBuilder:
    config = dd.DataDesignerConfigBuilder()

    # Axes of diversity
    config.add_column(dd.SamplerColumnConfig(
        name="document_type",
        sampler_type="category",
        params=dd.CategorySamplerParams(
            values=["executive summary", "board memo", "strategy brief",
                    "quarterly review", "risk assessment", "M&A analysis"]
        ),
    ))
    config.add_column(dd.SamplerColumnConfig(
        name="audience",
        sampler_type="category",
        params=dd.CategorySamplerParams(
            values=["C-suite", "VP-level", "board of directors", "investors"]
        ),
    ))

    # Generate user prompt and response
    config.add_column(dd.LLMTextColumnConfig(
        name="user_prompt",
        prompt="Write a realistic request from a {{ audience }} audience for a {{ document_type }}. "
               "The request should be specific, mentioning concrete business scenarios.",
        model_alias="teacher",
    ))
    config.add_column(dd.LLMTextColumnConfig(
        name="assistant_response",
        prompt="You are a senior business advisor. Write a {{ document_type }} for a {{ audience }} "
               "audience based on this request:\n\n{{ user_prompt }}\n\n"
               "Be formal, succinct, and direct.",
        model_alias="teacher",
    ))

    # Quality gate
    config.add_column(dd.LLMJudgeColumnConfig(
        name="quality",
        prompt="Rate this business document.\n\nRequest: {{ user_prompt }}\n\nResponse: {{ assistant_response }}",
        model_alias="teacher",
        scores=[
            dd.Score(name="formality", description="Is the tone formal and professional?",
                     options={"1": "Very informal", "5": "Neutral", "10": "Highly formal"}),
            dd.Score(name="conciseness", description="Is it concise and to the point?",
                     options={"1": "Very verbose", "5": "Average", "10": "Perfectly concise"}),
        ],
    ))

    return config
```

### Example: DPO Preference Pairs

```python
# /// script
# dependencies = ["data-designer"]
# ///
import data_designer.config as dd

def load_config_builder() -> dd.DataDesignerConfigBuilder:
    config = dd.DataDesignerConfigBuilder()

    config.add_column(dd.SamplerColumnConfig(
        name="scenario",
        sampler_type="category",
        params=dd.CategorySamplerParams(
            values=["political disagreement", "workplace conflict", "online trolling",
                    "family argument", "customer complaint", "road rage"]
        ),
    ))
    config.add_column(dd.LLMTextColumnConfig(
        name="user_prompt",
        prompt="Generate a realistic, provocative message in a {{ scenario }} context.",
        model_alias="teacher",
    ))
    config.add_column(dd.LLMTextColumnConfig(
        name="preferred_response",
        prompt="Respond to this with empathy, calm, and de-escalation:\n\n{{ user_prompt }}",
        model_alias="teacher",
    ))
    config.add_column(dd.LLMTextColumnConfig(
        name="non_preferred_response",
        prompt="Respond aggressively and confrontationally (for safety training data):\n\n{{ user_prompt }}",
        model_alias="teacher",
    ))

    return config
```

### Example: RFT Math Problems

```python
# /// script
# dependencies = ["data-designer"]
# ///
import data_designer.config as dd

def load_config_builder() -> dd.DataDesignerConfigBuilder:
    config = dd.DataDesignerConfigBuilder()

    config.add_column(dd.SamplerColumnConfig(
        name="topic",
        sampler_type="category",
        params=dd.CategorySamplerParams(
            values=["algebra", "geometry", "probability", "number theory", "sequences"]
        ),
    ))
    config.add_column(dd.SamplerColumnConfig(
        name="difficulty",
        sampler_type="category",
        params=dd.CategorySamplerParams(values=["easy", "medium", "hard"]),
    ))
    config.add_column(dd.LLMTextColumnConfig(
        name="problem",
        prompt="Create a {{ difficulty }} {{ topic }} math problem with an exact numerical answer.",
        model_alias="teacher",
    ))
    config.add_column(dd.LLMTextColumnConfig(
        name="answer",
        prompt="Solve this problem and return ONLY the numerical answer:\n\n{{ problem }}",
        model_alias="teacher",
    ))

    return config
```

### Converting Data Designer Output to Fine-Tuning Format

Data Designer outputs Parquet or CSV. Convert to JSONL for Azure AI Foundry:

```python
import pandas as pd, json

df = pd.read_parquet("output/dataset.parquet")

# SFT format
with open("train.jsonl", "w") as f:
    for _, row in df.iterrows():
        f.write(json.dumps({"messages": [
            {"role": "system", "content": "You are a senior business advisor."},
            {"role": "user", "content": row["user_prompt"]},
            {"role": "assistant", "content": row["assistant_response"]},
        ]}) + "\n")

# DPO format
with open("train_dpo.jsonl", "w") as f:
    for _, row in df.iterrows():
        f.write(json.dumps({
            "input": {"messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": row["user_prompt"]},
            ]},
            "preferred_output": [{"role": "assistant", "content": row["preferred_response"]}],
            "non_preferred_output": [{"role": "assistant", "content": row["non_preferred_response"]}],
        }) + "\n")

# RFT format (last message must be user, answer in extra field)
with open("train_rft.jsonl", "w") as f:
    for _, row in df.iterrows():
        f.write(json.dumps({
            "messages": [{"role": "user", "content": row["problem"]}],
            "answer": float(row["answer"]),
        }) + "\n")
```

Or use `scripts/convert_dataset.py` for automated conversion.

### Key Data Designer Features

| Feature | Description |
|---------|-------------|
| **SamplerColumnConfig** | Categorical, numeric, or custom samplers for diversity axes (`values=`, not `categories=`) |
| **LLMTextColumnConfig** | LLM-generated text with Jinja2 template variables (`{{ column_name }}`) |
| **LLMJudgeColumnConfig** | Automated quality scoring with `dd.Score` rubrics |
| **LLMStructuredColumnConfig** | LLM output constrained to a Pydantic schema |
| **CustomColumnGenerator** | Python functions for custom logic (code execution, API calls) |
| **SeedDataset** | Start from existing data and augment/expand it |
| **PersonSampling** | Realistic synthetic person data (names, demographics) |

### When to Use Data Designer vs. Custom Scripts

| Factor | Data Designer | Custom Scripts |
|--------|--------------|----------------|
| **Dataset size** | 500+ examples | < 500 examples |
| **Diversity control** | Strong (built-in samplers) | Manual category lists |
| **Quality filtering** | Built-in judges + filters | Custom grading code |
| **Reproducibility** | Config-driven, versioned | Ad-hoc scripts |
| **Setup time** | Moderate (install + configure) | Minimal |
| **Flexibility** | High (but within framework) | Unlimited |

## Quality Signals to Check

Before training, verify:

- [ ] **No duplicates**: Exact or near-duplicate examples waste budget
- [ ] **Balanced distribution**: Topics, difficulty, output lengths well-distributed
- [ ] **Consistent formatting**: All examples follow the same structure
- [ ] **Correct outputs**: Spot-check 20 random examples manually
- [ ] **Reasonable lengths**: No extremely short or extremely long outputs
- [ ] **Clean text**: No encoding errors, garbled text, or template artifacts

## Dataset Size vs. Quality Tradeoff

From experiments:
- **335 high-quality examples** (carefully curated) → best combined eval score (9.15)
- **1,576 examples** (broader but noisier) → higher correctness but lower conciseness (8.53)

**Takeaway**: A small, pristine dataset usually beats a large, noisy one. Quality filter aggressively.

## Reference

- [NVIDIA Data Designer Skills](https://github.com/NVIDIA-NeMo/DataDesigner/tree/main/skills/data-designer)
- [Data Designer Documentation](https://github.com/NVIDIA-NeMo/DataDesigner)
