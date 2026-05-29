# Model Router Fine-Tuning — Data Schema

This document describes the training data contract for fine-tuning the Azure Foundry **Model Router**. Each line in the JSONL training file is one prompt paired with **per-model correctness labels** that teach the router which underlying model is the best (cheapest correct) choice for that kind of prompt.

> Use this schema with the [`Demos/Zava_ModelRouter_FT`](../../Demos/Zava_ModelRouter_FT/) demo, which fine-tunes `model-router` end-to-end against the bundled [`zava_enterprise`](./zava_enterprise/) dataset.

## File Format

- **Format**: JSONL (one JSON object per line)
- **Encoding**: UTF-8

## Required Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `messages` | array | Chat-format prompt the router should learn to route. |
| `labels`   | object | Per-model binary correctness label (`0` or `1`). Determines which underlying models the router will learn to pick. |
| `usage`    | object | Per-model token usage observed when each candidate model answered the prompt. Used by the fine-tuning pipeline for **cost-aware routing**. |

## Example Row

```json
{
  "messages": [
    {"role": "user", "content": "What's the current return rate for our cordless drill collection?"}
  ],
  "labels": {
    "gpt-5_2025-08-07":      1,
    "gpt-5-mini_2025-08-07": 1,
    "gpt-5-nano_2025-08-07": 0
  },
  "usage": {
    "gpt-5_2025-08-07":      {"prompt_tokens": 15, "completion_tokens": 1222},
    "gpt-5-mini_2025-08-07": {"prompt_tokens": 15, "completion_tokens": 702},
    "gpt-5-nano_2025-08-07": {"prompt_tokens": 15, "completion_tokens": 1349}
  }
}
```

In the example above, both `gpt-5` and `gpt-5-mini` answered correctly while `gpt-5-nano` did not — so for similar prompts the fine-tuned router should learn to route to `gpt-5-mini` (correct **and** cheapest).

## Field Rules

### `messages`

- Must be a **non-empty** list.
- Each item must be an object with string `role` and string `content`.
- If the last message has role `assistant`, it will be stripped during import.

### `labels`

- Must be a **non-empty** dictionary.
- Keys must be **versioned LLM identifiers** in the form `<llm-name>_<version>` — see the canonical list in [Supported LLMs](#supported-llms).
- Values must be binary outcomes: `0`, `1`, `"0"`, or `"1"`. A string value is only accepted if it can be converted to integer `0` or `1`.
- `1` = the model is a successful choice for that sample. `0` = unsuccessful.
- The **first valid line** determines the required model-key set for the entire file. Every subsequent row must use **exactly** the same set of keys — otherwise the row is rejected.
- All keys must come from the [Supported LLMs](#supported-llms) catalog.

### `usage`

- Must be a **non-empty** dictionary on every row.
- The key set must **match `labels` exactly** for that row.
- All keys must come from the [Supported LLMs](#supported-llms) catalog and must be consistent across all rows in the file.
- Each value must be a dictionary containing at least `prompt_tokens` as an integer.
- `completion_tokens` is **strongly recommended** even though only `prompt_tokens` is validated — see [Why `completion_tokens` matters](#why-completion_tokens-matters).
- Any row missing `usage`, or failing any of the above checks, is rejected.

### Why `completion_tokens` matters

- The fine-tune pipeline's cost evaluation uses **both** prompt and completion tokens.
- If `completion_tokens` is missing, the pipeline treats output cost as 0 — understating total cost and making the model look cheaper than it really is.
- Routing decisions during inference are driven by accuracy and **input cost** only — but cost-related evaluation scores will appear better than reality if completion tokens are omitted.

## Supported LLMs

Keys in `labels` and `usage` must use the **versioned LLM identifiers** supported by Model Router. The canonical, always-up-to-date list lives in the Microsoft Learn documentation:

📖 **[Model Router → Supported models](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router#supported-models)**

You can fine-tune the router for **any subset** of those LLMs — include only those LLM keys (and only those) in every row of your training data; the deployed router will route between exactly that subset.

> We intentionally do not duplicate the list here — Microsoft Learn is the source of truth and updates as new models are added.

## Deployment Restrictions

One restriction is **specific to fine-tuned Model Router deployments** and worth knowing before you collect training data.

### The router's model set is whatever you labelled for

A fine-tuned router routes between the **exact set of models present in the training data** — i.e., the set of keys in `labels` (and `usage`). You choose that set up-front by deciding which LLMs to label for; the choice cannot be narrowed or widened at deployment time.

This is **not** a restriction on *which* LLMs you can target — any subset of the [Supported LLMs](#supported-llms) is allowed. It just means the subset must be **decided at data-collection time, not deployment time**.

Examples:
- Label for `gpt-5`, `gpt-5-mini`, `gpt-5-nano` → the deployed router picks between those three.
- Label for `claude-opus-4-7`, `gpt-5.5`, `DeepSeek-V3.2` → the deployed router picks between those three.
- Label for every LLM in the [supported catalog](#supported-llms) → the deployed router picks between all of them.

> **Recommended workflow:** Deploy a baseline Model Router with your desired model subset first to validate routing behavior, then collect labelled training data for exactly those models. See [Route to a model subset](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/model-router?tabs=foundry-responses#optional-route-to-a-model-subset).

### Fine-tuning job payload must set `trainingType: 1`

Model Router uses a dedicated training type. The fine-tuning job submission payload must include `"trainingType": 1` — for example:

```json
{
  "model": "model-router",
  "training_file": "file-abc123",
  "validation_file": "file-def456",
  "trainingType": 1
}
```

Omitting the field (or sending `0` / `"Standard"`) returns `400 invalidPayload — does not support fine-tuning with Standard TrainingType`.

## Validation Checklist

Before uploading, confirm each row in your JSONL satisfies:

- [ ] Valid JSON
- [ ] Has `messages` (non-empty list of `{role, content}` objects)
- [ ] Has `labels` (non-empty dict, keys in the supported LLM list, values in `{0, 1, "0", "1"}`)
- [ ] Has `usage` whose keys match `labels` exactly and each value has integer `prompt_tokens`
- [ ] The set of model keys in `labels` is **identical** to the first row of the file

The notebook in [`Demos/Zava_ModelRouter_FT`](../../Demos/Zava_ModelRouter_FT/) runs these checks locally before uploading.
