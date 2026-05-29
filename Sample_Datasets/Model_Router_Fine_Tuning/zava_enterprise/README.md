# Zava Enterprise — Model Router Fine-Tuning Dataset

A small labelled dataset of **enterprise operations prompts** used to fine-tune the Azure Foundry [Model Router](https://learn.microsoft.com/azure/ai-services/openai/how-to/model-router) for a fictional retail company (Zava). The prompts span the kind of internal-analyst questions you'd expect from a retail ops team: return rates, CSAT trends, inventory levels, vendor performance, store comparisons, etc.

> **The 3-model subset in this dataset is illustrative, not a limitation.**
> Each row here is labelled for **GPT-5 / GPT-5 mini / GPT-5 nano** as a representative subset, so the demo stays small and easy to read. Model Router fine-tuning supports **any subset of the [supported LLMs](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router#supported-models)** (spanning the GPT, Claude, Llama, DeepSeek, Grok, and gpt-oss families) — just label your own training data for whichever LLMs you want the router to choose between.

## Files

| File | Rows | Purpose |
|------|------|---------|
| `zava_enterprise_train.jsonl` | 200 | Training split |
| `zava_enterprise_test.jsonl`  | 100 | Held-out test split for validation / evaluation |

## Schema

Each line follows the [Model Router data format](../SCHEMA.md):

```json
{
  "messages": [{"role": "user", "content": "<enterprise prompt>"}],
  "labels":   {"<model_id>": 0 | 1, ...},
  "usage":    {"<model_id>": {"prompt_tokens": N, "completion_tokens": N}, ...}
}
```

## Models Labelled

Each row has correctness labels for **three** models from the GPT-5 family (chosen as a representative subset — see the callout above):

| Model | Versioned ID |
|-------|--------------|
| GPT-5         | `gpt-5_2025-08-07` |
| GPT-5 mini    | `gpt-5-mini_2025-08-07` |
| GPT-5 nano    | `gpt-5-nano_2025-08-07` |

Labels are **binary** — `1` if the model produced an acceptable answer for that prompt, `0` otherwise (graded with an LLM judge during data generation).

> **Subset → router behavior:** A fine-tuned Model Router routes between the exact set of LLMs present in `labels` (see [`SCHEMA.md`](../SCHEMA.md#deployment-restrictions)). Training on *this* dataset produces a router that routes only between the three GPT-5 family models. To target a different / wider set, build your own JSONL labelling for those LLMs.

## Sample Row

```json
{
  "messages": [
    {"role": "user", "content": "Show me the CSAT score trend for the last 30 days."}
  ],
  "labels": {
    "gpt-5_2025-08-07":      1,
    "gpt-5-mini_2025-08-07": 1,
    "gpt-5-nano_2025-08-07": 1
  },
  "usage": {
    "gpt-5_2025-08-07":      {"prompt_tokens": 16, "completion_tokens": 2241},
    "gpt-5-mini_2025-08-07": {"prompt_tokens": 16, "completion_tokens": 1499},
    "gpt-5-nano_2025-08-07": {"prompt_tokens": 16, "completion_tokens": 1485}
  }
}
```

All three models answered this prompt correctly, so the fine-tuned router should learn to send similar prompts to **gpt-5-nano** (correct **and** cheapest).

## Intended Use

This dataset is bundled with the [`Demos/Zava_ModelRouter_FT`](../../../Demos/Zava_ModelRouter_FT/) demo as an **end-to-end illustration** of the Model Router fine-tuning workflow. It is intentionally small — meant for learning and rapid iteration, not for producing a production-quality router. Combine with your own enterprise prompts and your chosen candidate-model set before deploying anywhere real.

## Provenance

The prompts were drawn from a Zava (fictional retail) enterprise-operations scenario; labels and usage statistics were produced by sweeping each prompt across the three candidate models and grading responses with an LLM judge. See the [Model Router fine-tuning documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning) for a full picture of the labelling workflow.
