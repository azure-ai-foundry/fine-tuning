# Zava Model Router Fine-Tuning Demo

**Technique**: Fine-tuning of **Azure Foundry Model Router**  
**Base Model**: `model-router`  
**Use Case**: Tailor the router to your domain so it consistently routes each prompt to the **cheapest model that can still answer it correctly** — cutting inference spend without sacrificing answer quality. This demo specialises the router on enterprise-operations prompts across GPT-5, GPT-5 mini, and GPT-5 nano.  
**Dataset**: [`zava_enterprise`](../../Sample_Datasets/Model_Router_Fine_Tuning/zava_enterprise/) — 200 training / 100 test enterprise-operations prompts labelled for three GPT-5 family models *as a representative subset* (see callout below)

> **The 3-model subset is just an example.** This demo labels for `gpt-5`, `gpt-5-mini`, and `gpt-5-nano` to keep the notebook compact. Model Router fine-tuning works with **any subset of the [supported LLMs](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router#supported-models)** (spanning GPT, Claude, Llama, DeepSeek, Grok, gpt-oss). Bring your own JSONL labelled for whichever LLMs you want the router to choose between.

## What You'll Learn

1. **Understand the Model Router data contract** — `messages` + per-model binary `labels` + per-model `usage`, see [`SCHEMA.md`](../../Sample_Datasets/Model_Router_Fine_Tuning/SCHEMA.md).
2. **Validate + upload** a JSONL training set to Azure Foundry.
3. **Submit and monitor** a fine-tuning job against the `model-router` base model.
4. **Deploy** the resulting custom router via the Azure Management REST API.
5. **Test** the deployed router with a sample prompt and see which underlying model it picked.

## ⚠️ Model Router Fine-Tuning — Key Constraint

> **The router's model set is fixed at data-collection time.** The deployed fine-tuned router routes between the **exact** set of LLMs that appear in `labels` (and `usage`) in your training data — and that subset is fixed when you build the JSONL, not at deploy time. To change which LLMs the router picks between, you need to relabel and re-train. The subset can be **any subset** of the [supported LLMs](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router#supported-models); this demo uses three GPT-5 family models as a representative example.

## Dataset Information

**Source**: [`zava_enterprise`](../../Sample_Datasets/Model_Router_Fine_Tuning/zava_enterprise/) — a synthetic, hand-curated dataset built for this demo around a fictional retail company (Zava). Prompts span the kind of internal-analyst questions a retail-ops team would ask: return rates, CSAT trends, inventory levels, vendor performance, store comparisons. Full provenance lives in the dataset's [`README.md`](../../Sample_Datasets/Model_Router_Fine_Tuning/zava_enterprise/README.md).

**Size**: 300 prompts total
- Training split: 200 examples (`zava_enterprise_train.jsonl`)
- Test split: 100 examples (`zava_enterprise_test.jsonl`)

**Models labelled** (representative subset — see the callout near the top of this README):

| Model | Versioned ID used in the JSONL |
|-------|--------------------------------|
| GPT-5         | `gpt-5_2025-08-07` |
| GPT-5 mini    | `gpt-5-mini_2025-08-07` |
| GPT-5 nano    | `gpt-5-nano_2025-08-07` |

You can fine-tune the router for **any subset** of the [supported LLMs](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router#supported-models) — just relabel your data for the LLMs you want the deployed router to choose between.

> **Note**: This dataset is intentionally small — meant for learning and rapid iteration, not for producing a production-quality router. For a real deployment, bring your own enterprise prompts and your chosen candidate-model set.

## Dataset Format

Each line is a Model Router training row: a chat-style prompt + per-model correctness `labels` + per-model token `usage`. See [`SCHEMA.md`](../../Sample_Datasets/Model_Router_Fine_Tuning/SCHEMA.md) for the full field-level contract and validation rules.

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

In this row, `gpt-5` and `gpt-5-mini` answered the prompt correctly but `gpt-5-nano` did not — so the fine-tuned router should learn to send similar prompts to **gpt-5-mini** (the cheapest *correct* choice for this kind of question).

## Prerequisites

- Azure AI Foundry project with fine-tuning enabled and `model-router` available
- Azure **AI Owner** role on the resource (for the deployment step)
- Python 3.9+
- A `.env` file at this demo's root (see [`.env.template`](./.env.template))

```bash
cd Demos/Zava_ModelRouter_FT
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.template .env                # then edit .env with your values
```

## Files

| File | Description |
|------|-------------|
| `Zava_ModelRouter_FineTuning.ipynb` | Main notebook — run cells top to bottom |
| `requirements.txt` | Python dependencies (`requests`, `python-dotenv`, `pandas`, `tqdm`) |
| `.env.template` | Copy to `.env` and fill in your Azure credentials |

Training and test data live in [`../../Sample_Datasets/Model_Router_Fine_Tuning/zava_enterprise/`](../../Sample_Datasets/Model_Router_Fine_Tuning/zava_enterprise/).

## How to Run

1. Set up your environment (see [Prerequisites](#prerequisites)).
2. Open `Zava_ModelRouter_FineTuning.ipynb` in VS Code or Jupyter and select your `.venv` kernel.
3. Run cells sequentially. The notebook will:
   - Preview + validate the Zava JSONL files locally (no API calls)
   - Upload `zava_enterprise_train.jsonl` and `zava_enterprise_test.jsonl` to Azure Foundry
   - Submit a fine-tuning job against `model-router`
   - Poll until the job finishes (minutes to hours depending on queue + dataset size)
   - Download `results.csv` and preview training metrics
   - Deploy the fine-tuned router
   - Send one prompt to the fine-tuned deployment and print which underlying model the router picked
4. Track job progress in [Azure AI Foundry](https://ai.azure.com/) → **Fine-tuning** while the notebook polls.

## Troubleshooting

### Common Issues

**File Not Ready Error**
- Uploaded files must finish server-side processing before the job is submitted. The notebook's upload helper waits for `processed` state automatically — re-run the submit cell if you see `File status is not 'ready'`.
- You can check file status in the [Azure AI Foundry portal](https://ai.azure.com/) under **Fine-tuning → Files**.

**Validator Errors**
- `Missing required field 'labels'` or `'usage'` — every row needs both fields, and their key sets must match across all rows. See [`SCHEMA.md`](../../Sample_Datasets/Model_Router_Fine_Tuning/SCHEMA.md).
- `labels[...] must be 0/1` — labels are binary; use `1` if the LLM answered the prompt correctly, `0` otherwise.

**Job Submission `400 invalidPayload — does not support fine-tuning with Standard TrainingType`**
- The Model Router base needs `"trainingType": 1` in the job payload. The notebook's `create_finetuning_job` helper includes it — if you've customised the call, make sure you didn't drop it.

**Deployment Authentication**
- The deploy step uses the Azure Management REST API, which requires an **Azure AD bearer token** (control plane) — not the data-plane API key used for upload/submit.
- Token expired → refresh with `az account get-access-token --resource https://management.azure.com/ --query accessToken -o tsv` and update `AZURE_AD_TOKEN` in `.env`.
- `403 Forbidden` → confirm you have the **Azure AI Owner** role on the resource (`Cognitive Services OpenAI Contributor` is not sufficient for FT deployment).
