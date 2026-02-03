# Distilling Sarcasm - Foundry SDK Version

This is an alternative implementation of the Sarcasm distillation demo using the **Azure AI Foundry SDK** instead of raw API keys.

## Overview

This demo teaches language models to generate sarcastic responses through distillation:
1. A **teacher model** (gpt-4.1) generates sarcastic training data
2. A **student model** (gpt-4.1-mini) is fine-tuned on this data
3. Evaluators measure sarcasm quality before and after training

## SDK Comparison

| Aspect | Original (API Key) | This Version (Foundry SDK) |
|--------|-------------------|---------------------------|
| **Package** | `openai` | `azure-ai-projects` + `openai` |
| **Auth** | API Key | `DefaultAzureCredential` |
| **Inference** | `client.chat.completions.create()` | `openai_client.chat.completions.create()` |
| **Evaluations** | `client.evals.*` | `openai_client.evals.*` (same!) |
| **Fine-tuning** | `client.fine_tuning.jobs.*` | `openai_client.fine_tuning.jobs.*` (same!) |

**Key Insight**: The APIs are nearly identical! The main difference is how you get the client:

```python
# Original (API Key)
client = OpenAI(base_url=..., api_key=...)

# Foundry SDK
project_client = AIProjectClient(endpoint=..., credential=DefaultAzureCredential())
openai_client = project_client.get_openai_client()  # Same API from here!
```

## Prerequisites

1. **Azure Subscription** with access to Azure AI Foundry
2. **Python 3.10+**
3. **Azure CLI** logged in (`az login`)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or: .venv\Scripts\activate  # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment template and fill in your values:
   ```bash
   cp .env.template .env
   ```

4. Configure `.env`:
   ```
   MICROSOFT_FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
   AZURE_INFERENCE_ENDPOINT=https://<deployment>.<region>.models.ai.azure.com
   AZURE_OPENAI_DEPLOYMENT=gpt-4.1
   AZURE_SUBSCRIPTION_ID=<your-subscription>
   AZURE_RESOURCE_GROUP=<your-rg>
   AZURE_AOAI_ACCOUNT=<your-account>
   BASE_MODEL=gpt-4.1-mini
   TEACHER_MODEL=gpt-4.1
   ```

## Running the Demo

Open `sarcasm_foundry.ipynb` in Jupyter and execute cells in order:

```bash
jupyter notebook sarcasm_foundry.ipynb
```

## Key Differences from Original

### Authentication
```python
# Original (API Key)
client = OpenAI(
    base_url=f"https://{resource}.openai.azure.com/openai/v1/",
    api_key=os.environ.get("FOUNDRY_API_KEY"),
)

# Foundry SDK (Azure Credential)
credential = DefaultAzureCredential()
project_client = AIProjectClient(endpoint=endpoint, credential=credential)
openai_client = project_client.get_openai_client()
```

### Inference
```python
# Original (OpenAI SDK)
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}]
)

# Foundry SDK (azure-ai-inference)
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage

client = ChatCompletionsClient(endpoint=..., credential=credential)
response = client.complete(messages=[UserMessage(content="Hello")])
```

### Evaluations (Same API!)
```python
# Both demos use identical evals API
eval = openai_client.evals.create(name="sarcasm-grader", ...)
run = openai_client.evals.runs.create(eval_id=eval.id, ...)
```

### Trade-offs

**Foundry SDK Advantages:**
- ✅ No API keys to manage (uses Azure managed identity)
- ✅ Better security with DefaultAzureCredential
- ✅ Consistent with other Azure AI services
- ✅ Unified project management

**Original (API Key) Advantages:**
- ✅ Simpler setup (just one API key)
- ✅ Works outside Azure environments
- ✅ Familiar OpenAI SDK patterns

## Troubleshooting

### Authentication Issues
- Ensure you're logged in: `az login`
- Check your role assignments include "Azure AI User" on the Foundry resource

### Model Not Found
- Verify the model deployment names in your `.env` match your Azure deployments

### Rate Limiting
- Reduce `sample_size` in evaluation cells if hitting rate limits

## Files

- `sarcasm_foundry.ipynb` - Main notebook
- `requirements.txt` - Python dependencies
- `.env.template` - Environment variable template
- `README.md` - This file

## Related

- [Original Sarcasm Demo](../DistillingSarcasm/) - Uses OpenAI SDK with API keys
- [CNN DailyMail Demo](../SFT_CNN_DailyMail/) - SFT example using Foundry SDK
