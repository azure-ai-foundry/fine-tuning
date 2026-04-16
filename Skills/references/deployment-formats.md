# Deployment Formats Reference

## Model Format and SKU Mapping

When deploying a fine-tuned model on Azure AI Foundry, you must specify the correct `model.format` and `sku.name`. A mismatch produces an unhelpful HTTP 500 error with no useful message.

| Base model family | `model.format` | `sku.name` | Endpoint type |
|-------------------|---------------|------------|---------------|
| gpt-4.1-mini | `"OpenAI"` | `"Standard"` | Project |
| gpt-4.1-nano | `"OpenAI"` | `"Standard"` | Project |
| o4-mini (RFT) | `"OpenAI"` | `"Standard"` | Project |
| gpt-oss-20b-11 | `"Microsoft"` | `"GlobalStandard"` | Cognitive Services |
| Ministral-3B | `"Mistral AI"` | `"GlobalStandard"` | Cognitive Services |
| Llama-3.3-70B | `"Meta"` | `"GlobalStandard"` | Cognitive Services |
| Qwen-3-32B | `"Alibaba"` | `"GlobalStandard"` | Cognitive Services |

**Format strings are case-sensitive.** `"Mistral AI"` works; `"mistral"` does not.

## Two Endpoint Types

Azure AI Foundry has two endpoint types, and each fine-tuned model must be called through the correct one:

### Project Endpoint (OpenAI models)
- URL pattern: `https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1/`
- Ends with `/v1/`
- Use `openai.OpenAI(base_url=..., api_key=...)` — NOT `AzureOpenAI`
- `AzureOpenAI` appends `?api-version=...` which breaks the `/v1/` path

### Cognitive Services Endpoint (OSS models)
- URL pattern: `https://<resource>.cognitiveservices.azure.com/openai/deployments/<name>/chat/completions?api-version=2025-04-01-preview`
- Use `openai.AzureOpenAI(azure_endpoint=..., api_key=..., api_version=...)`

## Deployment via ARM REST API

The most reliable deployment method — works for all model types. Uses the provider-specific `model.format` values from the table above.

### Create Deployment

```
PUT https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{rg}/providers/Microsoft.CognitiveServices/accounts/{account}/deployments/{deploy_name}?api-version=2024-10-01

Headers:
  Authorization: Bearer {arm_token}
  Content-Type: application/json

Body:
{
  "sku": {
    "name": "GlobalStandard",   // or "Standard" for OpenAI models
    "capacity": 100             // tokens-per-minute in thousands
  },
  "properties": {
    "model": {
      "format": "Microsoft",     // see table above — must match base model family
      "name": "gpt-oss-20b-11.ft-{jobid}-suffix",  // the fine-tuned model ID
      "version": "1"
    }
  }
}
```

## Deployment via `az cognitiveservices` CLI

The `az cognitiveservices account deployment create` command uses a **different** format string than the ARM REST API. For all OSS models, use `"OpenAI-OSS"` as the `--model-format`:

```bash
az cognitiveservices account deployment create \
  --name <resource> \
  --resource-group <rg> \
  --deployment-name <name> \
  --model-name <model> \
  --model-version "1" \
  --model-format "OpenAI-OSS" \
  --sku-capacity 100 \
  --sku-name "GlobalStandard"
```

| Base model family | ARM REST `model.format` | `az cognitiveservices` `--model-format` |
|-------------------|------------------------|-----------------------------------------|
| gpt-4.1-mini / nano | `"OpenAI"` | `"OpenAI"` |
| gpt-oss-20b | `"Microsoft"` | `"OpenAI-OSS"` |
| Ministral-3B | `"Mistral AI"` | `"OpenAI-OSS"` |
| Llama-3.3-70B | `"Meta"` | `"OpenAI-OSS"` |
| Qwen-3-32B | `"Alibaba"` | `"OpenAI-OSS"` |

> **Warning**: These two APIs use different format strings for OSS models. Using `"OpenAI-OSS"` in an ARM REST call (or `"Microsoft"` in `az cognitiveservices`) will fail with HTTP 500.

### Check Deployment Status

```
GET https://management.azure.com/.../deployments/{deploy_name}?api-version=2024-10-01

Response:
{
  "properties": {
    "provisioningState": "Succeeded"  // or "Creating", "Failed", "Canceled"
  }
}
```

### Delete Deployment

```
DELETE https://management.azure.com/.../deployments/{deploy_name}?api-version=2024-10-01
```

## Capacity and Quota

- **Standard (OpenAI models)**: Capacity = tokens-per-minute in thousands. `100` = 100K TPM.
- **GlobalStandard (OSS models)**: Capacity = tokens-per-minute in thousands. Same scale.
- **Evaluation workloads**: Set capacity ≥ 100 (100K TPM). At capacity=1, OSS FT models hit "Failed to load LoRA" errors. Even at capacity=100, expect intermittent LoRA weight loading failures (HTTP 500) — use 10+ retries with exponential backoff.
- **Quota is per-resource**: Deploying many models simultaneously may exhaust quota.
- **Deletion latency**: After deleting a deployment, wait 15–20 seconds before creating a new one — quota isn't freed instantly.

## Deployment Name Rules

- Max 64 characters
- Alphanumeric and hyphens only
- Must be unique within the resource
- Tip: Use a pattern like `ft-{model}-{experiment}-eval` for clarity

## Getting an ARM Token

```bash
az account get-access-token --query accessToken -o tsv
```

**Warning**: Tokens expire after ~60 minutes. Always refresh immediately before making a request.

## Common Deployment Errors

| Error | Cause | Fix |
|-------|-------|-----|
| HTTP 500, no message | Wrong `model.format` | Check the format table above |
| HTTP 409, deployment exists | Name collision | Use a unique deployment name |
| HTTP 403 | ARM token expired | Refresh with `az account get-access-token` |
| HTTP 400, "api-version not allowed" | Using `AzureOpenAI` client on `/v1/` endpoint | Switch to `openai.OpenAI` client |
| HTTP 429, quota exceeded | Too many deployments | Delete unused deployments, wait 20s |
| ProvisioningState: Failed | Model not available in region | Try a different region or check model availability |
