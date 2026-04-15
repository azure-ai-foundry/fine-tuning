# Foundry Fine-Tuning CLI (`azd ai finetuning`)

The Azure Developer CLI has a fine-tuning extension that handles job submission, monitoring, and deployment from the terminal — no Python SDK or REST calls needed.

## Setup

```bash
# Install azd (Windows)
winget install microsoft.azd

# Install the fine-tuning extension
azd ext install azure.ai.finetune

# Authenticate
azd auth login
```

Requires `azd` version **1.22.1+**. Verify with `azd version`.

## Initialize a Project

```bash
# Interactive mode (guided prompts)
azd ai finetuning init

# With project endpoint
azd ai finetuning init -e https://<account>.services.ai.azure.com/api/projects/<project>

# From a template
azd ai finetuning init -t https://github.com/microsoft-foundry/foundry-samples/blob/main/samples/cli/finetuning/supervised

# Clone config from an existing job
azd ai finetuning init -j ftjob-abc123
```

This creates a `fine-tune-job.yaml` config file in your working directory.

## Job YAML Config

The CLI uses a YAML file instead of Python code. Example:

```yaml
model: gpt-4.1-mini
training_file: training.jsonl
validation_file: validation.jsonl
method:
  type: supervised
hyperparameters:
  n_epochs: 2
  learning_rate_multiplier: 1.0
```

## Submit and Manage Jobs

```bash
# Submit
azd ai finetuning jobs submit -f ./fine-tune-job.yaml

# Quick submit (skip init — provide subscription and endpoint inline)
azd ai finetuning jobs submit -f ./job.yaml -s <subscription-id> -e <endpoint>

# List all jobs
azd ai finetuning jobs list

# Show details
azd ai finetuning jobs show -i <job-id>

# Pause / Resume / Cancel
azd ai finetuning jobs pause -i <job-id>
azd ai finetuning jobs resume -i <job-id>
azd ai finetuning jobs cancel -i <job-id>
```

## Deploy

```bash
azd ai finetuning jobs deploy \
    -i <job-id> \
    -d "my-deployment-name" \
    -c 100 \
    -m "OpenAI" \
    -s "Standard" \
    -v "1"
```

| Flag | Description |
|------|-------------|
| `-i` | Job ID |
| `-d` | Deployment name (max 64 chars) |
| `-c` | Capacity (TPM in thousands) |
| `-m` | Model provider/format (see `references/deployment-formats.md`) |
| `-s` | SKU name (`Standard` or `GlobalStandard`) |
| `-v` | Version (usually `"1"`) |

**Reminder**: The `-m` (model provider) flag maps to the same format values documented in `deployment-formats.md`:

| Base model | `-m` value | `-s` value |
|-----------|-----------|-----------|
| gpt-4.1-mini / nano | `"OpenAI"` | `"Standard"` |
| gpt-oss-20b-11 | `"Microsoft"` | `"GlobalStandard"` |
| Ministral-3B | `"Mistral AI"` | `"GlobalStandard"` |
| Llama-3.3-70B | `"Meta"` | `"GlobalStandard"` |
| Qwen-3-32B | `"Alibaba"` | `"GlobalStandard"` |

## When to Use CLI vs SDK vs REST

| Approach | Best for |
|----------|----------|
| **CLI (`azd ai finetuning`)** | Quick experiments, interactive use, single jobs, getting started fast |
| **Python SDK** | Automated pipelines, batch job submission, programmatic analysis |
| **REST API** | Models that fail with SDK (e.g., gpt-oss-20b-11), maximum control |

The CLI is the fastest way to get a single job running. Use the Python scripts when you need to automate sweeps, analyze curves programmatically, or integrate into a larger pipeline.

## Resources

- [Official docs](https://learn.microsoft.com/en-us/azure/foundry/fine-tuning/fine-tune-cli)
- [CLI samples](https://github.com/microsoft-foundry/foundry-samples/blob/main/samples/cli/finetuning/README.md)
- [azd documentation](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/)
