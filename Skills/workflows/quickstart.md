# Quickstart: Fine-Tune Your First Model

Go from zero to a fine-tuned model in 5 steps. This guide covers the simplest path — SFT (supervised fine-tuning) with an existing dataset.

> **Time estimate**: ~15 minutes of active work + 1-3 hours of training time.

## Prerequisites

- An Azure AI Foundry project with a model deployed (e.g., `gpt-4.1-mini`)
- Python 3.10+ with the `openai` package installed
- Your project endpoint URL and API key (find these in the Foundry portal under Project Settings)

```bash
pip install openai
```

## Step 1: Prepare your credentials

Set environment variables or create a `.env` file:

```bash
export OPENAI_BASE_URL="https://<your-resource>.services.ai.azure.com/api/projects/<your-project>/openai/v1/"
export AZURE_OPENAI_API_KEY="<your-key>"
```

Verify connectivity:
```python
from openai import OpenAI
import os

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["AZURE_OPENAI_API_KEY"])
resp = client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role": "user", "content": "Hello"}], max_tokens=10)
print(resp.choices[0].message.content)  # Should print a greeting
```

## Step 2: Prepare training data

Create a JSONL file where each line is a conversation. At minimum you need 10 examples, but 200-500 is recommended.

**`train.jsonl`** (one JSON object per line):
```json
{"messages": [{"role": "system", "content": "You are a helpful customer service agent."}, {"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Go to Settings > Security > Reset Password. You'll receive an email with a reset link within 5 minutes."}]}
{"messages": [{"role": "system", "content": "You are a helpful customer service agent."}, {"role": "user", "content": "What's your return policy?"}, {"role": "assistant", "content": "You can return any item within 30 days of purchase for a full refund, provided it's in original condition."}]}
```

Create a smaller **`val.jsonl`** with 10-20% of your examples for validation (different examples than training).

> **Tip**: See `examples/sample-data/sft_sample.jsonl` for a complete format reference.

Validate your data before uploading:
```bash
python scripts/validate/validate_sft.py train.jsonl
python scripts/validate/data_stats.py train.jsonl
```

## Step 3: Upload data and submit the job

```python
from openai import OpenAI
import os, time

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["AZURE_OPENAI_API_KEY"])

# Upload files
with open("train.jsonl", "rb") as f:
    train = client.files.create(file=f, purpose="fine-tune")
with open("val.jsonl", "rb") as f:
    val = client.files.create(file=f, purpose="fine-tune")

# Wait for processing
for _ in range(30):
    t = client.files.retrieve(train.id)
    v = client.files.retrieve(val.id)
    if t.status == "processed" and v.status == "processed":
        break
    time.sleep(10)

# Submit job
job = client.fine_tuning.jobs.create(
    model="gpt-4.1-mini",           # base model to fine-tune
    training_file=train.id,
    validation_file=val.id,
    suffix="my-first-ft",           # name suffix for the fine-tuned model
    method={"type": "supervised", "supervised": {
        "hyperparameters": {"n_epochs": 2, "learning_rate_multiplier": 1.0}
    }},
)
print(f"Job submitted: {job.id}")
```

Or use the script:
```bash
python scripts/submit_training.py --model gpt-4.1-mini --training-file train.jsonl --validation-file val.jsonl --type sft --suffix my-first-ft --epochs 2
```

## Step 4: Monitor and wait

```bash
python scripts/monitor_training.py --job-id <your-job-id>
```

Or check in the [Azure AI Foundry portal](https://ai.azure.com) under Fine-tuning > Jobs.

Training typically takes 1-3 hours depending on dataset size and model.

## Step 5: Deploy and test

Once the job succeeds, deploy the fine-tuned model:

```bash
python scripts/deploy_model.py --model-id <fine-tuned-model-name> --name my-ft-deployment --capacity 50
```

Then test it:
```python
resp = client.chat.completions.create(
    model="my-ft-deployment",  # your deployment name
    messages=[
        {"role": "system", "content": "You are a helpful customer service agent."},
        {"role": "user", "content": "How do I track my order?"},
    ],
)
print(resp.choices[0].message.content)
```

## What's next?

- **Evaluate properly**: Use `scripts/evaluate_model.py` to compare your fine-tuned model against the base model on a held-out test set
- **Try RFT**: For tasks with verifiable answers, reinforcement fine-tuning can push accuracy further — see `references/training-types.md`
- **Iterate**: If results aren't good enough, see `workflows/diagnose-poor-results.md` and `workflows/iterative-training.md`
- **Full guide**: For the complete workflow including data generation, quality scoring, and training curve analysis, see `workflows/full-pipeline.md`
