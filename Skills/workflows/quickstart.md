# Quickstart: Fine-Tune Your First Model

Go from zero to a fine-tuned model in 6 steps. This guide covers the simplest path — SFT (supervised fine-tuning) using synthetic data generated from a prompt.

> **Time estimate**: ~20 minutes of active work + 1-3 hours of training time.

## Prerequisites

- An Azure AI Foundry project with a model deployed (e.g., `gpt-4.1-mini`)
- Python 3.10+ with the `openai` package installed
- Your project endpoint URL and API key (find these in the Foundry portal under Project Settings)

```bash
pip install openai
```

## Step 1: Connect to your project

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

## Step 2: Generate training data

The fastest way to get started is to generate synthetic data from a prompt. Define your task and have the model create examples:

```python
import json

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["AZURE_OPENAI_API_KEY"])

SYSTEM_PROMPT = "You are a concise technical support agent. Answer in 1-2 sentences."

# Describe the kind of examples you want
generation_prompt = """Generate 50 diverse technical support conversations. 
Each should have a customer question and an ideal agent response.
The responses should be concise (1-2 sentences), accurate, and professional.
Cover topics like: password resets, billing issues, product setup, 
account changes, shipping status, and troubleshooting.

Return a JSON array where each element has "question" and "answer" fields."""

resp = client.chat.completions.create(
    model="gpt-4.1-mini",  # use your deployed model
    messages=[{"role": "user", "content": generation_prompt}],
    max_tokens=8000,
    temperature=1.0,
)

# Parse and convert to training format
import re
content = resp.choices[0].message.content
match = re.search(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
json_str = match.group(1) if match else content.strip().strip("`").replace("json\n", "")
examples = json.loads(json_str)

with open("train.jsonl", "w") as f:
    for ex in examples[:40]:  # 40 for training
        f.write(json.dumps({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]}) + "\n")

with open("val.jsonl", "w") as f:
    for ex in examples[40:]:  # 10 for validation
        f.write(json.dumps({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]}) + "\n")

print(f"Created train.jsonl ({40} examples) and val.jsonl ({len(examples)-40} examples)")
```

> **Tip**: For better results, generate more examples (200-500) across multiple prompts. See `workflows/dataset-creation.md` for advanced approaches.

Validate your data:
```bash
python scripts/validate/validate_sft.py train.jsonl
```

## Step 3: Baseline the base model

Before fine-tuning, see how the base model handles your task. This is your benchmark.

```python
# Test a few examples from your validation set
with open("val.jsonl") as f:
    test_examples = [json.loads(line) for line in f][:5]

print("Base model responses:\n")
for ex in test_examples:
    question = ex["messages"][1]["content"]
    expected = ex["messages"][2]["content"]
    
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=ex["messages"][:2],  # system + user only
        max_tokens=200,
    )
    actual = resp.choices[0].message.content
    
    print(f"Q: {question}")
    print(f"Expected: {expected}")
    print(f"Base model: {actual}")
    print()
```

Look for patterns: Is the base model too verbose? Wrong format? Missing domain knowledge? These are what fine-tuning will fix.

## Step 4: Upload data and submit the job

```python
import time

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

## Step 5: Monitor and wait

```bash
python scripts/monitor_training.py --job-id <your-job-id>
```

Or check in the [Azure AI Foundry portal](https://ai.azure.com) under Fine-tuning > Jobs.

Training typically takes 1-3 hours depending on dataset size and model.

## Step 6: Deploy, test, and compare

Once the job succeeds, deploy the fine-tuned model:

```bash
python scripts/deploy_model.py --model-id <fine-tuned-model-name> --name my-ft-deployment --capacity 50
```

Then compare base vs fine-tuned on the same questions:

```python
for ex in test_examples:
    question = ex["messages"][1]["content"]
    
    # Base model
    base_resp = client.chat.completions.create(
        model="gpt-4.1-mini", messages=ex["messages"][:2], max_tokens=200)
    
    # Fine-tuned model
    ft_resp = client.chat.completions.create(
        model="my-ft-deployment", messages=ex["messages"][:2], max_tokens=200)
    
    print(f"Q: {question}")
    print(f"Base:      {base_resp.choices[0].message.content}")
    print(f"Fine-tuned: {ft_resp.choices[0].message.content}")
    print()
```

## What's next?

- **Evaluate rigorously**: Use `scripts/evaluate_model.py` to score both models with an LLM judge on a held-out test set
- **Scale your data**: Generate 200-500 examples for better results — see `workflows/dataset-creation.md`
- **Try RFT**: For tasks with verifiable answers, reinforcement fine-tuning can push accuracy further — see `references/training-types.md`
- **Iterate**: If results aren't good enough, see `workflows/diagnose-poor-results.md`
- **Full guide**: For the complete workflow, see `workflows/full-pipeline.md`
