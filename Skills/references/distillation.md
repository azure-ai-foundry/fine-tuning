# Distillation (Teacher-Student Fine-Tuning)

Distillation is the practice of training a smaller, cheaper "student" model to replicate the behavior of a larger, more capable "teacher" model. The result is a model that performs close to the teacher at a fraction of the inference cost.

## When to Use Distillation

- You have a task where a large model (gpt-4.1, o3, gpt-4o) performs well
- You need lower latency or lower cost in production
- A smaller model (gpt-4.1-mini, gpt-4.1-nano) underperforms on the same task
- You can afford the one-time training cost to bridge the gap

## The Workflow

```
1. Assemble gold-standard examples (human-curated, 10–50)
2. Build a grader (LLM judge to score outputs)
3. Benchmark all candidate base models
4. Pick the Teacher (highest scoring) and Student (cheapest that's worth training)
5. Generate training data from the Teacher's outputs
6. Fine-tune the Student on the Teacher's data (SFT)
7. Evaluate the Student vs. its untrained peer on NEW data
8. Ship the Student if it matches or approaches the Teacher
```

## Step-by-Step

### 1. Build and Validate Your Grader

Before anything else, build a grading function and test it on your gold-standard examples. The grader will be your evaluation backbone throughout.

```python
# Example: LLM-judge grader for sarcasm + correctness
GRADER_PROMPT = """Rate this response on two dimensions:
1. Sarcasm level (1-10)
2. Answer correctness (1-10)

If the answer is factually wrong, both scores should be 0.

Response: {output}
Reference: {reference}

Return JSON: {"sarcasm": <int>, "correctness": <int>}"""
```

### 2. Benchmark Base Models

**First, verify your deployments exist.** Model names must match actual deployment names in your Azure resource. Send a trivial test request to each candidate — a 404 means the deployment doesn't exist.

Run every candidate model through your evaluation set and grade them:

```python
models = ["o3", "o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o"]
for model in models:
    scores = evaluate(model, test_set, grader)
    print(f"{model}: {scores}")
```

**Pick your Teacher**: The model with the highest scores.
**Pick your Student**: The smallest/cheapest model that's worth improving. Usually the worst performer that's in your budget for production inference.

### 3. Generate Training Data from the Teacher

Use `scripts/generate_distillation_data.py` for a complete, runnable pipeline:

```bash
python scripts/generate_distillation_data.py \
    --teacher gpt-4.1-mini \
    --system-prompt "You are a senior business writer." \
    --topics "earnings,risk,compliance,strategy" \
    --num-prompts 300 \
    --min-score 7.0 \
    --output-dir ./distillation_data
```

Or manually — run the Teacher on a diverse set of prompts and collect its outputs:

```python
training_data = []
for prompt in prompts:
    response = teacher_model.generate(prompt)
    training_data.append({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    })
```

**Quality filter**: Grade the Teacher's outputs and only keep high-scoring ones. Even the best model produces duds sometimes.

### 4. Fine-Tune the Student

Standard SFT using the Teacher's outputs as training data:

```python
job = client.fine_tuning.jobs.create(
    model="gpt-4.1-nano",  # the Student
    training_file=teacher_data_file_id,
    validation_file=val_file_id,
    method={"type": "supervised"}
)
```

### 5. Evaluate on New Data

Critical: evaluate on data the Teacher never saw during training data generation. Compare:
- **Student** (fine-tuned) vs. **Peer** (same base model, untrained)
- **Student** vs. **Teacher** (how close did we get?)

## Key Insights

- **You don't need the Student to match the Teacher exactly.** If gpt-4.1-nano gets 80% of gpt-4.1's quality at 10% of the cost, that's often a win.
- **More Teacher data isn't always better.** Quality-filter the Teacher's outputs. A smaller set of excellent examples beats a larger set of mediocre ones.
- **The grader is everything.** If your grader can't reliably distinguish good from bad, your entire distillation pipeline is unreliable.
- **Test on fresh prompts.** If you evaluate on the same prompts the Teacher used to generate training data, you're measuring memorization, not generalization.

## Cost Comparison Example

| | Teacher (gpt-4.1) | Student (gpt-4.1-nano) |
|-|-------------------|----------------------|
| Training cost | — | ~$2–5 (one time) |
| Inference cost/1M tokens | Higher | Much lower |
| Latency | Higher | Much lower |
| Quality (task-specific) | 9.5 | 8.5–9.0 (after distillation) |

## Reference

- [Distillation demo (microsoft-foundry/fine-tuning)](https://github.com/microsoft-foundry/fine-tuning/tree/main/Demos/DistillingSarcasm)
- [Build 2025 distillation demo](https://github.com/azure-ai-foundry/build-2025-demos/blob/main/Azure%20AI%20Model%20Customization/DistillationDemo/demo.ipynb)
