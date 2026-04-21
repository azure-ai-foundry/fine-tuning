# Evaluation Methodology

## Principles

1. **Always establish a baseline**: Evaluate the base (un-tuned) model first. Without a baseline, you can't measure improvement.
2. **Use a held-out test set**: Never evaluate on training or validation data. The model has seen those.
3. **Use the same test set for every model**: This is the only way to compare results fairly.
4. **Use task-specific graders**: Built-in generic evaluators (Coherence, Fluency) measure general quality and won't detect fine-tuning improvements. Use custom graders (Python, score model, string check) for task-specific evaluation.
5. **Measure cost alongside accuracy**: Report completion tokens per response when comparing models or checkpoints. A model that achieves the same accuracy with fewer tokens is strictly better — cheaper inference and lower latency. This is especially important for RFT, where models often learn to generate longer responses during training.

## Azure AI Evaluation SDK

Use the **Azure AI Evaluation SDK** (`azure-ai-evaluation`) for all evaluation. It provides custom graders, batch evaluation, and built-in guardrail metrics.

### Two-Layer Evaluation Strategy

| Layer | Purpose | Grader Type | When |
|-------|---------|-------------|------|
| **Task-specific** (primary) | Measure FT improvement | `AzureOpenAIScoreModelGrader`, `AzureOpenAIPythonGrader`, `AzureOpenAIStringCheckGrader` | Every eval |
| **General quality** (guardrail) | Verify model didn't degrade | `CoherenceEvaluator`, `FluencyEvaluator` | Spot-check only |

The generic built-in evaluators (Coherence, Fluency, TaskAdherence) often show **no difference** between base and fine-tuned models, even when domain-specific evaluation reveals clear improvement. They are guardrails, not metrics.

### Installation

```bash
pip install azure-ai-evaluation
```

### Model Configuration

The SDK needs a model config for AI-assisted evaluators. Use `OpenAIModelConfiguration` with the project `/v1/` endpoint:

```python
from azure.ai.evaluation import OpenAIModelConfiguration

model_config = OpenAIModelConfiguration(
    api_key="<your-api-key>",
    base_url="https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1/",
    model="gpt-4.1-mini",   # deployment name for the judge model
    type="openai",           # REQUIRED — tells SDK to use OpenAI client, not AzureOpenAI
)
```

> **Critical**: You must include `type="openai"` when using the project `/v1/` endpoint. Without it, the SDK throws `'' is not a supported connection type`.

### Built-in Evaluators (Guardrails Only)

These measure general quality — use as degradation checks, **not** as primary FT metrics:

| Category | Evaluators | Scale | Use Case |
|----------|-----------|-------|----------|
| **Quality** | `CoherenceEvaluator`, `FluencyEvaluator`, `SimilarityEvaluator` | 1–5 | Verify model didn't get worse |
| **NLP** | `F1ScoreEvaluator`, `RougeScoreEvaluator`, `BleuScoreEvaluator` | 0–1 | Exact/token overlap with reference |
| **Safety** | `ViolenceEvaluator`, `HateUnfairnessEvaluator`, etc. | 0–7 | Safety regression check |

### Custom Graders (Primary FT Evaluation)

These are the core tools for measuring fine-tuning improvement:

#### 1. Score Model Grader (LLM judge with task-specific rubric)

Best for: any task where quality is subjective (summarization, alignment, style).

```python
from azure.ai.evaluation import AzureOpenAIScoreModelGrader, evaluate

# Define a task-specific rubric — NOT generic "coherence"
summarization_grader = AzureOpenAIScoreModelGrader(
    model_config=model_config,
    name="summarization_quality",
    prompt="""Rate this news summary on a scale of 1-5.

Article: {{item.article}}
Summary: {{sample.output_text}}

Criteria:
- Captures ALL key facts (who, what, when, where)
- No hallucinated information not in the article
- Concise (under 3 sentences)
- Reads naturally

Score 1: Missing key facts or hallucinations
Score 3: Captures main point but misses details
Score 5: Perfect summary — all facts, no extras, concise

Return ONLY a number 1-5.""",
    output_type="numeric",
    pass_threshold=3,
)

result = evaluate(
    data="eval_data.jsonl",
    evaluators={"summary_quality": summarization_grader},
)
```

#### 2. Python Grader (programmatic/exact-match evaluation)

Best for: code generation, math, entity extraction, structured output, any verifiable task.

```python
from azure.ai.evaluation import AzureOpenAIPythonGrader, evaluate

# Entity extraction: check JSON validity + key coverage
entity_grader = AzureOpenAIPythonGrader(
    name="entity_extraction_accuracy",
    source="""
import json

def grade(item, sample):
    try:
        extracted = json.loads(sample["output_text"])
        reference = json.loads(item["ground_truth"])
    except (json.JSONDecodeError, KeyError):
        return {"score": 0, "reason": "Invalid JSON output"}

    # Check required keys exist
    required_keys = ["people", "organizations", "locations", "dates"]
    missing = [k for k in required_keys if k not in extracted]
    if missing:
        return {"score": 0.5, "reason": f"Missing keys: {missing}"}

    # Score by entity overlap
    total, matched = 0, 0
    for key in required_keys:
        ref_set = set(str(v).lower() for v in reference.get(key, []))
        ext_set = set(str(v).lower() for v in extracted.get(key, []))
        total += len(ref_set)
        matched += len(ref_set & ext_set)

    score = matched / total if total > 0 else 1.0
    return {"score": score, "reason": f"{matched}/{total} entities matched"}
""",
    pass_threshold=0.7,
)
```

#### 3. String Check Grader (pattern matching)

Best for: classification, format compliance, tool calling format.

```python
from azure.ai.evaluation import AzureOpenAIStringCheckGrader, evaluate

# Check that model output contains valid tool call JSON
tool_format_grader = AzureOpenAIStringCheckGrader(
    name="tool_call_format",
    input="{{sample.output_text}}",
    operation="like",          # or "eq", "starts_with", "contains"
    reference="function_call",
    pass_threshold=1,
)

# Check classification output matches expected label
classification_grader = AzureOpenAIStringCheckGrader(
    name="classification_accuracy",
    input="{{sample.output_text}}",
    operation="eq",
    reference="{{item.expected_label}}",
    pass_threshold=1,
)
```

#### 4. Text Similarity Grader (semantic match)

Best for: paraphrasing, translation, any task where output should be semantically similar to reference.

```python
from azure.ai.evaluation import AzureOpenAITextSimilarityGrader, evaluate

similarity_grader = AzureOpenAITextSimilarityGrader(
    model_config=model_config,
    name="semantic_similarity",
    input="{{sample.output_text}}",
    reference="{{item.ground_truth}}",
    pass_threshold=4,  # 1-5 scale
)
```

### Recommended Grader Sets by Training Type

**SFT (code generation, distillation):**
```python
evaluators = {
    "code_correctness": python_grader_that_runs_code,  # primary
    "f1": F1ScoreEvaluator(),                           # token overlap
    "coherence": CoherenceEvaluator(model_config=config),  # guardrail
}
```

**SFT (summarization):**
```python
evaluators = {
    "summary_quality": score_model_grader_with_rubric,  # primary
    "rouge": RougeScoreEvaluator(),                      # token overlap
    "fluency": FluencyEvaluator(model_config=config),    # guardrail
}
```

**DPO (alignment, safety):**
```python
evaluators = {
    "domain_quality": score_model_grader_with_domain_rubric,  # primary
    "coherence": CoherenceEvaluator(model_config=config),      # guardrail
}
```

**Tool Calling:**
```python
evaluators = {
    "tool_format": string_check_grader,     # format compliance
    "tool_accuracy": python_grader,          # correct tool + args
}
```

**Entity Extraction / Structured JSON:**
```python
evaluators = {
    "json_validity": python_grader_json_check,     # valid JSON
    "entity_accuracy": python_grader_entity_match, # entity overlap
}
```

### Batch Evaluation with evaluate()

The `evaluate()` function runs multiple graders over an entire dataset:

```python
from azure.ai.evaluation import evaluate, F1ScoreEvaluator

result = evaluate(
    data="eval_data.jsonl",
    evaluators={
        "task_grader": my_custom_score_grader,   # primary
        "f1": F1ScoreEvaluator(),                 # token overlap
    },
    output_path="./eval_results.json",
)

# Aggregate metrics
for metric, value in result["metrics"].items():
    print(f"{metric}: {value}")
```

Use `evaluator_config` with `column_mapping` if your data has different column names:

```python
result = evaluate(
    data="eval_data.jsonl",
    evaluators={"quality": my_score_grader},
    evaluator_config={
        "quality": {
            "column_mapping": {
                "query": "${data.prompt}",
                "response": "${data.model_output}",
                "ground_truth": "${data.reference}",
            }
        }
    },
)
```

## Evaluation Pipeline for Fine-Tuning

The `scripts/evaluate_model.py` script is a standalone custom evaluator that uses the OpenAI API directly (it does **not** wrap the Azure AI Evaluation SDK). It implements a 2-dimension LLM judge that:

```
1. Load held-out test set (JSONL with messages array, including per-example system prompts)
2. Generate responses from the deployed model for each prompt
3. Grade each response on correctness and conciseness using an LLM judge
4. Compute aggregate scores (weighted 70% correctness, 30% conciseness)
5. Save per-example and summary results to JSON
```

## Test Set Design

**Size**: 30–100 examples is sufficient. More helps reduce noise but costs more.

**Diversity**: Cover the full range of inputs:
- Easy, medium, and hard examples
- Edge cases and common cases
- Different sub-categories of your task

**Quality**: Reference answers must be gold-standard correct. A wrong reference will penalize correct model outputs.

## Interpreting Results

### AI-Assisted Quality Scores (1–5 scale)
| Range | Interpretation |
|-------|---------------|
| 1–2 | Poor — model is not useful |
| 3 | Adequate — functional with issues |
| 4 | Good — suitable for most use cases |
| 5 | Excellent — near-expert quality |

### NLP Scores (0–1 scale)
| Range | Interpretation |
|-------|---------------|
| < 0.3 | Very low overlap — likely wrong |
| 0.3–0.6 | Partial match |
| 0.6–0.8 | Good match |
| > 0.8 | Strong match |

### What the Numbers Tell You

- **High similarity, low coherence**: Output is correct but poorly structured. May need more diverse training data.
- **Low similarity, high fluency**: Model writes well but not what was asked. Check data quality.
- **Both low**: Fundamental data quality or model capacity issue.
- **Scores very close to baseline**: Fine-tuning didn't help. Check if your data teaches something the base model doesn't know.

### Statistical Significance

With 50+ eval examples, a difference of ~0.3 points (on 1–5 scale) is usually meaningful. Smaller differences may be noise.

## Evaluating RFT Models

RFT models need extra scrutiny because the training grader can be gamed:

1. **Evaluate with a DIFFERENT rubric than the training grader**. If the eval rubric matches the training grader, you're measuring overfitting to the grader, not actual quality.
2. **Use `F1ScoreEvaluator`** for exact-match accuracy on verifiable answers.
3. **Use `SimilarityEvaluator`** to catch cases where the answer is semantically correct but formatted differently.
4. **Compare against the base model** (not just other fine-tunes). RFT can sometimes make things worse.

## Custom Evaluators (Function-Based)

For quick inline evaluators, you can also use plain Python functions:

```python
from azure.ai.evaluation import evaluate

def code_correctness(*, response, ground_truth, **kwargs):
    """Check if generated code produces the same output as reference."""
    return {"code_correctness": 1.0 if response.strip() == ground_truth.strip() else 0.0}

result = evaluate(
    data="eval_data.jsonl",
    evaluators={"code_check": code_correctness},
)
```

This is useful for prototyping but prefer `AzureOpenAIPythonGrader` for production — it provides better error handling and pass/fail thresholds.

## Experiment Tracking

Track all experiments in a single JSON leaderboard:

```json
{
  "experiment-1-sft-code-gen": {
    "base_model": "gpt-4.1-nano",
    "training_type": "SFT",
    "dataset": "train.jsonl",
    "epochs": 2,
    "similarity": 4.8,
    "coherence": 4.5,
    "f1": 0.72,
    "eval_date": "2026-04-10"
  }
}
```

## Reference

- [Azure AI Evaluation SDK docs](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-evaluation-readme)
- [Evaluation metrics reference](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-metrics-built-in)
- [Evaluation samples](https://github.com/Azure-Samples/azureai-samples/tree/main/scenarios/evaluate)
