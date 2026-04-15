# Dataset Formats

## SFT Format (Supervised Fine-Tuning)

Standard chat-completion JSONL. Each line is a JSON object with a `messages` array.

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Write a haiku about rain."}, {"role": "assistant", "content": "Drops fall on the roof\nRhythmic drumming soothes the soul\nEarth drinks deeply now"}]}
```

**Rules:**
- Each line must be valid JSON (no trailing commas, no comments)
- `messages` array must contain at least one `user` and one `assistant` message
- `system` message is optional but recommended for consistency
- Multi-turn conversations are supported: alternate `user` and `assistant` messages
- The last message should be `assistant` (that's what the model learns to produce)

**Validation checklist:**
- [ ] File extension is `.jsonl`
- [ ] Each line parses as valid JSON
- [ ] Every example has `messages` key
- [ ] Every message has `role` and `content`
- [ ] No empty `content` fields
- [ ] Consistent system prompts across examples (or intentionally varied)

## DPO Format (Direct Preference Optimization)

DPO uses three top-level fields: `input`, `preferred_output`, and `non_preferred_output`.

```jsonl
{"input": {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain gravity."}]}, "preferred_output": [{"role": "assistant", "content": "Gravity is a fundamental force that attracts objects with mass toward each other."}], "non_preferred_output": [{"role": "assistant", "content": "Gravity is when stuff falls down."}]}
```

**Rules:**
- `input`: Object containing a `messages` array (system + user turns). May also include `tools` and `parallel_tool_calls` for tool-use training.
- `preferred_output`: Array of messages representing the preferred response (roles: `assistant` or `tool` only)
- `non_preferred_output`: Array of messages representing the non-preferred response (same role constraints)
- Both outputs must contain at least one `assistant` message
- Exactly two completions are compared — you cannot provide more
- The difference between preferred and non-preferred should be meaningful and consistent

**DPO-specific hyperparameters (set in the training job, not the data):**
- `beta` — Controls how strongly the model is pushed toward the preferred output (default: 0.1). Lower = more conservative alignment.
- `l2_multiplier` — L2 regularization to prevent the model from drifting too far from the base (default: 0.1).

**REST API example:**
```json
{
  "model": "gpt-4.1-mini-2025-04-14",
  "training_file": "file-abc123",
  "validation_file": "file-def456",
  "method": {
    "type": "dpo",
    "dpo": {
      "beta": 0.1,
      "l2_multiplier": 0.1
    }
  }
}
```

## RFT Format (Reinforcement Fine-Tuning)

RFT data uses the chat-completion format but with important differences from SFT:

1. **The last message MUST have role `user`** — the model generates its own response during training
2. **Extra fields** can be added alongside `messages` for the grader to access via `item.*`
3. Both training and validation datasets are **required**

```jsonl
{"messages": [{"role": "user", "content": "Write a Python function to reverse a string."}], "reference_code": "def reverse_string(s):\n    return s[::-1]", "expected_output": "olleh"}
{"messages": [{"role": "developer", "content": "Solve math problems step by step."}, {"role": "user", "content": "What is 15% of 240?"}], "answer": 36.0}
```

**⚠️ Common mistake**: Do NOT put an `assistant` message as the last message. Unlike SFT, RFT does not learn from example outputs — it generates its own and learns from the grader's reward signal.

The grader is defined in the training job config, not in the data file.

**⚠️ API version**: Python graders require `api-version=2025-04-01-preview` or later. Earlier versions reject the `python` grader type.

```python
# Use api_version="2025-04-01-preview" for RFT with Python graders
method={
    "type": "reinforcement",
    "reinforcement": {
        "grader": {
            "type": "python",
            "name": "my_grader",
            "source": "def grade(sample, item):\n    ..."
        }
    }
}
```

**Grader types:**
- `string_check`: Simple pass/fail text comparison (eq, ne, like, ilike). Best for classification/label tasks.
- `text_similarity`: Score based on fuzzy_match, BLEU, ROUGE, METEOR etc. Good for summarization.
- `python`: Custom Python function. Most reliable for code/math tasks. Has access to numpy, scipy, sympy, pandas, scikit-learn, etc.
- `score_model`: LLM judge (gpt-4o-2024-08-06 or o3-mini-2025-01-31). More flexible but nondeterministic.
- `multi`: Combine multiple graders with weighted arithmetic expression.

**Python grader template:**
```python
def grade(sample, item):
    """
    Args:
        sample: dict with 'output_text' (model's generation)
        item: dict with extra fields from your JSONL (e.g., 'reference_code', 'answer')
    Returns:
        float between 0.0 and 1.0
    """
    output = sample.get("output_text", "")
    reference = item.get("reference_code", "")
    # Your grading logic here
    return score
```

**Python grader constraints:**
- Code size: 256 KB max
- No network access
- Memory: 2 GB, Disk: 1 GB, CPU: 1 core
- Runtime: 2 minutes max
- Must always return a numeric value (handle all errors)

**Grader access to data fields:**
- `sample.output_text` → what the model generated
- `sample.output_json` → structured output (if using response_format)
- `item.*` → extra fields from your JSONL (e.g., `item.answer`, `item.reference_code`)
- Template variables use `{{item.field_name}}` syntax — **no spaces inside braces**, no array indexing

## Converting Between Formats

### SFT → RFT
The data is already compatible. Just ensure your assistant messages can serve as reference answers for your grader.

### SFT → DPO
You need to generate rejected responses. Common approaches:
1. Run the base model on the same prompts → use base outputs as `rejected`, your data as `chosen`
2. Intentionally degrade your good outputs (add errors, wrong style) → use as `rejected`
3. Have humans rank multiple model outputs

### DPO → SFT
Extract the `chosen` responses: `{"messages": [{"role": "user", ...}, chosen[0]]}`

See `scripts/convert_dataset.py` for automated conversion utilities.

## Dataset Size Guidelines

| Training type | Minimum | Sweet spot | Diminishing returns |
|--------------|---------|------------|-------------------|
| SFT | 50 | 300–500 | > 2,000 (unless very diverse) |
| DPO | 200 | 500–1,000 | > 5,000 |
| RFT | 100 | 200–500 | > 1,000 |

**Key finding**: More data doesn't always help. In experiments with code generation:
- 335 high-quality examples → combined score 9.15 (best overall)
- 1,576 examples (including lower quality) → higher correctness but lower conciseness, combined 8.53
- The larger dataset taught the model to be more thorough but also more verbose

## Data Quality Signals

Before training, check your data for:

1. **Diversity**: Are inputs varied enough? Clustering similar inputs wastes training budget.
2. **Consistency**: Do similar inputs get similar-quality outputs? Conflicting examples confuse the model.
3. **Correctness**: Are the outputs actually right? Even 5% error rate in training data can measurably hurt.
4. **Length distribution**: Wildly different output lengths can cause the model to be inconsistent.
5. **System prompt alignment**: If using a system prompt, does every example follow it?

Use `scripts/score_dataset.py` to automatically assess quality with an LLM judge.

## Splitting Data

- **Training set**: 80–90% of your data
- **Validation set**: 10–20% (minimum 20 examples)
- **Held-out test set**: Separate from both — this is what you evaluate on

**Critical**: Never evaluate on validation data. The model sees validation loss during training. Your final evaluation MUST use a held-out set that was never part of training or validation.
