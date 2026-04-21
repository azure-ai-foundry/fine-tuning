# RFT Grader Design Guide

How to build effective graders for reinforcement fine-tuning. The grader is the most important component of an RFT pipeline — it determines what the model learns to optimize.

## Grader Type Selection

| Grader Type | Best For | Tradeoffs |
|------------|---------|-----------|
| **Python grader** (recommended default) | Most tasks including tool-calling. Accesses `output_text` and `output_tools`. | Can't call external APIs or execute code. |
| **Multi grader** | Combining multiple scoring dimensions (e.g., string_check + score_model). | `score_model` component adds LLM inference cost per rollout. |
| **Endpoint grader** | Tasks requiring external API calls during grading (e.g., running test suites, querying a database). | HTTP latency, scaling requirements, and reliability risk. Under-provisioned endpoints can cause jobs to hang in post-training eval. |
| **String check** | Exact-match tasks (classification labels, yes/no, numeric answers). | Binary 0/1 only — no partial credit. |

### Python Grader as Default

Start with a Python grader unless you have a specific reason to use something else. Python graders are:
- **Fast** — no HTTP round-trips, no LLM inference cost per rollout
- **Deterministic** — same input always produces same score
- **Reliable** — no endpoint availability concerns, no post-training hang risk
- **Tool-aware** — `sample.output_tools` provides full tool call metadata

A common misconception is that endpoint graders are needed for tool-calling tasks. In practice, `sample.output_tools` in the Python grader gives you the tool call names and arguments, and `sample.output_text` gives you the model's final response after tool execution. This is sufficient for most agentic grading scenarios.

### When to Use Endpoint Graders

Use endpoint graders only when grading requires:
- Executing code (running unit tests, evaluating generated SQL against a database)
- Calling external APIs that aren't available in the Python sandbox
- Complex stateful logic that depends on external systems

If you do use an endpoint grader:
- **Scale appropriately** — use Always On, sufficient compute (S2+), and multiple instances
- **Handle errors gracefully** — always return `{"score": float}`, never crash
- **Test under load** before submitting — the training platform sends parallel requests
- **Align scoring logic exactly** with any other graders you use for evaluation

## Partial Credit Design

Binary pass/fail graders (score 0 or 1) give sparse reward — most rollouts get 0 and the model has no gradient to learn from. **Partial credit is critical** for incremental learning.

### Multi-Dimensional Scoring Pattern

Decompose your task into 2-4 independently scorable dimensions, each with a weight that reflects its importance:

```python
def grade(sample, item):
    output_text = sample.get("output_text", "") or ""
    expected = item.get("expected_answer", "")
    
    score = 0.0
    
    # Dimension 1: Core correctness (highest weight)
    if correct_action(output_text, expected):
        score += 0.4
    
    # Dimension 2: Precision (e.g., exact amounts, specific values)
    score += 0.3 * precision_score(output_text, expected)
    
    # Dimension 3: Reasoning quality (e.g., cited correct rules/facts)
    score += 0.2 * reasoning_score(output_text, expected)
    
    # Dimension 4: Process quality (e.g., used the right tools)
    if used_correct_tools(sample.get("output_tools", [])):
        score += 0.1
    
    return round(min(score, 1.0), 3)
```

**Why this works**: A response that gets the action right but the amount wrong scores 0.4 (not 0), giving the model a gradient to improve from. A response that also gets the amount right jumps to 0.7 — clear reward signal for improvement.

### Weight Guidelines

| Dimension | Typical Weight | Examples |
|-----------|---------------|----------|
| **Core correctness** | 0.3–0.5 | Right action, right answer, right classification |
| **Precision** | 0.2–0.3 | Exact amounts, specific values, correct format |
| **Reasoning/explanation** | 0.1–0.2 | Cited correct rules, justified the decision |
| **Process quality** | 0.05–0.1 | Used the right tools, followed the right steps |

## Grading Tool Calls in Python Graders

The `sample.output_tools` field provides tool call metadata without needing an endpoint grader:

```python
def grade(sample, item):
    output_tools = sample.get("output_tools", []) or []
    
    # Check which tools were called
    tool_names = [t.get("function", {}).get("name", "") for t in output_tools]
    
    # Check tool arguments
    for tool in output_tools:
        args = tool.get("function", {}).get("arguments", "")
        # args is a JSON string — parse and validate
```

### Alternative: Multi Grader with Template Variables

For simpler tool call checks, the multi grader supports template variables that reference tool calls directly:

```json
{
  "type": "multi",
  "graders": {
    "called_right_tool": {
      "type": "string_check",
      "input": "{{item.expected_tool}}",
      "reference": "{{sample.output_tools[0].function.name}}",
      "operation": "eq"
    },
    "correct_arguments": {
      "type": "text_similarity",
      "input": "{{item.expected_args}}",
      "reference": "{{sample.output_tools[0].function.arguments}}",
      "evaluation_metric": "fuzzy_match"
    }
  },
  "calculate_output": "0.6 * called_right_tool + 0.4 * correct_arguments"
}
```

Use `text_similarity` with `fuzzy_match` for arguments rather than `string_check` — argument JSON may have different key ordering or whitespace.

## Threshold Calibration

The `pass_threshold` determines what grader score counts as pass (positive reward) vs fail (negative reward). This is the single most important hyperparameter for RFT learning signal.

### Calibration Workflow

1. Run the **base model** on your training/validation set
2. Score every output with your grader
3. Compute pass rates at multiple thresholds:

```python
for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    pass_rate = sum(1 for s in scores if s >= threshold) / len(scores)
    fail_rate = 1 - pass_rate
    print(f"  @{threshold}: pass={pass_rate:.0%}, fail={fail_rate:.0%}")
```

4. Choose the threshold where **25-50% of base model rollouts fail**

Use `scripts/calibrate_grader.py` to automate this workflow:
```bash
python calibrate_grader.py --model o4-mini --data train.jsonl --grader grader.py --n 30
```

| Failure Rate | Signal Quality |
|-------------|----------------|
| < 10% | ❌ Too easy — model already passes, no learning signal |
| 10-25% | ⚠️ Weak signal — may converge slowly |
| **25-50%** | **✅ Good signal — enough failures to learn from** |
| 50-70% | ⚠️ Harsh — model gets mostly negative reward |
| > 70% | ❌ Too hard — sparse positive reward, training may diverge |

### Recalibrate When Data Changes

**Always re-run calibration when you change your dataset.** A threshold that worked for a smaller dataset may be too strict or too lenient after adding more examples — the base model's score distribution shifts with different data composition.

## Keeping Graders Consistent

If you use multiple graders (e.g., Python grader for training, endpoint grader for debugging, local eval script for checkpoint selection):

- **Use identical scoring logic** — same weights, same keywords, same dimension breakdown
- **Use identical default scores** — when no action is found, when no amounts are expected, etc.
- **Test with the same examples** — run 10 samples through all graders and verify scores match

Mismatched scoring between graders causes the model to learn different behavior than what your evaluation measures. For example, if one grader gives a 0.1 bonus for tool usage and another doesn't, the model may learn to call tools more/less depending on which grader trained it.
