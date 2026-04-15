# Agentic RFT — Tool Calling & Endpoint Graders

Train reasoning models (o4-mini, GPT-5) for agentic scenarios where the model invokes external tools during chain-of-thought reasoning.

> **Note**: Tool calling and endpoint graders are currently in **private preview** for GPT-5. o4-mini RFT is GA, but tool calling within RFT is preview-only. Features are subject to change.

## Tool Calling in RFT

During training, the model can learn when and how to call external tools mid-reasoning. The model issues function calls to your hosted endpoints, receives results, and continues reasoning.

### Defining Tools

```python
tools = [
    {
        "name": "search",
        "server_url": "https://your-function-app.azurewebsites.net/api/tools",
        "headers": {
            "Authorization": "Bearer <your-key>"
        }
    },
    {
        "name": "get_by_id",
        "server_url": "https://your-function-app.azurewebsites.net/api/tools",
        "headers": {
            "Authorization": "Bearer <your-key>"
        }
    }
]
```

### Submitting an Agentic RFT Job

```python
job = client.fine_tuning.jobs.create(
    model="o4-mini-2025-04-16",  # or "gpt-5-2025-08-07"
    training_file=train.id,
    validation_file=valid.id,
    method={
        "type": "reinforcement",
        "reinforcement": {
            "grader": grader,
            "tools": tools,
            "max_episode_steps": 10,
            "hyperparameters": {
                "eval_interval": 3,
                "eval_samples": 5,
                "compute_multiplier": 1.0,
                "reasoning_effort": "medium"
            }
        }
    }
)
```

### Tool Response Format

Your tool endpoint must return:

```json
{
    "type": "function_call_output",
    "call_id": "call_12345xyz",
    "output": "The result of the tool call...",
    "id": "fc_12345xyz"
}
```

### Tool Call Metrics

RFT jobs with tools generate additional training metrics (visible in the Foundry UI Metrics tab):
- **Tool calls per rollout**: How often each tool is called
- **Mean tokens per tool call**: Token usage per call
- **Tool execution latency**: Response time of your endpoints
- **Tool errors count**: Failed tool calls

### Tool Endpoint Requirements

| Constraint | Limit |
|-----------|-------|
| Recommended throughput | 50 QPS |
| Max input payload | 1 MB |
| Max return payload | 1 MB (413 error if exceeded) |
| Timeout | 10 minutes |
| Parallel calls | Supported — handle race conditions |
| Retry on 5xx | 3 attempts, then rollout discarded |
| On 4xx | Error serialized and shown to model |

## Endpoint Graders

Instead of Python or score_model graders, you can host your own grading logic as a REST endpoint. This gives maximum flexibility for complex or domain-specific evaluation.

### Defining an Endpoint Grader

```python
grader = {
    "type": "endpoint",
    "name": "my_grader",
    "url": "https://your-grading-endpoint.com/grade",
    "headers": {
        "Authorization": "Bearer <your-key>"
    },
    "rate_limit": 50,         # max requests/second to your endpoint
    "pass_threshold": 8,      # minimum passing score
}
```

### Endpoint Request Format

Your endpoint receives POST requests:

```json
{
    "sample": { "output_text": "..." },
    "item": { "reference": "...", "other_fields": "..." },
    "trace_id": "trace_1a2b3c"
}
```

- `sample`: Model's generation (chat completions format)
- `item`: The full training example (your JSONL fields)
- `trace_id`: UUID consistent across all tool calls in a single rollout

### Endpoint Response Format

Return a JSON score:

```json
{"score": 0.85}
```

### Endpoint Grader Limits

| Constraint | Limit |
|-----------|-------|
| Recommended throughput | 50 RPS |
| Max input payload | 1 MB |
| Timeout | 10 minutes |
| Retry on failure | 3 attempts, then rollout discarded |

## RFT Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `reasoning_effort` | `"low"`, `"medium"`, `"high"` — controls depth of reasoning | `"medium"` |
| `compute_multiplier` | Scales training compute (higher = more expensive, faster convergence) | `1.0` |
| `eval_interval` | Evaluate every N training steps | Varies |
| `eval_samples` | Number of validation examples per eval | Varies |
| `max_episode_steps` | Max tool calls + reasoning steps per rollout | `10` |

### Cost Control with RFT Hyperparameters

- Start with `reasoning_effort: "low"` and small `eval_samples` to estimate costs
- Increase `compute_multiplier` only if convergence is too slow
- Lower `eval_interval` means more frequent validation (more grader cost but better monitoring)

## When to Use Agentic RFT

- Your model needs to **decide when to call tools** (not just follow instructions)
- The task involves **multi-step reasoning** with external data lookups
- You need the model to learn **tool selection** — choosing the right tool for the job
- Standard RFT (without tools) can't capture the agentic behavior you need

## Reference

- [Agentic RFT demos](https://github.com/microsoft-foundry/fine-tuning/tree/main/Demos/Agentic_RFT_PrivatePreview)
- [Zava Retail Agent demo (SFT + RFT with tools)](https://github.com/microsoft-foundry/fine-tuning/tree/main/Demos/ZavaRetailAgent)
- [RFT Countdown (basic RFT)](https://github.com/microsoft-foundry/fine-tuning/tree/main/Demos/RFT_Countdown)
