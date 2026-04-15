# Training Types: SFT vs DPO vs RFT

## Quick Decision Matrix

| Factor | SFT | DPO | RFT |
|--------|-----|-----|-----|
| **Best for** | Teaching a new skill or format | Aligning preferences/style | Improving reasoning chains |
| **Data needed** | Input–output pairs | Chosen/rejected pairs | Prompts + grading function |
| **Data volume** | 50–5,000 examples | 500–5,000 pairs | 200–2,000 prompts |
| **Effort to prepare data** | Low | High (need contrasting pairs) | Medium (need grader, not outputs) |
| **Risk of regression** | Low | Medium | High (sensitive to grader quality) |
| **Typical improvement** | 5–30% on task metrics | Subtle style/safety shifts | 0–15% on reasoning tasks |
| **Supported models** | Most models | Select models | o4-mini, o3-mini |

## Supervised Fine-Tuning (SFT)

**What it does**: Trains the model to mimic your examples. Given input X, produce output Y.

**When to use**:
- You have high-quality input–output pairs
- The task is well-defined (code generation, classification, extraction, summarization)
- You want reliable, repeatable outputs in a specific format or style

**Data format**: Chat-completion JSONL with `messages` array (system, user, assistant).

**Key insight from experiments**: SFT on ~300–500 high-quality examples often outperforms SFT on 1,500+ lower-quality examples. Quality > quantity, always.

## Direct Preference Optimization (DPO)

**What it does**: Teaches the model to prefer one style of output over another using contrasting pairs. Adjusts model weights based on human preferences without requiring a reward model.

**When to use**:
- You want to adjust tone, verbosity, safety, or style
- You have examples of "good" and "bad" outputs for the same input
- SFT already works but the outputs need refinement
- You have preference data from user logs, A/B tests, or manual annotation

**Data format**: JSONL with `input` (system + user messages), `preferred_output`, and `non_preferred_output`. See `references/dataset-formats.md` for the exact format.

**DPO-specific hyperparameters**:
- `beta` (default 0.1): Controls alignment strength. Lower = more conservative.
- `l2_multiplier` (default 0.1): Regularization to prevent drifting from base model.

**Key advantage over RLHF**: DPO is computationally lighter — no reward model fitting needed. Uses simple binary preference data.

## Reinforcement Fine-Tuning (RFT)

**What it does**: The model generates its own outputs and learns from a grading signal — no reference outputs needed.

**When to use**:
- The task has objectively verifiable answers (code execution, math, logic)
- You can write a programmatic or LLM-based grader
- You want to improve the model's reasoning, not just its outputs

**Data format**: JSONL with `messages` (user prompt + optional reference for grading). Grader is defined in the `method` config, not the data.

**Critical lessons from experiments**:
- RFT is **extremely sensitive to grader quality**. A grader that's even slightly miscalibrated will produce bad models.
- The train–validation gap on the grader should be < 0.05. A gap > 0.2 means the model is gaming the grader.
- Python graders (syntax checking, test execution) are more reliable than LLM-judge graders.
- Multi-graders (syntax check + LLM score) can reduce gaming but add complexity.
- RFT may **not** improve over a strong base model (e.g., o4-mini) — always compare against the baseline.

## Choosing a Path

```
Start here:
│
├─ Do you have labeled input–output pairs?
│  ├─ Yes → SFT
│  └─ No
│     ├─ Can you write a grading function? → RFT
│     └─ Can you rank "good" vs "bad" outputs? → DPO
│
After SFT:
│
├─ Results good enough? → Ship it
├─ Need style refinement? → DPO on top of SFT model
└─ Reasoning needs improvement? → RFT (if model supports it)
```

## Model Compatibility (Azure AI Foundry)

| Model | SFT | DPO | RFT | Vision FT |
|-------|-----|-----|-----|-----------|
| gpt-4.1 | ✅ | ✅ | ❌ | ✅ |
| gpt-4.1-mini | ✅ | ✅ | ❌ | ❌ |
| gpt-4.1-nano | ✅ | ❌ | ❌ | ❌ |
| gpt-4o (2024-08-06) | ✅ | ✅ | ❌ | ✅ |
| o4-mini | ❌ | ❌ | ✅ | ❌ |
| o3-mini | ❌ | ❌ | ✅ | ❌ |
| gpt-oss-20b-11 | ✅ | ❌ | ❌ | ❌ |
| Ministral-3B | ✅ | ❌ | ❌ | ❌ |
| Llama-3.3-70B | ✅ | ❌ | ❌ | ❌ |
| Qwen-3-32B | ✅ | ❌ | ❌ | ❌ |

DPO can be applied on top of an already SFT-fine-tuned model (same model/version required). See `references/dataset-formats.md` for the correct DPO data format.

Vision fine-tuning follows the same SFT workflow but with image data in messages. See `references/vision-fine-tuning.md`.

*Check Azure AI Foundry docs for the latest model availability — this changes frequently.*

## Advanced Techniques

### Distillation

Use a strong model to generate training data for a weaker, cheaper model. This is standard SFT where the training data comes from a "teacher" model rather than humans.

**When to use**: You need production inference to be cheap/fast, but only a large model performs well enough.

See `references/distillation.md` for the full workflow.

### Agentic RFT (Tool Calling)

Train reasoning models to invoke external tools during chain-of-thought. The model learns *when* to call tools and *which* tool to use.

**When to use**: Building AI agents that need to search databases, call APIs, or use custom tools during reasoning.

**Requirements**: Hosted tool endpoints (50 QPS recommended), endpoint grader or Python grader.

See `references/agentic-rft.md` for configuration details.
