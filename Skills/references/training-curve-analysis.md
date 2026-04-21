# Training Curve Analysis

## What to Look At

After a training job completes, download its result CSV. It contains step-level metrics:

| Column | What it means |
|--------|---------------|
| `step` | Training step number |
| `train_loss` | Loss on the training batch (should decrease) |
| `train_mean_token_accuracy` | Token-level accuracy on training data |
| `valid_loss` | Loss on validation set (key metric) |
| `valid_mean_token_accuracy` | Token-level accuracy on validation data |
| `full_valid_loss` | Full-pass validation loss (more accurate, less frequent) |
| `full_valid_mean_token_accuracy` | Full-pass token accuracy |

**Primary metric**: `valid_loss` (or `full_valid_loss`). Lower is better.

## Reading the Curves

### Healthy Training
```
Loss
│
│\
│ \___________     ← Both train and valid loss decrease, then plateau
│   \_________ 
│
└──────────────── Steps
  train_loss  valid_loss
```

Both curves decrease together and plateau. The gap between them stays small.

### Overfitting
```
Loss
│
│\
│ \_____/‾‾‾‾    ← Valid loss starts INCREASING while train loss decreases
│   \_________ 
│
└──────────────── Steps
  train_loss  valid_loss
```

Validation loss decreases then turns upward. The model is memorizing training data instead of generalizing.

### Underfitting
```
Loss
│────────────     ← Both losses stay high and barely move
│
│
│
│
└──────────────── Steps
```

Neither curve moves much. Learning rate is too low, or the model capacity is insufficient for the task.

## Overfitting Detection

Calculate the **overfitting ratio** at each validation checkpoint:

```
overfitting_ratio = valid_loss / train_loss
```

| Ratio | Interpretation |
|-------|---------------|
| < 1.2 | Healthy — model generalizes well |
| 1.2–1.5 | Mild overfitting — acceptable for small datasets |
| 1.5–2.0 | Moderate overfitting — consider reducing epochs |
| > 2.0 | Severe overfitting — deploy an earlier checkpoint |

Also check: **Is the final validation loss > 20% above the best validation loss?** If yes, the model overfit during later training.

```python
val_losses = [cp.metrics.valid_loss for cp in checkpoints if cp.metrics.valid_loss]
best_val = min(val_losses)
final_val = val_losses[-1]
if final_val > best_val * 1.2:
    print(f"⚠️ OVERFIT: Best val_loss={best_val:.4f} at earlier checkpoint, final={final_val:.4f}")
```

## Finding the Best Checkpoint

Azure AI Foundry saves checkpoints at each epoch boundary. When overfitting is detected:

1. List checkpoints: `client.fine_tuning.jobs.checkpoints.list(job_id)`
2. Find the one with lowest `valid_loss`
3. Deploy that checkpoint's `fine_tuned_model_checkpoint` directly
4. Evaluate it before deciding whether to retrain

```python
checkpoints = client.fine_tuning.jobs.checkpoints.list(job_id)
best_cp = min(checkpoints.data, key=lambda cp: cp.metrics.valid_loss or float('inf'))
print(f"Best checkpoint: step {best_cp.step_number}, "
      f"valid_loss={best_cp.metrics.valid_loss:.4f}, "
      f"model={best_cp.fine_tuned_model_checkpoint}")
```

## What the Curves Tell You About Next Steps

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Train loss barely decreases | LR too low, or data too noisy | Increase LR, or clean data |
| Train loss crashes to near 0 | LR too high, or data too easy | Decrease LR, or add harder examples |
| Valid loss rises after epoch 2 | Overfitting | Deploy epoch-2 checkpoint, or reduce epochs |
| Valid loss plateaus after epoch 1 | Model learned quickly | Try epoch=1, or lower LR for refinement |
| Valid loss oscillates wildly | Batch size too small, or data inconsistent | Increase batch size, or audit data quality |
| Train and valid loss both stay high | Task too hard for this model | Try a larger model, or simplify the task |
| Large train-valid gap from start | Insufficient data or data mismatch | Add more diverse training data |

## Comparing Across Experiments

When comparing multiple training runs, always compare:

1. **Best validation loss** (not final loss — final may be overfit)
2. **At which step/epoch** the best loss occurred
3. **The overfitting ratio** at the final step
4. **Eval scores** (validation loss alone doesn't capture everything — a model with slightly higher loss may score better on your rubric)

Example analysis table:

| Run | Best val_loss | At epoch | Final val_loss | Overfit? | Combined eval |
|-----|--------------|----------|---------------|----------|---------------|
| R1 | 0.320 | 2 | 0.325 | No | 8.05 |
| R2 | 0.260 | 2 | 0.290 | Mild | 8.53 |
| R3 | 0.195 | 2 | 0.210 | Mild | 8.28 |

**Key insight**: Lower validation loss doesn't always mean higher eval scores. R2 has higher loss than R3 but scores better. This happens when the loss metric doesn't perfectly align with your evaluation rubric (which is common). Always evaluate — don't just trust loss.

## Downloading Training Results

```python
import openai

client = openai.AzureOpenAI(...)
job = client.fine_tuning.jobs.retrieve(job_id)

if job.result_files:
    content = client.files.content(job.result_files[0])
    with open("results.csv", "wb") as f:
        f.write(content.read())
```

## RFT-Specific Metrics

RFT training produces different metrics than SFT. The result CSV includes:

| Column | What it means |
|--------|---------------|
| `train_mean_reward` | Average reward across training rollouts (primary metric — should increase) |
| `full_valid_mean_reward` | Validation reward (check for overfitting) |
| `completion_tokens_mean` | Average response length per rollout |
| `reasoning_tokens_mean` | Average reasoning/thinking tokens per rollout (o-series models) |
| `mean_unresponsive_rewards` | Rollouts that produced no scoreable output |
| `errors/graders/.../train_sample_parse_error_count` | Rollouts where the grader couldn't parse the output |
| `errors/graders/.../train_other_error_count` | Grader logic errors (bugs) |
| `train_error_count_<tool_name>` | Tool call failures during training |

### Reading RFT Reward Curves

**Healthy RFT training**: `train_mean_reward` starts near 0 (or negative) and climbs steadily. Validation reward tracks similarly with a small gap.

**Warning signs**:
- Reward flat at ~0 for many steps → grader is broken or threshold is too strict
- Reward always negative → pass_threshold is too high, all rollouts fail
- Reward immediately high and stays flat → threshold is too lenient, all rollouts pass
- Large train-valid reward gap (>0.10) → possible reward hacking (see `references/reward-hacking-prevention.md`)

### Monitor Token Growth

During RFT, `completion_tokens_mean` and `reasoning_tokens_mean` often increase as the model learns to write more detailed responses and "think harder." Monitor these:

- **Moderate growth** (tokens double) → normal, model is becoming more thorough
- **Excessive growth** (tokens 3x+) → the grader may be incentivizing verbosity over precision. Check whether your scoring dimensions inadvertently reward length.
- **When comparing checkpoints or experiments**, factor in token cost. Equal accuracy at fewer tokens is strictly better — the model is cheaper and faster at inference.

### Grader Parse Errors vs Logic Errors

In agentic RFT (with tool calling), some grader parse errors are normal. These occur when the grader receives a rollout captured mid-reasoning (before the model finished its response). The key distinction:

- **`sample_parse_error_count`** — often high in agentic RFT. Training still works if reward is climbing. Don't panic.
- **`other_error_count`** — indicates bugs in your grader logic. Should be 0 or very low. If this is high, fix your grader before continuing.

### RFT Checkpoint Selection

For RFT, checkpoint selection works differently than SFT:

1. List checkpoints: `client.fine_tuning.jobs.checkpoints.list(job_id)`
2. **Don't rely solely on `valid_reward`** — it may not perfectly predict real-world performance (especially in agentic RFT where validation evals may not execute tools)
3. Deploy 2-3 candidates: the peak valid_reward checkpoint, the final checkpoint, and one mid-training
4. Evaluate each with your **real task harness** (including tool execution if agentic)
5. Compare on **both accuracy and token cost** — pick the model that maximizes accuracy per token

```python
checkpoints = client.fine_tuning.jobs.checkpoints.list(job_id)
for cp in checkpoints:
    m = cp.metrics
    tr = f"{m.train_mean_reward:.3f}" if m.train_mean_reward is not None else "n/a"
    vr = f"{m.full_valid_mean_reward:.3f}" if m.full_valid_mean_reward is not None else "n/a"
    ct = f"{m.completion_tokens_mean:.0f}" if m.completion_tokens_mean is not None else "n/a"
    print(f"Step {cp.step_number}: train_reward={tr}, valid_reward={vr}, tokens={ct}")
```
