# Hyperparameter Guide

## The Three Knobs

| Parameter | What it controls | Default | Typical range |
|-----------|-----------------|---------|---------------|
| **Epochs** | How many times the model sees each example | 2 | 1–5 |
| **Learning rate multiplier** | How aggressively weights change | 1.0 | 0.1–2.0 |
| **Batch size** | Examples processed together per step | Model-dependent | 4–32 |

## Starting Point

For your first experiment, use:
- **Epochs**: 2
- **Learning rate multiplier**: 1.0 (or the platform default)
- **Batch size**: Platform default

This gives you a clean baseline. Only change one variable at a time in subsequent experiments.

## Epochs

**What happens as you increase epochs:**
- Epoch 1: Model learns the broad patterns
- Epoch 2: Model refines — usually the sweet spot
- Epoch 3+: Risk of overfitting increases sharply
- Epoch 5+: Almost certainly overfitting unless dataset is very large

**Decision rule**: If validation loss increases after epoch 2, you don't need more epochs — you need better data.

**Dataset size vs. epochs** (rules of thumb):
| Dataset size | Recommended epochs |
|-------------|-------------------|
| < 100 examples | 3–5 (model needs more passes) |
| 100–500 examples | 2–3 |
| 500–2,000 examples | 1–2 |
| > 2,000 examples | 1 |

## Learning Rate Multiplier

**Higher LR** (1.5–2.0): Learns faster but risks overshooting. Good for:
- Large, diverse datasets
- When the task is very different from pre-training
- Quick experiments to test if a direction works

**Lower LR** (0.1–0.5): Learns slowly but preserves base model quality. Good for:
- Small datasets (< 200 examples)
- When you want to refine, not overwrite, base behavior
- When base model is already close to what you need

**Finding the right LR**:
1. Start at 1.0
2. If training loss drops fast but validation loss diverges → LR too high, try 0.5
3. If training loss barely moves → LR too low, try 1.5 or 2.0
4. If both losses converge smoothly → you're in the right range

**Key finding**: For large datasets (1,000+ examples), a lower LR multiplier (0.2–0.5) often produces better combined metrics than the default 1.0. The model has enough data to learn gradually.

## Batch Size

Most of the time, use the platform default. Change batch size only if:
- **Increasing**: You see very noisy training curves (helps smooth gradients)
- **Decreasing**: You have a tiny dataset and need more gradient updates per epoch

**Practical note**: On Azure AI Foundry, batch size may be constrained by the model. Not all models support all batch sizes.

## The Experiment Loop

```
Run 1: epochs=2, lr=1.0, batch=default
  │
  ├─ Good results? → Try lr=0.5 to see if refinement helps
  ├─ Overfitting (val loss rises)? → Reduce epochs to 1, or try lr=0.5
  ├─ Underfitting (val loss flat)? → Try lr=1.5 or epochs=3
  └─ Training looks good but eval is bad? → Data quality issue, not HP issue
```

## Checkpoint Trick for Overfitting

When you see overfitting (validation loss rises after epoch 2 of a 4-epoch run):

1. **Don't retrain yet** — deploy the epoch-2 checkpoint directly
2. Evaluate it. If it scores well, you've saved a full retraining cycle.
3. Only retrain with fewer epochs if no checkpoint is good enough

This works because Azure AI Foundry saves checkpoints at each epoch boundary. You can deploy any checkpoint, not just the final model.

**How to find checkpoints:**
```python
checkpoints = client.fine_tuning.jobs.checkpoints.list(job_id)
for cp in checkpoints.data:
    print(f"Step {cp.step_number}: val_loss={cp.metrics.valid_loss}")
    # cp.fine_tuned_model_checkpoint → deployable model ID
```

## Advanced: Hyperparameter Sweep Strategy

If you want to systematically explore, use this grid:

| Run | Epochs | LR | Why |
|-----|--------|----|-----|
| 1 | 2 | 1.0 | Baseline |
| 2 | 2 | 0.5 | Conservative refinement |
| 3 | 2 | 1.5 | Aggressive learning |
| 4 | 3 | 1.0 | More training time |
| 5 | 1 | 1.0 | Minimal intervention |

Then add batch size variation on the best-performing configuration.

**Cost-aware sweeping**: Each run costs money. Prioritize the runs most likely to help based on what your training curves tell you. Don't sweep blindly.

## Model-Specific Notes

**gpt-4.1-mini**: Responds well to low LR (0.5–1.0) with 2 epochs. Very capable base model — small nudges go a long way. Responds well to conservative hyperparameters.

**gpt-4.1-nano**: Needs slightly higher LR (1.0–1.5) and more epochs (2–3) due to smaller capacity. Excels at pattern tasks (structured extraction, code generation). Fine-tuned nano can sometimes surpass the teacher model on specific tasks.

**gpt-oss-20b-11**: Benefits from lower LR (0.2–0.5) and 2 epochs as a starting point. Responds well to larger datasets. Deployment can fail with InternalServerError — retry or use capacity=100.

**o4-mini (RFT)**: Hyperparameters are less tunable in RFT — the grader quality matters more than LR. Focus effort on the grader, not the HP sweep.

## OSS Model Hyperparameter Guide

OSS models (Ministral-3B, gpt-oss-20b, Llama-3.3-70B, Qwen-32B) behave differently from OpenAI models. Key differences:

1. **Require `trainingType: "globalStandard"`** in the API request body
2. **Need more epochs** than OpenAI models for the same task (especially smaller models)
3. **More prone to overfitting** — monitor validation loss carefully
4. **Deployment may fail** with InternalServerError (platform bug, retry with capacity=100)

### Starting points by model:

| Model | Recommended Start | Best Found | Notes |
|-------|------------------|------------|-------|
| **Ministral-3B** | 5ep, lr=1.0 | 10ep, lr=0.5 | Needs many epochs; small model capacity means slower convergence. 50ep massively overfits. |
| **gpt-oss-20b** | 2ep, lr=0.3 | 2ep, lr=0.3 | Lower LR is critical — lr=1.0 overfits quickly. Responds well to larger datasets. |
| **Llama-3.3-70B** | 3ep, lr=0.3 | 5ep, lr=0.5 | Large model, baseline was weak on code tasks (13%). FT improved it significantly (+11.8%). lr=2.0 caused catastrophic degradation. |
| **Qwen-32B** | 3ep, lr=0.3 | 3ep, lr=0.3 | Most fragile — more data hurt performance. 50ep caused collapse. Conservative HP only. |

> **Note:** These starting points were derived from code generation tasks. They transfer reasonably well to other task types, but task-specific tuning may still improve results.

### Key patterns for OSS models:
- **OSS models need 2-5× more epochs than nano** for the same task
- **Lower LR is safer** (0.3-0.5) — lr=1.0 works for Ministral but overfits gpt-oss-20b
- **More data doesn't always help** — large datasets (4K+) can sometimes degrade OSS model quality
- **Batch size rarely matters** — defaults are fine for OSS models
- **Always check for deployment bugs** before blaming HPs — OSS models have known deployment issues
