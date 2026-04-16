# Iterative Training Workflow

How to systematically improve a fine-tuned model through successive experiments.

## The Core Loop

```
┌─────────────────────────────────┐
│  1. Train with current config   │
│  2. Analyze training curves     │
│  3. Evaluate on held-out set    │
│  4. Diagnose what to change     │
│  5. Plan next experiment        │
└─────────┬───────────────────────┘
          │
          ▼
   Better than baseline?
     ├─ No → Go back to step 4
     └─ Yes → Good enough?
              ├─ No → Go back to step 4
              └─ Yes → Ship it
```

**Rule**: Change ONE variable per experiment. If you change the LR, dataset, AND epochs simultaneously, you won't know which change helped.

## Experiment Tracking

Keep a spreadsheet or JSON with every run:

| Run | Base model | Dataset | Epochs | LR | Batch | Best val_loss | Combined eval |
|-----|-----------|---------|--------|-----|-------|--------------|---------------|
| R1 | gpt-4.1-mini | v1 (335 ex) | 2 | 1.0 | default | 0.320 | 8.05 |
| R2 | gpt-4.1-mini | v1 (335 ex) | 2 | 0.5 | default | 0.310 | 9.15 |
| ... | ... | ... | ... | ... | ... | ... | ... |

## What to Try (In Priority Order)

### Priority 1: Data Quality
The highest-leverage change is almost always the data.

- **Filter low-quality examples**: Use `scripts/score_dataset.py`, raise the threshold
- **Fix inconsistencies**: Examples that contradict each other confuse the model
- **Add diversity**: If the model fails on certain input types, add training examples for those
- **Reduce noise**: Remove examples where the output is correct but not how you'd want it

### Priority 2: Hyperparameters
See `references/hyperparameters.md` for the full guide.

**Quick sweep strategy:**
1. Baseline: epochs=2, lr=1.0
2. If overfitting: try lr=0.5 or epochs=1
3. If underfitting: try lr=1.5 or epochs=3
4. Once you find a good LR: try batch_size=16 or batch_size=32

### Priority 3: Base Model
Different models have different strengths:

- **gpt-4.1-mini**: Best quality-per-dollar for most tasks
- **gpt-4.1-nano**: Fastest inference, good for simple tasks
- **gpt-oss-20b-11**: Strong on large datasets, reaches lowest absolute loss
- **Ministral-3B**: Lightweight, fast inference
- **Qwen-3-32B, Llama-3.3-70B**: Good for multilingual or specialized tasks

### Priority 4: Training Type
If SFT has plateaued and you need better reasoning → try RFT (if the model supports it).
If you need style alignment → try DPO.

Read `references/training-types.md` before switching.

## Diagnostic Decision Tree

After each run, diagnose:

```
Training curves look healthy (no overfitting)?
├─ Yes
│  ├─ Eval improved? → Great, try refining further
│  └─ Eval same/worse? → Data quality issue — filter or augment
│
└─ No (overfitting detected)
   ├─ Does an earlier checkpoint eval well?
   │  ├─ Yes → Deploy that checkpoint, skip retraining
   │  └─ No → Reduce epochs or lower LR
   │
   └─ Severe overfitting (ratio > 2.0)?
      ├─ Dataset too small? → Add more training data
      └─ Dataset large enough? → Lower LR dramatically (0.1–0.3)
```

## When to Stop

Stop iterating when any of these are true:
1. **You've beaten the baseline by a meaningful margin** (> 5% improvement) and the last 3 experiments didn't improve further
2. **You've hit diminishing returns**: each experiment improves by < 0.1 points
3. **The model is "good enough"** for your production use case
4. **You've exhausted your budget** (time or money)

## Multi-Model Strategy

For maximum coverage, run the same dataset through 2–3 different base models:

1. **gpt-4.1-mini** — your primary candidate (best general quality)
2. **gpt-oss-20b-11** — large-dataset specialist (if you have 500+ examples)
3. **gpt-4.1-nano** — fast inference option (if latency matters)

Often, different models are best at different sub-tasks. Your evaluation will reveal which model to deploy.

## Common Mistakes

1. **Not establishing a baseline first**: Without knowing the base model's score, you can't measure improvement.
2. **Changing multiple variables at once**: You learn nothing about what works.
3. **Overfitting to the eval set**: If you tune hyperparameters to maximize eval scores, you're overfitting to the eval set. Keep a completely separate "final test set" for the last check.
4. **Ignoring training curves**: The curves tell you what to change next. Reading them saves entire retraining cycles.
5. **More data without quality check**: Doubling your dataset size with lower-quality data often makes things worse.
6. **Not cleaning up deployments**: Leaving old deployments running wastes quota and money.
