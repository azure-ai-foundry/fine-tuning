# Experiment Review & Next Steps Workflow

After every fine-tuning experiment, follow this structured review to decide whether to iterate, deploy, or stop.

## Step 1: Retrieve Training Curves

```bash
python scripts/check_training.py \
  --base-url "$BASE_URL" --api-key "$API_KEY" \
  --job-id "<job-id>"
```

### What to look for:

| Signal | Meaning | Action |
|--------|---------|--------|
| Val loss decreasing throughout | Healthy training | May benefit from more epochs |
| Val loss rises after initial drop | Overfitting | Deploy earlier checkpoint, or retrain with fewer epochs / lower LR |
| Val/train loss both near zero (DPO) | DPO convergence — may be over-trained | Test for degeneration on edge cases |
| No val loss data (RFT) | Normal for reward-based training | Evaluate on held-out test set only |
| Train loss not decreasing | Underfitting | Increase LR multiplier or epochs |
| Val/train ratio > 3.0 consistently | Severe overfitting | Reduce epochs, increase data, or lower LR |

### Checkpoint strategy

- Checkpoints are saved at epoch boundaries (step = n_examples * epoch_number)
- If best val_loss is mid-epoch, deploy the **nearest checkpoint** (end of that epoch)
- If best val_loss is in epoch 1, deploy `ckpt-step-<epoch1_end>` — it's often better than the final model
- Checkpoint deployment: use the model ID with `:ckpt-step-<N>` suffix

## Step 2: Evaluate on Held-Out Test Set

Run evaluation comparing the fine-tuned model against the base model (and teacher, if distillation):

```bash
# For SFT / distillation
python scripts/evaluate_model.py \
  --base-url "$BASE_URL" --api-key "$API_KEY" \
  --deployment-name "<ft-deployment>" --test-file "<test.jsonl>" \
  --judge-model "gpt-4.1-mini"

# For RFT (math / code)
# Use exact-match accuracy on the answer field
# Compare FT vs base on same test set

# For DPO (alignment)
# Use domain-specific judge prompt (e.g., de-escalation quality)
# Test on adversarial/edge-case prompts, not just average cases
```

### Key metrics by training type:

| Type | Primary Metric | Secondary | Watch For |
|------|---------------|-----------|-----------|
| SFT | Combined quality score | Gap closure vs teacher | Regression on easy examples |
| DPO | Domain-specific judge score | Degeneration rate | Repetitive/garbage output on edge cases |
| RFT | Exact-match accuracy | Unique wins over base | Problems both miss (may be bad data) |

## Step 3: Diagnose Results

### Decision tree:

```
Did the model improve over base?
├── YES: By how much?
│   ├── Large improvement (>15% or >0.5 quality points)
│   │   └── Check for overfitting → if none, consider deploying
│   ├── Moderate improvement (5-15% or 0.2-0.5 points)
│   │   └── Review training curves → likely room to improve with more data or tuning
│   └── Small improvement (<5% or <0.2 points)
│       └── Consider: more data, different hyperparameters, or different approach
├── NO CHANGE:
│   └── Check: enough data? right task format? base model already strong?
└── WORSE:
    └── Check for: overtraining, degeneration, wrong data format, bad data quality
```

### Common patterns and fixes:

**Pattern: Overfitting (val loss rises)**
- Cause: Too many epochs for dataset size
- Fix: Retrain with fewer epochs, or deploy earlier checkpoint
- Rule of thumb: <500 examples → 1-2 epochs; 500-2000 → 2-3; >2000 → 3-5

**Pattern: DPO degeneration (repetitive tokens)**
- Cause: Over-optimization, especially on sensitive topics
- Fix: Deploy epoch-1 checkpoint; retrain with 1 epoch; increase beta (more conservative)
- Warning sign: training loss near zero before end of epoch 1

**Pattern: RFT problems both models miss**
- Cause: Often the generated reference answers are wrong, not the models
- Fix: Audit the "both miss" problems manually; fix data and retrain
- Also consider: tolerance threshold too tight, answer format mismatch

**Pattern: Small improvement despite good training curves**
- Cause: Base model already strong at the task; dataset too easy/homogeneous
- Fix: Generate harder examples; increase dataset diversity; try different base model

**Pattern: Good quality but high latency/cost**
- Cause: Fine-tuned a large model when distillation to smaller model would work
- Fix: Use the current FT model as teacher, distill to smaller model (nano)

## Step 4: Propose Next Experiment

Based on diagnosis, choose ONE of these experiment types:

### A. Earlier Checkpoint Deploy
When: Overfitting detected, earlier checkpoint likely better.
```bash
python scripts/deploy_model.py --name "<name>-ckpt" \
  --model-id "<model>:ckpt-step-<N>" \
  --sub "$SUB" --rg "$RG" --account "$ACCOUNT"
```
Then re-evaluate. No retraining needed.

### B. Hyperparameter Adjustment
When: Training curves suggest wrong LR or epochs.
```bash
python scripts/submit_training.py \
  --base-url "$BASE_URL" --api-key "$API_KEY" \
  --model "<base-model>" --training-file "<train.jsonl>" \
  --validation-file "<val.jsonl>" --epochs <N> --lr <multiplier> \
  --suffix "<experiment-name>"
```
Common adjustments:
- Overfitting → reduce epochs OR reduce LR multiplier (try 0.5x)
- Underfitting → increase epochs OR increase LR multiplier (try 2x)
- DPO degeneration → set epochs=1, increase beta (0.2 → 0.5)

### C. More/Better Data
When: Model improved but plateau'd, or both models miss same problems.
- Audit errors: are reference answers correct?
- Generate more diverse examples (different topics, harder difficulty)
- For DPO: ensure non-preferred responses are realistic (not cartoonishly bad)
- Re-split and retrain

### D. Different Training Type
When: Current approach has fundamental limits.
- SFT not aligning well → try DPO on preference pairs
- DPO degeneration → try SFT with curated good examples instead
- RFT plateau → try SFT on chain-of-thought traces from a stronger model

### E. Distillation Cascade
When: Quality is good but need lower cost/latency.
- Use current FT model as teacher
- Generate training data for smaller model
- Fine-tune smaller model (e.g., nano) via SFT distillation

## Step 5: Track Experiments

Maintain an experiment log. Each entry should record:

```
Experiment: experiment-2
  Parent: experiment-1 (ftjob-xxx)
  Change: Deploy epoch-1 checkpoint instead of final model
  Hypothesis: Epoch 1 has better val_loss, may score higher
  Base model: gpt-4.1-nano
  Training: N/A (checkpoint deploy)
  Result: [pending evaluation]
  Decision: [deploy / iterate / stop]
```

## Quick Reference: When to Stop

Stop iterating when:
- Quality meets your acceptance threshold
- Marginal improvement < 2% across last 2 experiments
- You've exhausted reasonable hyperparameter space
- Cost of further experiments exceeds value of improvement
- Base model is already near-ceiling for the task (e.g., DPO on an already-strong base model)

## Lessons Learned from Production Fine-Tuning

These patterns emerged from extensive testing across SFT, DPO, and RFT on Azure AI Foundry:

### SFT & Distillation Patterns

1. **SFT distillation is the most reliable pattern.** Teacher→student distillation (e.g., mini→nano) routinely achieves high gap closure across tasks like code generation, summarization, and structured extraction with just 200–300 examples and 2 epochs.

2. **Val loss overfitting doesn't always hurt.** A model well above its best val_loss can still outperform its epoch-1 checkpoint on downstream eval. Always evaluate on held-out data — don't just trust curves.

3. **Fine-tuned small models can beat the teacher.** On tasks with clear input→output patterns (summarization, entity extraction, code generation), fine-tuned nano models frequently surpass the teacher. The model learns a specific output style from training data that the larger teacher doesn't naturally produce.

4. **Small datasets (<100 examples) teach format only.** With very few examples, a fine-tuned model learns mechanical patterns (e.g., always produce tool calls, valid JSON) but does NOT improve task-specific accuracy. Need 200+ examples for domain knowledge.

5. **Well-defined pattern tasks distill best.** Structured extraction and code generation — tasks with clear input→output patterns — are ideal for distillation. Open-ended alignment tasks are not.

6. **Generate 15–20% more data than needed.** Content filters reject a portion of synthetic PII/security data. Also account for deduplication and quality filtering.

7. **Always baseline before fine-tuning.** Sometimes the base model already handles the task perfectly. Run evals on the base model first to confirm there's room for improvement.

### DPO Patterns

8. **DPO can make things worse.** When the base model already scores well (>4.5/5) on a task, DPO actively degrades quality with degenerate output. This can happen even at epoch 1 — it's not just overtraining.

9. **DPO consistently fails when the base is already strong.** If the base model scores >4.5/5, DPO will likely degrade rather than improve. DPO only helps when there's a clear gap between chosen and rejected that the base model doesn't already exploit.

### RFT Patterns

10. **The grader matters more than hyperparameters.** For RFT, focus effort on grader quality and alignment rather than HP sweeps.

### Evaluation Patterns

11. **Generic SDK evaluators cannot measure FT improvement.** Built-in Coherence/Fluency/TaskAdherence evaluators often show zero difference between base and FT. Use custom graders (PythonGrader, ScoreModelGrader, StringCheckGrader) for task-specific evaluation. Generic evals are only useful as degradation guardrails.

12. **Content safety can reject the FT model even with innocuous data.** Training may succeed but the model can be rejected at deployment for content safety violations triggered by PII-heavy documents. Workaround: remove sensitive document types and resubmit.

### OSS Model Patterns

13. **OSS FT deployment format must match the base model family.** Using the wrong format causes an unhelpful HTTP 500. See `references/deployment-formats.md` for the mapping.

14. **OSS FT models suffer intermittent LoRA weight loading failures.** Deploy with capacity ≥ 100 and use aggressive retries. This is a platform bug, not a model quality issue.

15. **Small OSS models can't memorize large label sets.** Models with ~3B parameters fail at classification with 50+ classes — they invent synonym labels. Increasing data doesn't help (model capacity limit). Use ≥20B models for many-class tasks.

### Data & Format Patterns

16. **Data Designer tips**: `CategorySamplerParams` uses `values=` (not `categories=`); use `LLMTextColumnConfig` (not `LLMColumnConfig`); set `AZURE_FOUNDRY_API_KEY` each new shell session.

17. **Classification eval MUST include the system prompt.** If your training data has a system prompt, replay it at eval time — otherwise the model reverts to generic assistant behavior.
