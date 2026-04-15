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
  --model "<ft-deployment>" --test-file "<test.jsonl>" \
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
  --model "<base-model>" --train-file "<train.jsonl>" \
  --val-file "<val.jsonl>" --epochs <N> --lr <multiplier> \
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
Experiment: S1-v2
  Parent: S1-v1 (ftjob-xxx)
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
- Base model is already near-ceiling for the task (DPO peacemaker case)

## Lessons from 6 Production Scenarios

These patterns emerged from testing the full pipeline end-to-end across SFT, DPO, and RFT:

1. **SFT distillation is the most reliable pattern.** mini→nano distillation achieved 58–100% teacher gap closure across 3 different tasks (NL→Python, Text→C#, PII redaction) with just 200–300 examples and 2 epochs.

2. **Val loss overfitting doesn't always hurt.** An S1 model 84% above its best val_loss still outperformed its epoch-1 checkpoint on downstream eval (8.90 vs 8.77). Always evaluate, don't just trust curves.

3. **DPO can make things worse.** When the base model already scores >9/10 on a task, DPO actively degraded quality (9.71→7.29) with degenerate output on sensitive topics. This happened at epoch 1 too — not just overtraining.

4. **Small datasets (<100 examples) teach format only.** 73 tool-calling examples taught nano to always produce tool calls (100% vs 80%) with valid JSON, but correct tool selection stayed at 40%.

5. **Well-defined pattern tasks distill best.** PII redaction (94% gap closure) and code generation (100% gap closure) — tasks with clear input→output patterns — are ideal for distillation. Open-ended alignment tasks are not.

6. **Generate 15–20% more data than needed.** Content filters reject ~14% of synthetic PII/security data. Also account for deduplication and quality filtering.

## Lessons from Expanded Experiments (S6v2–S12)

7. **Always baseline before fine-tuning.** S6v2 tool calling showed base nano already scored 100% correct tool / 80% correct args — identical to teacher mini. FT added no value. Run evals on the base model first.

8. **Content safety can reject the FT model even with innocuous data.** S9 entity extraction training succeeded but the model was rejected at deployment for "Hate/Fairness" — triggered by PII-heavy documents (medical records, legal contracts, resumes). Workaround: remove sensitive document types and resubmit.

9. **FT nano can surpass teacher mini.** S7 CNN DailyMail summarization: FT nano (ROUGE-1 0.363, judge relevance 4.4/5) beat both base nano (0.320, 4.0/5) and teacher mini (0.303, 4.2/5). This happens when the training data teaches a specific output style the teacher doesn't naturally produce.

10. **Generic SDK evaluators cannot measure FT improvement.** Built-in Coherence/Fluency/TaskAdherence showed zero difference between base and FT on peacemaker task. Use custom graders (PythonGrader, ScoreModelGrader, StringCheckGrader) for task-specific evaluation. Generic evals are only useful as degradation guardrails.

11. **Data Designer config uses `DataDesignerConfigBuilder` with `load_config_builder()`.** The DD package (`data_designer.config`) uses a builder pattern, not a declarative config class. Templates use `{{ variable }}` (Jinja2 syntax). The CLI takes a positional config arg, not `--config`.

12. **FT deployment "DeploymentNotReady" can persist after ARM shows Succeeded.** Delete and recreate the deployment if it stays stuck. There is no other workaround — the data plane lags behind the control plane.

13. **Entity extraction FT: quality beats format.** S9v2 FT nano achieved 0.781 entity F1 vs 0.698 base and 0.700 teacher — a second case of FT beating the teacher. However, JSON validity dropped to 90% (from 100% base). The model learned better entity recognition but slightly worse output formatting. Consider adding format-only examples to training data.

14. **DPO consistently fails when base is already strong.** S8 DPO Orca (mini) degraded from 4.96→3.33/5. This is the third DPO failure (S2, S2v2, S8). Pattern: if base model scores >4.5/5, DPO will make it worse. DPO only helps when there's a clear gap between chosen and rejected that the base model doesn't already exploit.

15. **PubMed summarization: third case of FT beating teacher.** S11 FT nano (ROUGE-1 0.460) surpassed both base nano (0.392) and teacher mini (0.421). With S7 CNN and S9v2 Entity, this makes 3/4 SFT distillation tasks where nano FT beat the mini teacher. The pattern: FT learns a specific output style from training data that the larger teacher doesn't naturally produce.

16. **NL→SQL needs more/better data for FT gains.** S12 FT nano showed flat keyword F1 (0.750 vs 0.743 base) and slightly worse judge scores (3.1 vs 3.6). Only 231 training examples from DD. SQL tasks may need 500+ examples with diverse schema complexity to show meaningful improvement.

## Lessons from OSS Cross-Task Experiments (S13)

17. **OSS FT deployment format must match the base model family.** Using `format: "OpenAI"` for OSS FT models causes an unhelpful HTTP 500. Correct formats: Ministral-3B → `"Mistral AI"`, gpt-oss-20b → `"Microsoft"`, Llama → `"Meta"`, Qwen → `"Alibaba"`. Always use `version: "1"` and `sku: "GlobalStandard"`.

18. **OSS FT models suffer intermittent LoRA weight loading failures.** After successful deployment, ~70-90% of requests may return HTTP 500 "Failed to get finetune weights path: TooManyRequests". Deploy with capacity ≥ 100 and use aggressive retries. This is a platform bug, not a model quality issue.

19. **Ministral-3B FT shows strong cross-task transfer.** S5 PII: FT 3.40 vs base 2.80 (+21%); S7 Summarization: FT ROUGE-1 0.390 vs base 0.297 (+31%), beating teacher mini (0.265). With just 5 epochs and lr=1.0, Ministral-3B learned summarization style better than even GPT-4.1-mini teacher. This is the 4th case of FT beating the teacher and the 1st with an OSS model.

20. **OSS HP patterns from text2py partially transfer to other tasks.** Ministral-3B with 5ep/lr1.0 (text2py best range) worked well for summarization. The conservative HP starting points identified from the 50-model text2py leaderboard are a reasonable starting point for other tasks, though task-specific tuning may still improve results.
