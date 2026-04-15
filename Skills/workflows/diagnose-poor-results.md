# Diagnosing Poor Results

What to do when your fine-tuned model performs worse than expected.

## Quick Diagnosis Checklist

Start from the top. The most common causes are listed first.

### 1. Is Your Baseline Actually Good?

**Symptom**: Fine-tuned model scores below your expectation, but you never checked the base model.

**Fix**: Evaluate the base model with `scripts/evaluate_model.py`. Many fine-tuning projects discover the base model is already quite good — the "improvement" headroom may be smaller than expected.

### 2. Is Your Data Clean?

**Symptom**: Training curves look normal but eval scores are poor or inconsistent.

**Checks**:
- Score a random sample with `scripts/score_dataset.py` — are quality scores high?
- Manually inspect 20 examples — are the outputs correct and consistent?
- Check for duplicates: `sort training.jsonl | uniq -d | wc -l`
- Check for conflicting examples (same input, different outputs)
- Verify the system prompt is consistent across examples

**Fix**: Filter to quality score ≥ 7, remove duplicates, fix inconsistencies. Even removing 10% of bad examples can dramatically improve results.

### 3. Are You Overfitting?

**Symptom**: Training loss drops to near-zero, validation loss rises, eval scores are mediocre.

**Checks**:
- Run `scripts/check_training.py` — look for overfitting warnings
- Check the overfitting ratio (valid_loss / train_loss) — anything > 1.5 is concerning

**Fix** (in order):
1. Deploy an earlier checkpoint (epoch 2 of a 4-epoch run)
2. Reduce epochs
3. Lower the learning rate multiplier
4. Add more diverse training data

### 4. Is the Model Correct But Verbose (or Vice Versa)?

**Symptom**: High correctness, low conciseness (or the reverse).

**Cause**: This is a common tradeoff with larger datasets. More examples teach thoroughness but also verbosity.

**Fix for verbose outputs**:
- Add concise examples to your dataset
- Include a system prompt like "Be concise. Provide code only, no explanations."
- Filter your dataset to keep only the most concise correct examples
- Try a smaller, higher-quality dataset

**Fix for terse/incorrect outputs**:
- Add more detailed examples
- Include explanations in your training data
- Increase dataset size with quality-filtered examples

### 5. Is the Evaluation Rubric Appropriate?

**Symptom**: The model seems good when you spot-check, but automated eval gives low scores.

**Checks**:
- Manually grade 10 examples yourself and compare to the LLM judge's scores
- Is the judge model strong enough? (Use gpt-4o or better)
- Is the rubric prompt clear and unambiguous?
- Does the reference answer in your test set match what you actually want?

**Fix**: Adjust the rubric prompt, update reference answers, or use a stronger judge model.

### 6. Is It a Deployment or Client Bug?

**Symptom**: Model produces garbage, empty outputs, or errors.

**Common causes**:
- **Wrong model format** in deployment → HTTP 500 (see `references/deployment-formats.md`)
- **Using `AzureOpenAI` on a project endpoint** → "api-version not allowed" error
- **Capacity too low** → timeouts on long outputs
- **Wrong deployment name** → calling the base model instead of the fine-tune

**Quick test**: Send the exact same prompt to the fine-tuned model manually (via curl or Python) and compare to what the eval script sees.

### 7. Is RFT Making Things Worse?

**Symptom**: RFT model scores below the base model.

**RFT-specific diagnoses**:
- **Train-val grader gap > 0.2**: Model is gaming the grader, not learning the task
- **Grader is too easy**: Model learns to produce outputs that score high on the grader but aren't actually good
- **Grader is too noisy**: Random grading signal teaches nothing useful

**Fix**:
1. Switch to a stricter or more deterministic grader (Python execution > LLM judge)
2. Add multi-criteria grading (syntax check + semantic check)
3. Increase validation set size to get a more reliable gap estimate
4. Consider whether RFT is appropriate — SFT might be better for your task

## Escalation Path

If none of the above helps:

1. **Try a different base model** — some models fine-tune better for certain tasks
2. **Increase dataset size dramatically** (2x–5x) with synthetic data
3. **Simplify the task** — fine-tune for a narrower sub-task first
4. **Consider prompt engineering instead** — sometimes a well-crafted system prompt beats fine-tuning
5. **Combine approaches** — prompt engineering + fine-tuning together

## Red Flags That Mean "Don't Fine-Tune"

- Base model already scores > 9.0 on your rubric (minimal headroom)
- Your task changes frequently (you'd need to retrain constantly)
- You have < 50 examples and can't generate synthetic data
- The "correct" output is highly subjective (fine-tuning can't resolve ambiguity in your own preferences)
