# Full Pipeline Workflow

End-to-end workflow for fine-tuning a model on Azure AI Foundry.

## Prerequisites

- Azure AI Foundry resource with fine-tuning enabled
- Python 3.10+ with `openai` and `requests` packages
- Azure CLI (`az`) authenticated
- A clear task definition: what should the model do differently after fine-tuning?

## Phase 1: Define the Task

Before touching any data or models, answer these questions:

1. **What task will the fine-tuned model perform?** (e.g., "translate natural language to Python code")
2. **What does a good output look like?** Write 5 examples by hand.
3. **What does a bad output look like?** Write 3 anti-examples.
4. **How will you measure success?** Define your evaluation dimensions (see `references/evaluation-methodology.md`).
5. **What base model will you start with?** Pick 1–3 candidates from the supported model list.

## Phase 2: Prepare the Dataset

### Option A: You Have Data
1. Convert to SFT JSONL format (see `references/dataset-formats.md`)
2. Split: 80% training, 10% validation, 10% held-out test
3. Score quality with `scripts/score_dataset.py`
4. Remove or fix low-quality examples

### Option B: You Need Synthetic Data
1. Use NVIDIA Data Designer to generate training data
   - Install: `pip install data-designer`
   - Configure: `data-designer config` (set model aliases)
   - Design the dataset columns, samplers, and validators
   - Preview with `data-designer preview <config>.py --save-results`
   - Generate with `data-designer create <config>.py --num-records <N>`
2. Convert the generated data to SFT JSONL format using `scripts/convert_dataset.py`
3. Score and filter using `scripts/score_dataset.py`

### Option C: Hybrid (Seed Data + Synthetic Expansion)
1. Use your existing data as a seed dataset in Data Designer
2. Generate synthetic variations to expand coverage
3. Merge, deduplicate, and quality-filter

**Checkpoint**: You should now have:
- `training.jsonl` — training set
- `validation.jsonl` — validation set
- `test.jsonl` — held-out test set (NEVER used for training)

## Phase 3: Establish Baselines

Before fine-tuning, evaluate the base model(s) on your held-out test set.

1. Deploy base model (or use an existing deployment)
2. Run `scripts/evaluate_model.py` against your test set
3. Record scores in your leaderboard JSON

This baseline is your "zero" — every fine-tuned model must beat it to be worth deploying.

## Phase 4: Choose Training Type

Read `references/training-types.md` for the full decision framework.

Quick rule:
- **Have input–output pairs?** → SFT
- **Can write a grading function?** → RFT (only for reasoning models)
- **Need style alignment?** → DPO

Most projects start with SFT. Only move to RFT/DPO if SFT isn't sufficient.

## Phase 5: Upload Data and Submit Training

1. Set up your client:
   ```python
   # See scripts/common.py for full auth helper
   import openai
   client = openai.OpenAI(
       base_url="https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1/",
       api_key="<your-api-key>"  # or use AZURE_OPENAI_API_KEY env var
   )
   ```

2. Upload training and validation files:
   ```python
   train_file = client.files.create(purpose="fine-tune", file=open("training.jsonl", "rb"))
   val_file = client.files.create(purpose="fine-tune", file=open("validation.jsonl", "rb"))
   client.files.wait_for_processing(train_file.id)
   client.files.wait_for_processing(val_file.id)
   ```

3. Submit first training job with default hyperparameters:
   ```python
   job = client.fine_tuning.jobs.create(
       model="gpt-4.1-mini",
       training_file=train_file.id,
       validation_file=val_file.id,
       method={"type": "supervised"}
   )
   ```

**Or use the Foundry CLI** (no Python needed):
   ```bash
   azd ai finetuning jobs submit -f ./fine-tune-job.yaml
   ```
   See `references/foundry-cli.md` for YAML config format and setup.

Or use `scripts/submit_training.py` for a more robust submission with error handling.

See `references/hyperparameters.md` for starting HP values.

## Phase 6: Monitor and Analyze Training

1. Wait for job completion:
   ```python
   while True:
       job = client.fine_tuning.jobs.retrieve(job.id)
       if job.status in ("succeeded", "failed", "cancelled"):
           break
       time.sleep(60)
   ```

2. Download and analyze training curves with `scripts/check_training.py`
3. Read `references/training-curve-analysis.md` to interpret the results
4. Check for overfitting — if detected, consider deploying an earlier checkpoint

## Phase 7: Evaluate Fine-Tuned Model

1. Deploy the fine-tuned model (see `references/deployment-formats.md` for format/SKU)
2. Run `scripts/evaluate_model.py` against the same held-out test set
3. Compare against baseline and previous experiments
4. Delete deployment after evaluation

## Phase 8: Iterate

Follow the experiment loop in `workflows/iterative-training.md`:
- Adjust hyperparameters based on training curves
- Try different data subsets or augmentations
- Test different base models
- Track everything in your leaderboard

## Phase 9: Ship

When you have a model that convincingly beats the baseline:
1. Deploy with production-appropriate capacity
2. Monitor in production with Application Insights
3. Periodically re-evaluate against your test set to catch regression
4. Consider periodic retraining as new data becomes available
