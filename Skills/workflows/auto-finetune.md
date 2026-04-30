# Autonomous Fine-Tuning Workflow (Experimental)

> **⚠️ Experimental**: This workflow automates the full fine-tuning loop. It's great for exploration and getting started quickly, but experienced practitioners may prefer the individual scripts (`submit_training.py`, `check_training.py`, `evaluate_model.py`, etc.) for finer control over each step.
>
> **SFT only**: This workflow currently supports supervised fine-tuning. For RFT (reinforcement fine-tuning), use the individual scripts — RFT requires manual grader design, threshold calibration, and reward curve monitoring that don't lend themselves to full automation. See `workflows/full-pipeline.md` and `references/grader-design.md`.

Automatically fine-tune a model given a task description and data. Inspired by the hierarchical manager + parallel candidates pattern from [AIBuildAI](https://arxiv.org/abs/2604.14455).

> **Who does what**: The coding agent (Copilot CLI / VS Code) acts as the **manager** — it makes strategic decisions (task interpretation, candidate design, iterate-or-ship). The Python scripts handle **data plumbing and execution**. The user provides the task and approves key decisions.

## Prerequisites

- Azure AI Foundry project with fine-tuning enabled
- Python 3.10+ with `openai`, `requests`, `azure-identity`
- A clear task description (what the model should do differently)
- Raw data (CSV, JSON, JSONL, or Parquet) — labeled or unlabeled

## Quick Start

### Autopilot (one command)

Run the full loop end-to-end — analyze, prepare, baseline, train, evaluate, iterate:

```bash
python scripts/auto_finetune.py auto \
  --data raw_data.csv \
  --description "Classify support tickets into categories" \
  --model gpt-4.1-mini \
  --work-dir ./my_ft_run \
  --max-iterations 3
```

### Step-by-step

Run each phase individually for more control:

```bash
# 1. Analyze data and generate task spec
python scripts/auto_finetune.py analyze --data raw.csv --output task_spec.json

# 2. Generate training data (if needed — uses teacher model)
python scripts/auto_finetune.py generate --task-spec task_spec.json --num-examples 200

# 3. Prepare (convert, filter, split into train/val/test)
python scripts/auto_finetune.py prepare --task-spec task_spec.json --data raw.csv

# 4. Baseline the base model
python scripts/auto_finetune.py baseline --task-spec task_spec.json --test-file ./prepared/test.jsonl

# 5. Design candidate experiments
python scripts/auto_finetune.py candidates --task-spec task_spec.json --data-dir ./prepared

# 6. Execute (submit training jobs, monitor until done)
python scripts/auto_finetune.py execute --plan candidate_plan.json

# 7. Evaluate (deploy candidates, score with LLM judge, cleanup)
python scripts/auto_finetune.py evaluate --runs runs.json --task-spec task_spec.json --test-file ./prepared/test.jsonl

# 8. Review (compare to baseline, decide SHIP or ITERATE)
python scripts/auto_finetune.py review --leaderboard leaderboard.json --baseline baseline.json --task-spec task_spec.json
```

## How It Works

### Phase Flow

```
analyze → prepare → baseline → candidates → execute → evaluate → review
                                    ↑                                 |
                                    └──── ITERATE (adjust HPs) ──────┘
                                          SHIP (deploy winner) ──────→ done
```

### Decision Logic

After each iteration, the review phase compares candidates to the baseline:

| Outcome | Criteria | Action |
|---------|----------|--------|
| **SHIP** | Best candidate beats baseline by ≥ `min_lift_pct` (default 5%) | Deploy winning model |
| **ITERATE** | All candidates regress or don't meet threshold | Design new candidates with adjusted HPs |
| **STOP** | Max iterations or budget reached | Report best result so far |

### Candidate Diagnostics

The review phase automatically diagnoses issues:

| Issue | Detection | Recommendation |
|-------|-----------|----------------|
| Catastrophic regression | Score < 50% of baseline | Lower learning rate drastically |
| Overfitting | Val/best ratio > 1.5 | Deploy earlier checkpoint, reduce epochs |
| Deployment failure | 0% pass rate, all errors | Redeploy with longer warmup |
| Marginal improvement | Score within 2% of baseline | More data or different base model |

## Training Tier Selection

| Model Type | Supported Tiers | Default |
|-----------|----------------|---------|
| OAI (gpt-4.1-mini, nano, etc.) | `developerTier`, `globalStandard`, `standard` | `developerTier` |
| OSS (qwen, llama, ministral, oss-20b) | `globalStandard` only | `globalStandard` (auto-overridden) |

Use `--tier` to specify:
```bash
python scripts/auto_finetune.py auto --data data.jsonl --tier globalStandard
```

> **Note**: OSS models automatically override to `globalStandard` regardless of `--tier` setting.

## Artifacts

Each phase produces JSON artifacts in the working directory:

| File | Produced by | Contains |
|------|-------------|----------|
| `task_spec.json` | analyze | Task definition, eval rubric, stopping criteria |
| `baseline.json` | baseline | Base model scores per dimension |
| `candidate_plan_iterN.json` | candidates | Experiment design (models, HPs) |
| `runs_iterN.json` | execute | Job IDs, status, training metrics |
| `leaderboard_iterN.json` | evaluate | Scored candidates with pass rates |
| `review_iterN.json` | review | SHIP/ITERATE decision with diagnostics |

## Limitations

- **SFT only** — does not support RFT or DPO (use individual scripts)
- **No checkpoint selection** — uses final model, not best-validation checkpoint
- **Single eval dimension** — uses combined score, not multi-objective optimization
- **No cost tracking** — tracks tokens but not dollar costs across iterations
