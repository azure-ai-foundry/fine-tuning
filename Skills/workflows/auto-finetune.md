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

# 2b. Alternative: pull data from the Foundry Data Generation API (traces, agent spec, OpenAPI tools, ...)
#     See "Data sources: local teacher vs Foundry Data Generation API" below.
python scripts/auto_finetune.py foundry-generate --task-spec task_spec.json \
    --source traces --agent-name my-agent --hours 24 --max-samples 200

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

## Data sources: local teacher vs Foundry Data Generation API

The `generate` phase (the second phase of the autopilot) has two backends:

| Backend | Subcommand / flag | When to use |
|---------|-------------------|-------------|
| **Local teacher (default)** | `generate` / `--datagen-backend local` | Cold-start from a free-text description. Works without a Foundry project endpoint; has built-in quality scoring (`--min-quality`), difficulty mixing, and dedup against existing data. |
| **Foundry Prompt** | `foundry-generate --source prompt-inline` / `--datagen-backend foundry-prompt` | Cold-start when the project has a strong teacher; uses the service's QA pair generation. (Implementation note: the autopilot routes this through `prompt-file` internally — uploads description as `user_data` — to avoid a known service-side fast-fail on inline Prompt+SFT on some projects.) |
| **Foundry File** | `foundry-generate --source file` / `--datagen-backend foundry-file --datagen-file-id …` | You have a corpus already uploaded as `user_data` (typically when corpus is large or has been curated). |
| **Foundry Agent** | `foundry-generate --source agent` / `--datagen-backend foundry-agent --datagen-agent-name …` | You have a deployed agent (with instructions/tools) but no traffic yet — bootstrap from the registered agent spec. |
| **Foundry Traces** | `foundry-generate --source traces` / `--datagen-backend foundry-traces --datagen-agent-name … --datagen-hours …` | You have real production traffic flowing through a deployed agent. Best for distillation: target the actual queries your users send. |
| **Tool-use SFT** (OpenAPI spec) | `foundry-generate --source file --recipe tool-use` | Tool-calling fine-tune. Upload an OpenAPI 3.0/3.1 spec as `user_data` first. See `workflows/synthetic-datagen.md`. |

### Who picks the backend — user or agent?

`--datagen-backend` defaults to **`auto`** — the autopilot infers the backend from the companion flags you pass, in this order:

| If you pass… | Inferred backend |
|---|---|
| `--datagen-file-id <id>` | `foundry-file` |
| `--datagen-agent-name <name>` + `--datagen-hours <n>` | `foundry-traces` |
| `--datagen-agent-name <name>` (no hours) | `foundry-agent` |
| `--project-endpoint <url>` (and nothing more specific) | `foundry-prompt` |
| (none of the above) | `local` |

Pass an explicit `--datagen-backend local|foundry-prompt|foundry-file|foundry-agent|foundry-traces` to override inference. The autopilot prints the chosen backend in Phase 2 so you can see how the inference resolved.

### Direct invocation (any backend)

```bash
# Local teacher loop (default)
python scripts/auto_finetune.py generate --task-spec task_spec.json --num-examples 200

# Foundry Data Generation API (explicit subcommand)
python scripts/auto_finetune.py foundry-generate \
  --task-spec task_spec.json \
  --source traces --recipe traces --scenario sft \
  --agent-name my-deployed-agent --agent-version 3 --hours 24 \
  --max-samples 200 --project-endpoint $AZURE_AI_PROJECT_ENDPOINT
```

### Autopilot with backend inference

Just pass the relevant flags — the backend resolves automatically:

```bash
# Inferred → foundry-traces (because --datagen-agent-name + --datagen-hours)
python scripts/auto_finetune.py auto \
  --description "Distil retail-agent responses into gpt-4.1-nano" \
  --model gpt-4.1-nano \
  --datagen-agent-name retail-agent --datagen-agent-version 3 \
  --datagen-hours 168 \
  --teacher gpt-4.1-mini \
  --project-endpoint $AZURE_AI_PROJECT_ENDPOINT \
  --work-dir ./distil_run

# Inferred → foundry-file (because --datagen-file-id)
python scripts/auto_finetune.py auto \
  --description "Q&A from internal HR policy" \
  --datagen-file-id file-abc123 \
  --project-endpoint $AZURE_AI_PROJECT_ENDPOINT \
  ...

# Inferred → local (no datagen-* flags)
python scripts/auto_finetune.py auto --description "Classify tickets" --model gpt-4.1-mini
```

Both backends write `<output-dir>/generated_data.jsonl` in chat-SFT format, so the rest of the pipeline (`prepare`, `baseline`, `candidates`, …) is identical regardless of which one runs. See `references/data-generation-api.md` for the full API surface (including the **error table** with known service-level constraints and workarounds) and `workflows/synthetic-datagen.md` / `workflows/traces-to-dataset.md` for end-to-end walkthroughs.

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
