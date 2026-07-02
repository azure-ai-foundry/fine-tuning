# Azure AI Fine-Tuning Skill

A coding agent skill that guides you through the full fine-tuning lifecycle on [Azure AI Foundry](https://ai.azure.com/) — from dataset preparation through training, evaluation, and deployment.

## What is a Skill?

A **skill** is a structured set of instructions, reference documentation, and reusable scripts that a coding agent (GitHub Copilot, Claude Code, etc.) can read and follow to perform complex multi-step tasks. Instead of the agent relying solely on its training data, a skill gives it up-to-date, task-specific knowledge and working code.

## What This Skill Covers

| Stage | What the agent does |
|-------|-------------------|
| **Dataset creation** | Generate synthetic training data, curate examples, or augment with LLM rephrasings |
| **Dataset validation** | Validate JSONL schema, token limits, and format for SFT / DPO / RFT |
| **Base model evaluation** | Benchmark the un-tuned model to establish a baseline |
| **Training type selection** | Choose between SFT, DPO, and RFT based on your task |
| **Grader calibration** | For RFT: test your grader on base model outputs and find the optimal pass_threshold |
| **Job submission** | Submit training jobs via SDK, REST API, or `azd` CLI |
| **Job monitoring** | Poll running jobs with real-time event streaming |
| **Training curve analysis** | Detect overfitting, monitor token growth, recommend checkpoints |
| **Iterative experimentation** | Plan successive runs based on results |
| **Model deployment** | Deploy fine-tuned models with the correct format and SKU |
| **Model evaluation** | Score outputs with custom LLM judges, compare accuracy and token cost |
| **Resource cleanup** | Delete old files and deployments to reclaim quota |
| **Autonomous fine-tuning** | *(Experimental)* Full SFT loop — analyze, prepare, baseline, train, evaluate, iterate |

## Quick Start

### 1. Set up the skill

**Auto-discovery (recommended):**
If you cloned the [microsoft-foundry/fine-tuning](https://github.com/microsoft-foundry/fine-tuning) repo, coding agents auto-discover the skill via symlinks:
- GitHub Copilot → `.github/skills/azure-ai-fine-tuning`
- Claude Code → `.claude/skills/azure-ai-fine-tuning`
- Codex / other agents → `.agents/skills/azure-ai-fine-tuning`

Just open the repo and start asking questions — no manual setup needed.

> ⚠️ **Windows users:** Git on Windows does not create symlinks by default. If your agent can't find the skill, either:
> 1. Enable Developer Mode in Windows Settings → then re-clone with `git clone -c core.symlinks=true`
> 2. Or use manual setup (below) — copy `Skills/` into your project directly

**Manual setup:**
Copy the `Skills/` directory into your project and reference `SKILL.md` in your agent's instructions file (`copilot-instructions.md`, `CLAUDE.md`, etc.).

### 2. Set up your environment

**Option A: `uv` (zero-setup, recommended):**
All scripts have [PEP 723](https://peps.python.org/pep-0723/) inline dependency declarations. Just run them with `uv`:
```bash
uv run Skills/scripts/submit_training.py --help
```

**Option B: Manual install:**
```bash
cp Skills/.env.template Skills/.env
# Edit .env with your Azure AI Foundry endpoint and API key

pip install openai azure-identity requests
```

### 3. Start fine-tuning

Tell your agent something like:
- *"I want to fine-tune gpt-4.1-mini on customer support summarization"*
- *"Help me set up an RFT training job for math reasoning"*
- *"I have a JSONL dataset — validate it and submit a training job"*

The agent will read the appropriate workflow and guide you step by step.

## Directory Structure

```
Skills/
├── SKILL.md                          # Main skill file (agent entry point)
├── .env.template                     # Environment variable template
├── README.md                         # This file
├── references/
│   ├── training-types.md             # SFT vs DPO vs RFT comparison
│   ├── hyperparameters.md            # Learning rate, epochs, batch size guidance (SFT + RFT)
│   ├── dataset-formats.md            # JSONL format specs for each training type
│   ├── deployment-formats.md         # Model format, SKU, and version mapping
│   ├── evaluation-methodology.md     # Eval rubric design and grader types
│   ├── training-curve-analysis.md    # Reading training logs and curves (SFT + RFT)
│   ├── auto-evals.md                 # Reading auto-generated per-step evals (early progress signal)
│   ├── grader-design.md             # RFT grader design (type selection, partial credit, calibration)
│   ├── foundry-cli.md               # azd ai finetuning CLI reference
│   ├── vision-fine-tuning.md         # Image/video fine-tuning (gpt-4o, gpt-4.1)
│   ├── cost-management.md            # Training costs, tier selection, and budget planning
│   ├── distillation.md              # Teacher→student distillation workflow
│   ├── agentic-rft.md              # Tool calling + endpoint graders for RFT
│   ├── reward-hacking-prevention.md  # Preventing reward hacking in RFT
│   └── platform-bugs.md             # Known platform bugs and workarounds
├── workflows/
│   ├── quickstart.md                 # 6-step quickstart (fine-tune your first model)
│   ├── full-pipeline.md              # End-to-end workflow
│   ├── dataset-creation.md           # Data generation (manual, LLM augmentation, synthetic)
│   ├── iterative-training.md         # Training and HP tuning loop
│   ├── diagnose-poor-results.md      # Troubleshooting bad results
│   ├── experiment-review.md          # Post-experiment review and next steps
│   └── auto-finetune.md             # Autonomous fine-tuning workflow (experimental)
├── scripts/
│   ├── auto_finetune.py              # Autonomous SFT orchestrator (experimental)
│   ├── submit_training.py            # Submit SFT/DPO/RFT jobs (SDK + REST fallback)
│   ├── monitor_training.py           # Poll a running job until completion
│   ├── calibrate_grader.py           # RFT grader threshold calibration
│   ├── check_training.py             # Training curve analysis and checkpoints
│   ├── analyze_auto_evals.py         # Retrieve/analyze auto-generated per-step evals
│   ├── deploy_model.py               # Deploy via ARM REST API
│   ├── evaluate_model.py             # LLM judge evaluation
│   ├── convert_dataset.py            # Format conversion (SFT↔DPO↔RFT)
│   ├── generate_distillation_data.py # Generate distillation training data
│   ├── score_dataset.py              # Dataset quality scoring
│   ├── cleanup.py                    # Delete old files, deployments, pending jobs
│   ├── common.py                     # Shared auth helpers (auto-refreshing AAD tokens)
│   └── validate/
│       ├── validate_sft.py           # SFT JSONL validator
│       ├── validate_dpo.py           # DPO JSONL validator
│       ├── validate_rft.py           # RFT JSONL validator
│       └── data_stats.py             # Token counts and cost estimates
├── tests/
│   └── test_skills.py                # Compilation, security, and code quality tests
└── examples/
    └── sample-data/
        ├── sft_sample.jsonl          # SFT format reference
        ├── dpo_sample.jsonl          # DPO format reference
        └── rft_sample.jsonl          # RFT format reference
```

## Prerequisites

- **Azure subscription** with access to [Azure AI Foundry](https://ai.azure.com/)
- **Azure AI Services resource** with fine-tuning enabled
- **Python 3.9+** with `openai`, `azure-identity`, and `tiktoken`
- **Azure CLI** (`az`) authenticated for resource management
- *(Optional)* `azd` CLI with the `azure.ai.finetune` extension for CLI-based workflows

## Supported Models

| Model | SFT | DPO | RFT | Vision | Notes |
|-------|-----|-----|-----|--------|-------|
| gpt-4.1 | ✅ | ✅ | ❌ | ✅ | Best general-purpose FT model (SFT + DPO + Vision) |
| gpt-4.1-mini | ✅ | ❌ | ❌ | ❌ | SFT only |
| gpt-4.1-nano | ✅ | ❌ | ❌ | ❌ | SFT only; good distillation target |
| gpt-4o | ✅ | ✅ | ❌ | ✅ | SFT + DPO + Vision |
| gpt-4o-mini | ✅ | ❌ | ❌ | ❌ | SFT only |
| o4-mini | ❌ | ❌ | ✅ | ❌ | RFT only with graders (hourly billing) |
| gpt-5 | ❌ | ❌ | ✅ | ❌ | RFT only with graders (hourly billing) |
| Ministral-3B | ✅ | ❌ | ❌ | ❌ | OSS; requires `globalStandard` training type |
| gpt-oss-20b | ✅ | ❌ | ❌ | ❌ | OSS; requires `globalStandard` training type |
| Llama-3.3-70B | ✅ | ❌ | ❌ | ❌ | OSS; requires `globalStandard` training type |
| Qwen-3-32B | ✅ | ❌ | ❌ | ❌ | OSS; requires `globalStandard` training type |

## Guidance Highlights

Key patterns encoded in this skill:

- **Start with SFT distillation** — the most reliable fine-tuning pattern, achieving high teacher gap closure with 200–500 examples
- **Always baseline first** — evaluate the base model before fine-tuning to confirm there's room for improvement
- **Calibrate your RFT grader** — target 25-50% failure rate on the base model; recalibrate when you change your dataset
- **Python grader as default for RFT** — fast, deterministic, and reliable; endpoint graders only when you need external API calls during grading
- **DPO is risky when the base is already strong** — it can degrade quality if the model already handles the task well
- **Measure cost alongside accuracy** — compare completion tokens per response, not just accuracy scores
- **Quality over quantity** — 200–500 high-quality examples is a good starting point

## Contributing

See the repo's [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. When adding to this skill:

1. Test any new guidance end-to-end before adding it
2. Keep scripts self-contained with inline documentation
3. Add sample data for any new format you introduce
4. Update `SKILL.md` reference tables when adding new files
