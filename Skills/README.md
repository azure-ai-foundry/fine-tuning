# Azure AI Fine-Tuning Skill

A coding agent skill that guides you through the full fine-tuning lifecycle on [Azure AI Foundry](https://ai.azure.com/) — from dataset preparation through training, evaluation, and deployment.

## What is a Skill?

A **skill** is a structured set of instructions, reference documentation, and reusable scripts that a coding agent (GitHub Copilot, Claude Code, etc.) can read and follow to perform complex multi-step tasks. Instead of the agent relying solely on its training data, a skill gives it up-to-date, task-specific knowledge and working code.

## What This Skill Covers

| Stage | What the agent does |
|-------|-------------------|
| **Dataset creation** | Generate synthetic training data using Data Designer, or prepare existing data |
| **Dataset validation** | Validate JSONL schema, token limits, and format for SFT / DPO / RFT |
| **Base model evaluation** | Benchmark the un-tuned model to establish a baseline |
| **Training type selection** | Choose between SFT, DPO, and RFT based on your task |
| **Job submission** | Submit training jobs via SDK, REST API, or `azd` CLI |
| **Training curve analysis** | Detect overfitting, recommend checkpoints |
| **Iterative experimentation** | Plan successive runs based on results |
| **Model deployment** | Deploy fine-tuned models with the correct format and SKU |
| **Model evaluation** | Score outputs with custom LLM judges |

## Quick Start

### 1. Install the skill

**GitHub Copilot (VS Code / CLI):**
Copy the `Skills/` directory into your project and reference `SKILL.md` in your Copilot instructions, or use it as a [Copilot custom instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions).

**Claude Code:**
Add the skill directory to your project and reference `SKILL.md` in your `CLAUDE.md` or system prompt.

**Any agent:**
Point your agent at `SKILL.md` as a context file. It will discover the workflows, references, and scripts from there.

### 2. Set up your environment

```bash
cp Skills/.env.template Skills/.env
# Edit .env with your Azure OpenAI endpoint, API key, and resource coordinates
```

Required Python packages:
```bash
pip install openai azure-identity tiktoken requests
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
├── SKILL.md                          # Main skill file (entry point)
├── .env.template                     # Environment variable template
├── README.md                         # This file
├── references/
│   ├── training-types.md             # SFT vs DPO vs RFT comparison
│   ├── hyperparameters.md            # Learning rate, epochs, batch size guidance
│   ├── dataset-formats.md            # JSONL format specs for each training type
│   ├── deployment-formats.md         # Model format, SKU, and version mapping
│   ├── evaluation-methodology.md     # Eval rubric design and grader types
│   ├── training-curve-analysis.md    # Reading training logs and curves
│   ├── foundry-cli.md               # azd ai finetuning CLI reference
│   ├── vision-fine-tuning.md         # Image/video fine-tuning (gpt-4o, gpt-4.1)
│   ├── cost-management.md            # Training costs and budget planning
│   ├── distillation.md              # Teacher→student distillation workflow
│   ├── agentic-rft.md              # Tool calling + endpoint graders for RFT
│   ├── reward-hacking-prevention.md  # Preventing reward hacking in RFT
│   └── platform-bugs.md             # Known platform bugs and workarounds
├── workflows/
│   ├── full-pipeline.md              # End-to-end workflow (start here)
│   ├── dataset-creation.md           # Data generation with Data Designer
│   ├── iterative-training.md         # Training and HP tuning loop
│   ├── diagnose-poor-results.md      # Troubleshooting bad results
│   └── experiment-review.md          # Post-experiment review and next steps
├── scripts/
│   ├── submit_training.py            # Submit SFT/RFT jobs (SDK + REST fallback)
│   ├── generate_distillation_data.py # Generate distillation training data
│   ├── check_training.py             # Training curve analysis
│   ├── deploy_model.py               # Deploy via ARM REST API
│   ├── evaluate_model.py             # LLM judge evaluation
│   ├── convert_dataset.py            # Format conversion (SFT↔DPO↔RFT)
│   ├── score_dataset.py              # Dataset quality scoring
│   ├── common.py                     # Shared auth helpers
│   └── validate/
│       ├── validate_sft.py           # SFT JSONL validator
│       ├── validate_dpo.py           # DPO JSONL validator
│       ├── validate_rft.py           # RFT JSONL validator
│       └── data_stats.py             # Token counts and cost estimates
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
| gpt-4.1-mini | ✅ | ✅ | ❌ | ❌ | Best general-purpose FT model |
| gpt-4.1-nano | ✅ | ✅ | ❌ | ❌ | Best for distillation targets |
| gpt-4o | ✅ | ✅ | ❌ | ✅ | Vision fine-tuning supported |
| gpt-4.1 | ✅ | ✅ | ❌ | ✅ | Vision fine-tuning supported |
| o4-mini | ❌ | ❌ | ✅ | ❌ | RFT with graders |
| o3-mini | ❌ | ❌ | ✅ | ❌ | RFT with graders |
| Ministral-3B | ✅ | ❌ | ❌ | ❌ | OSS; requires `globalStandard` training type |
| gpt-oss-20b | ✅ | ❌ | ❌ | ❌ | OSS; requires `globalStandard` training type |
| Llama-3.3-70B | ✅ | ❌ | ❌ | ❌ | OSS; requires `globalStandard` training type |
| Qwen-3-32B | ✅ | ❌ | ❌ | ❌ | OSS; requires `globalStandard` training type |

## Guidance Highlights

Key patterns encoded in this skill:

- **Start with SFT distillation** — the most reliable fine-tuning pattern, achieving high teacher gap closure with 200–500 examples
- **Always baseline first** — evaluate the base model before fine-tuning to confirm there's room for improvement
- **DPO is risky when the base is already strong** — it can degrade quality if the model already handles the task well
- **RFT is for verifiable tasks** — math, code with test suites, structured output with exact-match graders
- **Quality over quantity** — 200–500 high-quality examples is a good starting point

## Contributing

See the repo's [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. When adding to this skill:

1. Test any new guidance end-to-end before adding it
2. Keep scripts self-contained with inline documentation
3. Add sample data for any new format you introduce
4. Update `SKILL.md` reference tables when adding new files
