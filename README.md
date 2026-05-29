# Fine-Tuning in AI Foundry

This repository contains **14 end-to-end demos** and **sample datasets** for fine-tuning models on [Azure AI Foundry](http://ai.azure.com/). Use this repo to explore practical fine-tuning workflows and access ready-to-use data for your own projects.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Demos](#-demos)
- [Sample Datasets](#-sample-datasets)
- [AI Agent Skills](#-ai-agent-skills)
- [Prerequisites](#-prerequisites)
- [Contributing](#contributing)

---

## 🚀 Quick Start

**New to fine-tuning?** Start here:

1. **[SFT_Bug_Detection](Demos/SFT_Bug_Detection/)** - Best first demo! Fine-tune GPT-4.1-mini to detect code bugs — beat GPT-5.4 quality at 9x lower cost
2. **[Sample_Datasets](Sample_Datasets/)** - Understand data formats for SFT, DPO, and RFT

**Want an AI coding assistant to guide you?**
- Open this repo in VS Code with Copilot, or use Claude/Codex — the agent skill auto-discovers and walks you through the full workflow
- Or follow **[Skills/workflows/quickstart.md](Skills/workflows/quickstart.md)** — fine-tune your first model in 6 steps (no demo notebook needed)

**Ready for advanced techniques?**
- **[DPO_Intel_Orca](Demos/DPO_Intel_Orca/)** - Direct Preference Optimization
- **[RFT_Countdown](Demos/RFT_Countdown/)** - Reinforcement Fine-Tuning

---

## 🎯 Demos

Explore end-to-end fine-tuning experiences in the **[Demos](Demos/)** folder:

| Demo | Technique | Use Case | Difficulty |
|------|-----------|----------|------------|
| [SFT_Bug_Detection](Demos/SFT_Bug_Detection/) | SFT | Code bug detection (beats GPT-5.4 teacher) | ⭐ Beginner |
| [SFT_CNN_DailyMail](Demos/SFT_CNN_DailyMail/) | SFT | News summarization | ⭐ Beginner |
| [SFT_PubMed_Summarization](Demos/SFT_PubMed_Summarization/) | SFT | Medical paper summarization | ⭐ Beginner |
| [DPO_Intel_Orca](Demos/DPO_Intel_Orca/) | DPO | Preference optimization | ⭐⭐ Intermediate |
| [RFT_Countdown](Demos/RFT_Countdown/) | RFT | Math puzzle solving | ⭐⭐ Intermediate |
| [DistillingSarcasm](Demos/DistillingSarcasm/) | Distillation | Knowledge transfer | ⭐⭐ Intermediate |
| [Image_Breed_Classification_FT](Demos/Image_Breed_Classification_FT/) | Vision SFT | Dog breed classification | ⭐⭐ Intermediate |
| [Image_FT_Chart_Analysis](Demos/Image_FT_Chart_Analysis/) | Vision SFT | Chart understanding | ⭐⭐ Intermediate |
| [Video_FT_Action_Recognition](Demos/Video_FT_Action_Recognition/) | Vision SFT | Video action detection | ⭐⭐⭐ Advanced |
| [Zava_ModelRouter_FT](Demos/Zava_ModelRouter_FT/) | Model Router FT | Fine-tune the Model Router on enterprise prompts (GPT-5 / mini / nano as a representative subset) | ⭐⭐ Intermediate |
| [ZavaRetailAgent](Demos/ZavaRetailAgent/) | SFT + RFT | Retail customer service agent | ⭐⭐⭐ Advanced |
| [Agentic_RFT_PrivatePreview](Demos/Agentic_RFT_PrivatePreview/) | RFT | Agentic workflows with tools | ⭐⭐⭐ Advanced |
| [Evaluation](Demos/Evaluation/) | Evaluation | Multimodal model evaluation | ⭐⭐ Intermediate |

👉 See **[Demos/README.md](Demos/README.md)** for detailed descriptions of each demo.

---

## 📊 Sample Datasets

Ready-to-use datasets for testing fine-tuning techniques in the **[Sample_Datasets](Sample_Datasets/)** folder:

| Technique | Dataset | Description |
|-----------|---------|-------------|
| **SFT** | [Text-GSM8K](Sample_Datasets/Supervised_Fine_Tuning/Text-GSM8K/) | Grade school math problems |
| **SFT** | [Text-Bug-Detection](Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/) | Code bug detection and fix suggestions |
| **SFT** | [Multimodal-chartqa](Sample_Datasets/Supervised_Fine_Tuning/Multimodal-chartqa/) | Chart interpretation |
| **SFT** | [Tool-Calling](Sample_Datasets/Supervised_Fine_Tuning/Tool-Calling/) | Function calling patterns |
| **DPO** | [orca_dpo_pairs](Sample_Datasets/Direct_Preference_Optimization/orca_dpo_pairs/) | Preference alignment |
| **RFT** | [ClauseMatching](Sample_Datasets/Reinforcement_Fine_Tuning/ClauseMatching/) | Legal contract analysis |
| **RFT** | [MedMCQ](Sample_Datasets/Reinforcement_Fine_Tuning/MedMCQ/) | Medical Q&A |
| **Model Router FT** | [Zava Enterprise](Sample_Datasets/Model_Router_Fine_Tuning/zava_enterprise/) | Enterprise-operations prompts with per-model correctness labels for `gpt-5` / `gpt-5-mini` / `gpt-5-nano` (representative subset — see the canonical [Model Router supported models list](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router#supported-models)) |

👉 See **[Sample_Datasets/README.md](Sample_Datasets/README.md)** for data format details and when to use each technique.

> ⚠️ **Note**: These datasets are for **learning and experimentation only**—not for production use. Training jobs may incur costs on your Azure subscription.

---

## 🤖 AI Agent Skills

This repo includes a fine-tuning skill that coding agents can auto-discover and use to help you submit, monitor, and evaluate fine-tuning jobs.

| Agent | Skill Path | Auto-discovery |
|-------|-----------|----------------|
| **GitHub Copilot** (VS Code / CLI) | [.github/skills/azure-ai-fine-tuning](.github/skills/azure-ai-fine-tuning) | ✅ Automatic |
| **Claude Code** | [.claude/skills/azure-ai-fine-tuning](.claude/skills/azure-ai-fine-tuning) | ✅ Automatic |
| **Codex / other agents** | [.agents/skills/azure-ai-fine-tuning](.agents/skills/azure-ai-fine-tuning) | ✅ Automatic |

All three paths are symlinks to the canonical skill at **[Skills/](Skills/)**, which includes:
- **SKILL.md** — Agent instructions covering SFT, DPO, and RFT workflows
- **12 scripts** — submit, monitor, calibrate, check, deploy, evaluate, validate, score, convert, generate, cleanup, and shared utilities
- **14 reference docs** — grader design, hyperparameters, dataset formats, agentic RFT, cost management, and more
- **6 guided workflows** — quickstart, full pipeline, dataset creation, iterative training, diagnosis, experiment review
- **Sample data** — SFT, DPO, and RFT example JSONL files

### Using with GitHub Copilot (VS Code)

1. Open this repo in VS Code with Copilot Chat enabled.
2. Ask a fine-tuning task (e.g., *"help me submit an SFT job with my dataset"*).
3. Copilot auto-discovers the skill from `.github/skills/` and follows the workflow.

### Using with Copilot CLI

```bash
cd /path/to/this/repo
copilot
# Then ask: "Submit an SFT fine-tuning job with my training data"
```

### Using with Claude Code

```bash
cd /path/to/this/repo
claude
# Then ask: "Fine-tune gpt-4.1-mini on my dataset"
```

Scripts support `uv` for zero-setup execution (PEP 723 inline dependencies):
```bash
uv run Skills/scripts/submit_training.py --help
```

---

## ✅ Prerequisites

Before running any demo, ensure you have:

- **Azure subscription** with access to [Azure AI Foundry](http://ai.azure.com/)
- **Python 3.9+** installed
- **Jupyter Notebook** or VS Code with Jupyter extension
- Required **Azure role assignments** (see individual demo READMEs)

Each demo includes a `requirements.txt` and `.env.template` for setup.

👉 **New here?** See the **[Getting Started Guide](GETTING_STARTED.md)** for step-by-step setup instructions.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues and pull requests.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
