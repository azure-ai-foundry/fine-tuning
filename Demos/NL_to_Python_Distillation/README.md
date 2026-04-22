# NL-to-Python Distillation with NVIDIA Data Designer

> Distill GPT-5.4 into GPT-4.1-mini for text-to-Python code generation — using synthetic data from NVIDIA Data Designer.

## What this demo shows

1. **Synthetic data generation** with [NVIDIA Data Designer](https://developer.nvidia.com/data-designer) — generating diverse, quality-scored (instruction, code) training pairs without manual labeling
2. **Distillation** — training a smaller, cheaper student model (GPT-4.1-mini) to match a larger teacher model (GPT-5.4)
3. **End-to-end fine-tuning pipeline** on Azure AI Foundry — data generation, quality filtering, training, deployment, and evaluation

## Why this matters

Creating training data for code generation is expensive. NVIDIA Data Designer automates it by orchestrating an LLM to generate diverse examples, then scoring and filtering them automatically. The result: high-quality training data at a fraction of the cost of manual curation.

## Results

Using ~85 curated training examples (filtered from 100 generated), 1 epoch of fine-tuning with learning rate multiplier 1.3:

| Model | Correctness | Conciseness | Combined | Pass@8 |
|-------|-------------|-------------|----------|--------|
| **GPT-5.4** (teacher) | 7.5 | 7.0 | 7.2 | 50% |
| **GPT-4.1-mini** (base) | 7.0 | 7.2 | 7.1 | 25% |
| **GPT-4.1-mini** (fine-tuned) | — | — | — | — |

> *Note: The fine-tuned model scores vary by run. In the original 2,000-example run documented in the notebook, the fine-tuned model matched or exceeded GPT-5.4 quality. Results above are from a 100-example test run; scale up `NUM_RECORDS` for best results.*

## Pipeline overview

| Step | What | Tool |
|------|------|------|
| 1 | Setup & configuration | Azure AI Foundry SDK |
| 2 | Generate synthetic training data | NVIDIA Data Designer |
| 3 | Score, filter & split data | Data Designer judges + Python |
| 4 | Baseline evaluation | Azure AI Foundry Evals SDK |
| 5 | Fine-tune GPT-4.1-mini | Azure AI Foundry |
| 6 | Deploy & evaluate fine-tuned model | Azure AI Foundry |
| 7 | Compare results | Python analysis |
| 8 | Analyze training curve | GPT-5.4 as AI research assistant |

## Prerequisites

- An **Azure AI Foundry** resource with `gpt-5.4` and `gpt-4.1-mini` deployed
- TPM quota for both models
- Python 3.11+

## Setup

1. Create and populate environment variables (or a `.env` file):

```properties
AZURE_API_KEY=<YOUR AZURE OPENAI KEY>
AZURE_OPENAI_BASE_URL=https://<YOUR_RESOURCE>.openai.azure.com/openai/v1
AZURE_PROJECT_ENDPOINT=https://<YOUR_RESOURCE>.services.ai.azure.com/api/projects/<YOUR_PROJECT>
AZURE_SUBSCRIPTION_ID=<YOUR SUBSCRIPTION ID>
AZURE_RESOURCE_GROUP=<YOUR RESOURCE GROUP>
AZURE_ACCOUNT_NAME=<YOUR RESOURCE NAME>
```

2. Set up a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Open the notebook:

```bash
code Text_to_Python_Fine_Tuning.ipynb
```

## Key configuration

- **`NUM_RECORDS`**: Number of synthetic examples to generate (default: 2000, set lower for testing)
- **Quality threshold**: Average judge score ≥ 3.0/4.0 to include in training data
- **Hyperparameters**: 1 epoch, learning rate multiplier 1.3, batch size 1

## What Data Designer does

Data Designer uses a pipeline architecture to generate training data:

- **Sampler columns** steer diversity across industries, complexity levels, and coding concepts
- **LLM columns** generate instructions (high temperature) and code solutions (low temperature)
- **Judge columns** score each example on relevance, Pythonic style, readability, and efficiency
- **Validator columns** check Python syntax via AST parsing
- **Schema transform** converts to chat messages JSONL format for fine-tuning

## Files

| File | Description |
|------|-------------|
| `Text_to_Python_Fine_Tuning.ipynb` | Complete end-to-end notebook |
| `requirements.txt` | Python dependencies |
