# SFT Bug Detection Demo

**Technique**: Supervised Fine-Tuning (SFT) — Distillation  
**Use Case**: Teaching GPT-4.1-mini to identify bugs in code, explain them, and suggest fixes  
**Base Model**: GPT-4.1-mini  
**Teacher Model**: GPT-5.4 (used for data generation and as evaluation judge)  
**Dataset**: 224 training / 20 validation examples across 10 bug categories  

## What You'll Learn

1. **Baseline evaluation** — Measure the base model's bug detection ability before training
2. **Fine-tuning** — Submit an SFT job with optimal hyperparameters (2 epochs, lr=0.8)
3. **Evaluation** — Compare the fine-tuned model against both the base model and the teacher
4. **ROI analysis** — See how the fine-tuned mini model beats the teacher on pass rate while costing 9x less

## Key Results

| Model | Combined Score | Pass Rate | Input $/1M | Output $/1M |
|-------|---------------|-----------|------------|-------------|
| gpt-4.1-mini (base) | 8.87 | 85.7% | $0.40 | $1.60 |
| **gpt-4.1-mini FT** | **9.15** | **96.4%** | **$0.40** | **$1.60** |
| gpt-5.4 (teacher) | 9.29 | 89.3% | $2.50 | $15.00 |

The fine-tuned model **beats the teacher on pass rate** (96.4% vs 89.3%) while costing **9x less** per token.

## Prerequisites

- Azure AI Foundry project with fine-tuning access
- Python 3.9+
- `pip install -r requirements.txt`
- Copy `.env.template` to `.env` and fill in your Azure credentials

## Files

| File | Description |
|------|-------------|
| `Bug_Detection_Fine_Tuning.ipynb` | Main notebook — run cells sequentially |
| `requirements.txt` | Python dependencies |
| `.env.template` | Environment variable template |

Training data is in `../../Sample_Datasets/Supervised_Fine_Tuning/Text-Bug-Detection/`.

## Bug Categories Covered

The dataset covers 10 types of bugs across Python, JavaScript, Java, and C++:

1. Off-by-one errors
2. Null/undefined reference
3. Type mismatches
4. Resource leaks
5. Race conditions
6. Buffer overflows
7. Integer overflow
8. Logic errors
9. Unhandled exceptions
10. Security vulnerabilities (SQL injection, XSS)
