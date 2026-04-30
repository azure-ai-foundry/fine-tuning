# Cost Management

Fine-tuning has two cost components: a **one-time training cost** and **ongoing hosting/inference costs**. Understanding both helps you budget experiments and avoid surprises.

## Training Costs

### SFT & DPO

Charged by tokens × epochs (per-token billing):

```
Cost = training_tokens × epochs × price_per_token
```

- **Token estimation**: Use [tiktoken](https://github.com/openai/tiktoken), or roughly 1 word ≈ 4 tokens
- Smaller/newer models have lower per-token training prices
- You are NOT charged for: queue time, failed jobs, jobs cancelled before training starts, or data safety checks
- **Exception:** Some models are **RFT-only** and don't support SFT or DPO. These are billed **hourly**, not per-token:
  - `o4-mini` (Azure RFT, ~$100/hour)
  - `gpt-5` (Azure RFT, hourly — verify rate on pricing page)

  See the [RFT section](#rft-reinforcement-fine-tuning) below.

### Training Tiers

| Tier | Cost | Queue Priority | Data Residency | Supported Models |
|------|------|---------------|----------------|-----------------|
| **Developer** | 50% off | Spot capacity — fast when available, may pause/resume | ❌ | OpenAI models only (gpt-4.1-mini, nano, etc.) |
| **Global Standard** | 10–30% off | Priority capacity — reliable scheduling | ❌ | All models (OpenAI + OSS) |
| **Regional Standard** | Baseline | Standard queue | ✅ Guaranteed (NCUS, EUS2, Sweden Central) | OpenAI models only |

### How to pick a tier

```
Start with developerTier (cheapest, often fastest for experiments)
│
├── Job starts quickly? → Great, use it
│
└── Job stuck in queue?
    ├── OpenAI model? → Resubmit on globalStandard (priority capacity)
    └── OSS model? → globalStandard is the only option
│
└── Need data residency (EU/US compliance)?
    └── Use standard (regional) — OpenAI models only
```

**Key rules:**
- **OSS models** (gpt-oss-20b, Ministral-3B, Qwen3-32B, Llama-3.3-70B) only support `globalStandard`
- **o4-mini RFT** only supports `globalStandard` (no developerTier)
- **developerTier** runs on spot capacity — jobs may be paused and resumed automatically. Fine for experiments, not ideal if you need predictable completion time.
- If a standard tier job is stuck ("waiting for jobs ahead"), cancel and resubmit on a different tier — they have separate capacity pools.

### Setting the tier

**Via REST API** (add `trainingType` to the request body):
```python
body = {
    "model": "gpt-4.1-mini",
    "training_file": train_id,
    "validation_file": val_id,
    "hyperparameters": {"n_epochs": 2, "learning_rate_multiplier": 1.0},
    "trainingType": "developerTier",  # or "globalStandard" or "standard"
}
requests.post(f"{endpoint}/openai/fine_tuning/jobs?api-version=2025-04-01-preview",
              headers={"api-key": key}, json=body)
```

**Via SDK** — the SDK does not support `trainingType` directly. Use the REST API for tier selection, or set the default in the Azure portal.

### RFT (Reinforcement Fine-Tuning)

RFT is the **only** fine-tuning method supported for `o4-mini` and `gpt-5` — these models do **not** support SFT or DPO. RFT is charged by **time**, not tokens:

```
Cost = training_hours × hourly_rate + grader_token_costs
```

- Hourly rate examples (verify on pricing page):
  - `o4-mini` RFT: ~$100/hour
  - `gpt-5` RFT: hourly (verify rate on pricing page)
- Model grader tokens (if using `score_model`) billed separately at data zone rates
- **Per-job cap: $5,000** — training pauses and creates a deployable checkpoint. You can review results and decide whether to resume.

**Cost example**: 4 hours of `o4-mini` RFT + `gpt-4o-mini` grader (5M input, 4.9M output tokens) ≈ $407

### RFT Cost Control Strategies

| Strategy | How |
|----------|-----|
| Start small | Use `reasoning_effort: Low`, smaller validation sets |
| Limit validation | Reduce `eval_samples` and validation frequency |
| Choose smallest grader | Use the cheapest model that meets quality needs |
| Tune `compute_multiplier` | Balance convergence speed vs. cost |
| Monitor and cancel | Pause or cancel in the portal/API at any time |

### Job Failures and Cancellations

- **Service errors**: You're not billed for lost work
- **User cancellation**: Charged for work completed up to that point
- **Partial failure**: Only billed up to the last successful checkpoint

## Hosting & Inference Costs

After training, you pay to keep the model deployed:

| Deployment Type | Hosting Fee | Token Rate | Data Residency | Best For |
|----------------|------------|------------|----------------|----------|
| **Standard** | $1.70/hour | Same as base model | ✅ | Production with data residency needs |
| **Global Standard** | $1.70/hour | Same as base model | ❌ | Higher throughput production |
| **Regional Provisioned** | PTU/hour | None (PTU-based) | ✅ | Latency-sensitive workloads |
| **Developer Tier** | Free | Same as Global Standard | ❌ | Evaluation & POC (auto-removed after 24h) |

### Developer Tier for Evaluation

**Use Developer Tier deployments when evaluating model candidates.** No hosting fee, and auto-removed after 24 hours. Perfect for running your eval pipeline without incurring hosting costs.

### Hosting Cost Example

A fine-tuned chatbot handling 10,000 conversations/month:
- Hosting: $1.70/hour × 24h × 30 days = **$1,224**
- Input tokens (20M): 20 × $1.10 = **$22**
- Output tokens (40M): 40 × $4.40 = **$176**
- **Total: ~$1,422/month**

## Cost-Aware Experiment Planning

### Training Budget Rules of Thumb

Approximate **SFT training cost per 1M trained tokens** (where `trained_tokens = dataset_tokens × epochs`). All values below are from the Azure OpenAI pricing page at the **globalStandard** tier. `developerTier` is 50% off globalStandard. `standard` (regional, with data residency) can be 10–30% higher than globalStandard.

**OpenAI models** (support all three tiers):

| Model | globalStandard (baseline) | developerTier (–50%) | standard / regional (+10–30%) |
|-------|---------------------------|----------------------|--------------------------------|
| gpt-4.1-nano  | $1.50/M  | $0.75/M  | ~$1.65–$1.95/M |
| gpt-4.1-mini  | $5.00/M  | $2.50/M  | ~$5.50–$6.50/M |
| gpt-4.1       | $25.00/M | $12.50/M | ~$27.50–$32.50/M |
| gpt-4o-mini   | $3.00/M  | $1.50/M  | ~$3.30–$3.90/M |
| gpt-4o        | $25.00/M | $12.50/M | ~$27.50–$32.50/M |
| o4-mini (RFT only — no SFT/DPO) | $100/hour (time-based) |||
| gpt-5 (RFT only — no SFT/DPO) | hourly (verify rate on pricing page) |||

**OSS / open-weight models** (globalStandard only — no developerTier or regional support):

| Model | Training | Hosting | Inference Input | Inference Output |
|-------|----------|---------|------------------|-------------------|
| Mistral Ministral 3B | $1.00/M  | $0.65/hour | $0.05/M  | $0.15/M  |
| Qwen3 32B            | $3.20/M  | $0.30/hour | $0.30/M  | $1.20/M  |
| Llama 3.3 70B        | $4.50/M  | $0.30/hour | $0.71/M  | $0.71/M  |
| GPT OSS 20B          | $3.60/M  | $0.30/hour | $0.07/M  | $0.30/M  |

**Worked examples** (multiply $/M × trained_tokens / 1M):

| Model | 500K tokens × 2 epochs (developerTier or globalStandard) | 1M tokens × 2 epochs (globalStandard) |
|-------|-----------------------------------------------------------|---------------------------------------|
| gpt-4.1-nano  | ~$0.75 (dev)  | ~$3.00  |
| gpt-4.1-mini  | ~$2.50 (dev)  | ~$10.00 |
| gpt-4.1       | ~$12.50 (dev) | ~$50.00 |
| Ministral 3B  | ~$1.00 (gS)   | ~$2.00  |
| GPT OSS 20B   | ~$3.60 (gS)   | ~$7.20  |
| Qwen3 32B     | ~$3.20 (gS)   | ~$6.40  |
| Llama 3.3 70B | ~$4.50 (gS)   | ~$9.00  |
| o4-mini (RFT) | ~$400 (4 hrs) | ~$400 (time-based) |

> ⚠️ **Always verify against the live [Azure OpenAI pricing page](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service)** before making business decisions. Prices change and OSS-model fine-tuning rates aren't always publicly listed.

The Skills scripts (`scripts/auto_finetune.py`, `scripts/validate/data_stats.py`) source these values from a single table in `scripts/common.py` (`SFT_TRAINING_PRICE_PER_M_TOKENS` and `SFT_TIER_MULTIPLIER`). The code uses a midpoint multiplier (+20%) for the regional tier; update that table to keep all cost estimators in sync.

### Minimizing Experiment Costs

1. **Start with the cheapest model**: gpt-4.1-mini or gpt-4.1-nano for SFT experiments
2. **Use Developer tier training**: 50% discount, fine for experiments
3. **Use Developer tier hosting for eval**: Free hosting, auto-deleted after 24h
4. **Don't leave deployments running**: Delete after evaluation completes
5. **Fewer epochs first**: Start with 1–2 epochs, only increase if underfitting
6. **Smaller dataset first**: Validate your approach on 100 examples before scaling to 1,000+
7. **For RFT**: Start with `reasoning_effort: Low` and small validation sets to estimate time/cost

### The $5,000 RFT Safety Net

RFT jobs are capped at $5,000 per job. When reached:
1. Training pauses automatically
2. A deployable checkpoint is created
3. You can evaluate the checkpoint
4. Resume if needed (no further cap — billing continues)

This means you won't accidentally burn through your budget on a single runaway RFT job.

## Reference

- [Official cost management docs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning-cost-management)
- [Azure OpenAI pricing page](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service)
