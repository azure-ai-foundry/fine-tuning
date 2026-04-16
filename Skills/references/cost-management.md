# Cost Management

Fine-tuning has two cost components: a **one-time training cost** and **ongoing hosting/inference costs**. Understanding both helps you budget experiments and avoid surprises.

## Training Costs

### SFT & DPO

Charged by tokens × epochs:

```
Cost = training_tokens × epochs × price_per_token
```

- **Token estimation**: Use [tiktoken](https://github.com/openai/tiktoken), or roughly 1 word ≈ 4 tokens
- Smaller/newer models have lower per-token training prices
- You are NOT charged for: queue time, failed jobs, jobs cancelled before training starts, or data safety checks

### Training Tiers

| Tier | Discount | Data Residency | Notes |
|------|----------|---------------|-------|
| Regional Standard | Baseline | ✅ Guaranteed | Default |
| Global Standard | 10–30% off | ❌ | Good for non-sensitive workloads |
| Developer | 50% off | ❌ | Runs on spot capacity — may be paused/resumed automatically |

Developer tier jobs may take longer but are significantly cheaper for experimentation.

### RFT (Reinforcement Fine-Tuning)

RFT is charged by **time**, not tokens:

```
Cost = training_hours × hourly_rate + grader_token_costs
```

- Hourly rate: ~$100/hour for o4-mini (check pricing page for current rates)
- Model grader tokens (if using `score_model`) billed separately at data zone rates
- **Per-job cap: $5,000** — training pauses and creates a deployable checkpoint. You can review results and decide whether to resume.

**Cost example**: 4 hours of training + o3-mini grader (5M input, 4.9M output tokens) ≈ $427

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

| Model | ~500K tokens, 2 epochs | ~1M tokens, 2 epochs |
|-------|----------------------|---------------------|
| gpt-4.1-mini | ~$2 | ~$4 |
| gpt-4.1 | Higher | Higher |
| o4-mini (RFT, 4 hrs) | ~$400 | ~$400 (time-based) |

*These are illustrative — always check the [Azure pricing page](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service).*

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
