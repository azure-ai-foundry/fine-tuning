# Synthetic data generation for document Q&A fine-tuning

This demo fine-tunes a small model on **synthetic Q&A pairs generated from a source document** (PDF or markdown) using the Foundry Data Generation API's `SimpleQnA` recipe. The pipeline includes a chunked-datagen workaround for large source docs and an LLM-judge quality filter that drops fragmented or off-topic generated rows before training.

## What it shows

End-to-end, from a single source document:

1. **Extract** text from a PDF (or use any markdown/plain-text source)
2. **Chunk** large sources into pieces and run **parallel datagen jobs** — works around Foundry's per-source saturation (~100–150 unique Q&A pairs per single job regardless of `max_samples`)
3. **Quality filter** — LLM-judge each row on `non_fragmented` / `non_empty` / `on_topic` (1-5) and drop rows that score below threshold
4. **Score** the base model on a held-out test set
5. **Submit** one fine-tuning job (winning hyperparameters: 1 epoch, lr=0.5 — low and slow to prevent overfitting)
6. **Monitor** training to completion
7. **Deploy** the fine-tuned model
8. **Evaluate** it on the same test set and report the lift

Evaluation is driven by the **Foundry evaluations SDK** (`azure-ai-evaluation`) with a custom LLM-judge correctness evaluator (scores 1-10 against the gold reference).

## When this scenario works (and when it doesn't)

Q&A models from a reference document are **harder than they look** for SFT. The questions usually require *factual recall* the small model lacks. If the test asks "what's the implicit default value for HEADING when the PAGE clause omits it?", no amount of SFT will inject that fact into a small model — it would need to look it up.

**SFT shines** when the target task is *output-style*: enforcing a format, terminology, persona, or structured output. **SFT struggles** when the target task is *pure factual knowledge* — for that, RAG (let the small model look up the doc at inference) is usually the better tool.

The notebook reports the honest lift and, if SFT doesn't meet a 5% lift threshold, suggests alternatives (RAG, larger student, different task framing).

## Prerequisites

- An Azure AI Foundry project with:
  - One **teacher** model deployment (e.g. `gpt-4.1`, `gpt-5.4`) — used for both Q&A generation and the quality-filter / correctness judge
  - One **student** model deployment that supports fine-tuning (e.g. `gpt-4.1-mini`)
- Azure CLI (`az login`) for authentication and FT model deployment
- Python 3.11+ with:

```bash
pip install openai>=2.0 azure-ai-projects>=2.2.0 azure-identity>=1.21 azure-ai-evaluation>=1.0 pypdf>=4.0
```

## Files in this folder

| File | Purpose |
|------|---------|
| `notebook.ipynb` | End-to-end runnable walkthrough — **fully self-contained**, no external scripts required |
| `fixtures/sample_source.md` | A small sample document (CC0 COBOL intro from Wikipedia) for quick smoke runs — the real demo uses your own PDF |

## Run it

```bash
export AZURE_AI_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export OPENAI_BASE_URL="https://<resource>.openai.azure.com/openai/v1"
export AZURE_OPENAI_API_KEY="<key>"
export AZURE_SUBSCRIPTION_ID="<subscription-id>"
export AZURE_RESOURCE_GROUP="<resource-group>"

jupyter notebook notebook.ipynb
```

Full run is ~30–60 minutes (most of it datagen + FT training).

## Bring your own document

Replace `SOURCE_DOC` in the first cell with the path to your PDF or markdown. The notebook handles PDF text extraction automatically and auto-scales the chunking (1 chunk per ~150KB).

### Recommended teacher TPM headroom

Foundry datagen reads each chunk into the teacher's context many times per generated Q&A pair. For parallel chunked datagen:
- 200K TPM teacher → concurrency=1, chunks ≤100KB
- 500K TPM teacher → concurrency=2, chunks ≤150KB (the notebook's default)
- 1M+ TPM teacher → concurrency=3–5, chunks ≤200KB

If you hit rate limits, lower `concurrency` or shrink `n_chunks` in the notebook.
