# Synthetic data generation for document Q&A fine-tuning

This demo fine-tunes a small model on **synthetic Q&A pairs generated from a source document** (e.g. a technical reference PDF) using the Foundry Data Generation API's `SimpleQnA` recipe. The pipeline includes an LLM-judge quality filter that drops fragmented or off-topic generated rows before training.

## What it shows

End-to-end, from a single source document:

1. **Extract** text from a PDF (or use any markdown/plain-text source)
2. **Upload** the text to your Foundry project as `user_data`
3. **Generate** ~2000 Q&A pairs with the `SimpleQnA` recipe (the teacher model writes both the question and the gold answer from the source)
4. **Content-safety pre-screen** (Phase 2b, opt-in) — drop rows that trip Azure FT's data-safety check
5. **Quality filter** (Phase 2c, opt-in) — LLM-judge each row on `non_fragmented` / `non_empty` / `on_topic` and drop rows that score below threshold
6. **Submit** three fine-tuning candidates
7. **Evaluate** each FT'd model on a held-out test split
8. **Ship** the candidate that meets the lift threshold — or surface a clear diagnostic if none do

## Honest result on the included NIST COBOL reference

Distilling 824 pages of the NIST COBOL programming guide into a Q&A model — this is a **hard** scenario for fine-tuning:

| Candidate | Base | Score | Pass rate | Lift |
|-----------|------|-------|-----------|------|
| Baseline `gpt-4.1-mini` | — | 5.26 | 21% | — |
| `alt-mini` | `gpt-4.1-mini` (3ep, lr=1.0) | 5.46 | 21% | -0.9% |
| `conservative` / `high-lr` | nano variants | cancelled (silent hang) | — | — |

**Decision: ITERATE** — no candidate beat the baseline by the +5% threshold.

The autopilot's diagnostic concludes: *"All candidates regressed. Check: (1) are the training labels actually correct? (2) does the eval judge match the training task? (3) try a larger base model."*

**Why this is still a useful demo**: It shows the autopilot's full pipeline including the diagnostic loop. When SFT alone doesn't yield lift, the framework tells you why and proposes the next iteration's HPs/data changes — rather than silently shipping a regression.

**When this scenario works well**: Q&A datasets where the questions exercise patterns the small model can actually learn from supervised training — e.g. domain-specific terminology, output format, fixed response style. Pure factual knowledge tasks like "ask any question about COBOL" are harder because the small model genuinely lacks the knowledge.

## Prerequisites

- An Azure AI Foundry project with:
  - One **teacher** model deployment (e.g. `gpt-4.1`, `gpt-5.4`) — used for both Q&A generation and the quality-filter judge
  - At least one **student** model deployment that supports fine-tuning
- An Azure Content Safety endpoint + key (only if you want the Phase 2b pre-screen)
- The `microsoft-foundry/fine-tuning` skill checked out locally
- Python 3.11+ with `openai>=2.0`, `azure-ai-projects>=2.2.0`, `pypdf`

## Files in this folder

| File | Purpose |
|------|---------|
| `notebook.ipynb` | End-to-end runnable walkthrough |
| `fixtures/sample_source.md` | A small sample document (CC0 COBOL intro from Wikipedia) for quick smoke runs — the real demo uses your own PDF |

## Run it

```bash
export AZURE_AI_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export OPENAI_BASE_URL="https://<resource>.openai.azure.com/openai/v1"
export AZURE_OPENAI_API_KEY="<key>"

# Optional pre-screen:
export AZURE_CONTENT_SAFETY_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
export AZURE_CONTENT_SAFETY_KEY="<key>"

export FINETUNING_SKILL_PATH="/path/to/microsoft-foundry/fine-tuning/Skills"

jupyter notebook notebook.ipynb
```

Full run is ~30-60 minutes (most of it datagen + FT training).

## Bring your own document

Replace `SOURCE_DOC` in the first cell with the path to your PDF or markdown. The notebook handles PDF text extraction automatically. Better results when:
- The source has clear structure (headings, sections) — the teacher uses these to anchor question generation
- The target task is *output-style* (format, terminology, persona) rather than *raw knowledge*
- You have 500+ examples in the final filtered training set
