# Traces → SFT distillation

This demo fine-tunes a small student model (`gpt-4.1-nano`) to mimic a larger hosted agent's tool-using behavior, using **real production traces** from your Foundry agent as training data. No labels required.

## What it shows

1. **Pull** real conversation traces from a deployed Foundry agent via the `foundry-traces` recipe of the Data Generation API
2. **Transform** the raw export into Azure-FT-ready JSONL (the helper script `transform_traces_jsonl.py` applies five fixes that the Foundry export currently requires for tool-using agents)
3. **Submit** three fine-tuning candidates against the cleaned traces
4. **Evaluate** each FT'd model with **structural tool-call comparison** (tool name + arguments) — not just text similarity, because for tool-using agents the actual signal is "did it call the right tool with the right args"
5. **Ship** the candidate that beats the un-tuned baseline

## Result on the included Zava retail agent

Pulling 720 hours of real traces from a deployed Zava-style Post-Purchase Resolution Desk agent (gpt-4.1-mini behind it, ~100 conversations across 5 sessions) and distilling into `gpt-4.1-nano`:

| Candidate | Base | Score | Pass rate | Lift vs nano baseline |
|-----------|------|-------|-----------|----------------------|
| Baseline `gpt-4.1-nano` | — | 7.38 | 60% | — |
| Baseline `gpt-4.1-mini` | — | 7.92 | 60% | — |
| `conservative` ← **shipped** | `gpt-4.1-nano` (3ep, lr=1.0) | **8.60** | **100%** | **+16.5%** |
| `high-lr` | `gpt-4.1-nano` (3ep, lr=2.0) | 8.60 | 100% | +16.5% |
| `alt-mini` | `gpt-4.1-mini` (3ep, lr=1.0) | 8.32 | 80% | +5.0% (vs mini) |

The `gpt-4.1-nano` student, fine-tuned on traces from a `gpt-4.1-mini` teacher agent, now matches the teacher's tool selection on **every test row** — at ~10× lower cost per token.

## Prerequisites

- An Azure AI Foundry project with a **deployed hosted agent** that has historical traces in App Insights. If you don't have one yet, you can:
  - Use one of your existing agents (any hosted agent emits traces automatically)
  - Or stand up a new one — see `fixtures/push_prompts.py` for a script that pushes 500-1K diverse retail prompts through any deployed agent to populate trace history
- One **student** model deployment that supports fine-tuning (e.g. `gpt-4.1-nano`, `gpt-4.1-mini`)
- The `microsoft-foundry/fine-tuning` skill checked out locally
- Python 3.11+ with `openai>=2.0`, `azure-ai-projects>=2.2.0`, `azure-identity>=1.21`

## Files in this folder

| File | Purpose |
|------|---------|
| `notebook.ipynb` | End-to-end runnable walkthrough |
| `fixtures/push_prompts.py` | Standalone script that pushes diverse retail prompts through any hosted agent (optional, for populating trace history before the demo) |
| `fixtures/zava_system_prompt.md` | The agent's system prompt — required by `transform_traces_jsonl.py` because Foundry's trace export does not currently emit system messages at the row level |
| `fixtures/zava_tools.json` | The agent's tool catalog (OpenAI chat-completions format) — required by `transform_traces_jsonl.py` for the same reason |

## Run it

```bash
export AZURE_AI_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export OPENAI_BASE_URL="https://<resource>.openai.azure.com/openai/v1"
export AZURE_OPENAI_API_KEY="<key>"
export FINETUNING_SKILL_PATH="/path/to/microsoft-foundry/fine-tuning/Skills"

# (Optional) populate trace history first:
python fixtures/push_prompts.py \
    --agent-name <your-hosted-agent> \
    --agent-version <version> \
    --num-prompts 500

# Wait ~90 seconds for traces to land in App Insights, then:
jupyter notebook notebook.ipynb
```

Full run is ~30-50 minutes depending on FT queue depth.

## Why the transform step exists

The Foundry traces export currently emits chat JSONL with several issues that Azure FT preprocessing rejects:

1. **Overlapping snapshots** — Each LangGraph node invocation gets its own span; the worker stitches them into one messages list, so one row contains N overlapping snapshots of the same conversation.
2. **Fragment rows** — When the customer simulator ends right after the agent asks a clarifying question, you get 2-msg rows with no assistant tool_calls. Useless for tool-use FT.
3. **content="null"** — Assistant tool-call rows have `content` as the literal string `"null"` instead of JSON null. Azure FT rejects rows where content is present alongside tool_calls — the field must be **omitted** entirely.
4. **Consecutive assistant tool_call turns** — Sometimes two adjacent assistant spans each issue one tool call. Azure FT requires tool replies immediately after each assistant turn; the fix is to merge consecutive `asst(tc)` turns into one assistant message with parallel tool_calls.
5. **Missing system prompt + tools array** — The trace export does not currently emit the system message or the tools array at the row level; FT preprocessing for tool-using models needs both.

`transform_traces_jsonl.py` applies all five fixes. The notebook uses it via the autopilot's Phase 2a wiring (`--datagen-backend foundry-traces --traces-system-prompt-file <md> --traces-tools-file <json>`).

These are all upstream bugs in the Foundry traces export; the transform script is the workaround until they're fixed at the source.
