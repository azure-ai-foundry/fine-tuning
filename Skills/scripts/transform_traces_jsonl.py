# /// script
# dependencies = []
# ///
"""
transform_traces_jsonl.py — Transform Foundry "Traces → SFT" datagen output
into Azure-FT-ready chat JSONL.

Foundry's traces datagen emits `{"messages": [...]}` rows that have multiple
problems for fine-tuning a tool-using model:

  1. **Overlapping conversation snapshots** — Each chatbot LangGraph node
     invocation gets traced separately. The datagen worker stitches all spans
     into the same messages list, so one row contains N overlapping snapshots
     of the same conversation (e.g. [u1, a1] then [u1, a1, u2, a2] then
     [u1, a1, u2, a2, tool, a3]). Dedup with first-occurrence on
     (role, content, tool_call_id, tool_calls) collapses these.

  2. **Fragments** — When the customer simulator hits [END] right after the
     agent asks a clarifying question, you get 2-msg rows (user + assistant
     ask). Useless for tool-use SFT — drop any row that has no assistant
     tool_calls.

  3. **No system prompt** — Foundry traces export does NOT include the
     agent's system message. FT preprocessing requires one for tool-using
     models. You must supply it via --system-prompt-file.

  4. **No tools array** — Same issue. FT preprocessing requires the top-level
     `tools` array. Supply it via --tools-file (OpenAI tool-spec JSON).

  5. **parallel_tool_calls flag** — Some FT pipelines look for this top-level
     boolean; add it.

Reference: the Foundry team's example notebook
https://github.com/william-liang-MSFT/fine-tuning/blob/williamliang/agent-traces-distillation/Demos/AgentTracesDistillation/agent_traces_to_sft.ipynb

Usage:

  python scripts/transform_traces_jsonl.py \
      --jsonl raw_traces_dg.jsonl \
      --system-prompt-file system_prompt.md \
      --tools-file tools.json \
      --out cleaned_sft.jsonl

Output: a JSONL file ready for `openai.fine_tuning.jobs.create(training_file=...)`.

Exit code 0 on success; 1 if 0 valid rows remain.
"""

import argparse
import json
import sys


def first_line(s: str) -> str:
    return (s or "").split("\n", 1)[0].strip()


def dedup_messages(msgs):
    """Collapse overlapping snapshots via first-occurrence dedup."""
    def key(m):
        tcs = m.get("tool_calls") or []
        tc_key = tuple(
            (
                tc.get("id"),
                (tc.get("function") or {}).get("name"),
                (tc.get("function") or {}).get("arguments"),
            )
            for tc in tcs
        )
        return (m.get("role"), m.get("content") or "", m.get("tool_call_id"), tc_key)

    seen = set()
    out = []
    for m in msgs:
        k = key(m)
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    return out, len(msgs) - len(out)


def is_fragment(msgs) -> bool:
    """A row is a fragment if it has no assistant tool_calls (no SFT signal)."""
    return not any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in msgs
    )


def merge_consecutive_asst_tool_calls(msgs) -> int:
    """Merge runs of consecutive assistant messages that all have tool_calls
    into a single assistant message with the tool_calls array combined.

    Azure FT requires tool replies to follow each assistant tool_call turn
    immediately. The Foundry traces export sometimes emits sequences like
    `[..., asst(tc:1), asst(tc:1), tool, tool, ...]` where two separate
    assistant spans each issued one call. The semantically equivalent
    OpenAI-compatible form is one assistant message with parallel tool_calls:
    `[..., asst(tc:[t1,t2]), tool, tool, ...]`. This merges them so the row
    passes preprocessing without losing data.

    Returns the number of messages collapsed.
    """
    out = []
    merged = 0
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            combined = dict(m)
            combined["tool_calls"] = list(m["tool_calls"])
            j = i + 1
            while j < len(msgs) and msgs[j].get("role") == "assistant" and msgs[j].get("tool_calls"):
                combined["tool_calls"].extend(msgs[j]["tool_calls"])
                merged += 1
                j += 1
            out.append(combined)
            i = j
        else:
            out.append(m)
            i += 1
    if merged:
        msgs[:] = out
    return merged


def fix_null_content(msgs) -> int:
    """For assistant tool-call rows: REMOVE the content field entirely.

    Azure FT preprocessing rejects rows where content is present (null or empty)
    alongside tool_calls. The docs example simply omits the content key when
    tool_calls is set. Foundry traces export emits content="null" (string) on
    these rows; we strip the key entirely.
    """
    n = 0
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            if "content" in m:
                m.pop("content")
                n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.strip())
    p.add_argument("--jsonl", required=True, help="Raw Foundry traces datagen JSONL output")
    p.add_argument("--system-prompt-file", required=True,
                   help="Path to a text/markdown file containing the system prompt to prepend to every row")
    p.add_argument("--tools-file", required=True,
                   help="Path to a JSON file containing the OpenAI tool-spec tool definitions (top-level array)")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--parallel-tool-calls", action="store_true", default=True,
                   help="Set parallel_tool_calls=true on each row (default: true). Pass --no-parallel-tool-calls to disable.")
    p.add_argument("--no-parallel-tool-calls", dest="parallel_tool_calls", action="store_false")
    args = p.parse_args()

    with open(args.system_prompt_file, encoding="utf-8") as fh:
        system_prompt = fh.read().strip()
    with open(args.tools_file, encoding="utf-8") as fh:
        tools = json.load(fh)
    if not isinstance(tools, list):
        sys.exit("--tools-file must contain a JSON array of tool definitions")

    print(f"System prompt:  {len(system_prompt):,} chars from {args.system_prompt_file}")
    print(f"Tools:          {len(tools)} definitions from {args.tools_file}")
    print()

    n_in = n_out = n_empty = n_fragment = n_dedup = n_null_fixed = n_merged = 0
    with open(args.jsonl, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            n_in += 1
            try:
                ex = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  line {n_in}: malformed JSON, skipping ({e})", file=sys.stderr)
                continue

            msgs = ex.get("messages", [])
            if not msgs:
                n_empty += 1
                continue

            # Step 1: dedup
            msgs, dropped = dedup_messages(msgs)
            n_dedup += dropped

            # Step 2: strip content from assistant tool-call rows
            n_null_fixed += fix_null_content(msgs)

            # Step 2b: merge consecutive asst tool_call messages into one with
            # parallel tool_calls (Azure FT rejects "asst(tc), asst(tc), tool, tool")
            n_merged += merge_consecutive_asst_tool_calls(msgs)

            # Step 3: drop fragments (no assistant tool_calls)
            if is_fragment(msgs):
                n_fragment += 1
                continue

            # Step 4: replace/prepend system message
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {"role": "system", "content": system_prompt}
            else:
                msgs = [{"role": "system", "content": system_prompt}] + msgs

            # Step 5: build output row with tools + parallel_tool_calls
            out_row = {"messages": msgs, "tools": tools}
            if args.parallel_tool_calls:
                out_row["parallel_tool_calls"] = True
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Summary:")
    print(f"  read:                  {n_in}")
    print(f"  empty (no messages):   {n_empty}")
    print(f"  fragments dropped:     {n_fragment}")
    print(f"  duplicate msgs:        {n_dedup}")
    print(f"  asst-tc rows stripped: {n_null_fixed}")
    print(f"  consecutive asst-tc merged: {n_merged}")
    print(f"  written:               {n_out}")
    print(f"  output:                {args.out}")

    if n_out == 0:
        print("\nWARNING: 0 rows written — check input format and inputs.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
