#!/usr/bin/env python3
"""Quality-filter generated SFT/QnA JSONL.

Use after datagen and before submitting to fine-tune training. Scores each
prompt/response pair on three axes with an LLM judge:
  - non_fragmented:  the response is a complete, coherent answer (not cut off)
  - non_empty:       the response actually answers the prompt (not blank/punt)
  - on_topic:        the response addresses the prompt's subject

Drops rows that score < threshold (default 4 out of 5) on any axis.
Writes a `--drop-out` JSONL with only passing rows, plus a `--report` JSON
with per-row scores and reasons.

Designed for QnA-style datasets where messages = [system?, user, assistant]
with text content. For tool-use rows (assistant has tool_calls, no content),
the script passes them through unchanged — judging tool calls needs different
heuristics and is out of scope here.

Usage:

  python quality_filter.py \\
      --jsonl generated.jsonl \\
      --base-url https://<r>.openai.azure.com/openai/v1 \\
      --api-key $env:AZURE_OPENAI_API_KEY \\
      --judge gpt-4.1-mini \\
      --threshold 4 \\
      --drop-out filtered.jsonl \\
      --report quality_report.json

Exit code:
  0 — all rows passed
  1 — at least one row dropped (filtered.jsonl still written)
  2 — fatal error (no output written)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

JUDGE_PROMPT = """You are a quality reviewer for AI fine-tuning data. Score this prompt/response pair on three axes (1-5 each). Return ONLY a JSON object.

Prompt:
{prompt}

Response:
{response}

Score each axis 1 (worst) to 5 (best):
- non_fragmented: 5 = complete coherent answer; 1 = cut off mid-sentence or truncated
- non_empty: 5 = substantive, actually answers; 1 = blank, "I don't know", or refuses to engage
- on_topic: 5 = directly addresses the prompt; 1 = unrelated, off-topic, or hallucinated

Return JSON like: {{"non_fragmented": 5, "non_empty": 5, "on_topic": 5, "reason": "brief 1-line explanation"}}
"""


def _extract_pair(row: dict) -> tuple[str, str, bool]:
    """Pull (user prompt, assistant text response, is_tool_call_row) from a chat row."""
    msgs = row.get("messages") or []
    user = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
    asst = next((m for m in msgs if m.get("role") == "assistant"), None)
    if asst is None:
        return user, "", False
    text = asst.get("content") or ""
    is_tool_only = bool(asst.get("tool_calls")) and not text
    return user, text, is_tool_only


def _score_row(client, judge: str, prompt: str, response: str) -> dict:
    """Ask the judge to score a single prompt/response pair."""
    full = JUDGE_PROMPT.format(prompt=prompt[:4000], response=response[:4000])
    try:
        resp = client.chat.completions.create(
            model=judge,
            messages=[{"role": "user", "content": full}],
            temperature=0.0,
            max_completion_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            return {"non_fragmented": 0, "non_empty": 0, "on_topic": 0, "reason": "no_json_in_judge_response"}
        scores = json.loads(m.group())
        return {
            "non_fragmented": int(scores.get("non_fragmented", 0)),
            "non_empty": int(scores.get("non_empty", 0)),
            "on_topic": int(scores.get("on_topic", 0)),
            "reason": str(scores.get("reason", ""))[:200],
        }
    except Exception as e:
        return {"non_fragmented": 0, "non_empty": 0, "on_topic": 0, "reason": f"judge_error:{str(e)[:140]}"}


def filter_jsonl(
    jsonl_path: str,
    out_path: str,
    report_path: str | None,
    base_url: str,
    api_key: str,
    judge: str,
    threshold: int,
    concurrency: int = 4,
    progress_every: int = 25,
    project_endpoint: str | None = None,
) -> tuple[int, int, int]:
    """Filter a JSONL file. Returns (n_total, n_kept, n_dropped)."""
    # Use the standard project/Azure-OpenAI client resolution from common.py so
    # the judge endpoint works under all three of: /v1/ project endpoint,
    # plain OpenAI base URL, and classic Azure OpenAI.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import get_clients

    client, _method = get_clients(base_url=base_url, project_endpoint=project_endpoint, api_key=api_key)

    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"  Loaded {len(rows)} rows from {jsonl_path}")
    print(f"  Judge: {judge}  Threshold: {threshold}/5  Concurrency: {concurrency}")

    report: list[dict] = []
    keep_flags: list[bool] = [True] * len(rows)
    passthrough_tool = 0
    scored = 0

    def _job(i: int) -> tuple[int, dict | None]:
        prompt, response, is_tool_only = _extract_pair(rows[i])
        if is_tool_only:
            return i, None  # pass through tool-only rows
        if not prompt or not response:
            return i, {"non_fragmented": 1, "non_empty": 1, "on_topic": 1, "reason": "empty_prompt_or_response"}
        return i, _score_row(client, judge, prompt, response)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_job, i): i for i in range(len(rows))}
        for fut in as_completed(futures):
            i, scores = fut.result()
            if scores is None:
                passthrough_tool += 1
                report.append({"idx": i, "passthrough": "tool_call"})
                continue
            scored += 1
            entry = {"idx": i, **scores}
            report.append(entry)
            min_score = min(scores["non_fragmented"], scores["non_empty"], scores["on_topic"])
            if min_score < threshold:
                keep_flags[i] = False
            if scored % progress_every == 0:
                print(f"  scored {scored}/{len(rows) - passthrough_tool}")

    kept = sum(keep_flags)
    dropped = len(rows) - kept
    print(f"\n  Kept {kept}/{len(rows)} rows  (dropped {dropped}; {passthrough_tool} tool-call rows passed through)")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if keep_flags[i]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {out_path}")

    if report_path:
        report.sort(key=lambda e: e["idx"])
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "input": os.path.abspath(jsonl_path),
                "output": os.path.abspath(out_path),
                "judge": judge,
                "threshold": threshold,
                "n_total": len(rows),
                "n_kept": kept,
                "n_dropped": dropped,
                "n_passthrough_tool": passthrough_tool,
                "rows": report,
            }, f, indent=2)
        print(f"  Wrote {report_path}")

    return len(rows), kept, dropped


def main() -> int:
    p = argparse.ArgumentParser(description="Quality-filter generated SFT/QnA JSONL.")
    p.add_argument("--jsonl", required=True, help="Input JSONL (one chat row per line).")
    p.add_argument("--drop-out", required=True, help="Output JSONL with only passing rows.")
    p.add_argument("--report", default=None, help="Optional per-row report JSON.")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                   help="OpenAI-compatible endpoint base URL.")
    p.add_argument("--project-endpoint", default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"),
                   help="Alternative: Azure AI project endpoint (/v1/-style). Either --base-url or --project-endpoint must resolve a working OpenAI client.")
    p.add_argument("--api-key", default=os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                   help="API key for the judge model endpoint.")
    p.add_argument("--judge", default="gpt-4.1-mini",
                   help="Judge model deployment name (default: gpt-4.1-mini).")
    p.add_argument("--threshold", type=int, default=4,
                   help="Min acceptable score on any axis, 1-5 (default 4).")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Parallel judge requests (default 4).")
    args = p.parse_args()

    if not (args.base_url or args.project_endpoint) or not args.api_key:
        print("ERROR: --base-url or --project-endpoint required, plus --api-key (or set OPENAI_BASE_URL / AZURE_AI_PROJECT_ENDPOINT + AZURE_OPENAI_API_KEY).",
              file=sys.stderr)
        return 2

    try:
        n_total, n_kept, n_dropped = filter_jsonl(
            jsonl_path=args.jsonl,
            out_path=args.drop_out,
            report_path=args.report,
            base_url=args.base_url,
            project_endpoint=args.project_endpoint,
            api_key=args.api_key,
            judge=args.judge,
            threshold=args.threshold,
            concurrency=args.concurrency,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    return 0 if n_dropped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
