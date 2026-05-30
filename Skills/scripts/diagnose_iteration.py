#!/usr/bin/env python3
"""Deep iteration diagnosis: when the autopilot returns ITERATE, look at the
actual training and test data + per-candidate scoring patterns to suggest a
concrete next-iteration recommendation.

The default `cmd_review` diagnostic is heuristic-only — it looks at training/val
curves and decides things like "Overfitting detected, deploy earlier checkpoint"
or "Regressed by N%, try lower LR". That's useful but doesn't catch deeper
issues like:

  - Training labels are fragmented / off-topic / repetitive
  - The eval rubric judges something different from what the training data teaches
  - Test set is too small / too narrow / dominated by one question type
  - Baseline is already poor because the task is genuinely hard for any small model

This script samples real rows from train.jsonl / val.jsonl / test.jsonl, asks an
LLM judge to assess data quality on three axes (coherence, alignment with task
description, alignment with eval rubric), and combines that with the existing
candidate diagnostics to produce a synthesized root-cause + specific next-step.

Usage:
    python diagnose_iteration.py \\
        --work-dir ./auto_ft_run \\
        --base-url https://<r>.openai.azure.com/openai/v1 \\
        --api-key $env:AZURE_OPENAI_API_KEY \\
        --judge gpt-4.1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


JUDGE_PROMPT = """You are a fine-tuning engineer reviewing why a supervised fine-tuning run failed to beat its baseline. The task description, eval rubric, and sample data are below. Output ONLY a JSON object with your assessment.

## Task description
{task_description}

## Eval rubric (what the judge scores against at eval time)
{eval_rubric}

## Dataset sizes (the FT job actually trained on the full train set, not just the sample below)
{dataset_sizes}

## Sample training data (3 rows sampled randomly from train.jsonl — full set is much larger)
{train_sample}

## Sample test data (3 rows from test.jsonl that the FT'd model needs to do well on)
{test_sample}

## Outcome
- Baseline `{baseline_model}` scored {baseline_combined}/10 on the test set (pass rate {baseline_pass}%)
- Best candidate `{best_candidate}` ({best_model}) scored {best_combined}/10 (lift {best_lift}%)
- Candidate notes: {candidate_notes}

## Your job
Assess THREE things:

1. `data_quality` (1-5): Are the training rows substantive, coherent, on-topic for the task? Flag fragmented responses, refusals, off-topic answers, near-duplicates.
2. `train_test_alignment` (1-5): Do the training rows teach the same skill the test set evaluates? If train has Q&A and test asks for step-by-step reasoning, they're misaligned.
3. `rubric_train_alignment` (1-5): Does the eval rubric reward the behavior that the training labels demonstrate? If rubric prizes terse correctness but training shows verbose explanations, they're misaligned.

Then pick ONE of these as the primary root cause and write a concrete next-iteration recommendation.

Root cause options:
  - "data_quality"            — training data itself is bad
  - "train_test_mismatch"     — train teaches X, test asks for Y
  - "rubric_mismatch"         — eval rubric judges Y, training labels demonstrate X
  - "task_genuinely_hard"     — even the strong baseline scored poorly; small models can't learn this from SFT in one round
  - "needs_more_data"         — quality OK but volume too low for the task complexity
  - "wrong_hps"               — data fine, model fine, but HPs are off (LR too high, too many epochs, etc.)
  - "wrong_base_model"        — student is too small for the task; try a bigger base

Output schema (no markdown, no preamble):
{{
  "data_quality": <int 1-5>,
  "train_test_alignment": <int 1-5>,
  "rubric_train_alignment": <int 1-5>,
  "root_cause": "<one of the options above>",
  "rationale": "<1-3 sentences explaining what you saw and why>",
  "next_step": "<one specific actionable change for the next iteration>"
}}
"""


def _format_row(row: dict, max_chars: int = 600) -> str:
    """Compact representation of a chat row for the judge to read."""
    out_lines = []
    for m in row.get("messages") or []:
        role = m.get("role", "?")
        if m.get("tool_calls"):
            calls = ", ".join(
                f'{(tc.get("function") or {}).get("name", "?")}({(tc.get("function") or {}).get("arguments", "")[:120]})'
                for tc in m["tool_calls"]
            )
            out_lines.append(f"  [{role}] TOOL_CALLS: {calls}")
        else:
            content = (m.get("content") or "").strip()
            if len(content) > max_chars:
                content = content[:max_chars] + "…"
            out_lines.append(f"  [{role}] {content}")
    return "\n".join(out_lines)


def _sample_rows(jsonl_path: Path, n: int, seed: int) -> list[dict]:
    if not jsonl_path.exists():
        return []
    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return []
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def _load_artifacts(work_dir: Path) -> dict:
    """Read the autopilot artifacts; return a dict with what we found."""
    bundle: dict = {}
    for name in ("task_spec.json", "baseline.json", "runs_iter1.json",
                 "leaderboard_iter1.json", "review_iter1.json"):
        p = work_dir / name
        if p.exists():
            bundle[name] = json.loads(p.read_text(encoding="utf-8"))

    # Look for prepared/ subdir first, then fall back to work-dir root
    for sub in ("prepared", "."):
        base = work_dir / sub
        if (base / "train.jsonl").exists():
            bundle["train_path"] = base / "train.jsonl"
            bundle["val_path"] = base / "val.jsonl" if (base / "val.jsonl").exists() else None
            bundle["test_path"] = base / "test.jsonl"
            break

    return bundle


def _count_jsonl_rows(p) -> int:
    if p is None or not p.exists():
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _build_judge_prompt(bundle: dict, sample_seed: int) -> str:
    spec = bundle.get("task_spec.json", {})
    baseline = bundle.get("baseline.json", {})
    review = bundle.get("review_iter1.json", {})

    train_path = bundle.get("train_path")
    val_path = bundle.get("val_path")
    test_path = bundle.get("test_path")
    train_count = _count_jsonl_rows(train_path)
    val_count = _count_jsonl_rows(val_path)
    test_count = _count_jsonl_rows(test_path)

    train_rows = _sample_rows(train_path or Path(""), 3, sample_seed)
    test_rows = _sample_rows(test_path or Path(""), 3, sample_seed + 1)

    rubric = spec.get("eval_rubric", {})
    dims = rubric.get("dimensions", [])
    rubric_text = "\n".join(
        f"  - {d['name']} (weight={d.get('weight', 1.0)}): {d.get('description', '(no description)')}"
        for d in dims
    ) or "  (no dimensions defined)"

    candidates = review.get("candidates", []) if isinstance(review, dict) else []
    if candidates:
        best = max(candidates, key=lambda c: c.get("combined", 0))
        notes = "; ".join(
            f"{c['candidate']}={c.get('combined', 0):.2f}(lift {c.get('lift_pct', 0):+.1f}%, issue={c.get('issue', 'none')})"
            for c in candidates
        )
    else:
        best = {}
        notes = "(no candidates evaluated)"

    dataset_sizes_line = f"  - train.jsonl: {train_count} rows\n  - val.jsonl: {val_count} rows\n  - test.jsonl: {test_count} rows"

    return JUDGE_PROMPT.format(
        task_description=(spec.get("description") or "(none)")[:1200],
        eval_rubric=rubric_text,
        dataset_sizes=dataset_sizes_line,
        train_sample="\n\n".join(_format_row(r) for r in train_rows) or "(no train data found)",
        test_sample="\n\n".join(_format_row(r) for r in test_rows) or "(no test data found)",
        baseline_model=baseline.get("best_model", "?"),
        baseline_combined=f"{baseline.get('combined', 0):.2f}",
        baseline_pass=f"{baseline.get('pass_rate', 0):.1f}",
        best_candidate=best.get("candidate", "?"),
        best_model=best.get("hyperparameters", {}).get("model", "?"),
        best_combined=f"{best.get('combined', 0):.2f}",
        best_lift=f"{best.get('lift_pct', 0):+.1f}",
        candidate_notes=notes,
    )


def diagnose(
    work_dir: Path,
    base_url: str,
    api_key: str,
    judge: str,
    project_endpoint: str | None = None,
    seed: int = 42,
) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import get_clients  # type: ignore

    bundle = _load_artifacts(work_dir)
    if "task_spec.json" not in bundle:
        return {"error": f"No task_spec.json found in {work_dir}"}

    prompt = _build_judge_prompt(bundle, seed)
    client, _ = get_clients(base_url=base_url, project_endpoint=project_endpoint, api_key=api_key)

    print(f"  Diagnosing {work_dir.name} with judge={judge}...")
    try:
        resp = client.chat.completions.create(
            model=judge,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=600,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"error": f"judge call failed: {e}"}

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"error": "judge did not return JSON", "raw": text[:500]}

    try:
        diagnosis = json.loads(match.group())
    except Exception as e:
        return {"error": f"JSON parse failed: {e}", "raw": text[:500]}

    return diagnosis


def render_diagnosis(diagnosis: dict) -> None:
    if "error" in diagnosis:
        print(f"\n  ⚠️  Deep diagnosis unavailable: {diagnosis['error']}")
        if "raw" in diagnosis:
            print(f"     judge raw: {diagnosis['raw']}")
        return

    dq = diagnosis.get("data_quality", 0)
    tt = diagnosis.get("train_test_alignment", 0)
    rt = diagnosis.get("rubric_train_alignment", 0)
    root = diagnosis.get("root_cause", "unknown")
    rationale = diagnosis.get("rationale", "")
    next_step = diagnosis.get("next_step", "")

    print(f"\n  --- Deep Diagnosis ---")
    print(f"    data_quality           {dq}/5")
    print(f"    train_test_alignment   {tt}/5")
    print(f"    rubric_train_alignment {rt}/5")
    print(f"    root_cause:  {root}")
    print(f"    rationale:   {rationale}")
    print(f"    NEXT STEP:   {next_step}")


def main() -> int:
    p = argparse.ArgumentParser(description="Deep iteration diagnosis for autopilot ITERATE outcomes.")
    p.add_argument("--work-dir", required=True, type=Path,
                   help="Autopilot work directory (must contain task_spec.json, baseline.json, review_iter*.json, and prepared/train.jsonl + test.jsonl).")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--project-endpoint", default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"))
    p.add_argument("--api-key", default=os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--judge", default="gpt-4.1",
                   help="Judge model deployment (default: gpt-4.1). A strong model gives better diagnoses than a small one.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional: write the diagnosis JSON to this path.")
    args = p.parse_args()

    if not (args.base_url or args.project_endpoint) or not args.api_key:
        print("ERROR: --base-url (or --project-endpoint) and --api-key required.", file=sys.stderr)
        return 2

    diagnosis = diagnose(
        work_dir=args.work_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        judge=args.judge,
        project_endpoint=args.project_endpoint,
        seed=args.seed,
    )
    render_diagnosis(diagnosis)
    if args.out:
        args.out.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
        print(f"\n  Wrote {args.out}")
    return 0 if "error" not in diagnosis else 1


if __name__ == "__main__":
    sys.exit(main())
