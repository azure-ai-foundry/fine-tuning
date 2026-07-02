# /// script
# dependencies = [
#   "openai>=1.40",
#   "azure-identity",
#   "azure-ai-projects",
# ]
# ///
"""
analyze_auto_evals.py — Retrieve and analyze the auto-generated evals attached to a fine-tuning job.

Azure AI Foundry automatically attaches an OpenAI **Evals** object to every fine-tuning
job and runs it every `eval_interval` steps (default: every 5 steps). Each run scores the
validation set with the job's grader, so the pass-rate curve across steps is an **early
signal** of whether the job is learning — long before the final `fine_tuned_model` exists.

Object chain:
    fine_tuning.job.eval  ->  eval_<id>
      └─ runs (one per eval step)  ->  evalrun_<id>
           └─ output_items (one per validation sample)  ->  per-sample grader score

Usage:
  # Per-step pass-rate curve + trend analysis (the early signal):
  python analyze_auto_evals.py --job-id ftjob-abc123

  # Dump every per-sample result for one run to JSONL (inputs, outputs, scores):
  python analyze_auto_evals.py --job-id ftjob-abc123 --run-id evalrun_xyz --dump items.jsonl

Connection (see common.py). Preferred: project /v1/ endpoint + key.
  --base-url  https://<resource>.openai.azure.com/openai/v1   (or OPENAI_BASE_URL)
  --api-key   <key>                                            (or AZURE_OPENAI_API_KEY)
Falls back to Foundry SDK / DefaultAzureCredential when no key is given.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HelpOnErrorParser, get_clients


def _step_num(name):
    """Parse the trailing step number out of a run name like 'Step 115'."""
    try:
        return int(str(name).strip().split()[-1])
    except (ValueError, IndexError):
        return -1


def _get_eval_id(job):
    """The eval id lives on job.eval; fall back to model_extra for SDK version drift."""
    eval_id = getattr(job, "eval", None)
    if not eval_id and getattr(job, "model_extra", None):
        eval_id = job.model_extra.get("eval")
    return eval_id


def curve(client, job_id):
    """Print the per-step pass-rate curve and a short trend diagnosis."""
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Job: {job.id}")
    print(f"  Status: {job.status}")
    print(f"  Fine-tuned model: {job.fine_tuned_model}")

    eval_id = _get_eval_id(job)
    if not eval_id:
        print("\n  No eval attached to this job yet. Auto-evals appear once the first "
              "eval step completes (default: after step 5).")
        return
    print(f"  Eval: {eval_id}")

    # One run per eval step. Iterating the cursor page auto-paginates on long jobs.
    runs = list(client.evals.runs.list(eval_id, limit=100))

    if not runs:
        print("\n  No eval runs yet — check back after the first eval interval.")
        return

    rows = []
    for r in runs:
        rc = r.result_counts
        total = rc.total or 0
        rate = (rc.passed / total * 100) if total else 0.0
        rows.append((_step_num(r.name), r.name, r.id, rc.passed, rc.failed, total, rate, r.status))
    rows.sort(key=lambda x: x[0])

    print(f"\n  Auto-eval pass-rate curve ({len(rows)} runs):")
    print(f"  {'Step':>6} {'Passed':>8} {'Total':>7} {'Pass %':>8}  Run ID")
    print(f"  {'─'*6} {'─'*8} {'─'*7} {'─'*8}  {'─'*44}")
    best = None
    for step, name, rid, passed, failed, total, rate, status in rows:
        marker = ""
        if best is None or rate > best[2]:
            best = (step, name, rate, rid)
        note = "" if status == "completed" else f"  [{status}]"
        print(f"  {step:>6} {passed:>8} {total:>7} {rate:>7.1f}%  {rid}{note}")

    # Trend diagnosis — the "early signal".
    first_rate = rows[0][6]
    last_step, last_name, _, _, _, _, last_rate, _ = rows[-1]
    print(f"\n  First eval (step {rows[0][0]}): {first_rate:.1f}% pass")
    print(f"  Latest eval (step {last_step}): {last_rate:.1f}% pass")
    if best:
        print(f"  Best: {best[1]} — {best[2]:.1f}% pass ({best[3]})")

    delta = last_rate - first_rate
    if len(rows) < 2:
        print("\n  ⏳ Only one eval so far — need at least two to read a trend.")
    elif delta > 5:
        trailing = last_rate - rows[max(0, len(rows) - 4)][6]
        if trailing <= 0.5:
            print(f"\n  ⚠️  Improving overall (+{delta:.1f}%) but PLATEAUING in the last "
                  f"few evals. Consider whether more steps will help.")
        else:
            print(f"\n  ✅ Learning: pass rate up +{delta:.1f}% since the first eval. "
                  f"Good early signal.")
    elif delta < -5:
        print(f"\n  ⚠️  REGRESSING: pass rate down {delta:.1f}% since the first eval. "
              f"Possible over-training or reward hacking — inspect a recent run's "
              f"output_items with --run-id.")
    else:
        print(f"\n  ⚠️  FLAT ({delta:+.1f}%): little movement. The grader may be "
              f"miscalibrated, the task too hard, or hyperparameters off. Inspect "
              f"per-sample results with --run-id, and see references/grader-design.md.")

    print("\n  Inspect any step's per-sample results:")
    print(f"    python analyze_auto_evals.py --job-id {job_id} "
          f"--run-id {best[3] if best else '<evalrun_id>'} --dump results.jsonl")


def dump_items(client, job_id, run_id, out_path):
    """Write every per-sample output_item for one run to JSONL, or preview the first few."""
    job = client.fine_tuning.jobs.retrieve(job_id)
    eval_id = _get_eval_id(job)
    if not eval_id:
        print("No eval attached to this job.")
        return

    # Run-level pass/fail totals are already summarized on the run object — no need to
    # scan every sample just to count them.
    run = client.evals.runs.retrieve(run_id, eval_id=eval_id)
    rc = run.result_counts
    total = rc.total or 0
    rate = (rc.passed / total * 100) if total else 0.0
    print(f"\n  {run.name} ({run_id}): {rc.passed}/{total} passed ({rate:.1f}%)")

    # Preview only: show the first 10 samples and stop (the API is slow per item).
    if not out_path:
        print("\n  First samples (use --dump to write ALL to JSONL):")
        n = 0
        for item in client.evals.runs.output_items.list(run_id, eval_id=eval_id, limit=10):
            n += 1
            rec = item.model_dump()
            res = (rec.get("results") or [{}])[0]
            print(f"    {rec['id']}  passed={res.get('passed')}  score={res.get('score')}")
            if n >= 10:
                break
        return

    # Full dump: iterate all samples. NOTE: output_items carry the full sample (input,
    # model output, tool calls), so the API is slow and *times out* at large page sizes
    # (limit=100 → server-side 120s+ hang). Keep the page size small; the SDK
    # auto-paginates as we iterate the cursor.
    out = open(out_path, "w", encoding="utf-8")
    n = passed = 0
    scores = []
    for item in client.evals.runs.output_items.list(run_id, eval_id=eval_id, limit=20):
        n += 1
        rec = item.model_dump()
        res = (rec.get("results") or [{}])[0]
        if res.get("passed"):
            passed += 1
        if res.get("score") is not None:
            scores.append(res["score"])
        out.write(json.dumps(rec) + "\n")
        if n % 20 == 0:
            print(f"  ...{n} items", flush=True)
    out.close()
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Wrote {n} output items ({passed} passed"
          + (f", avg score {avg:.3f}" if scores else "") + f") to {out_path}")


def main():
    parser = HelpOnErrorParser(description="Analyze the auto-generated evals on a fine-tuning job")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                        help="Project /v1/ URL (preferred)")
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                        help="Azure OpenAI endpoint (fallback)")
    parser.add_argument("--project-endpoint", default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"),
                        help="Azure AI project endpoint (Foundry SDK)")
    parser.add_argument("--api-key", default=os.environ.get("AZURE_OPENAI_API_KEY"))
    parser.add_argument("--job-id", required=True, help="Fine-tuning job ID")
    parser.add_argument("--run-id", help="Dump per-sample output_items for this eval run")
    parser.add_argument("--dump", help="Write output_items to this JSONL path (with --run-id)")
    args = parser.parse_args()

    client, _ = get_clients(
        base_url=args.base_url, azure_endpoint=args.endpoint,
        project_endpoint=args.project_endpoint, api_key=args.api_key,
    )

    if args.run_id:
        dump_items(client, args.job_id, args.run_id, args.dump)
    else:
        curve(client, args.job_id)


if __name__ == "__main__":
    main()
