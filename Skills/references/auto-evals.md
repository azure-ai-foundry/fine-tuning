# Auto-Generated Evals (Early Progress Signal)

Azure AI Foundry automatically attaches an **OpenAI Evals** object to every fine-tuning job
and runs it every `eval_interval` steps (default: **every 5 steps**). Each run scores the
validation set with the job's grader, so the pass-rate curve across steps is an **early
signal** of whether the job is learning — visible long before the final `fine_tuned_model`
exists, and available for SFT, DPO, and RFT jobs.

This is complementary to `training-curve-analysis.md`:

| Source | Metric | Best for |
|--------|--------|----------|
| Result CSV (`check_training.py`) | `valid_loss`, token accuracy | Overfitting, checkpoint selection |
| Auto-evals (`analyze_auto_evals.py`) | grader **pass rate** per step | Early "is it learning?" signal, per-sample failure inspection |

## Object Model

```
fine_tuning.job.eval  ->  eval_<id>                 (one eval per job)
  └─ runs (one per eval step)  ->  evalrun_<id>      ("Step 5", "Step 10", ...)
       └─ output_items (one per validation sample)   (per-sample grader score + I/O)
```

## Retrieving via the API

Everything lives under the project `/openai/v1` endpoint with Bearer auth.

| Goal | Endpoint |
|------|----------|
| Find the eval id | `GET /fine_tuning/jobs/{job_id}` → `.eval` |
| Rollup across all steps | `GET /evals/{eval_id}` |
| Per-step pass/fail curve | `GET /evals/{eval_id}/runs?limit=100` → `.data[].result_counts` |
| One step's summary | `GET /evals/{eval_id}/runs/{run_id}` → `.per_testing_criteria_results` |
| Per-sample scores + model output | `GET /evals/{eval_id}/runs/{run_id}/output_items` |

Python SDK (`openai>=1.40`):

```python
job = client.fine_tuning.jobs.retrieve(job_id)
eval_id = job.eval
runs = client.evals.runs.list(eval_id, limit=100)            # one per step
items = client.evals.runs.output_items.list(run_id, eval_id=eval_id, limit=20)
```

Note the argument order: `runs.list(eval_id, ...)` but `output_items.list(run_id, *, eval_id=...)`.

## Using the Script

```bash
# Per-step pass-rate curve + trend diagnosis (the early signal)
python scripts/analyze_auto_evals.py --job-id ftjob-xxx

# Dump every per-sample result for one step to JSONL (inputs, outputs, tool calls, scores)
python scripts/analyze_auto_evals.py --job-id ftjob-xxx --run-id evalrun-yyy --dump results.jsonl
```

`--base-url` / `--api-key` fall back to `OPENAI_BASE_URL` / `AZURE_OPENAI_API_KEY`, or to
Foundry SDK / `DefaultAzureCredential` when no key is given (see `scripts/common.py`).

## Reading the Curve

- **Rising pass rate** → the job is learning. Even a noisy climb (e.g. 1% → 12% → 23%) is a
  good early signal well before the job finishes.
- **Flat near zero** → the grader may be miscalibrated (nothing can pass), the task may be
  too hard, or hyperparameters may be off. Inspect a run's `output_items` to see what the
  model actually produced, then revisit `grader-design.md`.
- **Rising then falling** → possible over-training or reward hacking. Inspect a late run's
  per-sample outputs and compare against `reward-hacking-prevention.md`.
- **Rising then plateauing** → returns are diminishing; more steps may not help.

Because these evals refresh every few steps, you can catch a doomed run (flat/near-zero
pass rate) early and cancel it instead of paying for the full job.

## Gotcha: Large `output_items` Pages Time Out

Each `output_item` carries the full sample — prompt, model output, tool calls, and grader
result — so the payload is large. Requesting `limit=100` on the `output_items` endpoint
routinely hangs past the 120s server timeout and returns nothing. Use a small page size
(the script uses `limit=20`); the SDK auto-paginates as you iterate the cursor. Expect a
full dump of a few hundred samples to take several minutes.
