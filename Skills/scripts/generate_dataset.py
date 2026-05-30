# /// script
# dependencies = [
#   "openai>=1.0",
#   "requests",
#   "azure-identity",
#   "azure-ai-projects>=2.2.0",
# ]
# ///
"""
generate_dataset.py — Generate fine-tuning or evaluation data via the
Foundry Data Generation API.

Wraps `project_client.beta.datasets.create_generation_job(...)` with a CLI
front end. Supports four sources (traces, prompt, agent, file), three recipes
(traces, qna, tool-use), and three scenarios (sft, rft, eval).

See:
  - Skills/references/data-generation-api.md  (full API surface)
  - Skills/workflows/traces-to-dataset.md     (traces → SFT/RFT/eval)
  - Skills/workflows/synthetic-datagen.md     (corpus / agent-spec → data)

Usage:
  # SFT from a corpus file you uploaded
  python generate_dataset.py \\
      --project-endpoint $env:AZURE_AI_PROJECT_ENDPOINT \\
      --source file --file-id file-abc123 \\
      --recipe qna --scenario sft \\
      --max-samples 15 --train-split 0.8 --teacher gpt-4.1-mini --download

  # Q&A from a doc file (auto-uploads as user_data under the hood)
  python generate_dataset.py \\
      --project-endpoint $env:AZURE_AI_PROJECT_ENDPOINT \\
      --source prompt-file --prompt-file refund_policy.md \\
      --recipe qna --scenario sft \\
      --max-samples 15 --train-split 0.8 --teacher gpt-4.1-mini --download

  # SFT from traces, last 24h
  python generate_dataset.py \\
      --project-endpoint $env:AZURE_AI_PROJECT_ENDPOINT \\
      --source traces --agent-name retail-agent --agent-version 3 \\
      --recipe traces --scenario sft \\
      --max-samples 200 --train-split 0.8 --hours 24 --download

  # Tool-use SFT from an OpenAPI 3.0 tool spec (uploaded as a .json file)
  python generate_dataset.py \\
      --project-endpoint $env:AZURE_AI_PROJECT_ENDPOINT \\
      --source file --file-id file-openapi123 \\
      --recipe tool-use --scenario sft \\
      --max-samples 20 --train-split 0.8 --teacher gpt-4.1-mini --download
  # NOTE: the uploaded .json MUST be a valid OpenAPI 3.0.x or 3.1.x spec —
  # NOT the OpenAI chat-completions tool format. Use
  # --tools-to-openapi-out openapi.json --tools-from openai_tools.json to
  # convert OpenAI tool-spec JSON to OpenAPI 3.0.

  # Convert an OpenAI-format tool catalog into OpenAPI 3.0 (no job submitted)
  python generate_dataset.py \\
      --tools-from openai_tools.json --tools-to-openapi-out openapi.json

  # Force REST API instead of the SDK
  python generate_dataset.py \\
      --project-endpoint $env:AZURE_AI_PROJECT_ENDPOINT \\
      --source file --file-id file-abc123 \\
      --recipe qna --scenario sft \\
      --max-samples 15 --train-split 0.8 --teacher gpt-4.1-mini \\
      --use-rest --download

Constraints enforced by the service:
- max_samples: 15-1000
- output name: <=50 characters (auto-generated if --output-name omitted)
- SimpleQnA / ToolUseFineTuning REQUIRE --teacher (a deployed model name)
- Tool-use recipe is SFT-only
- File source content should be >=1 KB and substantive enough to QA from
"""

import argparse
import io
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

# Force UTF-8 on stdout/stderr for Windows console emoji output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HelpOnErrorParser  # noqa: E402

import requests  # noqa: E402

# REST defaults for the Foundry Data Generation API.
REST_API_VERSION = "v1"
REST_AAD_SCOPE = "https://ai.azure.com/.default"
REST_FOUNDRY_FEATURES = "DataGenerationJobs=V1Preview"
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

# AAD credential is cached: DefaultAzureCredential is expensive to instantiate,
# and the underlying token cache transparently refreshes near expiry.
_CRED = None


def _credential():
    global _CRED
    if _CRED is None:
        from azure.identity import DefaultAzureCredential
        _CRED = DefaultAzureCredential()
    return _CRED


def _aad_token():
    # get_token() returns a cached token until ~5 min before expiry, so calling
    # this on every poll is safe — refresh happens transparently when needed.
    return _credential().get_token(REST_AAD_SCOPE).token


def _make_output_name(args):
    if args.output_name:
        if len(args.output_name) > 50:
            sys.exit(f"--output-name `{args.output_name}` exceeds 50-char service limit")
        return args.output_name
    run_id = f"{datetime.now(tz=timezone.utc).strftime('%y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
    src_short = {"prompt-inline": "pi", "prompt-file": "pf", "file": "fl",
                 "traces": "tr", "agent": "ag"}.get(args.source, args.source[:3])
    rec_short = {"traces": "tr", "qna": "qna", "tool-use": "tu"}.get(args.recipe, args.recipe[:3])
    scen_short = {"sft": "sft", "rft": "rft", "eval": "ev"}.get(args.scenario, args.scenario[:3])
    name = f"dg-{src_short}-{rec_short}-{scen_short}-{run_id}"
    return name[:50]


def _upload_inline(aoai, text, name_hint):
    print(f"📤 Uploading inline text as {name_hint} ({len(text)} bytes)")
    seed = aoai.files.create(file=(name_hint, io.BytesIO(text.encode("utf-8"))), purpose="user_data")
    for _ in range(60):
        seed = aoai.files.retrieve(file_id=seed.id)
        if seed.status in ("processed", "error"):
            break
        time.sleep(2)
    if seed.status != "processed":
        sys.exit(f"inline upload failed: status={seed.status}")
    print(f"   file_id={seed.id} bytes={seed.bytes}")
    return seed.id


def _normalize_iso(s):
    """Normalize a user-supplied ISO 8601 timestamp to UTC `Z` form."""
    if not s:
        return s
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return s  # let the service tell us it's malformed


# ── OpenAI tool-spec → OpenAPI 3.0 converter ────────────────────────────

def convert_openai_tools_to_openapi(src_path: str, dst_path: str) -> None:
    """Convert an OpenAI chat-completions tool catalog JSON into an OpenAPI 3.0 spec.

    The Data Generation API's tool-use recipe requires a `.json` file containing
    a valid OpenAPI 3.0.x or 3.1.x spec — NOT the OpenAI tool format
    (`[{"type":"function","function":{...}}]`). Use this helper to convert.

    Each function becomes `POST /<operationId>` with `requestBody.schema` taken
    from `function.parameters`. Tools with no parameters get no request body
    (an empty `{}` schema is rejected by the OpenAPI validator).
    """
    with open(src_path, encoding="utf-8") as fh:
        tools = json.load(fh)

    if not isinstance(tools, list):
        sys.exit(f"--tools-from must be a JSON array of {{type:'function',function:{{...}}}} objects; got {type(tools).__name__}")

    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": os.path.splitext(os.path.basename(src_path))[0] + " (converted)",
            "version": "1.0.0",
            "description": "Tool catalog (converted from OpenAI chat-completions tool format).",
        },
        "servers": [{"url": "https://example.invalid"}],
        "paths": {},
    }

    ops = 0
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        props = params.get("properties") or {}
        required_list = params.get("required") or []
        op = {
            "operationId": name,
            "summary": (fn.get("description") or name)[:120],
            "description": fn.get("description") or name,
            "responses": {
                "200": {
                    "description": "OK",
                    "content": {"application/json": {"schema": {"type": "object"}}},
                },
            },
        }
        if props:
            schema = {"type": "object", "properties": props}
            if required_list:
                schema["required"] = required_list
            op["requestBody"] = {
                "required": bool(required_list),
                "content": {"application/json": {"schema": schema}},
            }
        spec["paths"][f"/{name}"] = {"post": op}
        ops += 1

    with open(dst_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"Wrote {dst_path} with {ops} operations.")


# ── SDK source/options builders ─────────────────────────────────────────

def _build_source_sdk(args, aoai):
    from azure.ai.projects.models import (
        AgentDataGenerationJobSource,
        FileDataGenerationJobSource,
        PromptDataGenerationJobSource,
        TracesDataGenerationJobSource,
    )

    if args.source == "traces":
        if not args.agent_name:
            sys.exit("--agent-name required for --source traces")
        kwargs = {"agent_name": args.agent_name}
        if args.agent_version:
            kwargs["agent_version"] = args.agent_version
        if args.description:
            kwargs["description"] = args.description
        if args.hours is not None:
            end = datetime.now(timezone.utc)
            kwargs["end_time"] = end
            kwargs["start_time"] = end - timedelta(hours=args.hours)
        if args.start_time:
            kwargs["start_time"] = datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
        if args.end_time:
            kwargs["end_time"] = datetime.fromisoformat(args.end_time.replace("Z", "+00:00"))
        return TracesDataGenerationJobSource(**kwargs)

    if args.source == "prompt-inline":
        if not args.prompt:
            sys.exit("--prompt required for --source prompt-inline")
        kwargs = {"prompt": args.prompt}
        if args.description:
            kwargs["description"] = args.description
        return PromptDataGenerationJobSource(**kwargs)

    if args.source == "prompt-file":
        if not args.prompt_file:
            sys.exit("--prompt-file required for --source prompt-file")
        text = open(args.prompt_file, encoding="utf-8").read()
        file_id = _upload_inline(aoai, text, os.path.basename(args.prompt_file))
        kwargs = {"id": file_id}
        if args.description:
            kwargs["description"] = args.description
        return FileDataGenerationJobSource(**kwargs)

    if args.source == "file":
        if not args.file_id:
            sys.exit("--file-id required for --source file")
        kwargs = {"id": args.file_id}
        if args.description:
            kwargs["description"] = args.description
        return FileDataGenerationJobSource(**kwargs)

    if args.source == "agent":
        if not args.agent_name:
            sys.exit("--agent-name required for --source agent")
        kwargs = {"agent_name": args.agent_name}
        if args.agent_version:
            kwargs["agent_version"] = args.agent_version
        if args.description:
            kwargs["description"] = args.description
        return AgentDataGenerationJobSource(**kwargs)

    sys.exit(f"Unknown --source: {args.source}")


def _build_options_sdk(args):
    from azure.ai.projects.models import (
        DataGenerationModelOptions,
        SimpleQnADataGenerationJobOptions,
        ToolUseFineTuningDataGenerationJobOptions,
        TracesDataGenerationJobOptions,
    )

    common = {"max_samples": args.max_samples}
    if args.train_split is not None:
        common["train_split"] = args.train_split

    if args.recipe == "traces":
        # IMPORTANT: TracesDataGenerationJobOptions does NOT accept model_options.
        # The service replies 400 "Model options parameter is not applicable for
        # traces data generation type." Traces are real conversations — there is
        # no teacher model to invoke.
        if args.teacher:
            print(
                f"⚠️  --teacher {args.teacher!r} ignored: traces recipe doesn't use a teacher (the assistant response in each trace IS the teacher answer).",
                file=sys.stderr,
            )
        return TracesDataGenerationJobOptions(**common)

    if not args.teacher:
        sys.exit(f"--teacher required for --recipe {args.recipe}")
    common["model_options"] = DataGenerationModelOptions(model=args.teacher)

    if args.recipe == "qna":
        return SimpleQnADataGenerationJobOptions(**common)
    if args.recipe == "tool-use":
        if args.scenario != "sft":
            sys.exit("--recipe tool-use is SFT-only; use --scenario sft")
        return ToolUseFineTuningDataGenerationJobOptions(**common)
    sys.exit(f"Unknown --recipe: {args.recipe}")


def _scenario_sdk(args):
    from azure.ai.projects.models import DataGenerationJobScenario
    return {
        "sft": DataGenerationJobScenario.SUPERVISED_FINETUNING,
        "rft": DataGenerationJobScenario.REINFORCEMENT_FINETUNING,
        "eval": DataGenerationJobScenario.EVALUATION,
    }[args.scenario]


# ── SDK path ────────────────────────────────────────────────────────────

def run_sdk(args, output_name):
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import (
        DataGenerationJob,
        DataGenerationJobInputs,
        DataGenerationJobOutputOptions,
        DatasetDataGenerationJobOutput,
        FileDataGenerationJobOutput,
        JobStatus,
    )
    from azure.identity import DefaultAzureCredential

    with (
        DefaultAzureCredential() as cred,
        AIProjectClient(endpoint=args.project_endpoint, credential=cred) as pc,
        pc.get_openai_client() as aoai,
    ):
        source = _build_source_sdk(args, aoai)
        options = _build_options_sdk(args)
        scenario = _scenario_sdk(args)

        request = DataGenerationJob(inputs=DataGenerationJobInputs(
            name=output_name,
            scenario=scenario,
            sources=[source],
            options=options,
            output_options=DataGenerationJobOutputOptions(name=output_name),
        ))

        print(f"📤 Submitting '{output_name}' (source={args.source}, recipe={args.recipe}, scenario={args.scenario})")
        job = pc.beta.datasets.create_generation_job(job=request)
        print(f"   job.id = {job.id}  status = {job.status}")

        TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
        last = None
        print(f"   Polling every {args.poll_interval}s.")
        for i in range(args.max_polls):
            time.sleep(args.poll_interval)
            try:
                job = pc.beta.datasets.get_generation_job(job_id=job.id)
            except Exception as e:
                print(f"   ⚠️  poll error: {e}")
                continue
            if job.status != last:
                print(f"   t+{(i+1)*args.poll_interval}s  status={job.status}")
                last = job.status
            if job.status in TERMINAL:
                break

        if job.status != JobStatus.SUCCEEDED:
            err = job.error.message if job.error else "<no error>"
            print(f"❌ job ended in {job.status}: {err}", file=sys.stderr)
            sys.exit(1)

        samples = job.result.generated_samples if job.result else None
        print(f"✅ Generated {samples} samples")

        # Discover outputs and cache file metadata so download doesn't re-fetch.
        outputs_summary = []
        file_outputs = []
        for o in (job.result.outputs if job.result else None) or []:
            if isinstance(o, FileDataGenerationJobOutput):
                info = aoai.files.retrieve(file_id=o.id)
                file_outputs.append((o, info))
                idx = len(file_outputs)
                if args.train_split is not None and len(file_outputs) <= 2:
                    role = "train" if idx == 1 else "validation"
                else:
                    role = "combined"
                print(f"   📄 [{role}] file_id={o.id}  filename={info.filename}  bytes={info.bytes}")
                outputs_summary.append({"type": "file", "id": o.id, "filename": info.filename, "bytes": info.bytes})
            elif isinstance(o, DatasetDataGenerationJobOutput):
                print(f"   📚 dataset  name={o.name}  version={o.version}")
                outputs_summary.append({"type": "dataset", "name": o.name, "version": o.version})

        if args.download:
            for o, info in file_outputs:
                local_name = info.filename or f"{output_name}_{o.id}.jsonl"
                with open(local_name, "wb") as f:
                    f.write(aoai.files.content(o.id).content)
                print(f"   💾 saved {local_name}")

        print("\n" + json.dumps({
            "job_id": job.id,
            "status": str(job.status),
            "generated_samples": samples,
            "outputs": outputs_summary,
        }))


# ── REST path ───────────────────────────────────────────────────────────

def _build_source_rest(args, file_id_override=None):
    if args.source == "traces":
        if not args.agent_name:
            sys.exit("--agent-name required for --source traces")
        d = {"type": "traces", "agent_name": args.agent_name}
        if args.agent_version:
            d["agent_version"] = args.agent_version
        if args.description:
            d["description"] = args.description
        if args.hours is not None:
            end = datetime.now(timezone.utc)
            d["start_time"] = (end - timedelta(hours=args.hours)).isoformat().replace("+00:00", "Z")
            d["end_time"] = end.isoformat().replace("+00:00", "Z")
        if args.start_time:
            d["start_time"] = _normalize_iso(args.start_time)
        if args.end_time:
            d["end_time"] = _normalize_iso(args.end_time)
        return d

    if args.source == "prompt-inline":
        if not args.prompt:
            sys.exit("--prompt required for --source prompt-inline")
        d = {"type": "prompt", "prompt": args.prompt}
        if args.description:
            d["description"] = args.description
        return d

    if args.source in ("prompt-file", "file"):
        fid = file_id_override or args.file_id
        if not fid:
            sys.exit(f"file id required for --source {args.source}")
        d = {"type": "file", "id": fid}
        if args.description:
            d["description"] = args.description
        return d

    if args.source == "agent":
        if not args.agent_name:
            sys.exit("--agent-name required for --source agent")
        d = {"type": "agent", "agent_name": args.agent_name}
        if args.agent_version:
            d["agent_version"] = args.agent_version
        if args.description:
            d["description"] = args.description
        return d

    sys.exit(f"Unknown --source: {args.source}")


def _build_options_rest(args):
    type_map = {"traces": "traces", "qna": "simple_qna", "tool-use": "tool_use_fine_tuning"}
    if args.recipe not in type_map:
        sys.exit(f"Unknown --recipe: {args.recipe}")
    if args.recipe == "tool-use" and args.scenario != "sft":
        sys.exit("--recipe tool-use is SFT-only; use --scenario sft")
    if args.recipe in ("qna", "tool-use") and not args.teacher:
        sys.exit(f"--teacher required for --recipe {args.recipe}")

    d = {"type": type_map[args.recipe], "max_samples": args.max_samples}
    if args.train_split is not None:
        d["train_split"] = args.train_split
    if args.teacher and args.recipe != "traces":
        # Same constraint as SDK path: traces recipe rejects model_options
        # ("Model options parameter is not applicable for traces data generation type")
        d["model_options"] = {"model": args.teacher}
    elif args.teacher and args.recipe == "traces":
        print(
            f"⚠️  --teacher {args.teacher!r} ignored: traces recipe doesn't use a teacher (the assistant response in each trace IS the teacher answer).",
            file=sys.stderr,
        )
    return d


def _rest_headers():
    return {
        "Authorization": f"Bearer {_aad_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Foundry-Features": REST_FOUNDRY_FEATURES,
    }


def run_rest(args, output_name):
    seed_id = None
    if args.source == "prompt-file":
        from azure.identity import DefaultAzureCredential
        from azure.ai.projects import AIProjectClient
        with (
            DefaultAzureCredential() as cred,
            AIProjectClient(endpoint=args.project_endpoint, credential=cred) as pc,
            pc.get_openai_client() as aoai,
        ):
            seed_id = _upload_inline(
                aoai,
                open(args.prompt_file, encoding="utf-8").read(),
                os.path.basename(args.prompt_file),
            )

    source = _build_source_rest(args, file_id_override=seed_id)
    options = _build_options_rest(args)
    scenario = {"sft": "supervised_finetuning", "rft": "reinforcement_finetuning",
                "eval": "evaluation"}[args.scenario]

    body = {
        "inputs": {
            "name": output_name,
            "scenario": scenario,
            "sources": [source],
            "options": options,
            "output_options": {"name": output_name},
        }
    }

    base = args.project_endpoint.rstrip("/") + "/data_generation_jobs"
    print(f"📤 [REST] Submitting '{output_name}' (source={args.source}, recipe={args.recipe}, scenario={args.scenario})")
    r = requests.post(base, params={"api-version": REST_API_VERSION},
                      headers=_rest_headers(), json=body, timeout=(10, 60))
    if r.status_code != 201:
        print(f"❌ submit failed ({r.status_code}): {r.text[:1000]}", file=sys.stderr)
        sys.exit(1)
    job = r.json()
    job_id = job["id"]
    print(f"   job.id = {job_id}")

    get_url = f"{base}/{job_id}"
    last = None
    status = None
    print(f"   Polling every {args.poll_interval}s.")
    for i in range(args.max_polls):
        time.sleep(args.poll_interval)
        try:
            r = requests.get(get_url, params={"api-version": REST_API_VERSION},
                             headers=_rest_headers(), timeout=(10, 60))
            r.raise_for_status()
            job = r.json()
        except Exception as e:
            print(f"   ⚠️  poll error: {e}")
            continue
        status = (job.get("status") or "").lower()
        if status != last:
            print(f"   t+{(i+1)*args.poll_interval}s  status={status}")
            last = status
        if status in TERMINAL_STATUSES:
            break

    if status != "succeeded":
        err = (job.get("error") or {}).get("message") or "<no error>"
        print(f"❌ job ended in {status}: {err}", file=sys.stderr)
        sys.exit(1)

    result = job.get("result") or {}
    samples = result.get("generated_samples")
    print(f"✅ Generated {samples} samples")

    outputs_summary = []
    file_outputs = []
    for o in result.get("outputs", []):
        t = (o.get("type") or "").lower()
        if t == "file":
            file_outputs.append(o)
            print(f"   📄 file_id={o.get('id')}  filename={o.get('filename')}")
            outputs_summary.append({"type": "file", "id": o.get("id"), "filename": o.get("filename")})
        elif t == "dataset":
            print(f"   📚 dataset name={o.get('name')} version={o.get('version')}")
            outputs_summary.append({"type": "dataset", "name": o.get("name"), "version": o.get("version")})

    if args.download and file_outputs:
        from azure.identity import DefaultAzureCredential
        from azure.ai.projects import AIProjectClient
        with (
            DefaultAzureCredential() as cred,
            AIProjectClient(endpoint=args.project_endpoint, credential=cred) as pc,
            pc.get_openai_client() as aoai,
        ):
            for o in file_outputs:
                info = aoai.files.retrieve(file_id=o["id"])
                local_name = info.filename or f"{output_name}_{o['id']}.jsonl"
                with open(local_name, "wb") as f:
                    f.write(aoai.files.content(o["id"]).content)
                print(f"   💾 saved {local_name}")

    print("\n" + json.dumps({
        "job_id": job_id,
        "status": status,
        "generated_samples": samples,
        "outputs": outputs_summary,
    }))


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    p = HelpOnErrorParser(
        description="Generate fine-tuning or evaluation data via the Foundry Data Generation API."
    )

    p.add_argument("--project-endpoint", default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"),
                   help="https://<resource>.services.ai.azure.com/api/projects/<project>")
    p.add_argument("--output-name", default=None,
                   help="Output dataset/file name prefix (<=50 chars; auto-generated if omitted)")
    p.add_argument("--use-rest", action="store_true",
                   help="Use REST API instead of the SDK")

    p.add_argument("--source", default=None,
                   choices=["traces", "prompt-inline", "prompt-file", "file", "agent"],
                   help="Where the raw material comes from. Required unless using --tools-from.")
    p.add_argument("--agent-name", help="Deployed agent name (for traces, agent sources)")
    p.add_argument("--agent-version", help="Pin to a specific agent version (recommended for traces)")
    p.add_argument("--hours", type=int, default=None,
                   help="For traces: pull spans from the last N hours (now-hours .. now)")
    p.add_argument("--start-time", help="ISO 8601 UTC start time for traces (e.g. 2026-05-28T10:00:00Z)")
    p.add_argument("--end-time", help="ISO 8601 UTC end time for traces")
    p.add_argument("--prompt", help="Inline prompt text (for --source prompt-inline)")
    p.add_argument("--prompt-file", help="Path to a text file (for --source prompt-file). Uploaded as a file under the hood for better service errors.")
    p.add_argument("--description", help="Optional source description metadata")
    p.add_argument("--file-id", help="OpenAI file id (for --source file)")

    p.add_argument("--recipe", default=None, choices=["traces", "qna", "tool-use"],
                   help="Generation recipe (options class). Required unless using --tools-from.")
    p.add_argument("--scenario", default=None, choices=["sft", "rft", "eval"],
                   help="What the data is for: SFT, RFT, or evaluation. Required unless using --tools-from.")

    p.add_argument("--max-samples", type=int, default=100,
                   help="Samples to produce (15-1000 enforced by service; default 100)")
    p.add_argument("--train-split", type=float, default=None,
                   help="If set, splits into train/validation files (e.g. 0.8)")
    p.add_argument("--teacher", help="Teacher model deployment name (required for qna/tool-use)")

    p.add_argument("--poll-interval", type=int, default=10,
                   help="Seconds between polls (default 10)")
    p.add_argument("--max-polls", type=int, default=360,
                   help="Maximum number of poll iterations (default 360 = 1h at 10s)")
    p.add_argument("--download", action="store_true",
                   help="Download the produced JSONL file(s) to the cwd")

    p.add_argument("--tools-from", default=None,
                   help="Convert an OpenAI tool-spec JSON file (chat-completions format) to an OpenAPI 3.0 spec and write to --tools-to-openapi-out. No job is submitted. Use with --tools-to-openapi-out.")
    p.add_argument("--tools-to-openapi-out", default=None,
                   help="Output path for the OpenAPI 3.0 spec produced from --tools-from.")

    args = p.parse_args()

    # Tool-spec conversion is a standalone mode — no job submission
    if args.tools_from or args.tools_to_openapi_out:
        if not (args.tools_from and args.tools_to_openapi_out):
            p.error("--tools-from and --tools-to-openapi-out must be used together")
        convert_openai_tools_to_openapi(args.tools_from, args.tools_to_openapi_out)
        return

    if not args.project_endpoint:
        p.error("--project-endpoint required (or set AZURE_AI_PROJECT_ENDPOINT)")
    if not args.source:
        p.error("--source is required (unless using --tools-from)")
    if not args.recipe:
        p.error("--recipe is required (unless using --tools-from)")
    if not args.scenario:
        p.error("--scenario is required (unless using --tools-from)")
    if args.recipe == "tool-use" and args.scenario != "sft":
        p.error("--recipe tool-use is SFT-only; use --scenario sft")
    if args.recipe == "tool-use" and args.source not in ("file", "traces"):
        # Service requires a .json OpenAPI spec file or a traces stream with tool calls.
        # Agent/Prompt-only submissions return:
        #   "Tool use data generation requires exactly one .json file."
        p.error(
            "--recipe tool-use requires --source file (uploaded .json OpenAPI 3.x spec) "
            "or --source traces; got --source " + args.source
        )
    if not (15 <= args.max_samples <= 1000):
        p.error(f"--max-samples must be in [15, 1000]; got {args.max_samples}")
    if args.scenario == "rft" and args.recipe != "traces":
        # RFT requires verifiable ground-truth answers. In practice the service
        # only emits useful RFT data from the Traces recipe (where the production
        # assistant response is the target). Other recipes accept the scenario
        # at submit-time but typically fail in-flight with a generic error.
        print(
            f"⚠️  --scenario rft is reliable only with --recipe traces; "
            f"--recipe {args.recipe} usually fails in-flight",
            file=sys.stderr,
        )

    output_name = _make_output_name(args)

    if args.use_rest:
        run_rest(args, output_name)
    else:
        run_sdk(args, output_name)


if __name__ == "__main__":
    main()
