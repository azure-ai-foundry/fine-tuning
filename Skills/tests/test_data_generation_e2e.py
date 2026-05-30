"""End-to-end tests for the Foundry Data Generation API.

These tests hit the real service. Run with:

    pytest tests/test_data_generation_e2e.py -m live -v -s

Set FOUNDRY_PROJECT_ENDPOINT (required) and the optional FOUNDRY_AGENT_NAME /
FOUNDRY_AGENT_VERSION env vars described in conftest.py to point at your AI project.
"""

import io
import json
import os
import subprocess
import sys
import time

import pytest

from conftest import cleanup_job_outputs, make_run_id, poll_until_terminal, FIXTURES_DIR


# Live tests are individually marked so the CLI tests (last block) can run
# without --live.

# ── Smoke tests ──────────────────────────────────────────────────────────

@pytest.mark.live
def test_list_generation_jobs(project_client):
    """The project responds to list_generation_jobs (proves auth + capability header)."""
    jobs = list(project_client.beta.datasets.list_generation_jobs(limit=5))
    # We don't assert count — fresh projects have 0, busy ones may have many.
    assert isinstance(jobs, list)


@pytest.mark.live
def test_deployments_include_chat_model(project_client, teacher_model):
    """The configured teacher model is actually deployed at the project."""
    deployments = [d.name for d in project_client.deployments.list()]
    assert teacher_model in deployments, (
        f"Teacher model {teacher_model!r} not deployed. "
        f"Available: {deployments}. Set FOUNDRY_TEACHER_MODEL."
    )


# ── SimpleQnA SFT ────────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.slow
def test_simpleqna_sft_from_file(project_client, aoai_client, teacher_model,
                                  fixtures_dir, poll_interval, job_timeout):
    """Upload COBOL_Wikipedia.pdf, generate SFT JSONL via SimpleQnA + File source.

    Asserts:
      - Job reaches SUCCEEDED
      - generated_samples in expected range
      - Two FileDataGenerationJobOutput objects (train + valid) when train_split set
      - Each file is non-empty JSONL with {"messages": [...]} rows
    """
    from azure.ai.projects.models import (
        DataGenerationJob, DataGenerationJobInputs, DataGenerationJobOutputOptions,
        DataGenerationJobScenario, JobStatus,
        DataGenerationModelOptions, FileDataGenerationJobOutput,
        FileDataGenerationJobSource, SimpleQnADataGenerationJobOptions,
    )

    pdf_path = fixtures_dir / "COBOL_Wikipedia.pdf"
    assert pdf_path.exists(), f"Missing fixture {pdf_path}"

    run_id = make_run_id()
    output_name = f"e2e-qna-sft-file-{run_id}"[:50]

    job = None
    seed_file = None
    try:
        # Upload PDF
        with open(pdf_path, "rb") as f:
            seed_file = aoai_client.files.create(
                file=(pdf_path.name, f), purpose="user_data",
            )
        # Wait for processing
        import time as _t
        for _ in range(60):
            seed_file = aoai_client.files.retrieve(file_id=seed_file.id)
            if seed_file.status in ("processed", "error"):
                break
            _t.sleep(2)
        assert seed_file.status == "processed", f"file upload status={seed_file.status}"

        # Submit job
        req = DataGenerationJob(inputs=DataGenerationJobInputs(
            name=output_name,
            scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
            sources=[FileDataGenerationJobSource(
                id=seed_file.id, description="COBOL Wikipedia article (e2e test)",
            )],
            options=SimpleQnADataGenerationJobOptions(
                max_samples=15,
                train_split=0.8,
                model_options=DataGenerationModelOptions(model=teacher_model),
            ),
            output_options=DataGenerationJobOutputOptions(name=output_name),
        ))
        job = project_client.beta.datasets.create_generation_job(job=req)
        assert job.id
        assert job.status in (JobStatus.QUEUED, JobStatus.IN_PROGRESS)

        # Poll
        job = poll_until_terminal(project_client, job, poll_interval=poll_interval, timeout=job_timeout)
        if job.status != JobStatus.SUCCEEDED:
            err = job.error.message if job.error else "<no error>"
            pytest.fail(f"Job ended in {job.status}: {err}")

        # Verify outputs
        assert job.result.generated_samples >= 1, "no samples generated"
        file_outputs = [o for o in (job.result.outputs or []) if isinstance(o, FileDataGenerationJobOutput)]
        assert len(file_outputs) == 2, f"expected 2 file outputs (train+valid), got {len(file_outputs)}"

        # Sanity-check file contents
        for idx, o in enumerate(file_outputs):
            info = aoai_client.files.retrieve(file_id=o.id)
            assert info.bytes > 0, f"file {idx} is empty"
            content = aoai_client.files.content(o.id).content
            lines = [ln for ln in content.decode("utf-8").splitlines() if ln.strip()]
            assert lines, f"file {idx} has no JSONL rows"
            row = json.loads(lines[0])
            assert "messages" in row, f"file {idx} row missing 'messages': {row}"
            assert isinstance(row["messages"], list) and len(row["messages"]) >= 2
            roles = [m["role"] for m in row["messages"]]
            assert "user" in roles and "assistant" in roles
    finally:
        if seed_file:
            try: aoai_client.files.delete(file_id=seed_file.id)
            except Exception: pass
        cleanup_job_outputs(project_client, aoai_client, job)


# ── SimpleQnA EVAL ───────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.slow
def test_simpleqna_eval_from_prompt(project_client, aoai_client, teacher_model,
                                     poll_interval, job_timeout):
    """SimpleQnA + Prompt source + EVALUATION → Dataset output (not File)."""
    from azure.ai.projects.models import (
        DataGenerationJob, DataGenerationJobInputs, DataGenerationJobOutputOptions,
        DataGenerationJobScenario, JobStatus,
        DataGenerationModelOptions, DatasetDataGenerationJobOutput,
        PromptDataGenerationJobSource, SimpleQnADataGenerationJobOptions,
    )

    # Substantive inline doc (<10k chars).
    doc = (
        "# HTTP Semantics Reference\n\n"
        "HTTP methods include GET, POST, PUT, DELETE, HEAD, OPTIONS, PATCH. "
        "GET retrieves a representation; POST creates or causes side effects; "
        "PUT replaces; DELETE removes; HEAD returns only headers; OPTIONS "
        "describes communication options; PATCH applies partial updates.\n\n"
        "Status codes: 1xx informational; 2xx success (200 OK, 201 Created, "
        "204 No Content, 206 Partial Content); 3xx redirection (301, 302, 304, "
        "307, 308); 4xx client error (400, 401, 403, 404, 405, 409, 410, 415, "
        "422, 429); 5xx server error (500, 501, 502, 503, 504).\n\n"
        "Caching uses Cache-Control directives: public, private, no-cache, "
        "no-store, max-age, s-maxage. ETags enable validation via "
        "If-None-Match; Last-Modified pairs with If-Modified-Since. Vary tells "
        "caches which request headers affect the selected variant.\n\n"
        "Authentication schemes include Basic (base64), Bearer (OAuth2), and "
        "Digest (hash-based with nonces). Servers return 401 with "
        "WWW-Authenticate to request credentials.\n\n"
        "Content negotiation uses Accept, Accept-Language, Accept-Encoding. "
        "Servers respond with Content-Type, Content-Language, Content-Encoding."
    ) * 4
    assert len(doc) <= 10_000

    run_id = make_run_id()
    output_name = f"e2e-qna-eval-prompt-{run_id}"[:50]

    job = None
    try:
        req = DataGenerationJob(inputs=DataGenerationJobInputs(
            name=output_name,
            scenario=DataGenerationJobScenario.EVALUATION,
            sources=[PromptDataGenerationJobSource(prompt=doc, description="HTTP ref (e2e test)")],
            options=SimpleQnADataGenerationJobOptions(
                max_samples=15,
                model_options=DataGenerationModelOptions(model=teacher_model),
            ),
            output_options=DataGenerationJobOutputOptions(name=output_name),
        ))
        job = project_client.beta.datasets.create_generation_job(job=req)
        job = poll_until_terminal(project_client, job, poll_interval=poll_interval, timeout=job_timeout)
        if job.status != JobStatus.SUCCEEDED:
            err = job.error.message if job.error else "<no error>"
            pytest.fail(f"Job ended in {job.status}: {err}")

        assert job.result.generated_samples >= 1
        ds_outputs = [o for o in (job.result.outputs or []) if isinstance(o, DatasetDataGenerationJobOutput)]
        assert len(ds_outputs) >= 1, "expected at least one Dataset output for EVAL scenario"

        ds = ds_outputs[0]
        assert ds.name and ds.version
        # Confirm the dataset is resolvable
        resolved = project_client.datasets.get(name=ds.name, version=ds.version)
        assert resolved.id
    finally:
        cleanup_job_outputs(project_client, aoai_client, job)


# ── Traces SFT ───────────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.slow
def test_traces_sft_from_agent(project_client, aoai_client, agent_name, agent_version,
                                poll_interval, job_timeout):
    """Pull last 30 days of traces from the demo agent → SFT JSONL."""
    from datetime import datetime, timedelta, timezone
    from azure.ai.projects.models import (
        DataGenerationJob, DataGenerationJobInputs, DataGenerationJobOutputOptions,
        DataGenerationJobScenario, JobStatus,
        FileDataGenerationJobOutput,
        TracesDataGenerationJobSource, TracesDataGenerationJobOptions,
    )

    run_id = make_run_id()
    output_name = f"e2e-traces-sft-{run_id}"[:50]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)

    job = None
    try:
        req = DataGenerationJob(inputs=DataGenerationJobInputs(
            name=output_name,
            scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
            sources=[TracesDataGenerationJobSource(
                agent_name=agent_name,
                agent_version=agent_version,
                start_time=start,
                end_time=end,
                description=f"Traces from last 30 days for {agent_name} v{agent_version}",
            )],
            options=TracesDataGenerationJobOptions(
                max_samples=15,
                train_split=0.8,
            ),
            output_options=DataGenerationJobOutputOptions(name=output_name),
        ))
        job = project_client.beta.datasets.create_generation_job(job=req)
        job = poll_until_terminal(project_client, job, poll_interval=poll_interval, timeout=job_timeout)
        if job.status != JobStatus.SUCCEEDED:
            err = job.error.message if job.error else "<no error>"
            pytest.fail(f"Job ended in {job.status}: {err}  "
                        f"(if `generated 0 samples`, the agent had no traffic in the window)")

        # We don't assert >=15 samples — the agent might not have enough traffic.
        # But we do require at least one row, and outputs in the expected shape.
        assert job.result.generated_samples >= 1, (
            "0 samples generated. Agent has no traffic in the last 30 days, or "
            "App Insights hasn't ingested it yet."
        )
        file_outputs = [o for o in (job.result.outputs or []) if isinstance(o, FileDataGenerationJobOutput)]
        assert len(file_outputs) >= 1

        # Spot-check the first file
        info = aoai_client.files.retrieve(file_id=file_outputs[0].id)
        assert info.bytes > 0
        content = aoai_client.files.content(file_outputs[0].id).content
        lines = [ln for ln in content.decode("utf-8").splitlines() if ln.strip()]
        assert lines
        row = json.loads(lines[0])
        assert "messages" in row
    finally:
        cleanup_job_outputs(project_client, aoai_client, job)


# ── Tool-use SFT from OpenAPI spec ───────────────────────────────────────

@pytest.mark.live
@pytest.mark.slow
def test_tool_use_sft_from_openapi_spec(project_client, aoai_client, teacher_model,
                                          poll_interval, job_timeout, fixtures_dir):
    """ToolUseFineTuning + File source (OpenAPI 3.0 spec) → tool-calling SFT JSONL.

    The service requires exactly one .json file source for tool-use recipes — AND
    that file must validate as an OpenAPI 3.0.x or 3.1.x specification. Submitting
    the OpenAI chat-completions tool format fails in-flight with
    'Invalid or unsupported OpenAPI version'.

    This test:
      1. Converts retail_tools.json (OpenAI format) to OpenAPI 3.0 via the script's
         --tools-from converter.
      2. Uploads the OpenAPI spec as user_data.
      3. Submits a tool-use job and polls (can take 20-40 minutes).
      4. Validates the produced JSONL has `messages` + `tools` fields.

    Override the default 45-min timeout via E2E_TOOL_USE_TIMEOUT env var.
    """
    from azure.ai.projects.models import (
        FileDataGenerationJobSource,
        DataGenerationJob, DataGenerationJobInputs, DataGenerationJobOutputOptions,
        DataGenerationJobScenario, JobStatus,
        DataGenerationModelOptions, FileDataGenerationJobOutput,
        ToolUseFineTuningDataGenerationJobOptions,
    )

    run_id = make_run_id()
    output_name = f"e2e-tu-sft-{run_id}"[:50]

    # Convert retail_tools.json (OpenAI format) → OpenAPI 3.0 spec
    openapi_path = fixtures_dir / "retail_openapi.json"
    if not openapi_path.exists():
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_dataset.py")
        r = subprocess.run(
            [sys.executable, script,
             "--tools-from", str(fixtures_dir / "retail_tools.json"),
             "--tools-to-openapi-out", str(openapi_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        assert r.returncode == 0, f"converter failed: {r.stderr}"

    # Upload OpenAPI spec as user_data file
    with open(openapi_path, "rb") as fh:
        seed_file = aoai_client.files.create(
            file=("retail_openapi.json", fh), purpose="user_data",
        )

    deadline = time.time() + 60
    while time.time() < deadline:
        info = aoai_client.files.retrieve(file_id=seed_file.id)
        if info.status == "processed":
            break
        time.sleep(2)
    else:
        pytest.fail(f"Seed file did not process in time: {info.status}")

    job = None
    try:
        req = DataGenerationJob(inputs=DataGenerationJobInputs(
            name=output_name,
            scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
            sources=[FileDataGenerationJobSource(
                id=seed_file.id,
                description="Zava retail OpenAPI 3.0 spec (e2e tool-use test)",
            )],
            options=ToolUseFineTuningDataGenerationJobOptions(
                max_samples=15,
                train_split=0.8,
                model_options=DataGenerationModelOptions(model=teacher_model),
            ),
            output_options=DataGenerationJobOutputOptions(name=output_name),
        ))
        job = project_client.beta.datasets.create_generation_job(job=req)
        # Tool-use is the slowest path: typically 20-40 min on busy backends.
        # Override the shared timeout with at least 45 min for this test.
        tu_timeout = max(job_timeout, int(os.environ.get("E2E_TOOL_USE_TIMEOUT", "2700")))
        job = poll_until_terminal(project_client, job, poll_interval=poll_interval, timeout=tu_timeout)
        if job.status != JobStatus.SUCCEEDED:
            err = job.error.message if job.error else "<no error>"
            pytest.fail(f"Job ended in {job.status}: {err}")

        assert job.result.generated_samples >= 1
        file_outputs = [o for o in (job.result.outputs or []) if isinstance(o, FileDataGenerationJobOutput)]
        assert len(file_outputs) >= 1

        # Tool-calling rows should include the tools array
        content = aoai_client.files.content(file_outputs[0].id).content
        lines = [ln for ln in content.decode("utf-8").splitlines() if ln.strip()]
        assert lines, "no rows produced"
        row = json.loads(lines[0])
        assert "messages" in row
        assert "tools" in row, f"tool-use output missing 'tools' array: keys={list(row.keys())}"
        assert isinstance(row["tools"], list) and len(row["tools"]) > 0
    finally:
        cleanup_job_outputs(project_client, aoai_client, job)
        try:
            aoai_client.files.delete(seed_file.id)
        except Exception:
            pass


@pytest.mark.live
def test_tool_use_rejects_non_file_source(project_client, teacher_model, agent_name, agent_version):
    """Negative test: tool-use recipe MUST have a JSON file source.

    Agent source (without a .json file) should be rejected at submit time.
    """
    from azure.ai.projects.models import (
        AgentDataGenerationJobSource,
        DataGenerationJob, DataGenerationJobInputs, DataGenerationJobOutputOptions,
        DataGenerationJobScenario, DataGenerationModelOptions,
        ToolUseFineTuningDataGenerationJobOptions,
    )
    from azure.core.exceptions import HttpResponseError

    run_id = make_run_id()
    output_name = f"e2e-tu-neg-{run_id}"[:50]

    req = DataGenerationJob(inputs=DataGenerationJobInputs(
        name=output_name,
        scenario=DataGenerationJobScenario.SUPERVISED_FINETUNING,
        sources=[AgentDataGenerationJobSource(
            agent_name=agent_name,
            agent_version=agent_version,
        )],
        options=ToolUseFineTuningDataGenerationJobOptions(
            max_samples=15,
            model_options=DataGenerationModelOptions(model=teacher_model),
        ),
        output_options=DataGenerationJobOutputOptions(name=output_name),
    ))
    with pytest.raises(HttpResponseError) as exc:
        project_client.beta.datasets.create_generation_job(job=req)
    msg = str(exc.value).lower()
    assert "json" in msg and "tool" in msg, f"unexpected error: {exc.value}"


# ── REST path ────────────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.slow
def test_rest_simpleqna_sft(project_client, aoai_client, project_endpoint, teacher_model,
                             poll_interval, job_timeout):
    """REST POST /data_generation_jobs + poll + result inspection."""
    import time as _t
    import requests
    from azure.identity import DefaultAzureCredential

    run_id = make_run_id()
    output_name = f"e2e-rest-sft-{run_id}"[:50]

    # Upload a small substantive prompt as a file (cleanest path; covers most users)
    doc = (
        "# Reference Document\n\n"
        + ("HTTP/1.1 defines methods including GET, POST, PUT, DELETE, HEAD, OPTIONS, and PATCH. "
           "Status codes are grouped 1xx-5xx. Caching uses Cache-Control directives. "
           "Authentication schemes include Basic, Bearer, and Digest. "
           "Content negotiation uses Accept headers. ") * 60
    )
    seed_file = aoai_client.files.create(
        file=("ref.md", io.BytesIO(doc.encode("utf-8"))), purpose="user_data",
    )
    for _ in range(30):
        seed_file = aoai_client.files.retrieve(file_id=seed_file.id)
        if seed_file.status in ("processed", "error"): break
        _t.sleep(2)
    assert seed_file.status == "processed"

    with DefaultAzureCredential() as cred:
        token = cred.get_token("https://ai.azure.com/.default").token
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Foundry-Features": "DataGenerationJobs=V1Preview",
        }
        body = {
            "inputs": {
                "name": output_name,
                "scenario": "supervised_finetuning",
                "sources": [{"type": "file", "id": seed_file.id}],
                "options": {
                    "type": "simple_qna",
                    "max_samples": 15,
                    "train_split": 0.8,
                    "model_options": {"model": teacher_model},
                },
                "output_options": {"name": output_name},
            }
        }
        base = project_endpoint.rstrip("/") + "/data_generation_jobs"
        job_id = None
        try:
            r = requests.post(base, params={"api-version": "v1"}, headers=headers,
                              json=body, timeout=(10, 60))
            assert r.status_code == 201, f"submit failed {r.status_code}: {r.text[:500]}"
            job_id = r.json()["id"]

            # Poll
            import time as _t
            deadline = _t.monotonic() + job_timeout
            status = None
            while _t.monotonic() < deadline:
                _t.sleep(poll_interval)
                gr = requests.get(f"{base}/{job_id}", params={"api-version": "v1"},
                                  headers=headers, timeout=(10, 60))
                gr.raise_for_status()
                payload = gr.json()
                status = (payload.get("status") or "").lower()
                if status in ("succeeded", "failed", "cancelled"):
                    break
            assert status == "succeeded", (
                f"REST job ended in {status}: "
                f"{(payload.get('error') or {}).get('message')}"
            )

            result = payload.get("result") or {}
            assert (result.get("generated_samples") or 0) >= 1
            outputs = result.get("outputs") or []
            file_outs = [o for o in outputs if (o.get("type") or "").lower() == "file"]
            assert len(file_outs) == 2, f"expected 2 file outputs, got {len(file_outs)}"
            # Verify file resolvable
            info = aoai_client.files.retrieve(file_id=file_outs[0]["id"])
            assert info.bytes > 0

            # Cleanup via REST too — covers DELETE path
            for o in file_outs:
                try: aoai_client.files.delete(file_id=o["id"])
                except Exception: pass
            dr = requests.delete(f"{base}/{job_id}", params={"api-version": "v1"},
                                 headers=headers, timeout=(10, 60))
            # 200 or 204 both acceptable for delete
            assert dr.status_code in (200, 202, 204), f"delete returned {dr.status_code}"
        finally:
            try: aoai_client.files.delete(file_id=seed_file.id)
            except Exception: pass


# ── Negative / constraint tests ──────────────────────────────────────────

@pytest.mark.live
def test_file_source_rejected_for_simpleqna_eval(project_client, aoai_client, teacher_model):
    """Documented constraint: File source + SimpleQnA + EVAL is rejected at submit."""
    from azure.ai.projects.models import (
        DataGenerationJob, DataGenerationJobInputs, DataGenerationJobOutputOptions,
        DataGenerationJobScenario,
        DataGenerationModelOptions, FileDataGenerationJobSource,
        SimpleQnADataGenerationJobOptions,
    )
    from azure.core.exceptions import HttpResponseError

    # Upload a stub file just so we have a file id (won't actually be used)
    seed_file = aoai_client.files.create(
        file=("stub.txt", io.BytesIO(b"stub content " * 200)), purpose="user_data",
    )
    import time as _t
    for _ in range(20):
        seed_file = aoai_client.files.retrieve(file_id=seed_file.id)
        if seed_file.status in ("processed", "error"): break
        _t.sleep(2)

    run_id = make_run_id()
    output_name = f"e2e-neg-{run_id}"[:50]
    try:
        with pytest.raises(HttpResponseError) as exc_info:
            project_client.beta.datasets.create_generation_job(job=DataGenerationJob(
                inputs=DataGenerationJobInputs(
                    name=output_name,
                    scenario=DataGenerationJobScenario.EVALUATION,
                    sources=[FileDataGenerationJobSource(id=seed_file.id)],
                    options=SimpleQnADataGenerationJobOptions(
                        max_samples=15,
                        model_options=DataGenerationModelOptions(model=teacher_model),
                    ),
                    output_options=DataGenerationJobOutputOptions(name=output_name),
                ),
            ))
        # Error message must mention the constraint
        msg = str(exc_info.value).lower()
        assert "prompt" in msg or "agent" in msg or "invalid" in msg, (
            f"unexpected error message: {exc_info.value}"
        )
    finally:
        try: aoai_client.files.delete(file_id=seed_file.id)
        except Exception: pass


@pytest.mark.live
@pytest.mark.slow
def test_auto_finetune_foundry_generate_e2e(project_client, project_endpoint, teacher_model, tmp_path):
    """auto_finetune.py foundry-generate ties task_spec → Foundry datagen → SFT JSONL.

    Validates the end-to-end wiring: cmd_foundry_generate shells out to
    generate_dataset.py and writes a normalised generated_data.jsonl that the
    prepare phase can consume.

    Uses prompt-file source (which uploads the prompt as a user_data file
    internally) for two reasons:
      1. Some projects/teacher deployments reject inline Prompt+SFT with the
         generic "Something went wrong" error; uploading as a file is the
         robust path. (See references/data-generation-api.md — error table.)
      2. File source proves the cmd_foundry_generate path also works when
         a real upload is in the loop.
    """
    auto_ft = os.path.join(os.path.dirname(__file__), "..", "scripts", "auto_finetune.py")

    work = tmp_path / "auto_ft_run"
    work.mkdir()
    task_spec = work / "task_spec.json"

    # Hand-write a minimal task spec (skip the analyze phase to keep test focused
    # on the datagen integration — analyze is covered separately by test_skills.py).
    task_spec.write_text(json.dumps({
        "task_name": "http-qa",
        "description": "Q&A about HTTP/1.1 mechanics — methods, status codes, caching, auth, content negotiation.",
        "data_mode": "prompt_only",
        "hypotheses": [{"task_type": "generation"}],
    }), encoding="utf-8")

    # Write a substantive document the service can mine for Q&A (≥1 KB substantive content)
    doc = work / "http_doc.md"
    doc.write_text(
        "# HTTP/1.1 Reference\n\n"
        + ("HTTP/1.1 defines the methods GET (retrieve), POST (submit), PUT (replace), "
           "DELETE (remove), HEAD (metadata only), OPTIONS (allowed methods), and PATCH "
           "(partial update). Status codes are grouped: 1xx informational (100 Continue), "
           "2xx success (200 OK, 201 Created, 204 No Content), 3xx redirection (301 Moved "
           "Permanently, 304 Not Modified), 4xx client error (400 Bad Request, 401 Unauthorized, "
           "403 Forbidden, 404 Not Found, 429 Too Many Requests), 5xx server error (500 "
           "Internal Server Error, 502 Bad Gateway, 503 Service Unavailable). Caching is "
           "controlled by Cache-Control directives: max-age, no-cache, no-store, private, "
           "public, must-revalidate, immutable. ETag and If-None-Match enable conditional "
           "requests. Authentication schemes: Basic (base64-encoded credentials), Bearer "
           "(used by OAuth 2.0), Digest (MD5 challenge-response). Content negotiation uses "
           "Accept, Accept-Language, Accept-Encoding (gzip, br, deflate), and the server "
           "responds with Content-Type and Vary. Connection management: Keep-Alive header, "
           "persistent connections by default. Range requests via Range header return 206 "
           "Partial Content. Chunked transfer encoding streams unknown-length responses.\n\n"
           ) * 5,
        encoding="utf-8",
    )

    out_dir = work / "generated"
    cmd = [
        sys.executable, auto_ft, "foundry-generate",
        "--task-spec", str(task_spec),
        "--source", "prompt-file",
        "--prompt-file", str(doc),
        "--recipe", "qna",
        "--scenario", "sft",
        "--max-samples", "15",
        "--train-split", "0.8",
        "--teacher", teacher_model,
        "--project-endpoint", project_endpoint,
        "--output-dir", str(out_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)
    assert r.returncode == 0, (
        f"foundry-generate failed (rc={r.returncode})\n"
        f"stdout (last 2000):\n{r.stdout[-2000:]}\n"
        f"stderr (last 1000):\n{r.stderr[-1000:]}"
    )

    merged = out_dir / "generated_data.jsonl"
    assert merged.exists(), f"merged output not written: stdout={r.stdout[-500:]}"
    lines = [ln for ln in merged.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 10, f"expected ≥10 rows in merged output, got {len(lines)}"
    row = json.loads(lines[0])
    assert "messages" in row, f"row missing 'messages': {list(row.keys())}"
    assert any(m.get("role") == "user" for m in row["messages"])
    assert any(m.get("role") == "assistant" for m in row["messages"])


# ── CLI validation tests (no live calls — pure argparse) ─────────────────

@pytest.mark.parametrize("override,expect_in_output", [
    ({"--max-samples": "5"}, "--max-samples must be in [15, 1000]"),
    ({"--max-samples": "5000"}, "--max-samples must be in [15, 1000]"),
    ({"--recipe": "tool-use", "--scenario": "eval"}, "tool-use is SFT-only"),
    ({"--recipe": "tool-use", "--source": "agent", "--agent-name": "x"}, "requires --source file"),
    ({"--recipe": "tool-use", "--source": "prompt-inline"}, "requires --source file"),
])
def test_cli_rejects_invalid_args(override, expect_in_output, project_endpoint):
    """CLI argparse-level validation. No network calls."""
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_dataset.py")
    args = {
        "--project-endpoint": project_endpoint,
        "--source": "prompt-inline",
        "--prompt": "stub",
        "--recipe": "qna",
        "--scenario": "sft",
        "--teacher": "gpt-4.1-mini",
        "--max-samples": "15",
    }
    args.update(override)
    cmd = [sys.executable, script]
    for k, v in args.items():
        cmd.extend([k, v])

    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=30)
    assert r.returncode != 0, (
        f"expected non-zero exit, got {r.returncode}\n"
        f"stdout={r.stdout}\nstderr={r.stderr}"
    )
    combined = r.stdout + r.stderr
    assert expect_in_output in combined, (
        f"expected {expect_in_output!r} in output\n"
        f"stdout={r.stdout}\nstderr={r.stderr}"
    )


def test_cli_rft_warning_emitted(project_endpoint):
    """CLI warns when --scenario rft is used with a non-traces recipe."""
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_dataset.py")
    cmd = [
        sys.executable, script,
        "--project-endpoint", "https://invalid.example.com/api/projects/x",
        "--source", "prompt-inline", "--prompt", "stub",
        "--recipe", "qna", "--scenario", "rft",
        "--teacher", "gpt-4.1-mini",
        "--max-samples", "15",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=30)
    combined = r.stdout + r.stderr
    assert "--scenario rft is reliable only with --recipe traces" in combined, (
        f"expected RFT warning in output\nstdout={r.stdout}\nstderr={r.stderr}"
    )


def test_cli_help_renders():
    """`generate_dataset.py --help` exits 0 and shows all flags."""
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_dataset.py")
    r = subprocess.run([sys.executable, script, "--help"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=15)
    assert r.returncode == 0
    out = r.stdout
    for required in ["--source", "--recipe", "--scenario", "--teacher",
                     "--max-samples", "--train-split", "--use-rest", "--download",
                     "--tools-from", "--tools-to-openapi-out"]:
        assert required in out, f"--help missing {required}"


def test_auto_finetune_foundry_generate_requires_endpoint(tmp_path):
    """auto_finetune.py foundry-generate must reject when --project-endpoint missing."""
    auto_ft = os.path.join(os.path.dirname(__file__), "..", "scripts", "auto_finetune.py")
    task_spec = tmp_path / "ts.json"
    task_spec.write_text(json.dumps({"task_name": "t", "description": "d"}), encoding="utf-8")
    # Clear env var that would otherwise satisfy --project-endpoint
    env = {k: v for k, v in os.environ.items() if k != "AZURE_AI_PROJECT_ENDPOINT"}
    r = subprocess.run(
        [sys.executable, auto_ft, "foundry-generate",
         "--task-spec", str(task_spec),
         "--source", "prompt-inline",
         "--teacher", "gpt-4.1-mini",
         "--max-samples", "15"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, env=env,
    )
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "project-endpoint" in combined.lower(), (
        f"expected project-endpoint error, got:\n{combined}"
    )


def test_cli_tools_to_openapi_conversion(tmp_path):
    """`generate_dataset.py --tools-from X --tools-to-openapi-out Y` converts OpenAI
    chat-completions tool format into a valid OpenAPI 3.0 spec."""
    src = tmp_path / "tools.json"
    src.write_text(json.dumps([
        {"type": "function", "function": {
            "name": "get_order",
            "description": "Get order by id",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }},
        {"type": "function", "function": {
            "name": "list_orders",
            "description": "List all orders",
            "parameters": {"type": "object", "properties": {}},
        }},
    ]), encoding="utf-8")
    dst = tmp_path / "spec.json"
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_dataset.py")
    r = subprocess.run(
        [sys.executable, script, "--tools-from", str(src),
         "--tools-to-openapi-out", str(dst)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
    )
    assert r.returncode == 0, f"converter failed: {r.stderr}"
    spec = json.loads(dst.read_text(encoding="utf-8"))
    assert spec["openapi"].startswith("3.0")
    assert "/get_order" in spec["paths"]
    assert "/list_orders" in spec["paths"]
    # Tool with params produces a requestBody
    assert "requestBody" in spec["paths"]["/get_order"]["post"]
    # Tool with no params must NOT produce an empty requestBody (OpenAPI validator rejects)
    assert "requestBody" not in spec["paths"]["/list_orders"]["post"]
