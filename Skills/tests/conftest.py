"""Shared fixtures and pytest config for skill tests.

Live tests (marked `@pytest.mark.live`) hit the real Foundry API and require:
  - `FOUNDRY_PROJECT_ENDPOINT` env var (an Azure AI project endpoint URL)
  - AAD auth via `az login` or `DefaultAzureCredential`-compatible env vars
  - At least one chat model deployed at the project

Optional overrides:
  - `FOUNDRY_TEACHER_MODEL`    (default: gpt-4.1)
  - `FOUNDRY_AGENT_NAME`       (default: demo1-retail-agent-langraph-responses)
  - `FOUNDRY_AGENT_VERSION`    (default: 5)
  - `SKIP_LIVE_E2E=1`          (skip all live tests)
  - `E2E_POLL_INTERVAL`        (seconds between job polls; default 10)
  - `E2E_JOB_TIMEOUT`          (seconds; default 600)

Skip with `pytest -m "not live"` for the consistency tests only.
"""

import os
import pathlib
import sys
import time

import pytest

# Force UTF-8 for emoji output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


TESTS_DIR = pathlib.Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"

# Default test target — known-good project with a hosted agent that has traffic + tools.
DEFAULT_ENDPOINT = "https://REDACTED-FOUNDRY-RESOURCE.services.ai.azure.com/api/projects/REDACTED-FOUNDRY-PROJECT"
DEFAULT_AGENT_NAME = "demo1-retail-agent-langraph-responses"
DEFAULT_AGENT_VERSION = "5"
DEFAULT_TEACHER = "gpt-4.1"


# ── pytest config ────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: hits the real Foundry Data Generation API (requires FOUNDRY_PROJECT_ENDPOINT)",
    )
    config.addinivalue_line("markers", "slow: takes >30 seconds to run")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SKIP_LIVE_E2E", "").lower() in ("1", "true", "yes"):
        skip = pytest.mark.skip(reason="SKIP_LIVE_E2E=1")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip)


# ── env helpers ──────────────────────────────────────────────────────────

def _env(name, default):
    val = os.environ.get(name)
    return val if val else default


@pytest.fixture(scope="session")
def project_endpoint():
    return _env("FOUNDRY_PROJECT_ENDPOINT", DEFAULT_ENDPOINT)


@pytest.fixture(scope="session")
def teacher_model():
    return _env("FOUNDRY_TEACHER_MODEL", DEFAULT_TEACHER)


@pytest.fixture(scope="session")
def agent_name():
    return _env("FOUNDRY_AGENT_NAME", DEFAULT_AGENT_NAME)


@pytest.fixture(scope="session")
def agent_version():
    return _env("FOUNDRY_AGENT_VERSION", DEFAULT_AGENT_VERSION)


@pytest.fixture(scope="session")
def poll_interval():
    return int(_env("E2E_POLL_INTERVAL", "10"))


@pytest.fixture(scope="session")
def job_timeout():
    return int(_env("E2E_JOB_TIMEOUT", "600"))


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES_DIR


# ── SDK client fixtures (session-scoped) ─────────────────────────────────

@pytest.fixture(scope="session")
def credential():
    from azure.identity import DefaultAzureCredential
    cred = DefaultAzureCredential()
    yield cred
    try:
        cred.close()
    except Exception:
        pass


@pytest.fixture(scope="session")
def project_client(project_endpoint, credential):
    pytest.importorskip("azure.ai.projects", minversion="2.2.0a")
    from azure.ai.projects import AIProjectClient
    pc = AIProjectClient(endpoint=project_endpoint, credential=credential)
    yield pc
    try:
        pc.close()
    except Exception:
        pass


@pytest.fixture(scope="session")
def aoai_client(project_client):
    aoai = project_client.get_openai_client()
    yield aoai
    try:
        aoai.close()
    except Exception:
        pass


# ── Helpers used by tests ────────────────────────────────────────────────

def make_run_id():
    import uuid
    from datetime import datetime, timezone
    return f"{datetime.now(tz=timezone.utc).strftime('%y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"


def poll_until_terminal(project_client, job, *, poll_interval=10, timeout=600):
    """Poll a data-generation job until terminal state. Returns the refreshed job."""
    from azure.ai.projects.models import JobStatus
    TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = project_client.beta.datasets.get_generation_job(job_id=job.id)
        if job.status in TERMINAL:
            return job
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job.id} did not reach terminal state within {timeout}s")


def cleanup_job_outputs(project_client, aoai_client, job):
    """Best-effort cleanup of files, datasets, and the job record."""
    from azure.ai.projects.models import (
        DatasetDataGenerationJobOutput,
        FileDataGenerationJobOutput,
    )
    if job is None:
        return
    if job.result and job.result.outputs:
        for o in job.result.outputs:
            try:
                if isinstance(o, FileDataGenerationJobOutput) and o.id:
                    aoai_client.files.delete(file_id=o.id)
                elif isinstance(o, DatasetDataGenerationJobOutput) and o.name and o.version:
                    project_client.datasets.delete(name=o.name, version=o.version)
            except Exception:
                pass
    try:
        project_client.beta.datasets.delete_generation_job(job_id=job.id)
    except Exception:
        pass
