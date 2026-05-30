"""Dataset helpers for the Zava Model Router fine-tuning demo.

Two responsibilities:

1. **Local validation** — ``validate_jsonl`` checks structural well-formedness
   of a JSONL training/validation file (binary 0/1 labels, consistent key set
   across rows, ``usage`` present with matching keys). The Model Router
   accepts a curated set of LLM ids that grows over time — see
   https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router#supported-models
   for the canonical catalog. The server rejects any unsupported LLM ids at
   job submission, so this helper deliberately does not enumerate the catalog.

2. **Upload + processing wait** — ``upload_file`` POSTs a JSONL file to the
   Files endpoint with ``purpose=fine-tune``; ``wait_for_file_processed``
   polls until the file's status reaches ``processed`` (or errors out).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import requests


def validate_jsonl(path: str | Path) -> Tuple[int, List[str], List[str]]:
    """Validate a Model Router training/validation JSONL file.

    Returns a tuple of ``(row_count, sorted_model_keys, errors)``. The caller
    decides whether to raise on a non-empty ``errors`` list.
    """
    errors: List[str] = []
    n = 0
    first_keys: set[str] | None = None

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            n += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON — {e}")
                continue

            if not isinstance(r.get("messages"), list) or not r["messages"]:
                errors.append(f"Line {i}: missing/empty 'messages'")
            else:
                for m in r["messages"]:
                    if not (
                        isinstance(m, dict)
                        and isinstance(m.get("role"), str)
                        and isinstance(m.get("content"), str)
                    ):
                        errors.append(f"Line {i}: malformed message item")
                        break

            if not isinstance(r.get("labels"), dict) or not r["labels"]:
                errors.append(f"Line {i}: missing/empty 'labels'")
                continue

            keys = set(r["labels"].keys())
            if first_keys is None:
                first_keys = keys
            elif keys != first_keys:
                errors.append(
                    f"Line {i}: labels key set differs from first row "
                    f"(diff: {sorted(keys ^ first_keys)})"
                )

            for k, v in r["labels"].items():
                if v not in (0, 1, "0", "1"):
                    errors.append(f"Line {i}: labels[{k!r}] must be 0/1, got {v!r}")

            # 'usage' is REQUIRED — non-empty dict with the same key set as 'labels'
            if "usage" not in r:
                errors.append(f"Line {i}: missing required field 'usage'")
            else:
                u = r["usage"]
                if not isinstance(u, dict) or not u:
                    errors.append(f"Line {i}: 'usage' must be a non-empty dict")
                elif set(u.keys()) != keys:
                    errors.append(f"Line {i}: usage keys != labels keys")
                else:
                    for k, vv in u.items():
                        if not isinstance(vv, dict) or not isinstance(
                            vv.get("prompt_tokens"), int
                        ):
                            errors.append(
                                f"Line {i}: usage[{k!r}] missing integer 'prompt_tokens'"
                            )

    return n, sorted(first_keys or []), errors

def upload_file(
    path: str | Path,
    project_endpoint: str,
    api_key: str,
) -> dict:
    """Upload a JSONL file with purpose=fine-tune to the project's Files endpoint.

    Returns the parsed JSON response (``id``, ``status``, ...). Raises for
    non-2xx responses, printing the server message first for easier debugging.
    """
    url = f"{project_endpoint}/openai/v1/files"
    headers = {"api-key": api_key}
    with open(path, "rb") as f:
        resp = requests.post(
            url,
            headers=headers,
            files={"file": (Path(path).name, f, "application/octet-stream")},
            data={"purpose": "fine-tune"},
        )
    if not resp.ok:
        print(f"Upload failed ({resp.status_code}): {resp.text}")
    resp.raise_for_status()
    return resp.json()


def wait_for_file_processed(
    file_id: str,
    project_endpoint: str,
    api_key: str,
    timeout: int = 120,
    poll_interval: int = 2,
) -> None:
    """Poll the file's status until it reaches ``processed`` (or errors out).

    Raises ``RuntimeError`` if the file ends in a terminal failure state
    (``error`` / ``deleted``). Prints a warning and returns if ``timeout`` is
    reached while still processing — the caller can decide whether to proceed.
    """
    url = f"{project_endpoint}/openai/v1/files/{file_id}"
    headers = {"api-key": api_key}
    elapsed = 0
    while elapsed < timeout:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        status = resp.json().get("status", "")
        if status == "processed":
            return
        if status in {"error", "deleted"}:
            raise RuntimeError(f"File {file_id} ended in status: {status}")
        time.sleep(poll_interval)
        elapsed += poll_interval
    print(f"Warning: file {file_id} still processing after {timeout}s")
