# /// script
# dependencies = [
#   "openai>=1.0",
#   "requests",
#   "azure-identity",
# ]
# ///
"""
auto_rft.py — Autonomous RFT (Reinforcement Fine-Tuning) orchestrator.

Manages the full RFT lifecycle for o4-mini and gpt-5 models:
  validate → prepare → calibrate → baseline → execute → evaluate → review → iterate

RFT differs fundamentally from SFT:
  - Requires a Python grader that scores model outputs (grade(sample, item) → float)
  - Supports tool-calling (agentic tasks) during training
  - Billed hourly, not per-token — $102/hour for o4-mini
  - Only 1 experiment per iteration (expensive)
  - Pass threshold calibration is mandatory (grader must have signal)

Usage:
  # Full auto loop
  python auto_rft.py auto \\
      --data train.jsonl --grader grader.py --tools tools.json \\
      --model o4-mini --work-dir ./rft_run

  # Step-by-step
  python auto_rft.py validate --data train.jsonl --grader grader.py --tools tools.json
  python auto_rft.py prepare --data train.jsonl --work-dir ./rft_run
  python auto_rft.py calibrate --data ./rft_run/prepared/val.jsonl --grader grader.py \\
      --tools tools.json --model o4-mini
  python auto_rft.py baseline --data ./rft_run/prepared/test.jsonl --grader grader.py \\
      --tools tools.json --model o4-mini
  python auto_rft.py execute --train-file-id file-xxx --val-file-id file-yyy \\
      --grader grader.py --tools tools.json --model o4-mini --pass-threshold 0.8
"""

import hashlib
import importlib.util
import json
import os
import random
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass  # Stream not reconfigurable (older Python or non-tty); default encoding is fine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HelpOnErrorParser, get_clients, upload_file


# ── Constants ────────────────────────────────────────────────────────────

_RFT_MODELS = ("o4-mini", "gpt-5")
_DEFAULT_HYPERPARAMS = {
    "learning_rate_multiplier": 1.0,
    "n_epochs": 3,
    "compute_multiplier": 1.5,
    "eval_interval": 5,
    "eval_samples": 10,
    "reasoning_effort": "medium",
}
# Ideal failure rate range for grader calibration (25-50%, target 35%)
_IDEAL_FAIL_LOW = 0.25
_IDEAL_FAIL_HIGH = 0.50
_IDEAL_FAIL_TARGET = 0.35


# ── Helpers ──────────────────────────────────────────────────────────────

def _atomic_json_write(path, data):
    """Write JSON atomically via temp-file + rename."""
    import tempfile
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", dir=dir_name,
                                     delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _is_rft_model(model_id):
    """Check if model supports RFT."""
    m = model_id.lower().split(".ft-")[0]
    return any(m == r or m.startswith(r + "-") for r in _RFT_MODELS)


def _load_grader(grader_path):
    """Load a Python grader file and return the grade function.

    The grader must define: grade(sample, item) -> float
    where sample = {"output_text": str, "output_tools": list}
    and item = {all fields from the JSONL line except "messages"}.
    """
    if not os.path.isfile(grader_path):
        raise FileNotFoundError(f"Grader file not found: {grader_path}")

    spec = importlib.util.spec_from_file_location("grader", grader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "grade") or not callable(mod.grade):
        raise ValueError(f"Grader file must define a callable 'grade(sample, item)' function")

    return mod.grade


def _smoke_test_grader(grade_fn):
    """Validate grader by running it on a synthetic payload.

    Returns True if grader works, raises on failure.
    """
    sample = {"output_text": "Test response", "output_tools": []}
    item = {"expected_resolution": "Test expected"}
    try:
        score = grade_fn(sample, item)
        if not isinstance(score, (int, float)):
            raise ValueError(f"Grader returned {type(score).__name__}, expected float")
        if score < 0 or score > 1:
            print(f"  ⚠️  Grader returned {score} — expected 0.0-1.0 range")
        return True
    except TypeError as e:
        raise ValueError(f"Grader signature error: {e}. Must be grade(sample, item) -> float") from e


def _load_tools(tools_path):
    """Load tools JSON file. Expected format:
    [{"name": "tool_name", "server_url": "https://...", "headers": {}}]
    """
    if not tools_path:
        return []
    if not os.path.isfile(tools_path):
        raise FileNotFoundError(f"Tools file not found: {tools_path}")
    with open(tools_path, encoding="utf-8") as f:
        tools = json.load(f)
    if not isinstance(tools, list):
        raise ValueError("Tools file must contain a JSON array")
    for t in tools:
        if "name" not in t or "server_url" not in t:
            raise ValueError(f"Each tool must have 'name' and 'server_url'. Got: {list(t.keys())}")
    return tools


def _load_jsonl(path):
    """Load JSONL file, return list of dicts."""
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Skipping malformed JSON on line {i+1}: {e}")
    return data


def _deduplicate(examples):
    """Deduplicate by hashing the messages array."""
    seen = set()
    unique = []
    for ex in examples:
        key = hashlib.md5(json.dumps(ex.get("messages", []), sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def _write_jsonl(path, data):
    """Write list of dicts to JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── RFT Agent Runner ─────────────────────────────────────────────────────

def _run_rft_agent(client, model, messages, tools_schema=None, max_turns=8):
    """Run a multi-turn agent loop with tool calling.

    Returns a trajectory dict:
    {
        "output_text": str,          # Final assistant response
        "output_tools": list,        # All tool calls made [{function: {name, arguments}}]
        "tool_results": list,        # Tool call results
        "turns": int,                # Number of turns
        "finish_reason": str,
        "error": str or None,
        "latency_s": float,
    }
    """
    start = time.time()
    all_tool_calls = []
    all_tool_results = []
    conversation = list(messages)  # copy

    for turn in range(max_turns):
        try:
            kwargs = {
                "model": model,
                "messages": conversation,
                "max_completion_tokens": 8192,
            }
            if tools_schema:
                kwargs["tools"] = tools_schema
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            return {
                "output_text": f"ERROR: {e}",
                "output_tools": all_tool_calls,
                "tool_results": all_tool_results,
                "turns": turn + 1,
                "finish_reason": "error",
                "error": str(e),
                "latency_s": time.time() - start,
            }

        choice = resp.choices[0]
        msg = choice.message

        # If the model made tool calls, execute them
        if msg.tool_calls:
            # Append the assistant message with tool calls
            conversation.append(msg)
            for tc in msg.tool_calls:
                all_tool_calls.append({
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                })
                # Execute tool via the server_url
                tool_result = _execute_tool_call(tc, tools_schema)
                all_tool_results.append(tool_result)
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
            continue

        # No tool calls — final response
        content = msg.content or ""
        return {
            "output_text": content,
            "output_tools": all_tool_calls,
            "tool_results": all_tool_results,
            "turns": turn + 1,
            "finish_reason": choice.finish_reason or "stop",
            "error": None,
            "latency_s": time.time() - start,
        }

    # Max turns exceeded
    last_content = ""
    if conversation and hasattr(conversation[-1], "content"):
        last_content = conversation[-1].content or ""
    return {
        "output_text": last_content or "ERROR: max turns exceeded",
        "output_tools": all_tool_calls,
        "tool_results": all_tool_results,
        "turns": max_turns,
        "finish_reason": "max_turns",
        "error": "Max turns exceeded",
        "latency_s": time.time() - start,
    }


def _execute_tool_call(tool_call, tools_config):
    """Execute a tool call against its configured server_url.

    Looks up the tool by name in tools_config, POSTs the arguments to server_url.
    Returns the tool result as a string.
    """
    import requests as req

    tool_name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        args = {}

    # Find matching tool config
    tool_cfg = None
    for t in (tools_config or []):
        # tools_config may be OpenAI-format or our simplified format
        cfg_name = t.get("name") or t.get("function", {}).get("name", "")
        if cfg_name == tool_name:
            tool_cfg = t
            break

    if not tool_cfg:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    server_url = tool_cfg.get("server_url", "")
    headers = tool_cfg.get("headers", {})
    if not server_url:
        return json.dumps({"error": f"No server_url for tool: {tool_name}"})

    try:
        resp = req.post(server_url, json=args,
                        headers={**headers, "Content-Type": "application/json"},
                        timeout=(10, 30))
        if resp.status_code == 200:
            return resp.text
        return json.dumps({"error": f"Tool returned {resp.status_code}: {resp.text[:200]}"})
    except Exception as e:
        return json.dumps({"error": f"Tool call failed: {e}"})


def _build_tools_schema(tools_config):
    """Build OpenAI-compatible tools schema for chat completions.

    For RFT inference/eval, we need the function schema that the model sees.
    We auto-generate a minimal schema from the tool name since the full
    schema lives on the server side during training.
    """
    if not tools_config:
        return None

    schema = []
    for t in tools_config:
        name = t.get("name", "unknown")
        schema.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Call the {name} tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            },
        })
    return schema


# ── Phase 1: VALIDATE ────────────────────────────────────────────────────

def cmd_validate(args):
    """Validate all inputs before starting the RFT pipeline."""
    print("\n" + "=" * 60)
    print("  VALIDATE")
    print("=" * 60)

    errors = []

    # 1. Model check
    model = args.model
    if not _is_rft_model(model):
        errors.append(f"Model '{model}' does not support RFT. Use o4-mini or gpt-5.")
    else:
        print(f"  ✅ Model: {model}")

    # 2. Data check
    data_path = args.data
    if not os.path.isfile(data_path):
        errors.append(f"Data file not found: {data_path}")
    else:
        data = _load_jsonl(data_path)
        if len(data) < 10:
            errors.append(f"Data has only {len(data)} examples — need at least 10")
        else:
            # Check format: messages array required
            bad = [i for i, ex in enumerate(data[:20]) if "messages" not in ex]
            if bad:
                errors.append(f"Lines {bad[:5]} missing 'messages' field")
            else:
                print(f"  ✅ Data: {len(data)} examples, JSONL format")
                # Check for expected_* field (used by grader)
                extra_fields = set()
                for ex in data[:20]:
                    extra_fields.update(k for k in ex if k != "messages")
                if extra_fields:
                    print(f"     Extra fields: {', '.join(sorted(extra_fields))}")
                else:
                    print(f"     ⚠️  No extra fields (grader item dict will be empty)")

    # 3. Grader check
    grader_path = args.grader
    try:
        grade_fn = _load_grader(grader_path)
        _smoke_test_grader(grade_fn)
        print(f"  ✅ Grader: {grader_path} (smoke test passed)")
    except Exception as e:
        errors.append(f"Grader error: {e}")

    # 4. Tools check
    tools_path = getattr(args, "tools", None)
    if tools_path:
        try:
            tools = _load_tools(tools_path)
            print(f"  ✅ Tools: {len(tools)} tool(s) — {', '.join(t['name'] for t in tools)}")
        except Exception as e:
            errors.append(f"Tools error: {e}")
    else:
        print(f"  ℹ️  No tools file — non-agentic RFT")

    if errors:
        print(f"\n  ❌ Validation failed:")
        for e in errors:
            print(f"     • {e}")
        sys.exit(1)
    else:
        print(f"\n  ✅ All checks passed")


# ── Phase 2: PREPARE ─────────────────────────────────────────────────────

def cmd_prepare(args):
    """Load, validate, deduplicate, and split data for RFT."""
    data_path = args.data
    work_dir = args.work_dir
    prepared_dir = os.path.join(work_dir, "prepared")

    print("\n" + "=" * 60)
    print("  PREPARE")
    print("=" * 60)

    data = _load_jsonl(data_path)
    print(f"  Loaded {len(data)} examples")

    # Deduplicate
    unique = _deduplicate(data)
    if len(unique) < len(data):
        print(f"  Removed {len(data) - len(unique)} duplicates → {len(unique)} unique")
    data = unique

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(data)

    # Split: 85% train, 10% val, 5% test
    n = len(data)
    n_val = max(10, int(n * 0.10))
    n_test = max(5, int(n * 0.05))
    n_train = n - n_val - n_test

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    os.makedirs(prepared_dir, exist_ok=True)
    train_path = os.path.join(prepared_dir, "train.jsonl")
    val_path = os.path.join(prepared_dir, "val.jsonl")
    test_path = os.path.join(prepared_dir, "test.jsonl")

    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)
    _write_jsonl(test_path, test)

    print(f"  train: {len(train)} → {train_path}")
    print(f"  val:   {len(val)} → {val_path}")
    print(f"  test:  {len(test)} → {test_path}")

    # Upload train + val
    print(f"\n  Uploading to Azure...")
    client, _ = get_clients(
        base_url=getattr(args, "base_url", None),
        project_endpoint=getattr(args, "project_endpoint", None),
        api_key=getattr(args, "api_key", None),
    )
    train_id = upload_file(client, train_path)
    val_id = upload_file(client, val_path)

    manifest = {
        "train_file": train_id,
        "val_file": val_id,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "train_examples": len(train),
        "val_examples": len(val),
        "test_examples": len(test),
    }
    manifest_path = os.path.join(prepared_dir, "manifest.json")
    _atomic_json_write(manifest_path, manifest)
    print(f"  ✅ Manifest: {manifest_path}")

    return manifest


# ── Phase 3: CALIBRATE ───────────────────────────────────────────────────

def cmd_calibrate(args):
    """Run base model on validation samples, score with grader, find optimal pass_threshold."""
    print("\n" + "=" * 60)
    print("  CALIBRATE GRADER")
    print("=" * 60)

    model = args.model
    n_samples = getattr(args, "n_samples", 30)
    grade_fn = _load_grader(args.grader)
    tools = _load_tools(getattr(args, "tools", None))
    tools_schema = _build_tools_schema(tools)

    # Load validation data
    val_data = _load_jsonl(args.data)
    if len(val_data) > n_samples:
        random.seed(42)
        val_data = random.sample(val_data, n_samples)

    print(f"  Model: {model}")
    print(f"  Samples: {len(val_data)}")
    print(f"  Tools: {len(tools)}")

    client, _ = get_clients(
        base_url=getattr(args, "base_url", None),
        project_endpoint=getattr(args, "project_endpoint", None),
        api_key=getattr(args, "api_key", None),
    )

    print(f"\n  Running {model} on {len(val_data)} examples...\n")
    scores = []
    for i, ex in enumerate(val_data):
        messages = ex["messages"]
        item = {k: v for k, v in ex.items() if k != "messages"}

        trajectory = _run_rft_agent(client, model, messages, tools_schema)

        if trajectory["error"]:
            print(f"  [{i+1:3d}] ❌ {trajectory['error'][:60]}")
            scores.append(0.0)
            continue

        sample = {
            "output_text": trajectory["output_text"],
            "output_tools": trajectory["output_tools"],
        }

        try:
            score = grade_fn(sample, item)
        except Exception as e:
            print(f"  [{i+1:3d}] ❌ Grader error: {e}")
            scores.append(0.0)
            continue

        status = "✅" if score >= 0.8 else ("⚠️" if score >= 0.5 else "❌")
        user_msg = messages[-1]["content"][:55] if messages else ""
        tools_used = len(trajectory["output_tools"])
        print(f"  [{i+1:3d}] {score:.3f} {status}  tools={tools_used}  {user_msg}")
        scores.append(score)

        time.sleep(0.5)

    # Analyze thresholds
    scored = [s for s in scores if s is not None]
    if not scored:
        print("\n  ❌ No examples scored. Check model access and grader.")
        return None

    avg = sum(scored) / len(scored)
    print(f"\n{'='*60}")
    print(f"  CALIBRATION ({len(scores)} examples)")
    print(f"  Average score: {avg:.1%}")
    print(f"{'='*60}")

    print(f"\n  {'Threshold':>10} {'Pass Rate':>10} {'Fail Rate':>10} {'Signal':>20}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*20}")

    best_threshold = None
    best_distance = float("inf")

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        pass_rate = sum(1 for s in scored if s >= threshold) / len(scored)
        fail_rate = 1 - pass_rate

        if _IDEAL_FAIL_LOW <= fail_rate <= _IDEAL_FAIL_HIGH:
            signal = "✅ Good"
            distance = abs(fail_rate - _IDEAL_FAIL_TARGET)
            if distance < best_distance:
                best_distance = distance
                best_threshold = threshold
        elif fail_rate < 0.10:
            signal = "❌ Too easy"
        elif fail_rate < _IDEAL_FAIL_LOW:
            signal = "⚠️ Weak signal"
        elif fail_rate <= 0.70:
            signal = "⚠️ Harsh"
        else:
            signal = "❌ Too hard"

        print(f"  {threshold:>10.2f} {pass_rate:>9.0%} {fail_rate:>9.0%} {signal:>20}")

    if best_threshold:
        fail_at_best = sum(1 for s in scored if s < best_threshold) / len(scored)
        print(f"\n  ✅ Recommended pass_threshold: {best_threshold}")
        print(f"     (~{fail_at_best:.0%} failure rate)")
    else:
        # Fallback: use threshold closest to 35% failure
        fallback = min(
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            key=lambda t: abs((1 - sum(1 for s in scored if s >= t) / len(scored)) - _IDEAL_FAIL_TARGET)
        )
        best_threshold = fallback
        print(f"\n  ⚠️ No threshold in ideal range. Using closest: {best_threshold}")

    result = {
        "model": model,
        "n_samples": len(scores),
        "avg_score": round(avg, 4),
        "recommended_threshold": best_threshold,
        "scores": [round(s, 4) for s in scores],
    }

    output = getattr(args, "output", None)
    if output:
        _atomic_json_write(output, result)
        print(f"\n  Output: {output}")

    return result


# ── Phase 4: BASELINE ────────────────────────────────────────────────────

def cmd_baseline(args):
    """Evaluate base model on test set with grader + tools."""
    print("\n" + "=" * 60)
    print("  BASELINE")
    print("=" * 60)

    model = args.model
    grade_fn = _load_grader(args.grader)
    tools = _load_tools(getattr(args, "tools", None))
    tools_schema = _build_tools_schema(tools)

    test_data = _load_jsonl(args.data)
    print(f"  Model: {model}")
    print(f"  Test examples: {len(test_data)}")
    print(f"  Tools: {len(tools)}")

    client, _ = get_clients(
        base_url=getattr(args, "base_url", None),
        project_endpoint=getattr(args, "project_endpoint", None),
        api_key=getattr(args, "api_key", None),
    )

    results = _evaluate_with_grader(client, model, test_data, grade_fn, tools_schema, label="BASELINE")

    output = getattr(args, "output", None)
    if output:
        _atomic_json_write(output, results)
        print(f"\n  Output: {output}")

    return results


def _evaluate_with_grader(client, model, data, grade_fn, tools_schema, label="EVAL"):
    """Run model on data with tools, score with grader, return metrics."""
    print(f"\n  Evaluating {model} on {len(data)} examples...")
    scores = []
    errors = 0
    tool_usage = 0

    for i, ex in enumerate(data):
        messages = ex["messages"]
        item = {k: v for k, v in ex.items() if k != "messages"}

        trajectory = _run_rft_agent(client, model, messages, tools_schema)

        if trajectory["error"]:
            scores.append(0.0)
            errors += 1
            continue

        if trajectory["output_tools"]:
            tool_usage += 1

        sample = {
            "output_text": trajectory["output_text"],
            "output_tools": trajectory["output_tools"],
        }

        try:
            score = grade_fn(sample, item)
            scores.append(float(score))
        except Exception:
            scores.append(0.0)
            errors += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(data)}] scored")

    scored = [s for s in scores if s > 0]
    avg = sum(scored) / len(scored) if scored else 0

    # Pass rates at various thresholds
    pass_rates = {}
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pass_rates[str(t)] = round(sum(1 for s in scores if s >= t) / len(scores), 4) if scores else 0

    results = {
        "model": model,
        "label": label,
        "n": len(data),
        "errors": errors,
        "avg_score": round(avg, 4),
        "pass_rates": pass_rates,
        "tool_usage_rate": round(tool_usage / len(data), 4) if data else 0,
        "scores": [round(s, 4) for s in scores],
    }

    print(f"\n  {label}: {model}")
    print(f"    Avg score: {avg:.3f}")
    print(f"    Pass@0.8:  {pass_rates.get('0.8', 0):.0%}")
    print(f"    Pass@0.9:  {pass_rates.get('0.9', 0):.0%}")
    print(f"    Tool usage: {results['tool_usage_rate']:.0%}")
    print(f"    Errors: {errors}/{len(data)}")

    return results


# ── Phase 5: EXECUTE ─────────────────────────────────────────────────────

def cmd_execute(args):
    """Submit and monitor a single RFT job."""
    print("\n" + "=" * 60)
    print("  EXECUTE RFT")
    print("=" * 60)

    model = args.model
    train_id = args.train_file_id
    val_id = args.val_file_id
    pass_threshold = args.pass_threshold
    suffix = getattr(args, "suffix", None) or "auto-rft"

    # Load grader source
    grader_path = args.grader
    with open(grader_path, encoding="utf-8") as f:
        grader_source = f.read()

    # Load tools
    tools = _load_tools(getattr(args, "tools", None))

    # Build hyperparameters
    hp = dict(_DEFAULT_HYPERPARAMS)
    if getattr(args, "lr", None):
        hp["learning_rate_multiplier"] = args.lr
    if getattr(args, "epochs", None):
        hp["n_epochs"] = args.epochs
    if getattr(args, "compute_multiplier", None):
        hp["compute_multiplier"] = args.compute_multiplier
    if getattr(args, "reasoning_effort", None):
        hp["reasoning_effort"] = args.reasoning_effort

    print(f"  Model: {model}")
    print(f"  Train: {train_id}")
    print(f"  Val: {val_id}")
    print(f"  Pass threshold: {pass_threshold}")
    print(f"  Hyperparams: lr={hp['learning_rate_multiplier']}, epochs={hp['n_epochs']}, "
          f"compute={hp['compute_multiplier']}, reasoning={hp['reasoning_effort']}")

    client, _ = get_clients(
        base_url=getattr(args, "base_url", None),
        project_endpoint=getattr(args, "project_endpoint", None),
        api_key=getattr(args, "api_key", None),
    )

    # Build grader config
    grader_config = {
        "type": "python",
        "name": "auto_rft_grader",
        "source": grader_source.strip(),
        "pass_threshold": pass_threshold,
    }

    # Build method
    method = {
        "type": "reinforcement",
        "reinforcement": {
            "grader": grader_config,
            "hyperparameters": hp,
        },
    }
    if tools:
        method["reinforcement"]["tools"] = tools
        method["reinforcement"]["max_episode_steps"] = 5

    # Submit
    print(f"\n  Submitting RFT job...")
    start_time = time.time()
    try:
        job = client.fine_tuning.jobs.create(
            model=model,
            training_file=train_id,
            validation_file=val_id,
            suffix=suffix,
            method=method,
        )
        print(f"  ✅ Job: {job.id} | Status: {job.status}")
    except Exception as e:
        print(f"  ❌ Submission failed: {e}")
        return None

    # Monitor
    print(f"\n  Monitoring job (this may take hours)...")
    poll_interval = 60
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job.id)
        except Exception as e:
            print(f"  ⚠️  Poll error: {e}")
            time.sleep(poll_interval)
            continue

        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.0f}m"
        status = job.status

        if status in ("succeeded", "failed", "cancelled"):
            break

        print(f"  [{elapsed_str}] {status}...")
        time.sleep(poll_interval)

    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.0f}m"

    result = {
        "job_id": job.id,
        "status": job.status,
        "model": model,
        "fine_tuned_model": getattr(job, "fine_tuned_model", None),
        "suffix": suffix,
        "hyperparameters": hp,
        "pass_threshold": pass_threshold,
        "elapsed_s": round(elapsed),
        "elapsed_str": elapsed_str,
    }

    if job.status == "succeeded":
        print(f"\n  ✅ Job succeeded in {elapsed_str}")
        print(f"     Fine-tuned model: {result['fine_tuned_model']}")
    else:
        print(f"\n  ❌ Job {job.status} after {elapsed_str}")
        error = getattr(job, "error", None)
        if error:
            result["error"] = str(error)
            print(f"     Error: {error}")

    output = getattr(args, "output", None)
    if output:
        _atomic_json_write(output, result)
        print(f"  Output: {output}")

    return result


# ── Phase 6: EVALUATE (post-training) ────────────────────────────────────

def cmd_evaluate_post(args):
    """Deploy fine-tuned model, evaluate on test set, compare to baseline."""
    print("\n" + "=" * 60)
    print("  EVALUATE (post-training)")
    print("=" * 60)

    # Import deployment helpers from auto_finetune
    from auto_finetune import _deploy_model_arm, _delete_deployment_arm, _detect_azure_resource

    model_id = args.model_id  # fine-tuned model ID
    grade_fn = _load_grader(args.grader)
    tools = _load_tools(getattr(args, "tools", None))
    tools_schema = _build_tools_schema(tools)
    test_data = _load_jsonl(args.test_file)
    capacity = getattr(args, "capacity", 100)

    # Detect Azure resource for deployment
    sub, rg, account = _detect_azure_resource(
        base_url=getattr(args, "base_url", None)
    )
    if not all([sub, rg, account]):
        print(f"  ❌ Could not detect Azure resource. Need az CLI logged in.")
        return None

    print(f"  Azure: {account} ({rg})")
    print(f"  Model: {model_id}")
    print(f"  Test: {len(test_data)} examples")

    # Deploy
    deploy_name = f"eval-rft-{int(time.time()) % 10000}"
    print(f"\n  Deploying as '{deploy_name}' (capacity={capacity})...")
    ok = _deploy_model_arm(sub, rg, account, deploy_name, model_id, capacity=capacity)
    if not ok:
        print(f"  ❌ Deployment failed")
        return None

    # Wait for warmup
    client, _ = get_clients(
        base_url=getattr(args, "base_url", None),
        project_endpoint=getattr(args, "project_endpoint", None),
        api_key=getattr(args, "api_key", None),
    )

    print(f"  Warming up (may take 5-10 min)...")
    for attempt in range(12):
        try:
            client.chat.completions.create(
                model=deploy_name,
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=5,
            )
            print(f"  ✅ Model ready")
            break
        except Exception:
            time.sleep(30)
    else:
        print(f"  ⚠️  Warmup timed out, proceeding anyway")

    # Evaluate
    results = _evaluate_with_grader(client, deploy_name, test_data, grade_fn, tools_schema,
                                     label="FINE-TUNED")

    # Cleanup deployment
    print(f"\n  Cleaning up deployment '{deploy_name}'...")
    _delete_deployment_arm(sub, rg, account, deploy_name)

    output = getattr(args, "output", None)
    if output:
        results["fine_tuned_model"] = model_id
        _atomic_json_write(output, results)
        print(f"  Output: {output}")

    return results


# ── Phase 7: REVIEW ──────────────────────────────────────────────────────

def cmd_review(args):
    """Compare fine-tuned model to baseline and decide next action."""
    print("\n" + "=" * 60)
    print("  REVIEW")
    print("=" * 60)

    baseline = json.load(open(args.baseline, encoding="utf-8"))
    eval_results = json.load(open(args.eval_results, encoding="utf-8"))
    run_info = json.load(open(args.run_info, encoding="utf-8")) if getattr(args, "run_info", None) else {}

    base_avg = baseline.get("avg_score", 0)
    ft_avg = eval_results.get("avg_score", 0)
    lift = ft_avg - base_avg
    lift_pct = (lift / base_avg * 100) if base_avg > 0 else 0

    base_p80 = baseline.get("pass_rates", {}).get("0.8", 0)
    ft_p80 = eval_results.get("pass_rates", {}).get("0.8", 0)

    elapsed = run_info.get("elapsed_str", "?")
    hp = run_info.get("hyperparameters", {})

    print(f"\n  Baseline ({baseline.get('model', '?')}): avg={base_avg:.3f}, P@0.8={base_p80:.0%}")
    print(f"  Fine-tuned: avg={ft_avg:.3f}, P@0.8={ft_p80:.0%}")
    print(f"  Lift: {lift:+.3f} ({lift_pct:+.1f}%)")
    print(f"  Training time: {elapsed}")
    print(f"  HPs: lr={hp.get('learning_rate_multiplier', '?')}, "
          f"epochs={hp.get('n_epochs', '?')}, "
          f"reasoning={hp.get('reasoning_effort', '?')}")

    # Decision logic
    if lift_pct >= 5:
        decision = "SHIP"
        reason = f"Significant improvement: {lift_pct:+.1f}% lift"
    elif lift_pct >= 1:
        decision = "ITERATE"
        reason = f"Modest improvement ({lift_pct:+.1f}%), try different hyperparams"
    elif lift_pct > -2:
        decision = "ITERATE"
        reason = f"Flat result ({lift_pct:+.1f}%), adjust approach"
    else:
        decision = "STOP"
        reason = f"Regression ({lift_pct:+.1f}%), check grader alignment"

    # Suggest next hyperparams if iterating
    next_hp = None
    if decision == "ITERATE":
        next_hp = dict(hp)
        if lift_pct < 0:
            # Regression: lower LR
            next_hp["learning_rate_multiplier"] = max(0.3, hp.get("learning_rate_multiplier", 1.0) * 0.5)
            next_hp["n_epochs"] = min(hp.get("n_epochs", 3), 2)
        elif lift_pct < 3:
            # Modest: try higher reasoning or compute
            if hp.get("reasoning_effort") == "medium":
                next_hp["reasoning_effort"] = "high"
            else:
                next_hp["compute_multiplier"] = min(2.0, hp.get("compute_multiplier", 1.5) + 0.5)

    print(f"\n  {'='*50}")
    print(f"  Decision: {decision}")
    print(f"  Reason: {reason}")
    if next_hp and decision == "ITERATE":
        print(f"  Next HPs: lr={next_hp['learning_rate_multiplier']}, "
              f"epochs={next_hp['n_epochs']}, "
              f"reasoning={next_hp['reasoning_effort']}, "
              f"compute={next_hp['compute_multiplier']}")
    print(f"  {'='*50}")

    review = {
        "decision": decision,
        "reason": reason,
        "baseline_avg": base_avg,
        "ft_avg": ft_avg,
        "lift": round(lift, 4),
        "lift_pct": round(lift_pct, 2),
        "elapsed": elapsed,
        "hyperparameters": hp,
        "next_hyperparameters": next_hp,
    }

    output = getattr(args, "output", None)
    if output:
        _atomic_json_write(output, review)
        print(f"\n  Output: {output}")

    return review


# ── Full Auto Loop ───────────────────────────────────────────────────────

def cmd_auto(args):
    """Run the full RFT loop: validate → prepare → calibrate → baseline → execute → evaluate → review → iterate."""
    import argparse

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    max_iterations = getattr(args, "max_iterations", 2)

    print("=" * 60)
    print("  AUTO RFT")
    print("=" * 60)
    print(f"  Data:       {args.data}")
    print(f"  Grader:     {args.grader}")
    print(f"  Tools:      {getattr(args, 'tools', None) or '(none)'}")
    print(f"  Model:      {args.model}")
    print(f"  Work dir:   {work_dir}")
    print(f"  Max iters:  {max_iterations}")
    print("=" * 60)

    # ── Phase 1: VALIDATE ──
    print("\n\n" + "=" * 60)
    print("  PHASE 1: VALIDATE")
    print("=" * 60)
    validate_args = argparse.Namespace(
        data=args.data, grader=args.grader,
        tools=getattr(args, "tools", None), model=args.model,
    )
    cmd_validate(validate_args)

    # ── Phase 2: PREPARE ──
    print("\n\n" + "=" * 60)
    print("  PHASE 2: PREPARE")
    print("=" * 60)
    prepare_args = argparse.Namespace(
        data=args.data, work_dir=work_dir,
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
        project_endpoint=getattr(args, "project_endpoint", None),
    )
    manifest = cmd_prepare(prepare_args)

    # ── Phase 3: CALIBRATE ──
    print("\n\n" + "=" * 60)
    print("  PHASE 3: CALIBRATE")
    print("=" * 60)
    cal_output = os.path.join(work_dir, "calibration.json")
    cal_args = argparse.Namespace(
        data=manifest["val_path"], grader=args.grader,
        tools=getattr(args, "tools", None), model=args.model,
        n_samples=30, output=cal_output,
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
        project_endpoint=getattr(args, "project_endpoint", None),
    )
    cal_result = cmd_calibrate(cal_args)
    pass_threshold = cal_result["recommended_threshold"] if cal_result else 0.8

    # Allow user override
    if getattr(args, "pass_threshold", None):
        pass_threshold = args.pass_threshold
        print(f"  Using override pass_threshold: {pass_threshold}")

    # ── Phase 4: BASELINE ──
    print("\n\n" + "=" * 60)
    print("  PHASE 4: BASELINE")
    print("=" * 60)
    baseline_output = os.path.join(work_dir, "baseline.json")
    bl_args = argparse.Namespace(
        data=manifest["test_path"], grader=args.grader,
        tools=getattr(args, "tools", None), model=args.model,
        output=baseline_output,
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
        project_endpoint=getattr(args, "project_endpoint", None),
    )
    baseline = cmd_baseline(bl_args)

    # ── Iteration loop ──
    hp = dict(_DEFAULT_HYPERPARAMS)
    if getattr(args, "lr", None):
        hp["learning_rate_multiplier"] = args.lr
    if getattr(args, "epochs", None):
        hp["n_epochs"] = args.epochs
    if getattr(args, "compute_multiplier", None):
        hp["compute_multiplier"] = args.compute_multiplier
    if getattr(args, "reasoning_effort", None):
        hp["reasoning_effort"] = args.reasoning_effort

    for iteration in range(1, max_iterations + 1):
        print("\n\n" + "#" * 60)
        print(f"  ITERATION {iteration}/{max_iterations}")
        print("#" * 60)

        suffix = f"auto-rft-i{iteration}"
        run_output = os.path.join(work_dir, f"run_iter{iteration}.json")
        eval_output = os.path.join(work_dir, f"eval_iter{iteration}.json")
        review_output = os.path.join(work_dir, f"review_iter{iteration}.json")

        # ── Phase 5: EXECUTE ──
        print("\n" + "=" * 60)
        print(f"  PHASE 5: EXECUTE (iteration {iteration})")
        print("=" * 60)
        exec_args = argparse.Namespace(
            model=args.model, train_file_id=manifest["train_file"],
            val_file_id=manifest["val_file"], grader=args.grader,
            tools=getattr(args, "tools", None),
            pass_threshold=pass_threshold, suffix=suffix,
            lr=hp["learning_rate_multiplier"], epochs=hp["n_epochs"],
            compute_multiplier=hp["compute_multiplier"],
            reasoning_effort=hp["reasoning_effort"],
            output=run_output,
            base_url=getattr(args, "base_url", None),
            api_key=getattr(args, "api_key", None),
            project_endpoint=getattr(args, "project_endpoint", None),
        )
        run_result = cmd_execute(exec_args)

        if not run_result or run_result.get("status") != "succeeded":
            print(f"\n  ❌ Job failed in iteration {iteration}. Stopping.")
            break

        # ── Phase 6: EVALUATE ──
        print("\n" + "=" * 60)
        print(f"  PHASE 6: EVALUATE (iteration {iteration})")
        print("=" * 60)
        eval_args = argparse.Namespace(
            model_id=run_result["fine_tuned_model"],
            grader=args.grader, tools=getattr(args, "tools", None),
            test_file=manifest["test_path"], capacity=100,
            output=eval_output,
            base_url=getattr(args, "base_url", None),
            api_key=getattr(args, "api_key", None),
            project_endpoint=getattr(args, "project_endpoint", None),
        )
        eval_result = cmd_evaluate_post(eval_args)

        if not eval_result:
            print(f"\n  ❌ Evaluation failed in iteration {iteration}. Stopping.")
            break

        # ── Phase 7: REVIEW ──
        print("\n" + "=" * 60)
        print(f"  PHASE 7: REVIEW (iteration {iteration})")
        print("=" * 60)
        rev_args = argparse.Namespace(
            baseline=baseline_output, eval_results=eval_output,
            run_info=run_output, output=review_output,
        )
        review = cmd_review(rev_args)

        decision = review.get("decision", "STOP")

        if decision == "SHIP":
            print("\n\n" + "=" * 60)
            print("  ✅ SHIPPING — Fine-tuned model is ready!")
            print("=" * 60)
            print(f"  Model: {run_result['fine_tuned_model']}")
            print(f"  Lift: {review['lift_pct']:+.1f}%")
            print(f"  Training time: {run_result.get('elapsed_str', '?')}")
            return

        if decision == "STOP":
            print("\n\n" + "=" * 60)
            print("  ⏹️ STOPPING — No improvement path found")
            print("=" * 60)
            return

        # ITERATE — update hyperparams
        if review.get("next_hyperparameters"):
            hp = review["next_hyperparameters"]
            print(f"\n  🔄 Iterating with new HPs: lr={hp['learning_rate_multiplier']}, "
                  f"epochs={hp['n_epochs']}, reasoning={hp['reasoning_effort']}")

    # Fell through all iterations
    print("\n\n" + "=" * 60)
    print(f"  ⏹️ Completed {max_iterations} iteration(s)")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser():
    parser = HelpOnErrorParser(
        description="Autonomous RFT orchestrator — calibrate, train, evaluate, iterate",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_connection_args(p):
        p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
        p.add_argument("--api-key", default=os.environ.get("AZURE_OPENAI_API_KEY"))
        p.add_argument("--project-endpoint", default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"))

    def add_grader_tools(p):
        p.add_argument("--grader", required=True, help="Python grader file (must define grade(sample, item))")
        p.add_argument("--tools", default=None, help="Tools JSON file [{name, server_url, headers}]")

    # validate
    p = sub.add_parser("validate", help="Validate inputs (data, grader, tools, model)")
    p.add_argument("--data", required=True, help="Training data JSONL")
    p.add_argument("--model", default="o4-mini", help="Model (o4-mini or gpt-5)")
    add_grader_tools(p)

    # prepare
    p = sub.add_parser("prepare", help="Split and upload data")
    p.add_argument("--data", required=True, help="Training data JSONL")
    p.add_argument("--work-dir", default="./rft_run", help="Working directory")
    add_connection_args(p)

    # calibrate
    p = sub.add_parser("calibrate", help="Calibrate grader pass_threshold on validation data")
    p.add_argument("--data", required=True, help="Validation data JSONL")
    p.add_argument("--model", default="o4-mini")
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--output", default="calibration.json")
    add_grader_tools(p)
    add_connection_args(p)

    # baseline
    p = sub.add_parser("baseline", help="Evaluate base model on test set")
    p.add_argument("--data", required=True, help="Test data JSONL")
    p.add_argument("--model", default="o4-mini")
    p.add_argument("--output", default="baseline.json")
    add_grader_tools(p)
    add_connection_args(p)

    # execute
    p = sub.add_parser("execute", help="Submit and monitor RFT job")
    p.add_argument("--model", default="o4-mini")
    p.add_argument("--train-file-id", required=True, help="Uploaded training file ID")
    p.add_argument("--val-file-id", required=True, help="Uploaded validation file ID")
    p.add_argument("--pass-threshold", type=float, required=True)
    p.add_argument("--suffix", default="auto-rft")
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--compute-multiplier", type=float, default=1.5)
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--output", default="run.json")
    add_grader_tools(p)
    add_connection_args(p)

    # evaluate (post-training)
    p = sub.add_parser("evaluate", help="Deploy and evaluate fine-tuned model")
    p.add_argument("--model-id", required=True, help="Fine-tuned model ID")
    p.add_argument("--test-file", required=True, help="Test data JSONL")
    p.add_argument("--capacity", type=int, default=100)
    p.add_argument("--output", default="eval.json")
    add_grader_tools(p)
    add_connection_args(p)

    # review
    p = sub.add_parser("review", help="Compare fine-tuned model to baseline")
    p.add_argument("--baseline", required=True, help="Baseline results JSON")
    p.add_argument("--eval-results", required=True, help="Post-training eval JSON")
    p.add_argument("--run-info", default=None, help="Run info JSON (for hyperparams/timing)")
    p.add_argument("--output", default="review.json")

    # auto (full loop)
    p = sub.add_parser("auto", help="Full RFT loop: validate → prepare → calibrate → baseline → train → evaluate → iterate")
    p.add_argument("--data", required=True, help="Training data JSONL")
    p.add_argument("--model", default="o4-mini")
    p.add_argument("--work-dir", default="./rft_run")
    p.add_argument("--max-iterations", type=int, default=2)
    p.add_argument("--pass-threshold", type=float, default=None, help="Override calibrated threshold")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--compute-multiplier", type=float, default=None)
    p.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    add_grader_tools(p)
    add_connection_args(p)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    commands = {
        "validate": cmd_validate,
        "prepare": cmd_prepare,
        "calibrate": cmd_calibrate,
        "baseline": cmd_baseline,
        "execute": cmd_execute,
        "evaluate": cmd_evaluate_post,
        "review": cmd_review,
        "auto": cmd_auto,
    }

    commands[args.command](args)
