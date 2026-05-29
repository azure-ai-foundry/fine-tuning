# /// script
# dependencies = [
#   "openai>=1.0",
#   "requests",
#   "azure-identity",
# ]
# ///
"""
auto_finetune.py — Autonomous fine-tuning orchestrator (EXPERIMENTAL).

⚠️  EXPERIMENTAL: This tool automates the full SFT fine-tuning loop and is
best suited for exploration and quick prototyping. If you know what you're
doing (custom hyperparameters, checkpoint selection, RFT grader design),
use the individual scripts (submit_training.py, evaluate_model.py, etc.)
for finer control.

⚠️  SFT ONLY: This tool does NOT support RFT (reinforcement fine-tuning).
RFT requires manual grader design, threshold calibration, and reward curve
monitoring that don't lend themselves to full automation.
See workflows/full-pipeline.md and references/grader-design.md.

Manages the full SFT fine-tuning lifecycle: analyze data → prepare → baseline →
design candidates → execute → evaluate → review. Each phase is a subcommand
that reads/writes JSON artifacts, so the agent (or user) can run them
step-by-step or chain them.

Inspired by AIBuildAI's hierarchical manager + parallel candidates pattern.

Usage:
  python auto_finetune.py analyze --data raw.csv --output task_spec.json
  python auto_finetune.py prepare --task-spec task_spec.json --data raw.csv --output-dir ./prepared
  python auto_finetune.py baseline --task-spec task_spec.json --test-file ./prepared/test.jsonl
  python auto_finetune.py candidates --task-spec task_spec.json --data-dir ./prepared
  python auto_finetune.py execute --plan candidate_plan.json
  python auto_finetune.py evaluate --runs runs.json --test-file ./prepared/test.jsonl
  python auto_finetune.py review --leaderboard leaderboard.json --baseline baseline.json
"""

import hashlib
import json
import os
import random
import requests
import sys
import time
import uuid

# Fix Windows console encoding (cp1252 can't handle Unicode arrows/emoji)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HelpOnErrorParser, get_clients, estimate_training_cost, AZURE_PRICING_URL



# ── Model SKU/Tier Rules ─────────────────────────────────────────────────

# OSS models only support globalStandard; OAI models support standard,
# globalStandard, or developerTier.
_OSS_MODEL_PREFIXES = (
    "qwen", "llama", "ministral", "gpt-oss", "oss-20b",
    "phi-", "mistral", "deepseek",
)

# Models that only support RFT — cannot be used with auto-finetune (SFT only).
# Per Azure's supported model list, only `o4-mini` and `gpt-5` are RFT-only;
# they don't support SFT or DPO. Both are billed hourly, not per-token.
# Prefix match in `_is_rft_only_model` catches dated variants like
# `o4-mini-2025-04-16` and `gpt-5-2025-08-07`.
_RFT_ONLY_MODELS = ("o4-mini", "gpt-5")


def _is_rft_only_model(model_id):
    """Check if a model only supports RFT (not SFT)."""
    m = model_id.lower().split(".ft-")[0]  # strip fine-tune suffix
    return any(m == r or m.startswith(r + "-") for r in _RFT_ONLY_MODELS)


def _is_oss_model(model_id):
    """Check if a model is an OSS (open-source) model."""
    m = model_id.lower()
    return any(m.startswith(p) or p in m for p in _OSS_MODEL_PREFIXES)


def _resolve_tier(model_id, requested_tier):
    """Resolve the correct training tier for a model.

    OSS models only support globalStandard — override any other tier with a
    warning. OAI models support standard, globalStandard, or developerTier.
    """
    if _is_oss_model(model_id):
        if requested_tier and requested_tier != "globalStandard":
            print(f"  ⚠️  OSS model '{model_id}' only supports globalStandard "
                  f"(requested '{requested_tier}'). Overriding to globalStandard.")
        return "globalStandard"
    return requested_tier


def _parse_tiers(tier_arg):
    """Parse the --tier flag value into a list of tiers for round-robin assignment.

    Accepts a single tier ('globalStandard') or comma-separated list
    ('globalStandard,developerTier'). Returns a non-empty list of strings.
    Used to distribute candidates across tiers when capacity on one is
    constrained.
    """
    if not tier_arg:
        return ["globalStandard"]
    tiers = [t.strip() for t in tier_arg.split(",") if t.strip()]
    return tiers or ["globalStandard"]


# ── Helpers ──────────────────────────────────────────────────────────────

import re as _re_module


def _sanitize_name(name, max_len=40):
    """Sanitize a name for use as API suffix and filename.

    Strips control chars, replaces non-alphanumeric with hyphens,
    collapses runs of hyphens, and caps length.
    """
    s = _re_module.sub(r'[^A-Za-z0-9-]', '-', name or "auto")
    s = _re_module.sub(r'-+', '-', s).strip('-')
    return s[:max_len] or "auto"


def _atomic_json_write(path, data):
    """Write JSON atomically — write to temp file then rename to prevent corruption."""
    import tempfile
    dir_name = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", dir=dir_name,
                                     delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


# Judge model preference order (strongest first)
_JUDGE_CANDIDATES = ["gpt-5-4", "gpt-5.4", "gpt-4.1", "gpt-4o", "gpt-4.1-mini"]


def _detect_judge_model(args):
    """Probe the endpoint to find the best available judge model.

    Tries models in preference order, sends a trivial completion to verify
    the deployment exists. Returns the first that works, or falls back to
    the base model with a warning.
    """
    try:
        client, _ = get_clients(
            base_url=getattr(args, "base_url", None),
            project_endpoint=getattr(args, "project_endpoint", None),
            api_key=getattr(args, "api_key", None),
        )
    except Exception:
        print("  ⚠️  Could not connect to endpoint — defaulting judge to gpt-4.1-mini")
        return "gpt-4.1-mini"

    for model in _JUDGE_CANDIDATES:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=5,
            )
            print(f"  Judge model: {model} (auto-detected)")
            return model
        except Exception as e:
            # 400 = model exists but rejected our params (e.g., too few tokens) — still valid
            if "400" in str(e):
                print(f"  Judge model: {model} (auto-detected)")
                return model
            continue

    base = getattr(args, "model", "gpt-4.1-mini") or "gpt-4.1-mini"
    print(f"  ⚠️  No preferred judge found — falling back to base model ({base})")
    return base


# ── Phase 1: ANALYZE ──────────────────────────────────────────────────────

def cmd_analyze(args):
    """Read raw data (if provided) or just a description, and generate task_spec.json."""
    data_path = getattr(args, "data", None)

    # ── Prompt-only mode: no data file, just a description ──
    if not data_path:
        if not args.description:
            print("❌ Either --data or --description is required.")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"  TASK ANALYSIS (prompt-only)")
        print(f"{'='*60}")
        print(f"  Description: {args.description}")
        print(f"  No data file — will generate training data synthetically")

        judge_model = _detect_judge_model(args)

        task_spec = {
            "task_name": args.task_name or "custom-task",
            "description": args.description,
            "data_file": None,
            "total_rows": 0,
            "columns": [],
            "is_chat_format": False,
            "data_mode": "prompt_only",
            "hypotheses": [{"task_type": "generation", "confidence": 0.7}],
            "selected_hypothesis": 0,
            "requires_confirmation": True,
            "base_model": args.model or "gpt-4.1-mini",
            "eval_rubric": {
                "dimensions": [
                    {"name": "correctness", "weight": 0.7, "description": "Does the output correctly accomplish the task?"},
                    {"name": "quality", "weight": 0.3, "description": "Is the output well-written and appropriately concise?"},
                ],
                "pass_threshold": 8,
                "judge_model": judge_model,
            },
            "stopping_criteria": {
                "min_lift_pct": 5.0,
                "max_iterations": 3,
                "max_budget_usd": float(args.max_budget) if args.max_budget else 50.0,
            },
        }

        output = args.output or "task_spec.json"
        with open(output, "w", encoding="utf-8") as f:
            json.dump(task_spec, f, indent=2)

        print(f"\n  Data mode: prompt_only — all training data will be generated")
        print(f"  Output: {output}")
        return

    # ── Standard mode: analyze a data file ──
    ext = os.path.splitext(data_path)[1].lower()

    # Load sample rows
    if ext == ".csv":
        rows, columns = _load_csv_sample(data_path, n=20)
    elif ext in (".json", ".jsonl"):
        rows, columns = _load_json_sample(data_path, n=20)
    elif ext == ".parquet":
        rows, columns = _load_parquet_sample(data_path, n=20)
    else:
        print(f"Unsupported format: {ext}. Use .csv, .json, .jsonl, or .parquet")
        sys.exit(1)

    total_rows = _count_rows(data_path, ext)

    # Check if data is already in SFT chat format
    is_chat_format = _detect_chat_format(rows)

    # Detect labeled vs unlabeled
    hypotheses = _generate_hypotheses(columns, rows, is_chat_format, args.description)

    # Build task spec
    best = hypotheses[0] if hypotheses else None
    requires_confirmation = not best or best.get("confidence", 0) < 0.8

    if is_chat_format:
        data_mode = "chat_sft"
    elif best and best.get("output_col"):
        data_mode = "labeled"
    else:
        data_mode = "unlabeled"

    # Pick the best available judge model by probing the endpoint
    judge_model = _detect_judge_model(args)

    task_spec = {
        "task_name": args.task_name or os.path.splitext(os.path.basename(data_path))[0],
        "description": args.description or "",
        "data_file": os.path.abspath(data_path),
        "total_rows": total_rows,
        "columns": columns,
        "is_chat_format": is_chat_format,
        "data_mode": data_mode,
        "hypotheses": hypotheses,
        "selected_hypothesis": 0,
        "requires_confirmation": requires_confirmation,
        "base_model": args.model or "gpt-4.1-mini",
        "eval_rubric": {
            "dimensions": [
                {"name": "correctness", "weight": 0.7, "description": "Does the output correctly accomplish the task?"},
                {"name": "quality", "weight": 0.3, "description": "Is the output well-written and appropriately concise?"},
            ],
            "pass_threshold": 8,
            "judge_model": judge_model,
        },
        "stopping_criteria": {
            "min_lift_pct": 5.0,
            "max_iterations": 3,
            "max_budget_usd": float(args.max_budget) if args.max_budget else 50.0,
        },
    }

    output = args.output or "task_spec.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(task_spec, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print(f"  TASK ANALYSIS: {task_spec['task_name']}")
    print(f"{'='*60}")
    print(f"  Data: {data_path} ({total_rows} rows, {len(columns)} columns)")
    print(f"  Format: {'Chat SFT (already formatted)' if is_chat_format else ext}")
    print(f"  Data mode: {data_mode}")

    if hypotheses:
        print(f"\n  Task hypotheses:")
        for i, h in enumerate(hypotheses):
            marker = " ← selected" if i == 0 else ""
            print(f"    [{i}] {h['task_type']} (confidence: {h['confidence']:.0%}){marker}")
            if h.get("input_col"):
                print(f"        input: {h['input_col']} → output: {h.get('output_col', 'N/A')}")
    else:
        print(f"\n  ⚠️  Could not infer task type. Please specify manually.")

    if requires_confirmation:
        print(f"\n  ⚠️  CONFIRMATION REQUIRED: Review task_spec.json and confirm before proceeding.")
    if data_mode == "unlabeled":
        print(f"\n  ⚠️  No labels detected. You'll need to:")
        print(f"     (a) specify a label column, (b) generate synthetic labels, or (c) provide a gold eval set")

    print(f"\n  Output: {output}")
    print(f"  Next: Review/edit {output}, then run: auto_finetune.py prepare --task-spec {output} --data {data_path}")


def _load_csv_sample(path, n=20):
    import csv
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = []
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(row)
    return rows, columns


def _load_json_sample(path, n=20):
    rows = []
    columns = set()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                    columns.update(obj.keys())
            except json.JSONDecodeError:
                continue
    return rows, sorted(columns)


def _load_parquet_sample(path, n=20):
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas + pyarrow required for parquet. Install: pip install pandas pyarrow")
        sys.exit(1)
    df = pd.read_parquet(path).head(n)
    rows = df.to_dict(orient="records")
    columns = list(df.columns)
    return rows, columns


def _count_rows(path, ext):
    if ext == ".csv":
        with open(path, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f) - 1  # minus header
    elif ext in (".json", ".jsonl"):
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    elif ext == ".parquet":
        try:
            import pandas as pd
            return len(pd.read_parquet(path))
        except Exception:
            return -1
    return -1


def _detect_chat_format(rows):
    """Check if data is already in SFT chat-completion format."""
    if not rows:
        return False
    for row in rows[:5]:
        if "messages" in row:
            msgs = row["messages"]
            if isinstance(msgs, list) and any(isinstance(m, dict) and "role" in m for m in msgs):
                return True
    return False


def _generate_hypotheses(columns, rows, is_chat_format, description=""):
    """Generate task type hypotheses from schema + sample data."""
    hypotheses = []

    if is_chat_format:
        hypotheses.append({
            "task_type": "chat_sft",
            "input_col": "messages",
            "output_col": None,
            "confidence": 0.95,
            "notes": "Data is already in SFT chat format"
        })
        return hypotheses

    desc_lower = (description or "").lower()
    col_lower = [c.lower() for c in columns]

    # Classification signals
    classification_labels = {"label", "category", "class", "target", "output", "tag", "type",
                             "sentiment", "intent", "topic", "status", "priority", "severity",
                             "rating", "score", "verdict", "decision", "result"}
    for out_col in columns:
        if out_col.lower() in classification_labels:
            # Check cardinality
            vals = set(str(r.get(out_col, "")) for r in rows)
            if 2 <= len(vals) <= 50:
                input_cols = [c for c in columns if c != out_col]
                text_col = _find_text_column(input_cols, rows)
                hypotheses.append({
                    "task_type": "classification",
                    "input_col": text_col or input_cols[0] if input_cols else None,
                    "output_col": out_col,
                    "num_classes": len(vals),
                    "confidence": 0.85,
                })

    # Cardinality-based classification fallback: if one column has low cardinality
    # and another has long text, it's likely classification even with unknown column names
    if not any(h["task_type"] == "classification" for h in hypotheses) and len(columns) >= 2:
        text_col = _find_text_column(columns, rows)
        for col in columns:
            if col == text_col:
                continue
            vals = set(str(r.get(col, "")) for r in rows)
            if 2 <= len(vals) <= 30:
                hypotheses.append({
                    "task_type": "classification",
                    "input_col": text_col,
                    "output_col": col,
                    "num_classes": len(vals),
                    "confidence": 0.65,
                    "notes": f"Inferred from cardinality ({len(vals)} unique values in '{col}')",
                })
                break

    # Generation signals (input/output or prompt/response columns)
    for in_name, out_name in [("input", "output"), ("prompt", "response"), ("question", "answer"),
                               ("instruction", "code"), ("text", "summary"), ("source", "target")]:
        in_col = next((c for c in columns if c.lower() == in_name), None)
        out_col = next((c for c in columns if c.lower() == out_name), None)
        if in_col and out_col:
            hypotheses.append({
                "task_type": "generation",
                "input_col": in_col,
                "output_col": out_col,
                "confidence": 0.88,
            })

    # Fallback: description-based hints
    if "classif" in desc_lower:
        if not any(h["task_type"] == "classification" for h in hypotheses):
            hypotheses.append({"task_type": "classification", "confidence": 0.5, "input_col": None, "output_col": None})
    if any(kw in desc_lower for kw in ["generat", "translat", "summar", "code", "write"]):
        if not any(h["task_type"] == "generation" for h in hypotheses):
            hypotheses.append({"task_type": "generation", "confidence": 0.5, "input_col": None, "output_col": None})

    # Sort by confidence
    hypotheses.sort(key=lambda h: h.get("confidence", 0), reverse=True)
    return hypotheses


def _find_text_column(columns, rows):
    """Find the most likely text input column (longest average string)."""
    avg_lens = {}
    for col in columns:
        lens = [len(str(r.get(col, ""))) for r in rows]
        avg_lens[col] = sum(lens) / len(lens) if lens else 0
    if avg_lens:
        return max(avg_lens, key=avg_lens.get)
    return None


# ── Phase 1b: GENERATE ────────────────────────────────────────────────────

def cmd_generate(args):
    """Generate quality training data from a task spec using a teacher model.

    Produces diverse examples across difficulty levels, quality-scores each one,
    filters to a threshold, and writes SFT chat JSONL.
    
    If --existing-data is provided, loads those examples to avoid duplicates
    and merges new examples with the existing set.
    """
    with open(args.task_spec, encoding="utf-8") as f:
        spec = json.load(f)

    import re

    client, method = get_clients(
        base_url=args.base_url, project_endpoint=args.project_endpoint, api_key=args.api_key
    )

    teacher = args.teacher
    if not teacher:
        # Auto-detect best available teacher (prefer strongest model for data quality)
        teacher = _detect_judge_model(args)  # Same probe logic — strongest available model
        print(f"  Teacher model: {teacher} (auto-detected)")
    target_count = args.num_examples
    description = spec.get("description", "")
    task_name = spec.get("task_name", "task")
    schema = args.schema_file
    schema_text = ""
    if schema and os.path.exists(schema):
        with open(schema, encoding="utf-8") as f:
            schema_text = f.read()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load existing data for deduplication (if augmenting)
    existing_inputs = set()
    existing_examples = []
    if args.existing_data and os.path.exists(args.existing_data):
        with open(args.existing_data, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    existing_examples.append(ex)
                    # Extract user content as dedup key
                    user_msg = next((m["content"] for m in ex.get("messages", []) if m["role"] == "user"), "")
                    existing_inputs.add(user_msg.lower().strip()[:200])
        print(f"  Loaded {len(existing_examples)} existing examples for deduplication")

    # Build the generation prompt based on task type
    task_type = spec.get("hypotheses", [{}])[0].get("task_type", "generation")
    gen_prompt = _build_generation_prompt(task_type, description, schema_text, target_count)

    # Add dedup hint to prompt if we have existing data
    if existing_inputs:
        sample_existing = list(existing_inputs)[:5]
        gen_prompt += (
            f"\n\nIMPORTANT: You are augmenting an existing dataset. "
            f"Do NOT generate examples similar to these (showing {len(sample_existing)} of {len(existing_inputs)}):\n"
            + "\n".join(f"- {s[:100]}" for s in sample_existing)
            + "\nGenerate DIFFERENT examples covering new scenarios, edge cases, and patterns."
        )

    print(f"\n{'='*60}")
    print(f"  DATA GENERATION: {task_name}")
    print(f"{'='*60}")
    print(f"  Teacher model: {teacher}")
    print(f"  Target NEW examples: {target_count}")
    if existing_inputs:
        print(f"  Existing examples: {len(existing_inputs)} (will merge after dedup)")
    print(f"  Task type: {task_type}")
    if schema_text:
        print(f"  Schema: {schema} ({len(schema_text)} chars)")

    # Generate in batches of 20
    batch_size = 20
    batches_needed = (target_count + batch_size - 1) // batch_size
    all_examples = []

    for batch_num in range(batches_needed):
        remaining = target_count - len(all_examples)
        n = min(batch_size, remaining)
        if n <= 0:
            break

        # Difficulty distribution based on --difficulty flag
        diff_mode = getattr(args, 'difficulty', 'mixed') if hasattr(args, 'difficulty') else 'mixed'
        if diff_mode == "easy":
            difficulty = "simple" if batch_num < batches_needed * 0.7 else (
                "medium" if batch_num < batches_needed * 0.9 else "complex"
            )
        elif diff_mode == "hard":
            difficulty = "simple" if batch_num < batches_needed * 0.2 else (
                "medium" if batch_num < batches_needed * 0.6 else "complex"
            )
        else:  # mixed (default)
            difficulty = "simple" if batch_num < batches_needed * 0.4 else (
                "medium" if batch_num < batches_needed * 0.8 else "complex"
            )

        print(f"\n  Batch {batch_num + 1}/{batches_needed} ({difficulty}, {n} examples)...")

        batch_prompt = (
            f"{gen_prompt}\n\n"
            f"Generate exactly {n} examples at {difficulty} difficulty level.\n"
            f"Batch {batch_num + 1} — make these DIFFERENT from any prior batches.\n"
            f"{'Focus on ' + difficulty + ' patterns.' if difficulty != 'medium' else 'Mix of patterns.'}\n\n"
            f"Return a JSON object with an \"examples\" array. Each example has "
            f"\"input\" (the user's request) and \"output\" (the correct response)."
        )

        for attempt in range(3):
            try:
                # Use max_completion_tokens (works on all models including gpt-5.x)
                # Skip response_format entirely — parse JSON from response instead
                resp = client.chat.completions.create(
                    model=teacher,
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=1.0 if difficulty != "complex" else 0.8,
                    max_completion_tokens=4096,
                )

                content = resp.choices[0].message.content
                # Extract JSON from response (may have markdown fences)
                import re as _re
                json_match = _re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    parsed = json.loads(content)

                # Extract examples from various possible response shapes
                examples = parsed.get("examples", parsed.get("data", []))
                if isinstance(parsed, list):
                    examples = parsed

                valid = []
                for ex in examples:
                    inp = ex.get("input", ex.get("question", ex.get("prompt", "")))
                    out = ex.get("output", ex.get("answer", ex.get("response", ex.get("sql", ""))))
                    if inp and out and len(inp.strip()) > 5 and len(out.strip()) > 2:
                        valid.append({"input": inp.strip(), "output": out.strip()})

                all_examples.extend(valid)
                print(f"    Got {len(valid)} valid examples (total: {len(all_examples)})")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"    Retry {attempt + 1}: {str(e)[:60]}")
                    time.sleep(2)
                else:
                    print(f"    Failed after 3 attempts: {str(e)[:60]}")

        time.sleep(0.5)  # Rate limiting between batches

    if not all_examples:
        print("\n  No examples generated. Check teacher model deployment and API key.")
        sys.exit(1)

    # Quality scoring
    print(f"\n  Scoring {len(all_examples)} examples...")
    scored = []
    for i, ex in enumerate(all_examples):
        score = _score_example(client, teacher, ex["input"], ex["output"], description)
        ex["quality_score"] = score
        scored.append(ex)
        if (i + 1) % 25 == 0:
            avg = sum(e["quality_score"] for e in scored) / len(scored)
            print(f"    [{i+1}/{len(all_examples)}] avg quality: {avg:.1f}/10")

    # Filter by quality threshold
    threshold = args.min_quality
    filtered = [ex for ex in scored if ex["quality_score"] >= threshold]
    avg_quality = sum(ex["quality_score"] for ex in scored) / len(scored) if scored else 0

    print(f"\n  Quality filter (threshold={threshold}):")
    print(f"    Generated: {len(scored)}")
    print(f"    Passed: {len(filtered)} ({len(filtered)/len(scored)*100:.0f}%)")
    print(f"    Avg quality: {avg_quality:.1f}/10")

    if len(filtered) < 50:
        print(f"\n  Only {len(filtered)} examples passed quality filter. Consider:")
        print(f"    - Lowering --min-quality (currently {threshold})")
        print(f"    - Increasing --num-examples (currently {target_count})")
        print(f"    - Improving the task description")

    # Deduplicate (against both new examples and existing data)
    seen = set(existing_inputs)  # start with existing inputs as already-seen
    unique = []
    dupes_vs_existing = 0
    dupes_vs_new = 0
    for ex in filtered:
        key = ex["input"].lower().strip()[:200]
        if key in existing_inputs:
            dupes_vs_existing += 1
        elif key in seen:
            dupes_vs_new += 1
        else:
            seen.add(key)
            unique.append(ex)
    if dupes_vs_existing or dupes_vs_new:
        print(f"    Deduped: {len(filtered)} -> {len(unique)} ({dupes_vs_existing} matched existing, {dupes_vs_new} internal dupes)")
    filtered = unique

    # Build system prompt from task description + schema
    if schema_text:
        system_prompt = f"{description}\n\nSchema:\n{schema_text}"
    else:
        system_prompt = description or "You are a helpful assistant."

    # Convert to SFT chat format
    chat_examples = []
    for ex in filtered:
        chat_examples.append({"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]})

    # Merge with existing data if augmenting
    if existing_examples:
        merged = existing_examples + chat_examples
        print(f"\n  Merged: {len(existing_examples)} existing + {len(chat_examples)} new = {len(merged)} total")
    else:
        merged = chat_examples

    # Write output
    output_path = os.path.join(output_dir, "generated_data.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in merged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Also write raw scored data for inspection
    raw_path = os.path.join(output_dir, "generated_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"total_generated": len(scored), "passed_filter": len(filtered),
                   "avg_quality": round(avg_quality, 2), "threshold": threshold,
                   "examples": scored}, f, indent=2)

    print(f"\n  Output: {output_path} ({len(merged)} total examples{', ' + str(len(chat_examples)) + ' new' if existing_examples else ''})")
    print(f"  Raw scores: {raw_path}")
    print(f"\n  Next: auto_finetune.py prepare --task-spec {args.task_spec} --data {output_path}")


# ── Phase 2 alternative: FOUNDRY-GENERATE (Foundry Data Generation API) ──

def cmd_foundry_generate(args):
    """Generate training/eval data via the Foundry Data Generation API.

    Alternative to `cmd_generate` (which runs a custom teacher loop locally).
    Use this when:
      - You want to generate from real agent traces (--source traces)
      - You want tool-calling SFT data from an OpenAPI 3.0 spec (--source file
        --recipe tool-use)
      - You want an evaluation dataset (--scenario eval)
      - You want the service to handle quality control instead of the in-script scorer

    See workflows/synthetic-datagen.md and references/data-generation-api.md
    for the full API.

    Output is normalised to <output-dir>/generated_data.jsonl so the rest of
    the auto_finetune pipeline (prepare/baseline/candidates/etc.) just works.
    """
    import shutil
    import subprocess
    import tempfile

    if not args.project_endpoint:
        sys.exit("--project-endpoint required (or set AZURE_AI_PROJECT_ENDPOINT) for Foundry data generation")

    with open(args.task_spec, encoding="utf-8") as f:
        spec = json.load(f)

    task_name = spec.get("task_name", "task")
    description = spec.get("description", "")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Generate a short, unique output name (Foundry caps at 50 chars; needs [a-z0-9-])
    run_id = uuid.uuid4().hex[:8]
    short_task = "".join(c if c.isalnum() or c == "-" else "-" for c in task_name.lower())[:25].strip("-")
    output_name = f"af-{short_task}-{run_id}"[:50]

    script = os.path.join(os.path.dirname(__file__), "generate_dataset.py")
    if not os.path.exists(script):
        sys.exit(f"generate_dataset.py not found at {script}")

    cmd = [
        sys.executable, script,
        "--project-endpoint", args.project_endpoint,
        "--source", args.source,
        "--recipe", args.recipe,
        "--scenario", args.scenario,
        "--max-samples", str(args.max_samples),
        "--output-name", output_name,
        "--download",
    ]
    if args.train_split is not None:
        cmd += ["--train-split", str(args.train_split)]
    if args.teacher:
        cmd += ["--teacher", args.teacher]
    if args.api_key:
        # generate_dataset.py uses DefaultAzureCredential; --api-key is for OAI clients
        pass

    # Source-specific args (validated against generate_dataset.py's own argparse)
    if args.source == "prompt-inline":
        prompt = args.prompt or description
        if not prompt:
            sys.exit("--prompt or task spec 'description' required for --source prompt-inline")
        cmd += ["--prompt", prompt[:10000]]  # service cap
    elif args.source == "prompt-file":
        if not args.prompt_file:
            sys.exit("--prompt-file required for --source prompt-file")
        cmd += ["--prompt-file", args.prompt_file]
    elif args.source == "file":
        if not args.file_id:
            sys.exit("--file-id required for --source file (upload via openai.files.create first)")
        cmd += ["--file-id", args.file_id]
    elif args.source == "agent":
        if not args.agent_name:
            sys.exit("--agent-name required for --source agent")
        cmd += ["--agent-name", args.agent_name]
        if args.agent_version:
            cmd += ["--agent-version", args.agent_version]
    elif args.source == "traces":
        if not args.agent_name:
            sys.exit("--agent-name required for --source traces")
        cmd += ["--agent-name", args.agent_name]
        if args.agent_version:
            cmd += ["--agent-version", args.agent_version]
        if args.hours is not None:
            cmd += ["--hours", str(args.hours)]

    print(f"\n{'='*60}")
    print(f"  FOUNDRY DATA GENERATION: {task_name}")
    print(f"{'='*60}")
    print(f"  Source:      {args.source}")
    print(f"  Recipe:      {args.recipe}")
    print(f"  Scenario:    {args.scenario}")
    print(f"  Teacher:     {args.teacher or '(not required for traces)'}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Output name: {output_name}")
    print(f"{'='*60}\n")

    # Run from a temp cwd so --download lands in a known place, then merge
    tmpdir = tempfile.mkdtemp(prefix="foundry-datagen-")
    try:
        result = subprocess.run(
            cmd, cwd=tmpdir, capture_output=False, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            sys.exit(f"generate_dataset.py exited with {result.returncode}")

        # Discover downloaded files
        downloaded = sorted([
            os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
            if f.endswith(".jsonl")
        ])
        if not downloaded:
            sys.exit(f"No JSONL files downloaded to {tmpdir} — job may have produced 0 samples or a Dataset (EVAL) output")

        # Merge train+valid (if both) into a single generated_data.jsonl, while
        # normalising row-level quirks that the Foundry traces export produces and
        # that Azure FT preprocessing rejects. Specifically:
        #   - assistant messages with tool_calls sometimes have content="null"
        #     (the string) — must be JSON null or empty string for FT preprocess
        #     to accept the row.
        # See bugs_found table (run=test_agent_run, FT preprocessing failed).
        merged_path = os.path.join(output_dir, "generated_data.jsonl")
        total = 0
        normalised = 0
        dropped_malformed = 0
        with open(merged_path, "w", encoding="utf-8") as out:
            for path in downloaded:
                with open(path, encoding="utf-8") as src:
                    for line in src:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            fixed_count = _normalize_for_ft(row)
                            if fixed_count < 0:
                                # Row is fundamentally broken (e.g. assistant
                                # tool_call without matching tool reply). Drop it.
                                dropped_malformed += 1
                                continue
                            normalised += fixed_count
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            total += 1
                        except json.JSONDecodeError:
                            out.write(line.rstrip("\n") + "\n")
                            total += 1
                shutil.copy2(path, os.path.join(output_dir, os.path.basename(path)))

        print(f"\n  Merged {len(downloaded)} file(s) → {merged_path} ({total} examples)")
        if normalised:
            print(f"  Normalised {normalised} messages (e.g. content='null' → null) for FT compatibility")
        if dropped_malformed:
            print(f"  Dropped {dropped_malformed} malformed rows (asst tool_call without matching tool reply)")
        print(f"  Source files copied to {output_dir}")
        print(f"\n  Next: auto_finetune.py prepare --task-spec {args.task_spec} --data {merged_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _normalize_for_ft(row: dict) -> int:
    """Normalise a single chat-SFT row for Azure FT preprocessing acceptance.

    Returns the number of messages fixed. Currently fixes:
      - assistant messages with tool_calls that have `content: "null"` (string).
        Azure FT rejects these rows. Convert to actual None.

    Returns -1 if the row is fundamentally malformed and should be DROPPED.
    The caller should treat negative returns as "skip this row entirely."

    Drop conditions:
      - Assistant message with tool_calls but NOT immediately followed by tool
        reply messages with matching tool_call_id for EACH call. Foundry traces
        sometimes export truncated turns where a tool call was issued but the
        reply was never recorded. Azure FT rejects the whole file if any row
        has this — so we drop the offending rows.
    """
    fixed = 0
    msgs = row.get("messages", []) or []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            c = m.get("content")
            if c == "null":  # literal 4-char string from trace export bug
                m["content"] = None
                fixed += 1

    # Sequence check: every assistant tool_call.id must appear as
    # tool_call_id in a subsequent tool message before any other assistant turn.
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        expected_ids = {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}
        seen_ids = set()
        for fm in msgs[i + 1:]:
            if fm.get("role") == "tool" and fm.get("tool_call_id"):
                seen_ids.add(fm["tool_call_id"])
            elif fm.get("role") in ("assistant", "user", "system"):
                # Reached the next non-tool turn; tool replies must have come before
                break
        if not expected_ids.issubset(seen_ids):
            return -1
    return fixed


def _build_generation_prompt(task_type, description, schema_text, target_count):
    """Build the data generation prompt based on task type."""
    base = f"You are generating training data for a fine-tuning task.\n\nTask: {description}\n"

    if schema_text:
        base += f"\nSchema/Context:\n{schema_text}\n"

    if task_type == "classification":
        base += (
            "\nGenerate diverse classification examples. Each should have:\n"
            "- 'input': the text to classify\n"
            "- 'output': the correct class label\n"
            "Vary the inputs: different lengths, tones, edge cases, ambiguous examples.\n"
            "Ensure balanced class distribution."
        )
    elif task_type in ("code", "generation") and any(kw in description.lower() for kw in ["sql", "query", "database"]):
        base += (
            "\nGenerate diverse natural-language-to-SQL examples. Each should have:\n"
            "- 'input': a natural language question about the data\n"
            "- 'output': the correct SQL query\n"
            "Vary complexity: simple selects, joins, aggregations, subqueries, window functions, CTEs.\n"
            "Use realistic business questions a data analyst would ask.\n"
            "Reference actual table and column names from the schema."
        )
    elif task_type in ("code", "generation") and any(kw in description.lower() for kw in ["python", "code", "function", "program"]):
        base += (
            "\nGenerate diverse natural-language-to-code examples. Each should have:\n"
            "- 'input': a clear description of what the code should do\n"
            "- 'output': clean, correct, well-documented code\n"
            "Vary complexity: simple functions, class designs, algorithms, file I/O, API calls.\n"
            "Cover different domains: data processing, web, CLI tools, math, string manipulation."
        )
    else:
        base += (
            "\nGenerate diverse training examples. Each should have:\n"
            "- 'input': the user's request or question\n"
            "- 'output': the ideal response\n"
            "Vary inputs by length, complexity, tone, and edge cases.\n"
            "Make outputs high-quality and consistent in style."
        )

    return base


def _score_example(client, model, input_text, output_text, task_description, retries=2):
    """Score a single generated example on quality (1-10)."""
    import re
    prompt = (
        f"Rate this training example for a fine-tuning task on a scale of 1-10.\n\n"
        f"Task: {task_description}\n\n"
        f"Input: {input_text[:500]}\n\n"
        f"Output: {output_text[:500]}\n\n"
        f"Score on: correctness, diversity/usefulness, clarity.\n"
        f"Return ONLY a JSON object: {{\"score\": <int 1-10>}}"
    )
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_completion_tokens=50,
            )
            text = resp.choices[0].message.content
            match = re.search(r'\{[^}]+\}', text)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = json.loads(text)
            return int(parsed.get("score", 0))
        except Exception:
            if attempt < retries:
                time.sleep(1)
    return 0


def _data_governance_check(examples):
    """Basic data governance checks: encoding errors, empty content, suspicious patterns."""
    warnings = []
    for i, ex in enumerate(examples):
        text = json.dumps(ex)
        # Check for encoding errors
        if "\ufffd" in text:
            warnings.append(f"Example {i}: contains replacement characters (encoding error)")
        # Check for empty messages
        for msg in ex.get("messages", []):
            if msg.get("role") == "assistant" and not (msg.get("content") or "").strip():
                warnings.append(f"Example {i}: empty assistant response")
            if msg.get("role") == "user" and not (msg.get("content") or "").strip():
                warnings.append(f"Example {i}: empty user prompt")
        # Check for extremely long examples (may hit token limits)
        if len(text) > 32000:
            warnings.append(f"Example {i}: very long ({len(text)} chars) -- may exceed token limit")
    return warnings


# ── Phase 2: PREPARE ──────────────────────────────────────────────────────

def cmd_prepare(args):
    """Convert raw data to SFT format, quality-filter, split."""
    with open(args.task_spec, encoding="utf-8") as f:
        spec = json.load(f)

    # Check for unlabeled data — route to generate first
    if spec.get("data_mode") == "unlabeled":
        print("Data is unlabeled. To proceed, either:")
        print(f"  1. Edit {args.task_spec} to set input_col/output_col in the selected hypothesis")
        print(f"  2. Run: auto_finetune.py generate --task-spec {args.task_spec} --num-examples 200")
        print(f"     Then: auto_finetune.py prepare --task-spec {args.task_spec} --data ./generated/generated_data.jsonl")
        sys.exit(1)

    data_path = args.data or spec.get("data_file")
    if not data_path or not os.path.exists(data_path):
        print(f"Error: data file not found: {data_path}")
        sys.exit(1)

    output_dir = args.output_dir or "./prepared"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert to chat JSONL (or load if already chat format)
    # Generated data from prompt_only mode is already in chat format
    if spec.get("is_chat_format") or spec.get("data_mode") == "prompt_only":
        print("Data is already in chat format. Loading directly...")
        with open(data_path, encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]
    else:
        hyp = spec["hypotheses"][spec.get("selected_hypothesis", 0)]
        examples = _convert_to_chat(data_path, hyp, spec)

    print(f"Loaded {len(examples)} examples")

    if len(examples) == 0:
        print("❌ No valid examples found in dataset. Check your data file format.")
        sys.exit(1)

    # Step 1b: Data governance screen
    governance_warnings = _data_governance_check(examples)
    if governance_warnings:
        print(f"\n  Data governance warnings ({len(governance_warnings)}):")
        for w in governance_warnings[:5]:
            print(f"    - {w}")
        if len(governance_warnings) > 5:
            print(f"    ... and {len(governance_warnings) - 5} more")

    # Step 2: Deduplicate
    before = len(examples)
    examples = _deduplicate(examples)
    if len(examples) < before:
        print(f"Removed {before - len(examples)} duplicates → {len(examples)} unique")

    # Step 3: Split (early — protect blind test)
    n = len(examples)
    if n < 200:
        train_pct, val_pct = 0.70, 0.15
    elif n < 2000:
        train_pct, val_pct = 0.85, 0.10
    else:
        train_pct, val_pct = 0.90, 0.05

    random.seed(42)
    random.shuffle(examples)

    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],  # blind held-out
    }

    # Enforce minimums
    if len(splits["train"]) == 0:
        print("❌ Training split is empty after dedup and splitting. Need more data.")
        sys.exit(1)
    for name, minimum in [("train", 50), ("val", 20), ("test", 20)]:
        if len(splits[name]) < minimum:
            print(f"⚠️  {name} split has only {len(splits[name])} examples (minimum: {minimum})")
            if n < 90:
                print(f"   Dataset too small ({n} examples). Consider adding more data.")

    # Step 4: Write files + compute hashes
    manifest = {"splits": {}, "total": n}
    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        file_hash = _hash_file(path)
        manifest["splits"][name] = {
            "path": os.path.abspath(path),
            "count": len(data),
            "hash": file_hash,
        }
        print(f"  {name}: {len(data)} examples → {path}")

    # Save manifest
    manifest_path = os.path.join(output_dir, "data_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Data prepared in {output_dir}/")
    print(f"   Manifest: {manifest_path}")
    print(f"\n   ⚠️  test.jsonl is the BLIND held-out set. Do NOT use it for candidate design.")
    print(f"\n   Next: auto_finetune.py baseline --task-spec {args.task_spec} --test-file {output_dir}/test.jsonl")


def _convert_to_chat(data_path, hypothesis, spec):
    """Convert raw data to SFT chat format based on hypothesis."""
    ext = os.path.splitext(data_path)[1].lower()
    task_type = hypothesis.get("task_type", "generation")
    input_col = hypothesis.get("input_col")
    output_col = hypothesis.get("output_col")

    if not input_col or not output_col:
        print("Error: input_col and output_col required for conversion. Edit task_spec.json.")
        sys.exit(1)

    # Load all rows
    if ext == ".csv":
        import csv
        with open(data_path, encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    elif ext in (".json", ".jsonl"):
        with open(data_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif ext == ".parquet":
        import pandas as pd
        rows = pd.read_parquet(data_path).to_dict(orient="records")
    else:
        print(f"Unsupported: {ext}")
        sys.exit(1)

    system_prompt = spec.get("description", "You are a helpful assistant.")
    examples = []
    for row in rows:
        inp = str(row.get(input_col, "")).strip()
        out = str(row.get(output_col, "")).strip()
        if not inp or not out:
            continue
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": inp},
                {"role": "assistant", "content": out},
            ]
        })

    return examples


def _deduplicate(examples):
    """Remove exact duplicate examples based on content hash."""
    seen = set()
    unique = []
    for ex in examples:
        key = hashlib.md5(json.dumps(ex, sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def _hash_file(path):
    """SHA-256 hash of a file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


# ── Phase 3: BASELINE ────────────────────────────────────────────────────

def cmd_baseline(args):
    """Evaluate base model(s) on blind test set. Tests multiple models if --multi is set."""
    with open(args.task_spec, encoding="utf-8") as f:
        spec = json.load(f)

    test_file = args.test_file
    if not os.path.exists(test_file):
        print(f"Error: test file not found: {test_file}")
        sys.exit(1)

    client, method = get_clients(
        base_url=args.base_url, project_endpoint=args.project_endpoint, api_key=args.api_key
    )
    rubric = spec.get("eval_rubric", {})
    judge_model = rubric.get("judge_model", "gpt-4o")

    # Load test examples
    with open(test_file, encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    # Determine which models to baseline
    if args.multi:
        task_type = spec.get("hypotheses", [{}])[0].get("task_type", "chat_sft")
        recommendations = _recommend_models(task_type, spec.get("description", ""))
        models_to_test = [r["model"] for r in recommendations]
    else:
        models_to_test = [spec.get("base_model", "gpt-4.1-mini")]

    all_results = []
    for model in models_to_test:
        print(f"\nEvaluating base model '{model}' on {len(test_data)} test examples...")
        results = _evaluate_model_on_test(client, model, test_data, rubric, judge_model)
        results["model"] = model
        all_results.append(results)
        _print_eval_results("BASELINE", model, results)

    # Pick best model by combined score
    valid_results = [r for r in all_results if r.get("combined", 0) > 0]
    if valid_results:
        best = max(valid_results, key=lambda r: r["combined"])
        if len(valid_results) > 1:
            print(f"\n  🏆 Best base model: {best['model']} ({best['combined']:.2f})")
            # Update task_spec with recommended model
            spec["base_model"] = best["model"]
            spec["baseline_results"] = all_results
            with open(args.task_spec, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
            print(f"  Updated task_spec.json with base_model={best['model']}")

    output = args.output or "baseline.json"
    baseline_data = {
        "models_tested": [r["model"] for r in all_results],
        "results": all_results,
        "best_model": valid_results[0]["model"] if valid_results else None,
        "combined": valid_results[0]["combined"] if valid_results else 0,
        "pass_rate": valid_results[0].get("pass_rate", 0) if valid_results else 0,
    }
    # Also include top-level fields for backward compat with review phase
    if valid_results:
        best = max(valid_results, key=lambda r: r["combined"])
        baseline_data.update({
            "combined": best["combined"],
            "pass_rate": best.get("pass_rate", 0),
        })

    with open(output, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    # Adaptive difficulty check
    best_pass = best.get("pass_rate", 0) if valid_results else 0
    best_combined = best.get("combined", 0) if valid_results else 0
    if best_pass >= 95:
        print(f"\n  WARNING: Baseline pass rate is {best_pass}% — almost no headroom for FT improvement.")
        print(f"  The task may be too easy for {best.get('model', 'this model')}, or the eval rubric too lenient.")
        print(f"  Recommendations:")
        print(f"    1. Regenerate harder data:  auto_finetune.py generate --difficulty hard ...")
        print(f"    2. Use a stricter judge model (e.g., gpt-5.4 instead of gpt-4.1-mini)")
        print(f"    3. Tighten eval rubric descriptions to require more precision")
        print(f"    4. Consider a different task with more room for improvement")
    elif best_combined >= 9.0:
        print(f"\n  NOTE: Baseline combined score is {best_combined:.2f} — limited headroom.")
        print(f"  FT gains may be marginal. Consider stricter eval or harder data.")

    print(f"\n  Output: {output}")
    print(f"  Next: auto_finetune.py candidates --task-spec {args.task_spec} --data-dir ./prepared")


# ── Model Selection Intelligence ─────────────────────────────────────────

# Task-type → recommended models, ordered by cost (cheapest first)
MODEL_RECOMMENDATIONS = {
    "classification": [
        {"model": "gpt-4.1-nano", "reason": "Simple decision boundary, cheapest for classification"},
        {"model": "gpt-4.1-mini", "reason": "More nuanced classification, moderate cost"},
        {"model": "Ministral-3B-2501", "reason": "OSS lightweight option — fast inference, good for simple tasks"},
    ],
    "generation": [
        {"model": "gpt-4.1-mini", "reason": "Strong generation quality, good cost/performance"},
        {"model": "gpt-4.1-nano", "reason": "Budget option — try if mini is overkill"},
        {"model": "gpt-oss-20b-11", "reason": "OSS large model — strong on generation with big datasets (500+)"},
    ],
    "code": [
        {"model": "gpt-4.1-mini", "reason": "Best code generation quality for SFT"},
        {"model": "gpt-oss-20b-11", "reason": "OSS option — good code quality with large datasets"},
    ],
    "summarization": [
        {"model": "gpt-4.1-nano", "reason": "Summarization patterns are learnable by smaller models"},
        {"model": "gpt-4.1-mini", "reason": "Fallback if nano quality insufficient"},
        {"model": "Ministral-3B-2501", "reason": "OSS lightweight — fast inference for high-volume summarization"},
    ],
    "extraction": [
        {"model": "gpt-4.1-nano", "reason": "Structured extraction is a pattern-matching task — nano excels"},
        {"model": "gpt-4.1-mini", "reason": "Fallback for complex extraction schemas"},
        {"model": "Ministral-3B-2501", "reason": "OSS lightweight — good for simple extraction at scale"},
    ],
    "chat_sft": [
        {"model": "gpt-4.1-mini", "reason": "General-purpose, strong chat quality"},
        {"model": "gpt-4.1-nano", "reason": "Budget option for simple chat tasks"},
        {"model": "Qwen3-32B", "reason": "OSS large — strong multilingual and specialized chat"},
    ],
    "multilingual": [
        {"model": "gpt-4.1-mini", "reason": "Strong multilingual support"},
        {"model": "Qwen3-32B", "reason": "Excellent multilingual coverage, OSS"},
        {"model": "Llama-3.3-70B-Instruct", "reason": "Large OSS model — broad language coverage"},
    ],
    "reasoning": [
        {"model": "gpt-4.1-mini", "reason": "Good reasoning for SFT distillation"},
        {"model": "gpt-oss-20b-11", "reason": "Larger model capacity for complex reasoning"},
        {"model": "Llama-3.3-70B-Instruct", "reason": "OSS frontier — strong reasoning on large datasets"},
    ],
}


def _recommend_models(task_type, description=""):
    """Recommend base models based on task type and description."""
    desc_lower = (description or "").lower()

    # Detect task type from description keywords
    if any(kw in desc_lower for kw in ["code", "python", "sql", "javascript", "programming", "function", "bug"]):
        return MODEL_RECOMMENDATIONS.get("code", MODEL_RECOMMENDATIONS["generation"])

    if any(kw in desc_lower for kw in ["summar", "condense", "tldr", "brief", "abstract"]):
        return MODEL_RECOMMENDATIONS.get("summarization", MODEL_RECOMMENDATIONS["generation"])

    if any(kw in desc_lower for kw in ["extract", "parse", "json", "structured", "entity", "ner"]):
        return MODEL_RECOMMENDATIONS.get("extraction", MODEL_RECOMMENDATIONS["generation"])

    if any(kw in desc_lower for kw in ["translat", "multilingual", "language", "chinese", "spanish", "french", "german"]):
        return MODEL_RECOMMENDATIONS.get("multilingual", MODEL_RECOMMENDATIONS["chat_sft"])

    if any(kw in desc_lower for kw in ["reason", "math", "logic", "proof", "complex", "multi-step"]):
        return MODEL_RECOMMENDATIONS.get("reasoning", MODEL_RECOMMENDATIONS["generation"])

    return MODEL_RECOMMENDATIONS.get(task_type, MODEL_RECOMMENDATIONS["chat_sft"])


def _design_initial_candidates(primary_model, alt_model, train_count, alt_reason=None):
    """Design first-iteration candidates based on dataset size."""
    candidates = []

    if train_count < 200:
        candidates = [
            {"name": "conservative", "model": primary_model, "epochs": 3, "lr": 1.0,
             "rationale": "Standard 3-epoch with default LR"},
            {"name": "high-lr", "model": primary_model, "epochs": 3, "lr": 2.0,
             "rationale": "Aggressive LR -- may trigger phase transition on small data"},
        ]
        if alt_model:
            candidates.append({"name": f"alt-{alt_model.split('-')[-1]}", "model": alt_model, "epochs": 3, "lr": 1.0,
                              "rationale": f"Alternative model: {alt_reason}"})
        else:
            candidates.append({"name": "long-train", "model": primary_model, "epochs": 5, "lr": 0.5,
                              "rationale": "More passes with gentle LR"})
    elif train_count < 1000:
        candidates = [
            {"name": "conservative", "model": primary_model, "epochs": 2, "lr": 1.0,
             "rationale": "Safe baseline with 2 epochs"},
            {"name": "balanced", "model": primary_model, "epochs": 1, "lr": 1.3,
             "rationale": "Empirically best recipe for medium+ datasets"},
        ]
        if alt_model:
            candidates.append({"name": f"alt-{alt_model.split('-')[-1]}", "model": alt_model, "epochs": 2, "lr": 1.0,
                              "rationale": f"Alternative model: {alt_reason}"})
        else:
            candidates.append({"name": "aggressive", "model": primary_model, "epochs": 3, "lr": 2.0,
                              "rationale": "Higher LR may find phase transitions"})
    else:
        candidates = [
            {"name": "single-pass", "model": primary_model, "epochs": 1, "lr": 1.0,
             "rationale": "Single epoch, let data volume do the work"},
            {"name": "balanced", "model": primary_model, "epochs": 1, "lr": 1.3,
             "rationale": "Our winning recipe (Mini R8 LD-LowLR)"},
        ]
        if alt_model:
            candidates.append({"name": f"alt-{alt_model.split('-')[-1]}", "model": alt_model, "epochs": 1, "lr": 1.3,
                              "rationale": f"Alternative model: {alt_reason}"})
        else:
            candidates.append({"name": "two-pass", "model": primary_model, "epochs": 2, "lr": 0.5,
                              "rationale": "Two passes with gentle LR for thorough learning"})

    return candidates


def _design_iteration_candidates(prev_review, primary_model, alt_model, train_count):
    """Design iteration 2+ candidates based on previous review diagnostics.
    
    Decision tree:
    - All regressed → try gentler HPs, different model, or MORE DATA if already tried both
    - Some improved but below threshold → narrow HPs around winner, or MORE DATA if dataset is small
    - High variance (all similar scores) → data diversity issue, need MORE DATA
    - Overfitting on all candidates → MORE DATA or fewer epochs
    """
    diagnostics = prev_review.get("candidate_diagnostics", [])
    best_candidate = prev_review.get("best_candidate", "")
    best_score = prev_review.get("best_score", 0)
    lift = prev_review.get("lift_pct", 0)
    iteration = prev_review.get("iteration", 1)

    candidates = []
    data_recommendation = None

    # Check for patterns in the diagnostics
    catastrophic = [d for d in diagnostics if d.get("issue") == "catastrophic_regression"]
    regressions = [d for d in diagnostics if d.get("issue") == "regression"]
    improved = [d for d in diagnostics if (d.get("lift_pct", 0) or 0) > 0]
    scores = [d.get("score", 0) for d in diagnostics if d.get("score", 0) > 0]
    score_variance = (max(scores) - min(scores)) if len(scores) >= 2 else 0

    # Determine if we should recommend more data
    need_more_data = False
    if not improved and iteration >= 2:
        # Tried HPs twice, nothing worked → data is the bottleneck
        need_more_data = True
        data_recommendation = f"Two iterations without improvement. Generate more diverse data (current: {train_count} examples)."
    elif improved and lift > 0 and lift < 5 and train_count < 200:
        # Positive lift on small dataset → more data is highest leverage
        need_more_data = True
        data_recommendation = f"Positive lift ({lift:+.1f}%) on small dataset ({train_count} examples). More data is the highest-leverage next step."
    elif score_variance < 0.5 and len(scores) >= 2:
        # All candidates scored similarly → data diversity issue
        need_more_data = True
        data_recommendation = f"All candidates scored within {score_variance:.1f} points. Data diversity (not HPs) is likely the bottleneck."
    elif all(d.get("issue") == "catastrophic_regression" for d in diagnostics if d.get("issue")):
        # Everything catastrophically failed → possibly data quality issue
        need_more_data = True
        data_recommendation = f"All candidates catastrophically regressed. Inspect data quality, or generate more/harder data."

    if need_more_data:
        # Calculate recommended new dataset size
        if train_count < 100:
            target = 300
        elif train_count < 300:
            target = train_count * 2
        else:
            target = int(train_count * 1.5)

        candidates.append({
            "name": "MORE_DATA",
            "model": primary_model,
            "epochs": 0,
            "lr": 0,
            "rationale": data_recommendation,
            "action": "generate",
            "target_examples": target,
            "current_examples": train_count,
        })

    if not improved:
        # Nothing worked — try radically different approaches
        candidates.extend([
            {"name": "v2-gentle", "model": primary_model, "epochs": 1, "lr": 0.3,
             "rationale": "Very gentle LR -- previous attempts may have overfit"},
            {"name": "v2-more-epochs", "model": primary_model, "epochs": 5, "lr": 0.5,
             "rationale": "More training time with safe LR"},
        ])
        if alt_model and not need_more_data:
            candidates.append({"name": f"v2-alt-{alt_model.split('-')[-1]}", "model": alt_model, "epochs": 2, "lr": 1.0,
                              "rationale": "Try different base model entirely"})
    elif lift > 0:
        # Something worked but not enough — narrow around the winner
        best_lr = 1.0
        best_epochs = 2

        candidates.extend([
            {"name": "v2-refine-up", "model": primary_model, "epochs": best_epochs, "lr": best_lr * 1.3,
             "rationale": f"Slightly higher LR than best ({best_lr})"},
            {"name": "v2-refine-down", "model": primary_model, "epochs": best_epochs, "lr": best_lr * 0.7,
             "rationale": f"Slightly lower LR than best ({best_lr})"},
        ])
        if not need_more_data:
            candidates.append(
                {"name": "v2-more-epochs", "model": primary_model, "epochs": best_epochs + 1, "lr": best_lr,
                 "rationale": "Same LR, one more epoch for deeper learning"})
    else:
        # Fallback
        candidates.extend([
            {"name": "v2-safe", "model": primary_model, "epochs": 2, "lr": 0.5,
             "rationale": "Conservative retry"},
            {"name": "v2-balanced", "model": primary_model, "epochs": 1, "lr": 1.3,
             "rationale": "Proven balanced recipe"},
        ])

    # Filter out the MORE_DATA placeholder from actual training candidates
    # (it's a recommendation, not a trainable candidate)
    training_candidates = [c for c in candidates if c.get("name") != "MORE_DATA"]
    data_candidates = [c for c in candidates if c.get("name") == "MORE_DATA"]

    return training_candidates, data_candidates


# ── Phase 4: DESIGN CANDIDATES ───────────────────────────────────────────

def cmd_candidates(args):
    """Generate candidate experiment plan with intelligent model + HP selection.
    
    On iteration 2+, reads previous review diagnostics to adapt strategy.
    """
    with open(args.task_spec, encoding="utf-8") as f:
        spec = json.load(f)

    data_dir = args.data_dir or "./prepared"
    manifest_path = os.path.join(data_dir, "data_manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        train_count = manifest["splits"]["train"]["count"]
    else:
        train_path = os.path.join(data_dir, "train.jsonl")
        with open(train_path, encoding="utf-8") as f:
            train_count = sum(1 for line in f if line.strip())

    primary_model = spec.get("base_model", "gpt-4.1-mini")
    task_type = spec.get("hypotheses", [{}])[0].get("task_type", "chat_sft")
    description = spec.get("description", "")

    # Get model recommendations for this task
    recommendations = _recommend_models(task_type, description)
    alt_model = None
    for rec in recommendations:
        if rec["model"] != primary_model:
            alt_model = rec["model"]
            alt_reason = rec["reason"]
            break

    # Check if we have previous review diagnostics (iteration 2+)
    prev_review = None
    data_recommendations = []
    if args.review_file and os.path.exists(args.review_file):
        with open(args.review_file, encoding="utf-8") as f:
            prev_review = json.load(f)

    if prev_review:
        candidates, data_recommendations = _design_iteration_candidates(
            prev_review, primary_model, alt_model, train_count
        )
    else:
        candidates = _design_initial_candidates(
            primary_model, alt_model, train_count,
            alt_reason if alt_model else None
        )

    # Add common fields
    for c in candidates:
        c["batch_size"] = None
        c["training_file"] = os.path.abspath(os.path.join(data_dir, "train.jsonl"))
        c["validation_file"] = os.path.abspath(os.path.join(data_dir, "val.jsonl"))

    iteration = (prev_review["iteration"] + 1) if prev_review else (args.iteration or 1)

    plan = {
        "iteration": iteration,
        "task_name": spec.get("task_name", ""),
        "base_model": primary_model,
        "train_count": train_count,
        "candidates": candidates,
        "data_recommendations": [{"rationale": d["rationale"], "target_examples": d.get("target_examples", 0),
                                   "current_examples": d.get("current_examples", train_count)}
                                  for d in data_recommendations] if data_recommendations else [],
    }

    output = args.output or "candidate_plan.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  CANDIDATE PLAN (iteration {plan['iteration']})")
    print(f"{'='*60}")
    print(f"  Primary model: {primary_model}  |  Training data: {train_count} examples")
    if alt_model:
        print(f"  Alt model: {alt_model}")

    if data_recommendations:
        print(f"\n  DATA RECOMMENDATION:")
        for d in data_recommendations:
            print(f"    {d['rationale']}")
            print(f"    Action: auto_finetune.py generate --num-examples {d.get('target_examples', train_count * 2)} --difficulty hard ...")
        print()

    print(f"  {'Name':<15} {'Model':<18} {'Epochs':>6} {'LR':>6}  Rationale")
    print(f"  {'-'*15} {'-'*18} {'-'*6} {'-'*6}  {'-'*40}")
    for c in candidates:
        print(f"  {c['name']:<15} {c['model']:<18} {c['epochs']:>6} {c['lr']:>6.1f}  {c['rationale']}")

    print(f"\n  Output: {output}")
    print(f"  Next: Review plan, then: auto_finetune.py execute --plan {output}")


# ── Phase 5: EXECUTE ──────────────────────────────────────────────────────

def cmd_execute(args):
    """Submit all candidate jobs and monitor to completion."""
    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)

    client, method = get_clients(
        base_url=args.base_url, project_endpoint=args.project_endpoint, api_key=args.api_key
    )

    # Preflight checks
    print("Preflight checks...")
    train_path = plan["candidates"][0]["training_file"]
    val_path = plan["candidates"][0]["validation_file"]

    if not os.path.exists(train_path):
        print(f"  FAIL: Training file not found: {train_path}")
        sys.exit(1)
    if not os.path.exists(val_path):
        print(f"  FAIL: Validation file not found: {val_path}")
        sys.exit(1)

    with open(train_path, encoding="utf-8") as f:
        train_count = sum(1 for line in f if line.strip())
    with open(val_path, encoding="utf-8") as f:
        val_count = sum(1 for line in f if line.strip())
    print(f"  Data: {train_count} train, {val_count} val")

    if train_count < 10:
        print(f"  FAIL: Training data too small ({train_count} examples, minimum 10)")
        sys.exit(1)
    if val_count < 5:
        print(f"  WARNING: Validation set very small ({val_count} examples)")

    # Check file quota
    try:
        files = client.files.list()
        file_count = len(list(files))
        print(f"  File quota: {file_count}/100 used")
        if file_count >= 95:
            print(f"  WARNING: File quota nearly full ({file_count}/100). Run cleanup first.")
    except Exception:
        pass

    # Check unique models in candidates
    models_used = set(c["model"] for c in plan["candidates"])
    print(f"  Models: {', '.join(models_used)}")
    print(f"  Candidates: {len(plan['candidates'])}")
    print(f"  Preflight OK")

    # Upload files
    print("\nUploading training data...")
    with open(train_path, "rb") as f:
        train_upload = client.files.create(purpose="fine-tune", file=f)
    print(f"  Training file: {train_upload.id}")

    print("Uploading validation data...")
    with open(val_path, "rb") as f:
        val_upload = client.files.create(purpose="fine-tune", file=f)
    print(f"  Validation file: {val_upload.id}")

    # Wait for processing
    print("Waiting for file processing...")
    client.files.wait_for_processing(train_upload.id)
    client.files.wait_for_processing(val_upload.id)
    print("  Files ready.")

    # Submit each candidate
    runs = []
    requested_tier = getattr(args, "tier", None) or plan.get("tier", None)
    tier_pool = _parse_tiers(requested_tier)
    if len(tier_pool) > 1:
        print(f"  Tier mix: {tier_pool} (round-robin across candidates)")

    for cand_idx, c in enumerate(plan["candidates"]):
        print(f"\nSubmitting candidate '{c['name']}'...")
        suffix = _sanitize_name(f"{plan.get('task_name', 'auto')}-{c['name']}")

        # Block RFT — auto-finetune only supports SFT
        method = c.get("method", "supervised")
        if method in ("reinforcement", "rft"):
            print(f"  ❌ Skipping '{c['name']}': auto-finetune does not support RFT. "
                  "Use submit_training.py with manual grader design instead.")
            runs.append({"candidate": c["name"], "status": "skipped",
                         "error": "RFT not supported by auto-finetune"})
            continue

        # Resolve tier per model+candidate (round-robin across pool; OSS forced to globalStandard)
        chosen = tier_pool[cand_idx % len(tier_pool)]
        tier = _resolve_tier(c["model"], chosen)

        try:
            # SDK supports trainingType via extra_body. No REST fallback needed for tiers.
            # (REST fallback is still used for OSS models that error out with
            #  "does not support fine-tuning with Standard TrainingType".)
            create_kwargs = dict(
                model=c["model"],
                training_file=train_upload.id,
                validation_file=val_upload.id,
                suffix=suffix,
                method={"type": "supervised"},
                hyperparameters={
                    "n_epochs": c["epochs"],
                    "learning_rate_multiplier": c["lr"],
                },
            )
            if tier:
                create_kwargs["extra_body"] = {"trainingType": tier}

            try:
                job = client.fine_tuning.jobs.create(**create_kwargs)
            except Exception as sdk_err:
                msg = str(sdk_err)
                if "does not support fine-tuning with Standard TrainingType" in msg or \
                   ("training type" in msg.lower() and ("standard" in msg.lower() or "unsupported" in msg.lower())):
                    # OSS model — needs explicit REST path
                    api_key = args.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
                    base = args.base_url or os.environ.get("OPENAI_BASE_URL", "")
                    if not base or not api_key:
                        raise RuntimeError(
                            f"Model '{c['model']}' rejected tier '{tier}' via SDK and REST fallback "
                            f"is unavailable (need --base-url + --api-key). Original error: {sdk_err}"
                        ) from sdk_err
                    print(f"  ↩ SDK rejected for {c['model']}; retrying via REST with tier={tier}")
                    rest_endpoint = base.replace("/openai/v1", "").rstrip("/")
                    rest_url = f"{rest_endpoint}/openai/fine_tuning/jobs?api-version=2025-04-01-preview"
                    body = {
                        "model": c["model"],
                        "training_file": train_upload.id,
                        "validation_file": val_upload.id,
                        "suffix": suffix,
                        "hyperparameters": {
                            "n_epochs": c["epochs"],
                            "learning_rate_multiplier": c["lr"],
                        },
                        "trainingType": tier or "globalStandard",
                    }
                    resp = requests.post(rest_url, headers={"api-key": api_key, "Content-Type": "application/json"}, json=body, timeout=(10, 120))
                    if resp.status_code not in (200, 201):
                        try:
                            err_msg = resp.json().get("error", {}).get("message", "Unknown error")
                        except (ValueError, KeyError):
                            err_msg = resp.text[:200] if resp.text else "Unknown error"
                        raise RuntimeError(f"REST submission failed ({resp.status_code}): {err_msg}")
                    data = resp.json()
                    class _RESTJob:  # tiny shim
                        def __init__(self, d): self.id = d["id"]; self.status = d.get("status", "pending")
                    job = _RESTJob(data)
                else:
                    raise

            job_id = job.id
            job_status = job.status

            run = {
                "candidate": c["name"],
                "job_id": job_id,
                "status": job_status,
                "model": c["model"],
                "hyperparameters": {"n_epochs": c["epochs"], "lr": c["lr"]},
                "dataset_hash": _hash_file(train_path),
                "fine_tuned_model": None,
                "tier": tier or "service-default",
            }
            runs.append(run)
            tier_label = f" [{tier}]" if tier else ""
            print(f"  ✅ Job {job_id} ({job_status}){tier_label}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            runs.append({"candidate": c["name"], "status": "failed", "error": str(e)})

    # Monitor all jobs
    print(f"\nMonitoring {len([r for r in runs if r.get('job_id')])} jobs...")
    terminal = {"succeeded", "failed", "cancelled"}
    try:
        while True:
            all_done = True
            for run in runs:
                if not run.get("job_id") or run["status"] in terminal:
                    continue
                try:
                    job = client.fine_tuning.jobs.retrieve(run["job_id"])
                    run["status"] = job.status
                    if job.status == "succeeded":
                        run["fine_tuned_model"] = job.fine_tuned_model
                        run["trained_tokens"] = job.trained_tokens
                        if not job.fine_tuned_model:
                            print(f"  ⚠️  {run['candidate']}: succeeded but fine_tuned_model is None — retrying retrieve...")
                            time.sleep(10)
                            job = client.fine_tuning.jobs.retrieve(run["job_id"])
                            run["fine_tuned_model"] = job.fine_tuned_model
                            if not job.fine_tuned_model:
                                run["status"] = "failed"
                                run["error"] = "Job succeeded but fine_tuned_model is None"
                                print(f"  ❌ {run['candidate']}: succeeded but no model ID returned")
                                continue
                        # Download full training results CSV for curve analysis
                        try:
                            import re as _re, csv, io
                            if job.result_files:
                                content = client.files.content(job.result_files[0])
                                csv_text = content.read().decode("utf-8")
                                reader = csv.DictReader(io.StringIO(csv_text))
                                rows = list(reader)

                                # Extract key curve metrics (support alternate column names)
                                def _get_val(row, *keys):
                                    for k in keys:
                                        v = row.get(k)
                                        if v:
                                            return float(v)
                                    return None

                                train_losses = [(int(r["step"]), float(r["train_loss"])) for r in rows if r.get("train_loss")]
                                val_losses = [(int(r["step"]), _get_val(r, "valid_loss", "full_valid_loss", "eval_loss"))
                                              for r in rows if _get_val(r, "valid_loss", "full_valid_loss", "eval_loss") is not None]

                                if train_losses:
                                    run["final_train_loss"] = train_losses[-1][1]
                                    run["min_train_loss"] = min(tl for _, tl in train_losses)
                                    run["train_loss_std"] = (sum((tl - sum(tl for _, tl in train_losses)/len(train_losses))**2 for _, tl in train_losses) / len(train_losses)) ** 0.5

                                if val_losses:
                                    run["final_val_loss"] = val_losses[-1][1]
                                    run["best_val_loss"] = min(vl for _, vl in val_losses)
                                    best_step = min(val_losses, key=lambda x: x[1])[0]
                                    run["best_val_step"] = best_step
                                    # Overfitting ratio
                                    if run["best_val_loss"] > 0:
                                        run["overfit_ratio"] = round(run["final_val_loss"] / run["best_val_loss"], 2)

                                # Token accuracy
                                accs = [float(r["train_mean_token_accuracy"]) for r in rows if r.get("train_mean_token_accuracy")]
                                if accs:
                                    run["final_token_accuracy"] = accs[-1]

                                # Save full CSV path
                                csv_path = os.path.join(os.path.dirname(args.output or "runs.json"), f"results_{run['candidate']}.csv")
                                with open(csv_path, "w", encoding="utf-8") as csvf:
                                    csvf.write(csv_text)
                                run["results_csv"] = csv_path
                        except Exception:
                            # Fallback: get loss from events
                            try:
                                events = client.fine_tuning.jobs.list_events(run["job_id"], limit=20)
                                for e in events.data:
                                    if "training loss" in e.message:
                                        match = _re.search(r'training loss=([0-9.]+)', e.message)
                                        if match:
                                            run["final_train_loss"] = float(match.group(1))
                                            break
                            except Exception:
                                pass
                        print(f"  ✅ {run['candidate']}: succeeded → {job.fine_tuned_model}")
                    elif job.status in ("failed", "cancelled"):
                        run["error"] = str(getattr(job, "error", ""))
                        print(f"  ❌ {run['candidate']}: {job.status}")
                    else:
                        all_done = False
                except Exception:
                    all_done = False

            if all_done:
                break
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving partial progress...")
        print("    Note: Submitted jobs continue running on the server.")
        print("    Use check_training.py to monitor, or cleanup.py --cancel-pending to cancel.")

    # Save runs (always — even on interrupt) — atomic write to prevent corruption
    output = args.output or "runs.json"
    _atomic_json_write(output, {"iteration": plan.get("iteration", 1), "runs": runs})

    succeeded = sum(1 for r in runs if r["status"] == "succeeded")
    total_tokens = sum(r.get("trained_tokens", 0) or 0 for r in runs if r["status"] == "succeeded")

    # Per-run cost estimation (model + tier aware). Skip runs we don't have
    # pricing for rather than guessing — printed disclaimer makes the
    # remaining estimate clearly best-effort.
    cost_breakdown = []
    total_cost = 0.0
    unknown_models = set()
    for r in runs:
        if r["status"] != "succeeded":
            continue
        toks = r.get("trained_tokens", 0) or 0
        if not toks:
            continue
        est = estimate_training_cost(r.get("model", ""), r.get("tier", "standard"), toks)
        if est is None:
            unknown_models.add(r.get("model") or "unknown")
        else:
            total_cost += est["cost"]
            cost_breakdown.append((r["candidate"], r.get("model", "?"),
                                   r.get("tier", "standard"), toks, est["cost"],
                                   est["price_per_M"]))

    print(f"\n{'='*60}")
    print(f"  EXECUTION COMPLETE: {succeeded}/{len(runs)} succeeded")
    if total_tokens > 0:
        print(f"  Total trained tokens: {total_tokens:,}")
        if cost_breakdown:
            print(f"  Estimated cost: ~${total_cost:.2f}")
            for cand, model, tier, toks, c, ppM in cost_breakdown:
                print(f"    • {cand:<18s} {model:<22s} [{tier:<14s}] "
                      f"{toks:>10,}t  ~${c:>7.2f}  (@${ppM:.2f}/M)")
        if unknown_models:
            print(f"  ⚠️  Cost unknown for model(s): {', '.join(sorted(unknown_models))}")
        print(f"  ⚠️  Estimates are illustrative — verify on the Azure pricing page:")
        print(f"      {AZURE_PRICING_URL}")
    print(f"{'='*60}")
    print(f"  Output: {output}")
    print(f"  Next: auto_finetune.py evaluate --runs {output} --test-file ./prepared/test.jsonl")


# ── Phase 6: EVALUATE ────────────────────────────────────────────────────

def _deploy_model_arm(sub, rg, account, deploy_name, model_id, sku="GlobalStandard", capacity=100):
    """Deploy a fine-tuned model via ARM REST API. Returns True on success."""
    import subprocess, requests
    az = _find_az_cli()
    result = subprocess.run(
        [az, "account", "get-access-token", "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True,
    )
    token = result.stdout.strip()
    if not token:
        print(f"    Failed to get ARM token")
        return False

    url = (f"https://management.azure.com/subscriptions/{sub}"
           f"/resourceGroups/{rg}/providers/Microsoft.CognitiveServices"
           f"/accounts/{account}/deployments/{deploy_name}?api-version=2024-10-01")

    body = {
        "sku": {"name": sku, "capacity": capacity},
        "properties": {"model": {"format": "OpenAI", "name": model_id, "version": "1"}},
    }
    resp = requests.put(url, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
    }, json=body, timeout=(10, 120))

    if resp.status_code in (200, 201):
        return True
    # Retry with Standard SKU if GlobalStandard fails
    if sku == "GlobalStandard":
        body["sku"]["name"] = "Standard"
        resp = requests.put(url, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        }, json=body, timeout=(10, 120))
        if resp.status_code in (200, 201):
            return True
    print(f"    Deploy failed ({resp.status_code}): {resp.text[:100]}")
    return False


def _delete_deployment_arm(sub, rg, account, deploy_name):
    """Delete a deployment via ARM REST API."""
    import subprocess, requests
    az = _find_az_cli()
    result = subprocess.run(
        [az, "account", "get-access-token", "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True,
    )
    token = result.stdout.strip()
    url = (f"https://management.azure.com/subscriptions/{sub}"
           f"/resourceGroups/{rg}/providers/Microsoft.CognitiveServices"
           f"/accounts/{account}/deployments/{deploy_name}?api-version=2024-10-01")
    requests.delete(url, headers={"Authorization": f"Bearer {token}"}, timeout=(10, 30))


def _cleanup_eval_deployments(sub, rg, account, keep_names=None):
    """Delete stale eval-* and ckpt-* deployments to free up quota."""
    import subprocess, requests
    keep = set(keep_names or [])
    az = _find_az_cli()
    result = subprocess.run(
        [az, "account", "get-access-token", "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True,
    )
    token = result.stdout.strip()
    url = (f"https://management.azure.com/subscriptions/{sub}"
           f"/resourceGroups/{rg}/providers/Microsoft.CognitiveServices"
           f"/accounts/{account}/deployments?api-version=2024-10-01")
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=(10, 60))
    if resp.status_code != 200:
        return
    try:
        deployments = resp.json().get("value", [])
    except (ValueError, KeyError):
        return
    deleted = 0
    for d in deployments:
        name = d["name"]
        if (name.startswith("eval-") or name.startswith("ckpt-")) and name not in keep:
            del_url = url.replace("?", f"/{name}?")
            requests.delete(del_url, headers={"Authorization": f"Bearer {token}"}, timeout=(10, 30))
            print(f"      Cleaned up stale deployment: {name}")
            deleted += 1
            time.sleep(2)
    if deleted:
        print(f"      Freed {deleted} deployment(s). Waiting 30s for cleanup to propagate...")
        time.sleep(30)

def _find_az_cli():
    """Find the Azure CLI executable."""
    import shutil
    az = shutil.which("az")
    if az:
        return az
    # Common Windows paths
    for candidate in [
        r"C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
        r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
    ]:
        if os.path.exists(candidate):
            return candidate
    return "az"


def _detect_azure_resource(base_url=None):
    """Extract subscription, resource group, and account from Azure CLI context."""
    import subprocess
    az = _find_az_cli()

    # Get subscription
    r = subprocess.run([az, "account", "show", "--query", "id", "-o", "tsv"],
                       capture_output=True, text=True)
    sub = r.stdout.strip()

    # Try to find the cognitive services account from the base URL
    account = ""
    if base_url:
        import re
        m = re.search(r'https://([^.]+)\.(openai\.azure\.com|services\.ai\.azure\.com)', base_url)
        if m:
            account = m.group(1)

    # Find the resource group for this account
    rg = ""
    if account:
        r = subprocess.run(
            [az, "cognitiveservices", "account", "list", "--query",
             f"[?contains(name,'{account}')].resourceGroup | [0]", "-o", "tsv"],
            capture_output=True, text=True,
        )
        rg = r.stdout.strip()

    return sub, rg, account


def cmd_evaluate(args):
    """Deploy, evaluate, and score each candidate on blind test set."""
    with open(args.runs, encoding="utf-8") as f:
        runs_data = json.load(f)

    with open(args.task_spec, encoding="utf-8") as f:
        spec = json.load(f)

    test_file = args.test_file
    with open(test_file, encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    client, method = get_clients(
        base_url=args.base_url, project_endpoint=args.project_endpoint, api_key=args.api_key
    )
    rubric = spec.get("eval_rubric", {})
    judge_model = rubric.get("judge_model", "gpt-4o")

    # Detect Azure resource for deployment management
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "")
    sub, rg, account = _detect_azure_resource(base_url)
    can_deploy = bool(sub and rg and account)
    if can_deploy:
        print(f"  Azure resource: {account} ({rg})")
        # Proactively clean stale eval/ckpt deployments to free quota
        print(f"  Cleaning stale eval deployments...")
        _cleanup_eval_deployments(sub, rg, account, keep_names=[])
    else:
        print(f"  Warning: Could not detect Azure resource. Will try direct model inference (may 404 for FT models).")

    capacity = args.capacity
    deployed_names = []  # Track deployments we create for cleanup

    leaderboard = []
    try:
        for run in runs_data["runs"]:
            if run["status"] != "succeeded" or not run.get("fine_tuned_model"):
                continue

            model_id = run["fine_tuned_model"]
            name = run["candidate"]
            deploy_name = f"eval-{name}"[:64]  # ARM limit

            # Deploy the model (with retry on failure — may need to clean up old deployments)
            if can_deploy:
                print(f"\n  Deploying '{name}' as '{deploy_name}' (capacity={capacity})...")
                deploy_ok = _deploy_model_arm(sub, rg, account, deploy_name, model_id, capacity=capacity)

                if not deploy_ok:
                    # Retry: clean up stale eval deployments, then try again
                    print(f"    Deploy failed — cleaning up stale eval deployments and retrying...")
                    _cleanup_eval_deployments(sub, rg, account, deployed_names)
                    time.sleep(30)
                    deploy_ok = _deploy_model_arm(sub, rg, account, deploy_name, model_id, capacity=capacity)

                if not deploy_ok:
                    print(f"    ❌ Deployment failed after retry — skipping candidate '{name}'")
                    leaderboard.append({"candidate": name, "model_id": model_id, "combined": 0,
                                        "pass_rate": 0, "errors": len(test_data), "n_scored": 0,
                                        "n_total": len(test_data), "hyperparameters": run.get("hyperparameters", {}),
                                        "issue": "deployment_failure"})
                    continue

                deployed_names.append(deploy_name)
                eval_model = deploy_name

                # Probe until ready (escalating waits, handle any 400/creating error)
                deploy_ready = False
                for attempt in range(6):
                    wait = 150 if attempt < 2 else 120  # 2.5 min, 2.5 min, then 2 min each
                    time.sleep(wait)
                    try:
                        client.chat.completions.create(
                            model=eval_model,
                            messages=[{"role": "user", "content": "test"}],
                            max_completion_tokens=5,
                        )
                        deploy_ready = True
                        total = sum(150 if i < 2 else 120 for i in range(attempt + 1))
                        print(f"    ✅ Model ready after ~{total // 60} min")
                        break
                    except Exception as e:
                        err = str(e)
                        if "400" in err or "Creating" in err or "BadRequest" in err:
                            print(f"    ⏳ Warming up (attempt {attempt+1}/6)...")
                        elif "404" in err:
                            print(f"    ⏳ Not found yet (attempt {attempt+1}/6)...")
                        else:
                            # Unknown error — assume ready and let eval handle it
                            deploy_ready = True
                            break

                if not deploy_ready:
                    print(f"    ❌ Model not ready after ~12 min — skipping candidate '{name}'")
                    leaderboard.append({"candidate": name, "model_id": model_id, "combined": 0,
                                        "pass_rate": 0, "errors": len(test_data), "n_scored": 0,
                                        "n_total": len(test_data), "hyperparameters": run.get("hyperparameters", {}),
                                        "issue": "deployment_failure"})
                    continue
            else:
                eval_model = model_id  # Try direct model ID

            print(f"  Evaluating '{name}'...")
            results = _evaluate_model_on_test(client, eval_model, test_data, rubric, judge_model)
            results["candidate"] = name
            results["model_id"] = model_id
            results["deployment_name"] = deploy_name if can_deploy else None
            results["hyperparameters"] = run.get("hyperparameters", {})
            results["trained_tokens"] = run.get("trained_tokens")

            leaderboard.append(results)
            _print_eval_results(name, eval_model, results)

            # ── Checkpoint evaluation: check if an earlier checkpoint is better ──
            if run.get("overfitting_detected") or True:  # Always check — it's cheap compared to training
                try:
                    checkpoints = client.fine_tuning.jobs.checkpoints.list(run["job_id"])
                    if checkpoints.data and len(checkpoints.data) > 1:
                        best_cp = min(checkpoints.data, key=lambda cp: cp.metrics.valid_loss or float("inf"))
                        # Only eval if best checkpoint is significantly better than final
                        final_cp = checkpoints.data[0]  # Most recent = final
                        if best_cp.step_number != final_cp.step_number:
                            best_vl = best_cp.metrics.valid_loss or float("inf")
                            final_vl = final_cp.metrics.valid_loss or float("inf")
                            if final_vl > best_vl * 1.1:  # >10% worse = worth checking
                                cp_model = best_cp.fine_tuned_model_checkpoint
                                cp_name = f"{name}-ckpt{best_cp.step_number}"
                                cp_deploy = f"eval-{cp_name}"[:64]
                                print(f"\n  📊 Overfitting detected: final val_loss={final_vl:.4f} vs best={best_vl:.4f} at step {best_cp.step_number}")
                                print(f"  Evaluating checkpoint '{cp_name}'...")

                                if can_deploy:
                                    if _deploy_model_arm(sub, rg, account, cp_deploy, cp_model, capacity=capacity):
                                        deployed_names.append(cp_deploy)
                                        time.sleep(300)
                                        cp_results = _evaluate_model_on_test(client, cp_deploy, test_data, rubric, judge_model)
                                        cp_results["candidate"] = cp_name
                                        cp_results["model_id"] = cp_model
                                        cp_results["deployment_name"] = cp_deploy
                                        cp_results["hyperparameters"] = run.get("hyperparameters", {})
                                        cp_results["is_checkpoint"] = True
                                        leaderboard.append(cp_results)
                                        _print_eval_results(cp_name, cp_deploy, cp_results)
                                        _delete_deployment_arm(sub, rg, account, cp_deploy)
                                        deployed_names.remove(cp_deploy)
                except Exception as e:
                    print(f"  (Checkpoint check skipped: {str(e)[:80]})")

            # Delete deployment immediately after eval (keep quota clean)
            if can_deploy and deploy_name in deployed_names:
                print(f"  Cleaning up deployment '{deploy_name}'...")
                _delete_deployment_arm(sub, rg, account, deploy_name)
                deployed_names.remove(deploy_name)

    # Cleanup any remaining deployments (safety net)
    finally:
        for dn in list(deployed_names):
            print(f"  Cleaning up leftover deployment '{dn}'...")
            try:
                _delete_deployment_arm(sub, rg, account, dn)
            except Exception:
                print(f"  ⚠️  Failed to clean up '{dn}' — delete manually.")

    # Sort by combined score
    leaderboard.sort(key=lambda r: r.get("combined", 0), reverse=True)

    output_data = {
        "iteration": runs_data.get("iteration", 1),
        "test_file": os.path.abspath(test_file),
        "test_count": len(test_data),
        "candidates": leaderboard,
        "winner": leaderboard[0]["candidate"] if leaderboard else None,
    }

    output = args.output or "leaderboard.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  LEADERBOARD (iteration {output_data['iteration']})")
    print(f"{'='*60}")
    print(f"  {'Rank':<5} {'Candidate':<20} {'Combined':>8} {'Pass Rate':>10}")
    print(f"  {'-'*5} {'-'*20} {'-'*8} {'-'*10}")
    for i, r in enumerate(leaderboard):
        marker = " ← winner" if i == 0 else ""
        print(f"  {i+1:<5} {r['candidate']:<20} {r['combined']:>8.2f} {r.get('pass_rate', 0):>9.1f}%{marker}")

    print(f"\n  Output: {output}")
    print(f"  Next: auto_finetune.py review --leaderboard {output} --baseline baseline.json")


# ── Phase 7: REVIEW ──────────────────────────────────────────────────────

# NOTE: Earlier versions of this file shipped a hardcoded MODEL_PRICING dict
# (per-model input/output $ and TTFT seconds) plus a _compute_cost_comparison
# helper that printed "ROI vs Teacher: X% quality, Yx cheaper, Zx faster TTFT"
# in the SHIP summary. Those numbers were FAKE:
#   * MODEL_PRICING.ttft values were never measured anywhere — pure guesses
#     (e.g. gpt-5.4 ttft=181.2s — wildly off from reality).
#   * The "teacher" used for the comparison was hardcoded to gpt-5.4 regardless
#     of which teacher was actually configured for datagen.
#   * Pricing values would drift out of date with no warning.
# Removed in commit <pending> to stop fabricating numbers. If you want a real
# cost/latency comparison, measure it: run N inference calls against the FT
# model and the teacher (or any reference model) on the same prompts and
# record actual TTFT + token costs from the live Azure pricing page.


def cmd_review(args):
    """Compare candidates to baseline, decide: ship, iterate, or stop.
    
    Outputs rich diagnostics so the agent can design informed next-iteration candidates.
    """
    with open(args.leaderboard, encoding="utf-8") as f:
        lb = json.load(f)
    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)

    with open(args.task_spec, encoding="utf-8") as f:
        spec = json.load(f)

    # Load runs data for training metrics
    runs_data = {}
    if hasattr(args, 'runs') and args.runs and os.path.exists(args.runs):
        with open(args.runs, encoding="utf-8") as f:
            rd = json.load(f)
            for r in rd.get("runs", []):
                runs_data[r["candidate"]] = r

    stopping = spec.get("stopping_criteria", {})
    min_lift = stopping.get("min_lift_pct", 5.0)
    max_iter = stopping.get("max_iterations", 3)
    iteration = lb.get("iteration", 1)
    base_score = baseline.get("combined", 0)

    # Build per-model baseline lookup
    model_baselines = {}
    for r in baseline.get("results", []):
        model_baselines[r.get("model", "")] = r.get("combined", 0)
    # Fallback: if no per-model results, use the overall baseline for everything
    if not model_baselines:
        model_baselines["_default"] = base_score

    print(f"\n{'='*60}")
    print(f"  REVIEW (iteration {iteration})")
    print(f"{'='*60}")
    if len(model_baselines) > 1:
        print(f"  Baselines: {', '.join('%s=%.2f' % (m, s) for m, s in model_baselines.items())}")
    else:
        print(f"  Baseline: {base_score:.2f}")

    candidates = lb.get("candidates", [])
    if not candidates:
        print(f"\n  No candidates to review (all training/eval failed).")
        # Always write a review file so cmd_auto can keep going. Decision = STOP.
        review = {
            "iteration": lb.get("iteration", 1),
            "best": None,
            "lift_pct": 0.0,
            "decision": "STOP",
            "reason": "no_candidates",
            "rationale": "All candidate training or evaluation runs failed; nothing to compare to baseline. Inspect runs_iter*.json for per-job error messages.",
            "baselines": model_baselines,
            "diagnostics": [],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2)
        print(f"  Output: {args.output}")
        return

    best = candidates[0]
    best_score = best.get("combined", 0)
    # Use per-model baseline for the best candidate's lift
    best_run = runs_data.get(best.get("candidate", ""), {})
    best_base_model = best_run.get("model", "")
    best_base_score = model_baselines.get(best_base_model, model_baselines.get("_default", base_score))
    lift = ((best_score - best_base_score) / best_base_score * 100) if best_base_score > 0 else 0

    # Full candidate comparison
    print(f"\n  {'Candidate':<20} {'Combined':>8} {'Pass@8':>7} {'Lift':>8}  {'vs Base':>8}  HPs")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*8}  {'-'*8}  {'-'*30}")
    # Print baseline rows for each model
    for model, score in model_baselines.items():
        if model != "_default":
            bl_pass = next((r.get("pass_rate", 0) for r in baseline.get("results", []) if r.get("model") == model), 0)
            print(f"  {'BASE:'+model:<20} {score:>8.2f} {bl_pass:>6.1f}%  {'---':>8}  {'---':>8}")
    if "_default" in model_baselines and len(model_baselines) == 1:
        print(f"  {'BASELINE':<20} {base_score:>8.2f} {baseline.get('pass_rate', 0):>6.1f}%  {'---':>8}  {'---':>8}")

    diagnostics = []
    for c in candidates:
        c_score = c.get("combined", 0)
        c_pass = c.get("pass_rate", 0)
        c_errors = c.get("errors", 0)
        c_scored = c.get("n_scored", 0)
        hp = c.get("hyperparameters", {})
        hp_str = f"{hp.get('n_epochs', hp.get('epochs', '?'))}ep lr={hp.get('lr', '?')}"

        # Compare to THIS candidate's base model, not the overall baseline
        run_info = runs_data.get(c["candidate"], {})
        c_base_model = run_info.get("model", "")
        c_base_score = model_baselines.get(c_base_model, model_baselines.get("_default", base_score))
        c_lift = ((c_score - c_base_score) / c_base_score * 100) if c_base_score > 0 else 0

        marker = " <-- best" if c == best else ""
        base_tag = c_base_model.split("-")[-1] if c_base_model else ""
        print(f"  {c['candidate']:<20} {c_score:>8.2f} {c_pass:>6.1f}% {c_lift:>+7.1f}%  {'vs '+base_tag:>8}  {hp_str}{marker}")

        # Look up model and training curve data from runs
        run_info = runs_data.get(c["candidate"], {})
        train_loss = run_info.get("final_train_loss")
        best_val = run_info.get("best_val_loss")
        final_val = run_info.get("final_val_loss")
        overfit_ratio = run_info.get("overfit_ratio")
        best_val_step = run_info.get("best_val_step")
        loss_std = run_info.get("train_loss_std")
        c_model = run_info.get("model", c.get("model_id", ""))

        diag = {"candidate": c["candidate"], "score": c_score, "lift_pct": round(c_lift, 2),
                "hyperparameters": hp, "train_loss": train_loss, "pass_rate": c_pass,
                "model": c_model, "best_val_loss": best_val, "final_val_loss": final_val,
                "overfit_ratio": overfit_ratio, "best_val_step": best_val_step}

        if c_errors > 0 and c_scored == 0:
            diag["issue"] = "deployment_failure"
            diag["recommendation"] = (
                "All inference calls failed (likely deployment warmup too short or quota issue). "
                "Redeploy with longer warmup (5+ min) and retry evaluation."
            )
        elif c_lift < -50:
            lr = hp.get("lr", hp.get("learning_rate_multiplier", 1.0))
            if lr and float(lr) >= 2.0:
                diag["issue"] = "catastrophic_regression"
                diag["recommendation"] = (
                    f"Catastrophic regression with lr={lr}. The learning rate was too high — "
                    f"the model's weights were pushed too far. Try lr={float(lr)*0.25:.1f} or lower."
                )
            elif loss_std and loss_std > 1.0:
                diag["issue"] = "catastrophic_regression"
                diag["recommendation"] = (
                    f"Training was highly unstable (loss std={loss_std:.2f}). "
                    f"The model may be too small for this task. Try a larger model or simpler data."
                )
            else:
                diag["issue"] = "catastrophic_regression"
                diag["recommendation"] = (
                    "Severe regression. Check: (1) training data quality — are labels correct? "
                    "(2) eval rubric — does the judge model match what you're training for? "
                    "(3) try fewer epochs or lower LR."
                )
        elif c_lift < 0:
            diag["issue"] = "regression"
            # Use curve data for specific diagnosis
            rec_parts = []
            if overfit_ratio and overfit_ratio > 1.5:
                rec_parts.append(f"Overfitting detected (val/best ratio={overfit_ratio:.1f}). Deploy checkpoint at step {best_val_step}.")
            elif loss_std and loss_std > 0.8:
                rec_parts.append(f"Training unstable (loss std={loss_std:.2f}). Model may be too small or data has conflicting examples.")
            if train_loss and train_loss < 0.5 and c_lift > -10:
                rec_parts.append(f"Train loss is low ({train_loss:.2f}) but eval regressed — the model learned something, but the eval rubric may not align with training.")
            if not rec_parts:
                rec_parts.append(f"Regressed by {c_lift:.1f}%. Try: lower LR ({float(hp.get('lr', 1.0))*0.5:.1f}), or deploy an earlier checkpoint.")
            diag["recommendation"] = " ".join(rec_parts)
        elif c_lift < min_lift:
            diag["issue"] = "insufficient_lift"
            diag["recommendation"] = (
                f"Improved {c_lift:+.1f}% but needs {min_lift}%. "
                f"Try: more diverse training data, or narrow HPs around this config."
            )
        else:
            diag["issue"] = None
            diag["recommendation"] = f"Meets threshold ({c_lift:+.1f}% >= {min_lift}%)."
        diagnostics.append(diag)

    # Aggregate diagnostics
    deploy_failures = [d for d in diagnostics if d.get("issue") == "deployment_failure"]
    real_candidates = [d for d in diagnostics if d.get("issue") != "deployment_failure"]
    all_regressed = all(d["lift_pct"] <= 0 for d in real_candidates) if real_candidates else True
    any_catastrophic = any(d.get("issue") == "catastrophic_regression" for d in real_candidates)

    # ── Print per-candidate diagnostics ──
    print(f"\n  --- Per-Candidate Diagnosis ---")
    for d in diagnostics:
        icon = {"deployment_failure": "🔌", "catastrophic_regression": "💥",
                "regression": "📉", "insufficient_lift": "📊", None: "✅"}.get(d.get("issue"), "❓")
        tl = d.get("train_loss")
        bvl = d.get("best_val_loss")
        fvl = d.get("final_val_loss")
        oratio = d.get("overfit_ratio")
        bstep = d.get("best_val_step")

        # Build curve summary line
        curve_parts = []
        if tl is not None:
            curve_parts.append(f"final_train={tl:.2f}")
        if bvl is not None and fvl is not None:
            curve_parts.append(f"val={bvl:.2f}→{fvl:.2f}")
        if oratio is not None:
            curve_parts.append(f"overfit={oratio:.1f}x")
        if bstep is not None:
            curve_parts.append(f"best@step={bstep}")
        curve_str = ", ".join(curve_parts) if curve_parts else "no curve data"

        print(f"\n  {icon} {d['candidate']}:")
        print(f"     Score: {d['score']:.2f} ({d['lift_pct']:+.1f}% vs baseline)")
        print(f"     Curve: {curve_str}")
        print(f"     {d['recommendation']}")

    # ── Aggregate recommendations ──
    recommendations = []
    if deploy_failures:
        recommendations.append(
            f"{len(deploy_failures)} candidate(s) failed due to deployment issues (not model quality). "
            f"Redeploy with longer warmup and re-evaluate — the model may actually be good."
        )
    if all_regressed and not deploy_failures:
        recommendations.append(
            "All candidates regressed. Check: (1) are the training labels actually correct? "
            "(2) does the eval judge match the training task? (3) try a larger base model."
        )
    elif any_catastrophic:
        # Find the catastrophic ones and their LRs
        bad_lrs = [str(d["hyperparameters"].get("lr", "?")) for d in diagnostics
                   if d.get("issue") == "catastrophic_regression"]
        recommendations.append(
            f"Catastrophic regression at LR {', '.join(bad_lrs)}. "
            f"Use LR ≤ 1.0 for next iteration. Also try deploying earlier checkpoints."
        )
    if deploy_failures and real_candidates and all(d["lift_pct"] <= 0 for d in real_candidates):
        recommendations.append(
            "The candidates that DID deploy also regressed. "
            "Consider: more training data, different base model, or reviewing data quality."
        )
    if lift > 0 and lift < min_lift:
        recommendations.append(
            f"Best improved {lift:+.1f}% but target is {min_lift}%. "
            f"Try: more training data, narrow HPs around the winner, or augment existing data."
        )

    # ── Decision ──
    if lift >= min_lift:
        decision = "SHIP"
    elif iteration >= max_iter:
        decision = "STOP"
    else:
        decision = "ITERATE"

    # ── Print decision + recommendations + summary ──
    print(f"\n  {'='*55}")
    if decision == "SHIP":
        print(f"  ✅ DECISION: SHIP")
        print(f"  {'='*55}")
        print(f"  Winner: {best['candidate']} ({lift:+.1f}% lift, meets {min_lift}% threshold)")
        print(f"  Model:  {best.get('model_id', 'N/A')}")
    elif decision == "STOP":
        print(f"  ⏹️  DECISION: STOP (iteration {iteration}/{max_iter})")
        print(f"  {'='*55}")
        if lift > 0:
            print(f"  Best: {best['candidate']} at {lift:+.1f}% (below {min_lift}% threshold)")
        else:
            print(f"  No candidate beat the baseline after {iteration} iteration(s).")
    else:
        print(f"  🔄 DECISION: ITERATE (iteration {iteration}/{max_iter})")
        print(f"  {'='*55}")

    if recommendations:
        print(f"\n  Next steps:")
        for i, r in enumerate(recommendations, 1):
            print(f"    {i}. {r}")

    # Get training example count from runs data
    train_examples = "?"
    if runs_data:
        first_run = next(iter(runs_data.values()), {})
        if "dataset_hash" in first_run:
            # Try to read from manifest
            spec_dir = os.path.dirname(args.task_spec)
            manifest_path = os.path.join(spec_dir, "prepared", "data_manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                train_examples = manifest.get("splits", {}).get("train", {}).get("count", "?")

    # Save review with full diagnostics
    review = {
        "iteration": iteration,
        "baseline_score": base_score,
        "best_candidate": best["candidate"],
        "best_score": best_score,
        "lift_pct": round(lift, 2),
        "decision": decision,
        "candidate_diagnostics": diagnostics,
        "recommendations": recommendations,
        "train_examples": train_examples,
        "next_action": {
            "SHIP": "Deploy the winning model",
            "ITERATE": f"Design new candidates (iteration {iteration + 1}) addressing the recommendations above",
            "STOP": "Report findings; consider manual investigation or different approach",
        }.get(decision, ""),
    }

    output = args.output or "review.json"
    _atomic_json_write(output, review)

    print(f"\n  Output: {output}")


# ── Shared evaluation helper ─────────────────────────────────────────────

def _evaluate_model_on_test(client, model, test_data, rubric, judge_model):
    """Run a model on test data and grade with LLM judge."""
    import re

    dimensions = rubric.get("dimensions", [{"name": "correctness", "weight": 1.0}])
    pass_threshold = rubric.get("pass_threshold", 8)
    dim_names = [d["name"] for d in dimensions]
    weights = {d["name"]: d.get("weight", 1.0) for d in dimensions}

    all_scores = []
    errors = 0

    for i, ex in enumerate(test_data):
        msgs = ex.get("messages", [])
        # Extract system, user, reference
        system_msgs = [m for m in msgs if m["role"] == "system"]
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        reference = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        # Generate response
        gen_msgs = system_msgs + [{"role": "user", "content": user_msg}]
        try:
            resp = client.chat.completions.create(
                model=model, messages=gen_msgs, temperature=0.0, max_completion_tokens=2048,
            )
            output = resp.choices[0].message.content or ""
        except Exception as e:
            errors += 1
            all_scores.append({d: 0 for d in dim_names})
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(test_data)}] (error: {e})")
            continue

        # Grade with LLM judge
        dim_text = "\n".join(f"**{d['name']}** (1-10): {d.get('description', '')}" for d in dimensions)
        example_json = json.dumps({d: 8 for d in dim_names})
        judge_prompt = (
            f"Evaluate this model output.\n\n"
            f"## Prompt\n{user_msg}\n\n"
            f"## Reference\n{reference}\n\n"
            f"## Model Output\n{output}\n\n"
            f"## Dimensions\n{dim_text}\n\n"
            f"Return ONLY JSON: {example_json}"
        )
        try:
            judge_resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0, max_completion_tokens=200,
            )
            text = judge_resp.choices[0].message.content.strip()
            match = re.search(r'\{[^}]+\}', text)
            if match:
                scores = json.loads(match.group())
                # Clamp scores to valid 1-10 range
                scores = {d: max(1, min(10, int(scores.get(d, 0)))) for d in dim_names}
            else:
                scores = {d: 0 for d in dim_names}
        except Exception as e:
            if "DeploymentNotFound" in str(e) or "404" in str(e):
                if i == 0:
                    print(f"  ❌ Judge model '{judge_model}' not found. Check deployment or change eval_rubric.judge_model in task_spec.json")
                    return {"combined": 0, "pass_rate": 0, "errors": len(test_data), "n": len(test_data),
                            "error": f"Judge model '{judge_model}' not deployed"}
            scores = {d: 0 for d in dim_names}

        all_scores.append(scores)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_data)}] scored")

    # Aggregate
    valid = [s for s in all_scores if any(v > 0 for v in s.values())]
    if not valid:
        return {"combined": 0, "pass_rate": 0, "errors": errors, "n": len(test_data)}

    dim_avgs = {}
    for d in dim_names:
        vals = [s[d] for s in valid if s[d] > 0]
        dim_avgs[d] = sum(vals) / len(vals) if vals else 0

    total_weight = sum(weights.values())
    if total_weight <= 0:
        print(f"  ❌ Eval rubric weights sum to {total_weight} — check task_spec eval_rubric dimensions.")
        return {"combined": 0, "pass_rate": 0, "errors": errors, "n": len(test_data)}
    combined = sum(dim_avgs.get(d, 0) * weights.get(d, 0) for d in dim_names) / total_weight

    pass_count = sum(1 for s in valid if
                     sum(s.get(d, 0) * weights.get(d, 0) for d in dim_names) / total_weight >= pass_threshold)
    pass_rate = pass_count / len(valid) * 100 if valid else 0

    return {
        "model": model,
        "combined": round(combined, 2),
        "dimensions": {d: round(v, 2) for d, v in dim_avgs.items()},
        "pass_rate": round(pass_rate, 1),
        "n_scored": len(valid),
        "n_total": len(test_data),
        "errors": errors,
    }


def _print_eval_results(label, model, results):
    """Pretty-print evaluation results."""
    print(f"\n  {label}: {model}")
    print(f"    Combined: {results.get('combined', 0):.2f}")
    for dim, score in results.get("dimensions", {}).items():
        print(f"    {dim}: {score:.2f}")
    print(f"    Pass rate: {results.get('pass_rate', 0):.1f}%")
    print(f"    Scored: {results.get('n_scored', 0)}/{results.get('n_total', 0)} ({results.get('errors', 0)} errors)")


# ── Phase AUTO: Full autopilot loop ──────────────────────────────────────

def _infer_datagen_backend(args) -> str:
    """Pick the datagen backend from the user's flags.

    Resolution order:
      1. Explicit `--datagen-backend X` (anything other than 'auto') wins.
      2. Otherwise infer from companion flags:
         - --datagen-file-id set                            → foundry-file
         - --datagen-agent-name + --datagen-hours set       → foundry-traces
         - --datagen-agent-name set (no hours)              → foundry-agent
         - (no datagen-* hints)                             → local

    Note: `--project-endpoint` alone is NOT enough to switch to Foundry. Many
    users set AZURE_AI_PROJECT_ENDPOINT for unrelated reasons (e.g. judge
    model deployment); routing them through Foundry datagen by accident causes
    confusing failures like "File content is too small" (the autopilot writes
    the short task description to a tmp file, which is under the 1 KB minimum).
    Users must explicitly opt into Foundry via a datagen-* flag or
    --datagen-backend foundry-prompt.
    """
    explicit = getattr(args, "datagen_backend", "auto")
    if explicit and explicit != "auto":
        return explicit

    if getattr(args, "datagen_file_id", None):
        chosen, why = "foundry-file", "--datagen-file-id set"
    elif getattr(args, "datagen_agent_name", None) and getattr(args, "datagen_hours", None):
        chosen, why = "foundry-traces", "--datagen-agent-name + --datagen-hours set"
    elif getattr(args, "datagen_agent_name", None):
        chosen, why = "foundry-agent", "--datagen-agent-name set (no --datagen-hours)"
    else:
        chosen, why = "local", "no datagen-* flags (use --datagen-backend foundry-prompt to opt in)"

    print(f"  Datagen backend inferred: {chosen}  ({why})")
    return chosen


def cmd_auto(args):
    """Run the full fine-tuning loop: analyze → prepare → baseline → candidates → execute → evaluate → review → iterate."""
    import argparse

    # Validate: need either --data or --description
    if not getattr(args, "data", None) and not args.description:
        print("❌ Provide either --data (a data file) or --description (generate data from a prompt).")
        sys.exit(1)

    # Validate: reject RFT-only models (auto-finetune is SFT only)
    model = getattr(args, "model", "gpt-4.1-mini")
    if _is_rft_only_model(model):
        print(f"❌ Model '{model}' only supports RFT (reinforcement fine-tuning).")
        print(f"   auto-finetune is SFT only. Use submit_training.py --type rft instead.")
        print(f"   For SFT, try: gpt-4.1-mini, gpt-4.1-nano, or gpt-4o.")
        sys.exit(1)

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Paths for intermediate artifacts
    task_spec_path = os.path.join(work_dir, "task_spec.json")
    generated_dir = os.path.join(work_dir, "generated")
    prepared_dir = os.path.join(work_dir, "prepared")
    baseline_path = os.path.join(work_dir, "baseline.json")

    max_iterations = args.max_iterations
    data_path = getattr(args, "data", None)

    print("=" * 60)
    print("  AUTO FINE-TUNE")
    print("=" * 60)
    print(f"  Data:           {data_path or '(prompt-only — will generate)'}")
    print(f"  Description:    {args.description[:80] if args.description else '(none)'}")
    print(f"  Work dir:       {work_dir}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Model:          {args.model}")
    print("=" * 60)

    # ── Phase 1: ANALYZE ──
    print("\n\n" + "=" * 60)
    print("  PHASE 1: ANALYZE")
    print("=" * 60)
    analyze_args = argparse.Namespace(
        data=data_path, description=args.description, task_name=args.task_name,
        model=args.model, max_budget=args.max_budget, output=task_spec_path,
        base_url=args.base_url, api_key=args.api_key,
        project_endpoint=args.project_endpoint,
    )
    cmd_analyze(analyze_args)

    spec = json.load(open(task_spec_path, encoding="utf-8"))
    data_mode = spec.get("data_mode", "labeled")

    # ── Phase 2: GENERATE (if no data, unlabeled, or prompt-only) ──
    backend = _infer_datagen_backend(args)
    needs_generate = data_mode in ("unlabeled", "prompt_only")
    if needs_generate:
        print("\n\n" + "=" * 60)
        if data_mode == "prompt_only":
            print(f"  PHASE 2: GENERATE  [backend={backend}]")
            print("  (no data file — generating from description)")
        else:
            print(f"  PHASE 2: GENERATE  [backend={backend}]")
            print("  (unlabeled data detected)")
        print("=" * 60)

        if backend == "local":
            gen_args = argparse.Namespace(
                task_spec=task_spec_path, num_examples=args.num_examples,
                teacher=args.teacher, schema_file=args.schema_file,
                min_quality=args.min_quality, difficulty="mixed",
                existing_data=None, output_dir=generated_dir,
                base_url=args.base_url, api_key=args.api_key,
                project_endpoint=args.project_endpoint,
            )
            cmd_generate(gen_args)
        else:
            # foundry-* backend → delegate to cmd_foundry_generate via task_spec
            # NOTE: prompt-inline + qna + SFT has been observed to fail fast on some
            # projects (see references/data-generation-api.md error table). We default
            # to prompt-file when there's no agent/file-id input so the prompt is
            # uploaded as a user_data file under the hood (workaround for that bug).
            src_map = {
                "foundry-prompt": ("prompt-file",   "qna"),     # prompt-file ⇒ uploads as File internally
                "foundry-file":   ("file",          "qna"),
                "foundry-agent":  ("agent",         "qna"),
                "foundry-traces": ("traces",        "traces"),
            }
            source, recipe = src_map[backend]
            # Allow user override of recipe (e.g. tool-use with foundry-file when
            # the uploaded file is an OpenAPI 3.0 spec instead of a prose document)
            if getattr(args, "datagen_recipe", None):
                recipe = args.datagen_recipe
            scenario = getattr(args, "datagen_scenario", None) or "sft"
            prompt_file_for_foundry = None
            if source == "prompt-file":
                # Write spec.description to a temp file so generate_dataset.py can
                # upload it as user_data. Caller-visible artifact lives in work-dir.
                prompt_file_for_foundry = os.path.join(work_dir, "prompt_for_foundry.md")
                with open(prompt_file_for_foundry, "w", encoding="utf-8") as _pf:
                    _pf.write(spec.get("description", ""))
            fg_args = argparse.Namespace(
                task_spec=task_spec_path, source=source, recipe=recipe, scenario=scenario,
                max_samples=max(15, min(1000, args.num_examples)),
                train_split=None,  # let prepare phase split
                teacher=args.teacher,
                prompt=None,
                prompt_file=prompt_file_for_foundry,
                file_id=args.datagen_file_id,
                agent_name=args.datagen_agent_name,
                agent_version=args.datagen_agent_version,
                hours=(args.datagen_hours or 24) if source == "traces" else None,
                output_dir=generated_dir,
                base_url=args.base_url, api_key=args.api_key,
                project_endpoint=args.project_endpoint,
            )
            cmd_foundry_generate(fg_args)
        data_path = os.path.join(generated_dir, "generated_data.jsonl")
    elif data_mode == "chat_sft":
        print("\n  Data is already in SFT chat format — skipping generation.")
    else:
        print(f"\n  Data mode: {data_mode} — skipping generation.")

    # Optional: Azure Content Safety pre-screen between GENERATE and PREPARE.
    # Off by default. Use when Azure FT preprocessing has rejected your data
    # with "User data has failed data safety check" and you want to drop the
    # offending rows automatically. Shells out to scripts/content_safety_check.py
    # so the helper stays a separate concern (debugging tool, not core flow).
    if getattr(args, "content_safety_prescreen", False) and data_path and os.path.exists(data_path):
        import subprocess
        cs_endpoint = getattr(args, "content_safety_endpoint", None) or os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT")
        cs_key = getattr(args, "content_safety_key", None) or os.environ.get("AZURE_CONTENT_SAFETY_KEY")
        cs_threshold = getattr(args, "content_safety_threshold", None) or 2
        if not cs_endpoint or not cs_key:
            print("\n  ⚠️  --content-safety-prescreen set but AZURE_CONTENT_SAFETY_ENDPOINT/KEY unset — skipping.")
        else:
            print("\n\n" + "=" * 60)
            print(f"  PHASE 2b: CONTENT SAFETY PRE-SCREEN (threshold={cs_threshold})")
            print("=" * 60)
            cs_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_safety_check.py")
            cleaned = data_path.replace(".jsonl", ".safe.jsonl")
            cmd = [
                sys.executable, cs_script,
                "--jsonl", data_path,
                "--endpoint", cs_endpoint,
                "--api-key", cs_key,
                "--threshold", str(cs_threshold),
                "--drop-out", cleaned,
            ]
            r = subprocess.run(cmd, capture_output=False)
            if r.returncode in (0, 1) and os.path.exists(cleaned):
                # rc=1 means some rows were flagged but a clean file was written
                print(f"\n  Using cleaned data for downstream phases: {cleaned}")
                data_path = cleaned
            else:
                print(f"\n  ⚠️  Pre-screen failed (rc={r.returncode}); proceeding with original {data_path}")

    # ── Phase 3: PREPARE ──
    print("\n\n" + "=" * 60)
    print("  PHASE 3: PREPARE")
    print("=" * 60)
    prepare_args = argparse.Namespace(
        task_spec=task_spec_path, data=data_path, output_dir=prepared_dir,
    )
    cmd_prepare(prepare_args)

    test_file = os.path.join(prepared_dir, "test.jsonl")

    # ── Phase 4: BASELINE ──
    print("\n\n" + "=" * 60)
    print("  PHASE 4: BASELINE")
    print("=" * 60)
    baseline_args = argparse.Namespace(
        task_spec=task_spec_path, test_file=test_file, multi=False,
        output=baseline_path,
        base_url=args.base_url, api_key=args.api_key,
        project_endpoint=args.project_endpoint,
    )
    cmd_baseline(baseline_args)

    baseline = json.load(open(baseline_path, encoding="utf-8"))
    base_score = baseline.get("combined", 0)

    # Check headroom
    if base_score >= 9.5:
        print(f"\n  ⚠️  Base model already scores {base_score:.2f}/10 — minimal headroom for improvement.")
        print("  Consider whether fine-tuning is worth the cost.")

    # ── Iteration loop ──
    review_file = None
    for iteration in range(1, max_iterations + 1):
        print("\n\n" + "#" * 60)
        print(f"  ITERATION {iteration}/{max_iterations}")
        print("#" * 60)

        plan_path = os.path.join(work_dir, f"candidate_plan_iter{iteration}.json")
        runs_path = os.path.join(work_dir, f"runs_iter{iteration}.json")
        leaderboard_path = os.path.join(work_dir, f"leaderboard_iter{iteration}.json")
        review_path = os.path.join(work_dir, f"review_iter{iteration}.json")

        # ── Phase 5: CANDIDATES ──
        print("\n" + "=" * 60)
        print(f"  PHASE 5: DESIGN CANDIDATES (iteration {iteration})")
        print("=" * 60)
        cand_args = argparse.Namespace(
            task_spec=task_spec_path, data_dir=prepared_dir,
            iteration=iteration, review_file=review_file,
            output=plan_path,
        )
        cmd_candidates(cand_args)

        # Check for data augmentation recommendations
        plan = json.load(open(plan_path, encoding="utf-8"))

        # ── Baseline any alt models not yet baselined ──
        candidate_models = set(c["model"] for c in plan.get("candidates", []))
        baselined_models = set(baseline.get("models_tested", []))
        new_models = candidate_models - baselined_models
        if new_models:
            print(f"\n  Baselining {len(new_models)} new model(s): {', '.join(new_models)}")
            client_bl, _ = get_clients(
                base_url=args.base_url, project_endpoint=args.project_endpoint, api_key=args.api_key
            )
            rubric_bl = json.load(open(task_spec_path, encoding="utf-8")).get("eval_rubric", {})
            judge_bl = rubric_bl.get("judge_model", "gpt-4o")
            with open(test_file, encoding="utf-8") as f:
                test_bl = [json.loads(line) for line in f if line.strip()]
            for model in new_models:
                print(f"    Evaluating {model}...")
                r = _evaluate_model_on_test(client_bl, model, test_bl, rubric_bl, judge_bl)
                r["model"] = model
                baseline.setdefault("results", []).append(r)
                baseline["models_tested"] = list(set(baseline.get("models_tested", [])) | {model})
                _print_eval_results(f"BASELINE ({model})", model, r)
            # Save updated baseline
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline, f, indent=2)

        data_recs = plan.get("data_recommendations", [])
        if data_recs:
            print(f"\n  📊 Data augmentation recommended:")
            for rec in data_recs:
                print(f"     Target: {rec.get('target_examples', '?')} examples (currently {rec.get('current_examples', '?')})")
                print(f"     Reason: {rec.get('rationale', '')[:100]}")

            # Generate more data
            existing = os.path.join(prepared_dir, "train.jsonl")
            target = data_recs[0].get("target_examples", 400)
            print(f"\n  Generating {target} more examples (augmenting existing data)...")
            gen_args = argparse.Namespace(
                task_spec=task_spec_path, num_examples=target,
                teacher=args.teacher, schema_file=args.schema_file,
                min_quality=args.min_quality, difficulty="mixed",
                existing_data=existing, output_dir=generated_dir,
                base_url=args.base_url, api_key=args.api_key,
                project_endpoint=args.project_endpoint,
            )
            cmd_generate(gen_args)

            # Re-prepare with augmented data
            augmented_data = os.path.join(generated_dir, "generated_data.jsonl")
            prepare_args = argparse.Namespace(
                task_spec=task_spec_path, data=augmented_data, output_dir=prepared_dir,
            )
            cmd_prepare(prepare_args)

        # ── Phase 6: EXECUTE ──
        print("\n" + "=" * 60)
        print(f"  PHASE 6: EXECUTE (iteration {iteration})")
        print("=" * 60)
        exec_args = argparse.Namespace(
            plan=plan_path, output=runs_path,
            tier=getattr(args, "tier", None),
            base_url=args.base_url, api_key=args.api_key,
            project_endpoint=args.project_endpoint,
        )
        cmd_execute(exec_args)

        # ── Phase 7: EVALUATE ──
        print("\n" + "=" * 60)
        print(f"  PHASE 7: EVALUATE (iteration {iteration})")
        print("=" * 60)
        eval_args = argparse.Namespace(
            runs=runs_path, task_spec=task_spec_path,
            test_file=test_file, capacity=args.capacity,
            output=leaderboard_path,
            base_url=args.base_url, api_key=args.api_key,
            project_endpoint=args.project_endpoint,
        )
        cmd_evaluate(eval_args)

        # ── Phase 8: REVIEW ──
        print("\n" + "=" * 60)
        print(f"  PHASE 8: REVIEW (iteration {iteration})")
        print("=" * 60)
        rev_args = argparse.Namespace(
            leaderboard=leaderboard_path, baseline=baseline_path,
            task_spec=task_spec_path, output=review_path,
            runs=runs_path,
        )
        cmd_review(rev_args)

        # Read the decision
        review = json.load(open(review_path, encoding="utf-8"))
        decision = review.get("decision", "STOP")
        review_file = review_path

        if decision == "SHIP":
            print("\n\n" + "=" * 60)
            print("  ✅ SHIPPING — Winner found!")
            print("=" * 60)
            _print_auto_summary(work_dir, iteration, review, baseline)
            return

        if decision == "STOP":
            print("\n\n" + "=" * 60)
            print("  ⏹️ STOPPING — Max iterations or budget reached")
            print("=" * 60)
            _print_auto_summary(work_dir, iteration, review, baseline)
            return

        # ITERATE — loop continues
        print(f"\n  🔄 ITERATING — designing next round based on diagnostics...")

    # Fell through all iterations
    print("\n\n" + "=" * 60)
    print(f"  ⏹️ COMPLETED {max_iterations} iterations without meeting threshold")
    print("=" * 60)
    if review_file:
        review = json.load(open(review_file, encoding="utf-8"))
        _print_auto_summary(work_dir, max_iterations, review, baseline)


def _print_auto_summary(work_dir, iterations, final_review, baseline):
    """Print a rich experiment table summary of an auto run."""
    best_name = final_review.get("best_candidate", "none")
    best_score = final_review.get("best_score", 0)
    base_score = baseline.get("combined", 0)
    base_model = baseline.get("model", baseline.get("models_tested", [""])[0] if "models_tested" in baseline else "")
    if not base_model:
        # Try to infer from task spec
        spec_path = os.path.join(work_dir, "task_spec.json")
        if os.path.exists(spec_path):
            with open(spec_path, encoding="utf-8") as f:
                spec = json.load(f)
            base_model = spec.get("base_model", "unknown")
        else:
            base_model = "unknown"
    base_pass = baseline.get("pass_rate", 0)
    lift = final_review.get("lift_pct", 0)
    decision = final_review.get("decision", "?")
    diagnostics = final_review.get("candidate_diagnostics", [])
    recommendations = final_review.get("recommendations", [])

    # Infer task name from work_dir
    task_name = os.path.basename(work_dir.rstrip("/\\")) if work_dir else "unknown"

    icon = {"SHIP": "✅", "ITERATE": "🔄", "STOP": "⏹️"}.get(decision, "❓")

    print(f"\n{'='*60}")
    print(f"  {icon} AUTO FINE-TUNE SUMMARY")
    print(f"{'='*60}")
    print(f"  Task:         {task_name}")
    print(f"  Base model:   {base_model}")
    print(f"  Iterations:   {iterations}")
    print(f"  Decision:     {decision}")

    # Full experiment table
    print(f"\n  --- Experiment Results ---")
    header = (f"  {'Candidate':<20} {'Model':<15} {'Epochs':>6}  {'LR':>4}  "
              f"{'Train':>6} {'Val':>6} {'OvFit':>5}  {'Combined':>8}  {'Pass@8':>6}  {'Lift':>7}  {'Diagnosis'}")
    sep = (f"  {'-'*20} {'-'*15} {'-'*6}  {'-'*4}  "
           f"{'-'*6} {'-'*6} {'-'*5}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*19}")
    print(header)
    print(sep)

    # Baseline row
    print(f"  {'BASELINE':<20} {base_model:<15} {'-':>6}  {'-':>4}  "
          f"{'-':>6} {'-':>6} {'-':>5}  {base_score:>8.2f}  {base_pass:>5.1f}%  {'-':>7}  ")

    # Candidate rows
    issue_icons = {"deployment_failure": "🔌 Deploy fail",
                   "catastrophic_regression": "💥 LR too high",
                   "regression": "📉 Regression",
                   "insufficient_lift": "📊 Below threshold",
                   None: "✅ Ship it"}

    for d in diagnostics:
        hp = d.get("hyperparameters", {})
        epochs = str(hp.get("n_epochs", hp.get("epochs", "-")))
        lr = hp.get("lr", hp.get("learning_rate_multiplier", "-"))
        lr_str = f"{float(lr):.1f}" if lr and lr != "-" else "-"
        model = d.get("model", hp.get("model", base_model))
        train_loss = d.get("train_loss")
        tl_str = f"{train_loss:.2f}" if train_loss is not None else "-"
        fvl = d.get("final_val_loss")
        vl_str = f"{fvl:.2f}" if fvl is not None else "-"
        oratio = d.get("overfit_ratio")
        of_str = f"{oratio:.1f}x" if oratio is not None else "-"
        score = d.get("score", 0)
        c_pass = d.get("pass_rate", 0)
        lift_pct = d.get("lift_pct", 0)
        lift_str = f"{lift_pct:+.1f}%"
        diag_str = issue_icons.get(d.get("issue"), "❓ Unknown")

        pass_str = f"{c_pass:>5.1f}%"

        print(f"  {d['candidate']:<20} {model:<15} {epochs:>6}  {lr_str:>4}  "
              f"{tl_str:>6} {vl_str:>6} {of_str:>5}  {score:>8.2f}  {pass_str}  {lift_str:>7}  {diag_str}")

    # Recommendations
    if recommendations:
        print(f"\n  --- Next Steps ---")
        for i, r in enumerate(recommendations, 1):
            print(f"  {i}. {r}")

    # Proposed next iteration (only if ITERATE)
    if decision == "ITERATE":
        print(f"\n  --- Proposed Next Iteration ---")
        deploy_failures = [d for d in diagnostics if d.get("issue") == "deployment_failure"]
        catastrophic = [d for d in diagnostics if d.get("issue") == "catastrophic_regression"]
        regressions = [d for d in diagnostics if d.get("issue") == "regression"]
        close_calls = [d for d in diagnostics if d.get("issue") == "insufficient_lift"]
        real_candidates = [d for d in diagnostics if d.get("issue") != "deployment_failure"]
        all_regressed = all(d.get("lift_pct", 0) <= 0 for d in real_candidates) if real_candidates else True

        if deploy_failures:
            print(f"  1. Re-evaluate {len(deploy_failures)} deployment failure(s) with longer warmup:")
            for df in deploy_failures:
                print(f"     - {df['candidate']} (train_loss={df.get('train_loss', '?')})")

        if catastrophic:
            bad_lrs = set(str(d.get("hyperparameters", {}).get("lr", "?")) for d in catastrophic)
            print(f"  {'2' if deploy_failures else '1'}. Drop high LR candidates (lr={', '.join(bad_lrs)}) — caused catastrophic regression")

        if regressions or (all_regressed and not deploy_failures):
            n = len(deploy_failures) + (1 if catastrophic else 0) + 1
            # Find the closest-to-baseline candidate
            best_d = min(diagnostics, key=lambda d: abs(d.get("lift_pct", -999)))
            best_loss = best_d.get("train_loss")
            best_lr = best_d.get("hyperparameters", {}).get("lr", 1.0)
            best_model = best_d.get("model", base_model)
            train_n = final_review.get("train_examples", "?")

            print(f"  {n}. Submit new candidates:")
            print(f"     - {best_model} 2ep lr={float(best_lr)*0.5:.1f} — lower LR than best attempt ({best_d['candidate']})")
            print(f"     - {best_model} 1ep lr={float(best_lr):.1f} — fewer epochs to reduce overfitting")
            if best_loss and best_loss < 0.5:
                print(f"     - Deploy {best_d['candidate']} epoch-1 checkpoint — train_loss={best_loss:.2f} suggests the model learned, but may have overfit by epoch 3")
            print(f"  {n+1}. Generate more training data (currently {train_n} examples)")
            print(f"     - More diverse examples may help the model generalize instead of memorize")

        if close_calls:
            best_close = close_calls[0]
            print(f"  Narrow hyperparameters around {best_close['candidate']} ({best_close['lift_pct']:+.1f}% lift)")
            print(f"     - Try: same config with more data, or lr +/- 0.2")

    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser():
    parser = HelpOnErrorParser(
        description="Autonomous fine-tuning orchestrator (EXPERIMENTAL, SFT only) — analyze, prepare, train, evaluate, iterate",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared connection args
    def add_connection_args(p):
        p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
        p.add_argument("--api-key", default=os.environ.get("AZURE_OPENAI_API_KEY"))
        p.add_argument("--project-endpoint", default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"))

    # analyze
    p = sub.add_parser("analyze", help="Analyze raw data and generate task spec")
    p.add_argument("--data", default=None, help="Raw data file (CSV, JSON, JSONL, Parquet). Omit to generate from --description.")
    p.add_argument("--description", default="", help="Task description in natural language")
    p.add_argument("--task-name", default=None, help="Short task name (default: filename)")
    p.add_argument("--model", default="gpt-4.1-mini", help="Base model for fine-tuning")
    p.add_argument("--max-budget", default=50, help="Max budget in USD")
    p.add_argument("--output", default="task_spec.json")
    add_connection_args(p)

    # generate
    p = sub.add_parser("generate", help="Generate training data from task spec using teacher model")
    p.add_argument("--task-spec", required=True)
    p.add_argument("--num-examples", type=int, default=200, help="Target number of NEW examples to generate")
    p.add_argument("--teacher", default=None, help="Teacher model (default: base_model from task spec)")
    p.add_argument("--schema-file", default=None, help="Schema file (SQL tables, API spec, etc.)")
    p.add_argument("--min-quality", type=float, default=7.0, help="Minimum quality score to keep (1-10)")
    p.add_argument("--difficulty", default="mixed", choices=["easy", "mixed", "hard"],
                   help="Difficulty distribution: easy (70%% simple), mixed (40/40/20), hard (20/40/40)")
    p.add_argument("--existing-data", default=None,
                   help="Path to existing JSONL data — new examples will be deduped against this and merged")
    p.add_argument("--output-dir", default="./generated")
    add_connection_args(p)

    # foundry-generate (uses the Foundry Data Generation API instead of a local teacher loop)
    p = sub.add_parser("foundry-generate",
                       help="Generate data via the Foundry Data Generation API (traces / corpus / agent / OpenAPI spec → SFT or eval JSONL)")
    p.add_argument("--task-spec", required=True)
    p.add_argument("--source", required=True,
                   choices=["traces", "prompt-inline", "prompt-file", "file", "agent"],
                   help="Where Foundry pulls raw material from")
    p.add_argument("--recipe", default="qna", choices=["traces", "qna", "tool-use"],
                   help="Recipe to apply (default: qna)")
    p.add_argument("--scenario", default="sft", choices=["sft", "eval"],
                   help="What the data is for (default: sft). RFT requires Traces source — see workflows/traces-to-dataset.md.")
    p.add_argument("--max-samples", type=int, default=100,
                   help="Samples to produce (15-1000, enforced by service; default 100)")
    p.add_argument("--train-split", type=float, default=0.8,
                   help="Train/validation split for SFT (default 0.8). EVAL ignores this.")
    p.add_argument("--teacher", default=None,
                   help="Teacher model deployment name (required for qna/tool-use; not needed for traces)")
    # Source-specific args
    p.add_argument("--prompt", default=None,
                   help="Inline prompt text for --source prompt-inline (default: task_spec.description)")
    p.add_argument("--prompt-file", default=None, help="Path to a text file (for --source prompt-file)")
    p.add_argument("--file-id", default=None,
                   help="Pre-uploaded OpenAI file id (for --source file). For tool-use, the file MUST be an OpenAPI 3.0/3.1 spec.")
    p.add_argument("--agent-name", default=None, help="Deployed agent name (for traces/agent sources)")
    p.add_argument("--agent-version", default=None, help="Pin agent version (recommended for traces)")
    p.add_argument("--hours", type=int, default=None, help="For traces: pull spans from last N hours")
    p.add_argument("--output-dir", default="./generated")
    add_connection_args(p)

    # prepare
    p = sub.add_parser("prepare", help="Convert, filter, and split data")
    p.add_argument("--task-spec", required=True)
    p.add_argument("--data", default=None, help="Raw data file (default: from task spec)")
    p.add_argument("--output-dir", default="./prepared")

    # baseline
    p = sub.add_parser("baseline", help="Evaluate base model(s) on test set")
    p.add_argument("--task-spec", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--multi", action="store_true",
                   help="Test multiple recommended models (picks best for fine-tuning)")
    p.add_argument("--output", default="baseline.json")
    add_connection_args(p)

    # candidates
    p = sub.add_parser("candidates", help="Design candidate experiments")
    p.add_argument("--task-spec", required=True)
    p.add_argument("--data-dir", default="./prepared")
    p.add_argument("--iteration", type=int, default=1)
    p.add_argument("--review-file", default=None,
                   help="Previous review.json — enables smart iteration based on diagnostics")
    p.add_argument("--output", default="candidate_plan.json")

    # execute
    p = sub.add_parser("execute", help="Submit and monitor all candidates")
    p.add_argument("--plan", required=True)
    p.add_argument("--output", default="runs.json")
    p.add_argument("--tier", default="globalStandard",
                   help="Training tier(s) — single value or comma-separated list for round-robin. "
                        "'globalStandard' (default, works in all regions). 'developerTier' (cheapest, "
                        "capacity-limited). 'standard' (region-restricted; absent in many regions). "
                        "Example: --tier globalStandard,developerTier distributes candidates across "
                        "both tiers so one capacity bottleneck doesn't block the whole iteration. "
                        "OSS models auto-override to globalStandard.")
    add_connection_args(p)

    # evaluate
    p = sub.add_parser("evaluate", help="Deploy, evaluate, and clean up candidates")
    p.add_argument("--runs", required=True)
    p.add_argument("--task-spec", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--capacity", type=int, default=100,
                   help="Deployment capacity for eval (default: 100 for fast eval)")
    p.add_argument("--output", default="leaderboard.json")
    add_connection_args(p)

    # review
    p = sub.add_parser("review", help="Compare candidates to baseline and decide")
    p.add_argument("--leaderboard", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--task-spec", required=True)
    p.add_argument("--output", default="review.json")
    p.add_argument("--runs", default=None, help="runs.json for training metrics (optional)")

    # auto (full loop)
    p = sub.add_parser("auto", help="Run the full loop: analyze → prepare → baseline → train → evaluate → iterate")
    p.add_argument("--data", default=None, help="Raw data file (CSV, JSON, JSONL, Parquet). Omit to generate data from --description.")
    p.add_argument("--description", default="", help="Task description — required if --data is omitted")
    p.add_argument("--task-name", default=None, help="Short task name (default: filename or 'custom-task')")
    p.add_argument("--model", default="gpt-4.1-mini", help="Base model for fine-tuning")
    p.add_argument("--work-dir", default="./auto_ft_run", help="Working directory for all artifacts")
    p.add_argument("--max-iterations", type=int, default=3, help="Max training iterations before stopping")
    p.add_argument("--max-budget", type=float, default=50, help="Max budget in USD")
    p.add_argument("--num-examples", type=int, default=200, help="Target examples for data generation")
    p.add_argument("--teacher", default=None, help="Teacher model for data generation (default: auto-detect best available)")
    p.add_argument("--schema-file", default=None, help="Schema/context file for domain-aware generation")
    p.add_argument("--min-quality", type=float, default=7.0, help="Min quality score for generated data")
    p.add_argument("--capacity", type=int, default=100, help="Deployment capacity for eval")
    p.add_argument("--tier", default="globalStandard",
                   help="Training tier(s) — single value or comma-separated list for round-robin. "
                        "'globalStandard' (default, works in all regions) — recommended. "
                        "'developerTier' (cheapest, spot — may be capacity-limited). "
                        "'standard' (region-restricted; absent in many regions). "
                        "Example: --tier globalStandard,developerTier distributes candidates across "
                        "both tiers so one capacity bottleneck doesn't block the whole iteration. "
                        "OSS models (qwen, llama, ministral, oss-20b) auto-override to globalStandard. "
                        "Tier is sent via extra_body to the SDK, body parameter to REST.")
    p.add_argument("--datagen-backend", default="auto",
                   choices=["auto", "local", "foundry-prompt", "foundry-file",
                            "foundry-agent", "foundry-traces"],
                   help="Where Phase 2 (GENERATE) pulls data from. "
                        "'auto' (default) infers from companion flags: "
                        "--datagen-file-id → foundry-file; "
                        "--datagen-agent-name + --datagen-hours → foundry-traces; "
                        "--datagen-agent-name alone → foundry-agent; "
                        "nothing → local. "
                        "Project endpoint alone is NOT enough — must pass a "
                        "datagen-* flag or explicit --datagen-backend foundry-prompt. "
                        "'local' = in-process teacher loop (no project endpoint required); "
                        "'foundry-*' = Foundry Data Generation API.")
    p.add_argument("--datagen-file-id", default=None, help="OpenAI file id for --datagen-backend foundry-file")
    p.add_argument("--datagen-agent-name", default=None, help="Agent name for --datagen-backend foundry-agent or foundry-traces")
    p.add_argument("--datagen-agent-version", default=None, help="Pin agent version for traces (recommended)")
    p.add_argument("--datagen-hours", type=int, default=None,
                   help="Hours of traces to pull (default: 24 when foundry-traces backend is selected, unset otherwise — presence of this flag is one of the inference signals for backend=foundry-traces)")
    p.add_argument("--datagen-recipe", default=None, choices=["qna", "tool-use", "traces"],
                   help="Override the Foundry datagen recipe (default: inferred from --datagen-backend — file/agent/prompt → qna, traces → traces). Use 'tool-use' when the file is an OpenAPI 3.0 spec.")
    p.add_argument("--datagen-scenario", default=None, choices=["sft", "eval"],
                   help="Override the Foundry datagen scenario (default: sft).")
    # Optional content-safety pre-screen between Phase 2 (GENERATE) and Phase 3 (PREPARE).
    # Use when you've seen "User data has failed data safety check" rejections.
    p.add_argument("--content-safety-prescreen", action="store_true",
                   help="Score generated data against Azure Content Safety and drop rows above --content-safety-threshold before PREPARE. Off by default. Requires --content-safety-endpoint + --content-safety-key (or AZURE_CONTENT_SAFETY_ENDPOINT / AZURE_CONTENT_SAFETY_KEY env vars).")
    p.add_argument("--content-safety-endpoint", default=None,
                   help="Azure Content Safety endpoint, e.g. https://<resource>.cognitiveservices.azure.com")
    p.add_argument("--content-safety-key", default=None,
                   help="Azure Content Safety API key")
    p.add_argument("--content-safety-threshold", type=int, default=2,
                   help="Severity threshold for the pre-screen (0=safe, 2=low, 4=medium, 6=high). Default 2 because Azure FT preprocessing rejects at low severity in practice.")
    add_connection_args(p)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    commands = {
        "analyze": cmd_analyze,
        "generate": cmd_generate,
        "foundry-generate": cmd_foundry_generate,
        "prepare": cmd_prepare,
        "baseline": cmd_baseline,
        "candidates": cmd_candidates,
        "execute": cmd_execute,
        "evaluate": cmd_evaluate,
        "review": cmd_review,
        "auto": cmd_auto,
    }

    commands[args.command](args)
