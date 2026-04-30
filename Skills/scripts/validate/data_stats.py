#!/usr/bin/env python3
"""Compute dataset statistics for any fine-tuning JSONL file.

Adapted from foundry-ft agent. Auto-detects SFT/DPO/RFT format and reports
token estimates, role distribution, and rough cost estimates.
"""
import json
import os
import sys
from collections import Counter

# Fix Windows console encoding (cp1252 can't handle Unicode arrows/emoji)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Reuse the centralized SFT pricing table from common.py so cost estimates
# stay consistent across data_stats.py, auto_finetune.py, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from common import (
        SFT_TRAINING_PRICE_PER_M_TOKENS,
        SFT_TIER_MULTIPLIER,
        AZURE_PRICING_URL,
    )
except ImportError:
    # Fallback — keep estimates working even if common.py isn't importable
    # (e.g. when this script is copied standalone). Values match common.py.
    SFT_TRAINING_PRICE_PER_M_TOKENS = {
        "gpt-4.1-nano":   1.50,
        "gpt-4.1-mini":   5.00,
        "gpt-4.1":       25.00,
        "gpt-4o-mini":    3.00,
        "gpt-4o":        25.00,
        "ministral-3b":   1.00,
        "qwen3-32b":      3.20,
        "llama-3.3-70b":  4.50,
        "gpt-oss-20b":    3.60,
    }
    SFT_TIER_MULTIPLIER = {"globalstandard": 1.00, "developertier": 0.50, "standard": 1.20}
    AZURE_PRICING_URL = (
        "https://azure.microsoft.com/pricing/details/cognitive-services/openai-service"
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def extract_text(record: dict) -> str:
    """Extract all text content from a record regardless of format."""
    texts = []
    if "messages" in record:
        for msg in record["messages"]:
            if "content" in msg and msg["content"]:
                texts.append(str(msg["content"]))
    if "input" in record and "messages" in record["input"]:
        for msg in record["input"]["messages"]:
            if "content" in msg and msg["content"]:
                texts.append(str(msg["content"]))
    for field in ["preferred_output", "non_preferred_output"]:
        if field in record:
            for msg in record[field]:
                if "content" in msg and msg["content"]:
                    texts.append(str(msg["content"]))
    # Include any extra fields beyond messages/input/preferred_output/non_preferred_output
    known_structural = {"messages", "input", "preferred_output", "non_preferred_output"}
    for field in record:
        if field not in known_structural and isinstance(record[field], (str, int, float)):
            texts.append(str(record[field]))
    return " ".join(texts)


def data_stats(filepath: str) -> None:
    records = []
    format_type = "unknown"
    parse_errors = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1

    if not records:
        print(f"No valid records found in {filepath}")
        sys.exit(1)

    # Detect format
    first = records[0]
    if "input" in first and "preferred_output" in first:
        format_type = "DPO"
    elif "messages" in first:
        msgs = first["messages"]
        extra_fields = set(first.keys()) - {"messages"}
        last_role = msgs[-1].get("role") if isinstance(msgs, list) and msgs else None
        if extra_fields and last_role == "user":
            format_type = "RFT"
        else:
            format_type = "SFT"

    # Compute stats
    token_counts = [estimate_tokens(extract_text(r)) for r in records]
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(records)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)

    print(f"\n{'='*60}")
    print(f"Dataset Statistics: {filepath}")
    print(f"{'='*60}")
    print(f"Format:           {format_type}")
    print(f"Total records:    {len(records)}")
    print(f"Parse errors:     {parse_errors}")
    print(f"")
    print(f"Token Estimates (approx):")
    print(f"  Total:          {total_tokens:,}")
    print(f"  Average/record: {avg_tokens:,.0f}")
    print(f"  Min:            {min_tokens:,}")
    print(f"  Max:            {max_tokens:,}")

    if format_type == "SFT":
        role_counts = Counter()
        for r in records:
            for msg in r.get("messages", []):
                role_counts[msg.get("role", "unknown")] += 1
        print(f"\nRole Distribution:")
        for role, count in role_counts.most_common():
            print(f"  {role}: {count}")

        has_system = sum(1 for r in records if any(m.get("role") == "system" for m in r.get("messages", [])))
        print(f"\nRecords with system message: {has_system}/{len(records)}")

    elif format_type == "DPO":
        pref_lens = []
        non_pref_lens = []
        for r in records:
            pref_text = " ".join(m.get("content", "") for m in r.get("preferred_output", []))
            non_pref_text = " ".join(m.get("content", "") for m in r.get("non_preferred_output", []))
            pref_lens.append(estimate_tokens(pref_text))
            non_pref_lens.append(estimate_tokens(non_pref_text))
        print(f"\nPreferred output avg tokens:     {sum(pref_lens)/len(pref_lens):,.0f}")
        print(f"Non-preferred output avg tokens: {sum(non_pref_lens)/len(non_pref_lens):,.0f}")

    elif format_type == "RFT":
        grader_field_counts = Counter()
        grader_values = []
        for r in records:
            extra = set(r.keys()) - {"messages"}
            grader_field_counts.update(extra)
            for field in sorted(extra):
                grader_values.append(str(r[field]))
        unique = len(set(grader_values))
        avg_val_len = sum(len(v) for v in grader_values) / len(grader_values) if grader_values else 0
        print(f"\nGrader fields found:")
        for field, count in grader_field_counts.most_common():
            print(f"  • '{field}' — in {count}/{len(records)} records")
        print(f"Unique grader values: {unique}/{len(grader_values)}")
        print(f"Avg grader value length: {avg_val_len:.0f} chars")

    # Cost estimates per model family (per epoch, globalStandard tier baseline).
    # See Skills/references/cost-management.md and common.py for the source table.
    print(f"\n💰 Cost Estimates per Epoch (globalStandard tier baseline):")
    # Display in a stable, human-friendly order
    display_order = [
        "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
        "gpt-4o-mini", "gpt-4o",
        "ministral-3b", "qwen3-32b", "llama-3.3-70b", "gpt-oss-20b",
    ]
    # OSS models only support globalStandard — skip the developerTier column for them.
    oss_models = {"ministral-3b", "qwen3-32b", "llama-3.3-70b", "gpt-oss-20b"}
    for model in display_order:
        cost_per_1m = SFT_TRAINING_PRICE_PER_M_TOKENS.get(model)
        if cost_per_1m is None:
            continue
        per_epoch = (total_tokens / 1_000_000) * cost_per_1m
        if model in oss_models:
            # OSS: globalStandard only (no dev tier discount available)
            print(
                f"  {model:<14s}: ~${per_epoch:>7.2f}/epoch  "
                f"(2 epochs ~${per_epoch * 2:>7.2f}; globalStandard only)"
            )
        else:
            dev_tier_mult = SFT_TIER_MULTIPLIER.get("developertier", 0.5)
            dev_tier_per_epoch = per_epoch * dev_tier_mult
            print(
                f"  {model:<14s}: ~${per_epoch:>7.2f}/epoch  "
                f"(2 epochs ~${per_epoch * 2:>7.2f}; "
                f"developerTier 2 epochs ~${dev_tier_per_epoch * 2:>7.2f})"
            )
    print(f"  ⚠️  Estimates are illustrative — verify on the Azure pricing page:")
    print(f"      {AZURE_PRICING_URL}")

    # Dataset size guidance
    print(f"\n📊 Dataset size guidance:")
    if len(records) < 50:
        print(f"  ⚠️ Very small dataset ({len(records)} records). May only learn format, not domain knowledge.")
    elif len(records) < 200:
        print(f"  ⚠️ Small dataset. Good for initial experiments — evaluate results and add more data if needed.")
    elif len(records) <= 500:
        print(f"  ✅ Sweet spot for getting started (200-500). Evaluate results to decide if you need more.")
    elif len(records) <= 2000:
        print(f"  ✅ Good dataset size. Watch for diminishing returns — check if quality beats quantity.")
    else:
        print(f"  ⚠️ Large dataset ({len(records):,}). Larger isn't always better — especially for OSS models where 335-500 examples outperformed 4K.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_stats.py <path-to-jsonl>")
        sys.exit(1)
    data_stats(sys.argv[1])
