#!/usr/bin/env python3
"""Validate RFT (Reinforcement Fine-Tuning) JSONL files for Azure AI Foundry.

Adapted from foundry-ft agent with critical additions from our platform gotchas:
- Grader escaping warnings (\\n, \\t must be \\\\n, \\\\t in JSON strings)
- Content moderation risk detection ("chain of thought" triggers RAI filter)
- Reference answer diversity check
"""
import json
import sys
import re


RISKY_PHRASES = [
    "chain of thought", "step by step reasoning", "let me think",
    "think carefully", "reason through",
]


def validate_rft(filepath: str) -> None:
    errors = []
    warnings = []
    total = 0
    ref_answers = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total += 1
            raw_line = line
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON — {e}")
                continue

            if "messages" not in record:
                errors.append(f"Line {line_num}: Missing 'messages' field")
            else:
                msgs = record["messages"]
                if not isinstance(msgs, list) or len(msgs) == 0:
                    errors.append(f"Line {line_num}: 'messages' must be a non-empty array")
                elif not any(m.get("role") == "user" for m in msgs):
                    errors.append(f"Line {line_num}: 'messages' has no 'user' message")

            if "reference_answer" not in record:
                errors.append(f"Line {line_num}: Missing 'reference_answer' field")
            else:
                ref = str(record["reference_answer"]).strip()
                if not ref:
                    errors.append(f"Line {line_num}: 'reference_answer' is empty")
                else:
                    ref_answers.append(ref)

                # Check for grader escaping issues (CRITICAL platform gotcha)
                if "\\n" in ref and "\\\\n" not in raw_line:
                    warnings.append(
                        f"Line {line_num}: reference_answer contains literal newlines — "
                        "grader may fail. Use \\\\n in the JSON string."
                    )

            # Content moderation risk
            all_text = json.dumps(record).lower()
            for phrase in RISKY_PHRASES:
                if phrase in all_text:
                    warnings.append(
                        f"Line {line_num}: Contains '{phrase}' — may trigger Azure content moderation filter."
                    )
                    break

    # Diversity check
    if ref_answers:
        unique_answers = set(ref_answers)
        if len(unique_answers) == 1:
            warnings.append(
                f"All reference_answers are identical ('{list(unique_answers)[0][:50]}...') — "
                "grader may not learn effectively"
            )
        avg_len = sum(len(a) for a in ref_answers) / len(ref_answers)
        if avg_len > 500:
            warnings.append(
                f"Average reference_answer length is {avg_len:.0f} chars — "
                "consider using a model_grader instead of string_check"
            )

    print(f"\n{'='*60}")
    print(f"RFT Validation Report: {filepath}")
    print(f"{'='*60}")
    print(f"Total records: {total}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if ref_answers:
        unique_answers = set(ref_answers)
        print(f"Unique reference answers: {len(unique_answers)}/{len(ref_answers)}")

    if errors:
        print(f"\n❌ ERRORS (must fix):")
        for e in errors[:20]:
            print(f"  • {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")

    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for w in warnings[:10]:
            print(f"  • {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")

    # RFT-specific guidance
    if total > 0:
        print(f"\n💡 RFT tips:")
        print(f"  • Ensure your training grader matches your eval grader (alignment gotcha)")
        print(f"  • Start with reasoning_effort='medium', pass_rate_threshold=0.5")
        print(f"  • RFT only works with o3 and o4-mini — not GPT or OSS models")

    if not errors:
        print(f"\n✅ Data is valid for RFT fine-tuning!")
    else:
        print(f"\n❌ Fix {len(errors)} error(s) before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_rft.py <path-to-jsonl>")
        sys.exit(1)
    validate_rft(sys.argv[1])
