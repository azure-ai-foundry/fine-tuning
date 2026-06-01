# /// script
# dependencies = [
#   "requests",
# ]
# ///
"""
content_safety_check.py — Pre-screen an SFT JSONL file against Azure Content Safety.

When an Azure fine-tune job fails with "User data has failed data safety check"
the service doesn't tell you which rows tripped the classifier — only that some
did. This script scores each row offline so you can:

  - Identify which row(s) caused the failure
  - Decide whether to drop the offending rows, rewrite them, or relax the
    content filter on the deployment
  - Confirm a regenerated dataset is safe before re-uploading

Usage:

  python scripts/content_safety_check.py \
      --jsonl my_training_data.jsonl \
      --endpoint https://<resource>.cognitiveservices.azure.com \
      --api-key $env:AZURE_CONTENT_SAFETY_KEY

Optional:
  --threshold 4      # severity to flag (0=safe, 2=low, 4=medium, 6=high)
  --drop-out clean.jsonl   # write a copy with flagged rows removed
  --report report.json     # detailed per-row scores

If --endpoint and --api-key are omitted, falls back to
AZURE_CONTENT_SAFETY_ENDPOINT / AZURE_CONTENT_SAFETY_KEY env vars.

The Azure Content Safety API used here is:
  POST {endpoint}/contentsafety/text:analyze?api-version=2024-09-01

The check scores the concatenated user + assistant content from each row across
four categories (Hate, Sexual, SelfHarm, Violence). System prompts, tool
messages and tool_calls are excluded — they rarely trip the classifier and add
needless cost.

Exit code: 0 if no rows are flagged; 1 otherwise. Useful as a pre-submit gate
in scripts.
"""

import argparse
import json
import os
import sys

import requests


SEVERITY_LABELS = {0: "safe", 2: "low", 4: "medium", 6: "high"}


def screen_row(row: dict, endpoint: str, api_key: str, threshold: int):
    """Score a single SFT row. Returns dict with safe/reason/scores."""
    chunks = []
    for m in row.get("messages", []):
        if m.get("role") in ("user", "assistant"):
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                chunks.append(c)
    text = "\n".join(chunks).strip()
    if not text:
        return {"safe": True, "reason": "no scoreable content", "scores": {}}
    text = text[:10000]

    url = endpoint.rstrip("/") + "/contentsafety/text:analyze?api-version=2024-09-01"
    headers = {"Ocp-Apim-Subscription-Key": api_key, "Content-Type": "application/json"}
    body = {
        "text": text,
        "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
        "outputType": "FourSeverityLevels",
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=(10, 30))
    except Exception as e:
        return {"safe": None, "reason": f"API exception: {e}", "scores": {}}

    if resp.status_code != 200:
        return {"safe": None, "reason": f"API error {resp.status_code}: {resp.text[:200]}", "scores": {}}

    data = resp.json()
    scores = {c.get("category", "?"): c.get("severity", 0) for c in data.get("categoriesAnalysis", [])}
    bad = {c: s for c, s in scores.items() if s >= threshold}
    if bad:
        return {"safe": False, "reason": f"flagged {bad}", "scores": scores}
    return {"safe": True, "reason": f"max severity {max(scores.values()) if scores else 0}", "scores": scores}


def main():
    p = argparse.ArgumentParser(description=__doc__.strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", required=True, help="Path to SFT JSONL file to screen")
    p.add_argument("--endpoint", default=os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT"),
                   help="Azure Content Safety endpoint, e.g. https://<resource>.cognitiveservices.azure.com")
    p.add_argument("--api-key", default=os.environ.get("AZURE_CONTENT_SAFETY_KEY"),
                   help="Azure Content Safety API key (or set AZURE_CONTENT_SAFETY_KEY)")
    p.add_argument("--threshold", type=int, default=4,
                   help="Severity threshold to flag (0=safe, 2=low, 4=medium, 6=high). Default 4.")
    p.add_argument("--drop-out", default=None,
                   help="Path to write a JSONL containing only the rows that PASSED")
    p.add_argument("--report", default=None,
                   help="Path to write a detailed per-row JSON report")
    args = p.parse_args()

    if not args.endpoint or not args.api_key:
        p.error("--endpoint and --api-key required (or set AZURE_CONTENT_SAFETY_* env vars)")

    rows = []
    with open(args.jsonl, encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((ln_no, json.loads(line)))
            except json.JSONDecodeError as e:
                print(f"  line {ln_no}: malformed JSON, skipping ({e})", file=sys.stderr)

    print(f"Scoring {len(rows)} rows against Azure Content Safety...")
    print(f"  endpoint: {args.endpoint}")
    print(f"  threshold: {args.threshold} ({SEVERITY_LABELS.get(args.threshold, '?')})\n")

    flagged = []
    api_errors = []
    safe_rows = []
    report = []
    for i, (ln_no, row) in enumerate(rows):
        verdict = screen_row(row, args.endpoint, args.api_key, args.threshold)
        entry = {"line": ln_no, **verdict}
        report.append(entry)
        if verdict["safe"] is False:
            flagged.append((ln_no, verdict))
            print(f"  ❌ line {ln_no}: {verdict['reason']}")
        elif verdict["safe"] is None:
            api_errors.append((ln_no, verdict))
            print(f"  ⚠️  line {ln_no}: {verdict['reason']}")
        else:
            safe_rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  ... scored {i+1}/{len(rows)} ({len(flagged)} flagged, {len(api_errors)} errors)")

    print(f"\nSummary:")
    print(f"  passed: {len(safe_rows)}")
    print(f"  flagged: {len(flagged)}")
    print(f"  api errors: {len(api_errors)}")

    if flagged:
        print(f"\nFlagged rows (per-category severities >= {args.threshold}):")
        for ln_no, v in flagged:
            print(f"  line {ln_no}: {v['scores']}")

    if args.drop_out:
        with open(args.drop_out, "w", encoding="utf-8") as fh:
            for row in safe_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(safe_rows)} passing rows to {args.drop_out}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({
                "threshold": args.threshold,
                "total": len(rows),
                "passed": len(safe_rows),
                "flagged": len(flagged),
                "api_errors": len(api_errors),
                "results": report,
            }, fh, indent=2)
        print(f"Wrote detailed report to {args.report}")

    sys.exit(0 if not flagged and not api_errors else 1)


if __name__ == "__main__":
    main()
