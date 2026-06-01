#!/usr/bin/env python3
"""Chunk a large source text and run parallel Foundry datagen jobs per chunk.

The Foundry Data Generation API's `SimpleQnA` recipe saturates at ~100-150
unique Q&A pairs per source file, because the teacher self-deduplicates
aggressively against the source content. Asking for more samples in a single
job doesn't yield more once the source is "covered".

Chunking the source into N pieces and running N parallel jobs multiplies
the effective output: 10 chunks × ~100 pairs each ≈ 1000 unique pairs.

Usage:
  python chunk_and_generate.py \\
      --source-text nist-cobol.txt \\
      --chunks 10 \\
      --teacher gpt-5.4 \\
      --recipe qna \\
      --scenario sft \\
      --max-samples-per-chunk 100 \\
      --concurrency 3 \\
      --out merged.jsonl

This uploads 10 chunked files, submits 10 datagen jobs (concurrency-capped),
and concatenates the results into one JSONL.

⚠️ TPM headroom matters
-----------------------
The Foundry datagen service reads the source chunk into the teacher's
context many times per generated Q&A pair, so each parallel job uses
significant TPM. A 100 KB chunk fed to a teacher with 500K TPM can
support roughly 1-2 parallel jobs; 2-3 parallel jobs against the same
teacher quickly exceeds the quota and the service fails with
"Too many Rate limit errors". To run more chunks in parallel you must
either bump the teacher's TPM or use smaller chunks.

Recommended starting points:
  - 200K TPM teacher → concurrency=1, chunks of ≤100KB
  - 500K TPM teacher → concurrency=2, chunks of ≤150KB
  - 1M+ TPM teacher → concurrency=3-5, chunks of ≤200KB

If parallel jobs fail with rate limits, lower --concurrency or shrink
--chunks (smaller chunks = less per-job TPM usage).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _chunk_text(text: str, n_chunks: int, overlap_chars: int = 1000) -> list[str]:
    """Split text into n_chunks roughly-equal pieces with small overlap.

    Overlap helps the teacher see context boundaries; otherwise mid-section
    cuts can produce confusing fragments.
    """
    if n_chunks <= 1:
        return [text]
    target = len(text) // n_chunks
    chunks = []
    pos = 0
    for i in range(n_chunks):
        start = max(0, pos - (overlap_chars if i > 0 else 0))
        end = min(len(text), pos + target + overlap_chars)
        if i == n_chunks - 1:
            end = len(text)
        # Snap to nearest paragraph boundary to avoid cutting mid-sentence
        if end < len(text):
            nl = text.rfind("\n\n", pos, end)
            if nl > pos:
                end = nl
        chunks.append(text[start:end])
        pos = end
    return chunks


def _upload_chunk(client, chunk_text: str, chunk_idx: int, name_prefix: str) -> str:
    """Upload one chunk as a user_data file. Returns file_id."""
    import io
    fname = f"{name_prefix}-chunk-{chunk_idx:02d}.txt"
    buf = io.BytesIO(chunk_text.encode("utf-8"))
    f = client.files.create(file=(fname, buf), purpose="user_data")
    # Wait briefly for processing
    for _ in range(30):
        f = client.files.retrieve(file_id=f.id)
        if f.status == "processed":
            return f.id
        time.sleep(2)
    raise RuntimeError(f"chunk {chunk_idx} did not process within 60s")


def _run_one_datagen(args, file_id: str, chunk_idx: int, out_dir: Path) -> Path | None:
    """Shell out to generate_dataset.py for one chunk. Returns the resulting JSONL path."""
    script = Path(__file__).resolve().parent / "generate_dataset.py"
    chunk_out_name = f"chunk-{chunk_idx:02d}"
    cmd = [
        sys.executable, str(script),
        "--source", "file",
        "--file-id", file_id,
        "--recipe", args.recipe,
        "--scenario", args.scenario,
        "--max-samples", str(args.max_samples_per_chunk),
        "--teacher", args.teacher,
        "--output-name", chunk_out_name,
        "--download",
    ]
    if args.project_endpoint:
        cmd += ["--project-endpoint", args.project_endpoint]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    # SECURITY: api-key flows via env, not CLI args (which are visible to `ps`).
    sub_env = os.environ.copy()
    if args.api_key:
        sub_env["AZURE_OPENAI_API_KEY"] = args.api_key

    # Run from out_dir so --download writes there
    print(f"  [chunk {chunk_idx}] submitting datagen (file_id={file_id[-12:]})...", flush=True)
    r = subprocess.run(
        cmd, cwd=out_dir,
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=1800,
        env=sub_env,
    )
    if r.returncode != 0:
        print(f"  [chunk {chunk_idx}] FAILED rc={r.returncode}: {(r.stderr or '')[-300:]}", flush=True)
        return None
    # Find the produced JSONL
    candidates = list(out_dir.glob(f"{chunk_out_name}*_dg.jsonl"))
    if not candidates:
        print(f"  [chunk {chunk_idx}] no _dg.jsonl produced. stdout tail: {r.stdout[-200:]}", flush=True)
        return None
    p = candidates[0]
    n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
    print(f"  [chunk {chunk_idx}] ✅ {n} rows → {p.name}", flush=True)
    return p


def chunk_and_generate(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import get_clients  # type: ignore

    source_text = Path(args.source_text).read_text(encoding="utf-8")
    print(f"Source: {len(source_text):,} chars → splitting into {args.chunks} chunks")

    chunks = _chunk_text(source_text, args.chunks, overlap_chars=args.overlap_chars)
    print(f"Chunk sizes: min={min(len(c) for c in chunks):,}  max={max(len(c) for c in chunks):,}  median={sorted(len(c) for c in chunks)[len(chunks)//2]:,}")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = Path(args.source_text).stem[:20]

    client, _ = get_clients(base_url=args.base_url, project_endpoint=args.project_endpoint, api_key=args.api_key)

    # Upload all chunks (sequential — files API is fast)
    print(f"\nUploading {len(chunks)} chunks to Foundry...")
    file_ids: list[str | None] = [None] * len(chunks)
    for i, ch in enumerate(chunks):
        try:
            file_ids[i] = _upload_chunk(client, ch, i, name_prefix)
            print(f"  chunk {i}: uploaded ({len(ch):,} chars → {file_ids[i]})")
        except Exception as e:
            print(f"  chunk {i}: upload FAILED: {e}")

    # Run datagen jobs in parallel (concurrency capped)
    print(f"\nRunning datagen on each chunk (concurrency={args.concurrency})...")
    produced: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(_run_one_datagen, args, fid, i, out_dir): i
            for i, fid in enumerate(file_ids) if fid is not None
        }
        for fut in as_completed(futures):
            p = fut.result()
            if p is not None:
                produced.append(p)

    # Concatenate
    out_path = Path(args.out)
    total = 0
    with out_path.open("w", encoding="utf-8") as out:
        for p in sorted(produced):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        total += 1
    print(f"\nMerged {len(produced)} chunk outputs → {out_path} ({total} total rows)")

    # Optional: delete chunk files from Foundry to free quota
    if args.cleanup_uploads:
        print(f"\nDeleting {sum(1 for f in file_ids if f)} uploaded chunk files...")
        for fid in file_ids:
            if fid:
                try:
                    client.files.delete(fid)
                except Exception as e:
                    print(f"  delete {fid} failed: {e}")

    return 0 if total > 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    p.add_argument("--source-text", required=True, type=Path, help="Path to plain-text source file (e.g. extracted PDF text).")
    p.add_argument("--chunks", type=int, required=True, help="Number of chunks to split the source into.")
    p.add_argument("--overlap-chars", type=int, default=1000, help="Characters of overlap between adjacent chunks (default 1000).")
    p.add_argument("--recipe", default="qna", choices=["qna", "tool-use"])
    p.add_argument("--scenario", default="sft", choices=["sft", "eval"])
    p.add_argument("--max-samples-per-chunk", type=int, default=100, help="Foundry caps each job at 1000; for SimpleQnA the practical ceiling per source is ~100-150. Default 100.")
    p.add_argument("--teacher", required=True)
    p.add_argument("--concurrency", type=int, default=3, help="Parallel datagen jobs (default 3 — higher risks Foundry quota errors).")
    p.add_argument("--out", required=True, type=Path, help="Merged JSONL output path.")
    p.add_argument("--cleanup-uploads", action="store_true", help="Delete the uploaded chunk files from Foundry after generation (frees user_data quota).")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--project-endpoint", default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"))
    p.add_argument("--api-key", default=os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    args = p.parse_args()

    if not (args.base_url or args.project_endpoint) or not args.api_key:
        print("ERROR: --base-url (or --project-endpoint) and --api-key required.", file=sys.stderr)
        return 2

    return chunk_and_generate(args)


if __name__ == "__main__":
    sys.exit(main())
