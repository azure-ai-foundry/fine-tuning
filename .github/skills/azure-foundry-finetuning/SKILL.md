---
name: azure-foundry-finetuning-py
description: Submit fine-tuning jobs on Azure AI Foundry.
compatibility: Requires uv python package
license: MIT
metadata:
  author: Microsoft
  version: "1.0.0"
  package: azure-ai-projects
---

# Azure AI Foundry Fine-Tuning Py

## When to use this

This skill helps users submit supervised fine-tuning (SFT) jobs on Azure AI Foundry. Confirm with the user which arguments are used for the fine-tuning job before submission.

## Prerequisite enforcement

Before running any script in this skill, you must verify `uv` is available:

`command -v uv >/dev/null 2>&1 || pip install uv`

If `uv` is still unavailable after installation, stop and report the prerequisite failure. Do not run this skill's scripts without `uv`.

## Available scripts

Use --help flag to see the usage of each script, e.g. `uv run scripts/submit_sft.py --help`.

- **`scripts/submit_sft.py`** — Submits SFT fine-tuning jobs to Azure AI Foundry. After successfully submitting the job, ask user if they want to monitor the job, and if yes, call `monitor_ft_job.py` with the returned job ID.
- **`scripts/monitor_ft_job.py`** — Monitors a fine-tuning job by job ID. Stream the output of monitoring to the user until the job is completed, and report the final status of the job.