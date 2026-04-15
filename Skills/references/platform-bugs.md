# Azure AI Foundry Fine-Tuning — Platform Bugs & Issues

Bugs and inconsistencies discovered during end-to-end fine-tuning testing.
Updated as we find new issues.

---

## 1. Model catalog `fine_tune` capability flag wrong for OSS models
- **Severity**: Medium
- **Where**: `GET /openai/models?api-version=2025-04-01-preview`
- **Expected**: Ministral-3B, Qwen-32B, Llama-3.3-70B-Instruct, gpt-oss-20b should show `capabilities.fine_tune = true`
- **Actual**: All OSS models return `fine_tune = false` despite being listed as FT-supported in [official docs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning#supported-models)
- **Impact**: Any automation that checks capability flags before submitting FT jobs will skip OSS models
- **Workaround**: Hardcode known-supported model list instead of relying on API flags

## 2. Project endpoint SDK broken for file uploads and job management
- **Severity**: High
- **Where**: OpenAI Python SDK with project `/v1/` endpoint
- **Expected**: `client.files.create()` and `client.fine_tuning.jobs.retrieve()` should work
- **Actual**: Throws "API version not supported" for all tested versions
- **Impact**: Cannot use SDK for core FT operations through project endpoint
- **Workaround**: Use REST API directly with non-project endpoint (`/openai/`) and `2025-04-01-preview`

## 3. FT deployment "DeploymentNotReady" persists after ARM shows Succeeded
- **Severity**: High
- **Where**: Data plane `/chat/completions` after ARM `PUT deployments/` returns 200/201 with `provisioningState: Succeeded`
- **Expected**: Deployment should be callable once ARM reports success
- **Actual**: Returns 400 `DeploymentNotReady` indefinitely; sometimes never resolves
- **Impact**: Blocks evaluation pipeline; no programmatic way to detect when deployment is truly ready
- **Workaround**: Delete and recreate the deployment. Wait ~5 minutes after recreate for warmup.

## 4. Content safety false positive on entity extraction training data
- **Severity**: Medium
- **Where**: Fine-tuning job validation / model deployment
- **Expected**: Entity extraction data (names, dates, locations from business documents) should pass content safety
- **Actual**: Model rejected for "Hate/Fairness" — triggered by PII-dense document types (medical records, legal contracts, resumes)
- **Impact**: Training succeeded but the model was blocked at deployment
- **Workaround**: Remove medical, legal, and resume document types from training data. Resubmit.

## 5. FT deployments severely rate-limited (1K TPM / 1 RPM)
- **Severity**: Medium
- **Where**: Fine-tuned model deployments (Standard, capacity=1)
- **Expected**: Reasonable throughput for evaluation (at least 6-10 RPM)
- **Actual**: Effectively 1 request per minute; 429s with long retry-after headers
- **Impact**: Evaluating 10 samples takes ~10 minutes per model. Production use requires significant capacity planning.
- **Workaround**: Use exponential backoff (5-60s delays), limit eval to 10-15 samples for FT models

## 6. SDK generic evaluators show no differentiation
- **Severity**: Low (by design, but misleading)
- **Where**: Azure AI Evaluation SDK built-in evaluators (Coherence, Fluency, TaskAdherence)
- **Expected**: Should show some signal between base and FT models
- **Actual**: All models score identically (5/5 across the board) on style/alignment tasks
- **Impact**: Users who rely only on generic evaluators will conclude FT had no effect
- **Workaround**: Always use custom graders (PythonGrader, ScoreModelGrader, StringCheckGrader) for task-specific evaluation

---

## Non-Bug Gotchas

These aren't bugs — they're common stumbling blocks when using Azure AI Foundry tools.

## 7. `AZURE_FOUNDRY_API_KEY` env var lost between shell sessions
- **Where**: Data Designer CLI + any tool using env vars
- **Expected**: Env var persists across terminal sessions (or is documented as session-scoped)
- **Actual**: Must be re-set in every new PowerShell/terminal session
- **Impact**: DD jobs fail silently with auth errors if env var not set
- **Workaround**: Set `$env:AZURE_FOUNDRY_API_KEY` at the start of every shell session

## 8. Data Designer CLI arg parsing
- **Where**: `data-designer create` CLI command
- **Expected**: `--config <path>` flag (consistent with most CLIs)
- **Actual**: Positional argument: `data-designer create <config.py>` — no `--config` flag
- **Impact**: First-time users get confusing errors
- **Workaround**: Use positional arg syntax

## 9. OSS FT requires undocumented `trainingType: "globalStandard"` field
- **Severity**: High
- **Where**: `POST /openai/fine_tuning/jobs` for OSS models (Ministral-3B, gpt-oss-20b, Llama-3.3-70B, Qwen-32B)
- **Expected**: Same API schema as OpenAI models, or clear error message about required field
- **Actual**: Returns "does not support fine-tuning with Standard TrainingType" — must add `"trainingType": "globalStandard"` to the request body
- **Impact**: All OSS FT jobs fail on first attempt without this field. Not documented in the REST API reference.
- **Workaround**: Add `"trainingType": "globalStandard"` to the JSON body for any non-OpenAI model

## 10. OSS FT models may fail to deploy with InternalServerError
- **Severity**: High
- **Where**: ARM `PUT deployments/` for fine-tuned OSS models
- **Expected**: Deployment creation succeeds like OpenAI FT models
- **Actual**: InternalServerError on some OSS FT model deployments. Consistently affects gpt-oss-20b FT models. Ministral-3B FT works when using correct format.
- **Impact**: Models train successfully but cannot be deployed for inference
- **Workaround**: 
  - **Ministral-3B FT**: Use `format: "Mistral AI"`, `version: "1"`, `sku: "GlobalStandard"` (NOT `format: "OpenAI"`)
  - **gpt-oss-20b FT**: Use `format: "Microsoft"`, `version: "1"`, `sku: "GlobalStandard"` (NOT `format: "OpenAI"`)
  - Try capacity=100 if capacity=1 fails

## 11. OSS FT inference: "Failed to load LoRA" / TooManyRequests on weight loading
- **Severity**: High
- **Where**: Chat completions endpoint for deployed OSS FT models
- **Expected**: Model responds to inference requests after deployment succeeds
- **Actual**: HTTP 500 with `"Failed to get finetune weights path: Failed to get token for object_store: HttpResponse(TooManyRequests)"`. Affects both Ministral-3B FT and gpt-oss-20b FT intermittently.
- **Impact**: 90%+ of requests fail during bad periods, making evaluation unreliable. May resolve after 5-10 minutes of retry.
- **Workaround**: 
  - Deploy with capacity ≥ 100 (not 1) — reduces frequency
  - Use aggressive retry with exponential backoff (8+ retries)
  - Wait longer after deployment before first inference (2+ minutes)
  - Some requests will eventually succeed — expect only 10-30% success rate during bad periods

---

## 12. Wrong resource endpoint (common user error)
- **Severity**: N/A (not a platform bug)
- **Root cause**: Jobs were submitted to a secondary resource endpoint instead of the Foundry-connected primary resource. The wrong resource had different compute/config and jobs failed mid-training.
- **Lesson learned**: Always verify the OAI endpoint matches the resource connected to your Foundry project. Jobs submitted to the wrong resource will succeed via API but won't appear in the Foundry portal or telemetry. The two resources have completely separate job lists, file stores, and deployments.
- **How we discovered it**: No jobs appeared in the portal for several days. Comparing the OAI endpoint vs the Foundry endpoint revealed they return different job sets with no overlapping IDs. The correct resource had all prior successful experiments.
- **Symptoms that should have alerted us earlier**:
  - Portal showed no recent jobs despite successful API submissions
  - Some jobs "needed retries" — but had actually succeeded on the correct resource already
  - OSS jobs succeeded because they were submitted via REST to the correct endpoint path

---
