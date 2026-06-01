"""
common.py — Shared Azure AI Foundry authentication and client setup.

Supports three connection methods in order of preference:
1. /v1/ project endpoint (simplest, preferred)
2. Foundry SDK with DefaultAzureCredential (no API key needed, cloud-native)
3. Azure OpenAI endpoint (classic)

AAD tokens are auto-refreshed via azure.identity for long-running scripts
(monitor_training.py, generate_distillation_data.py, etc.).

Usage:
    from common import get_clients, upload_file

    # Method 1: Project /v1/ endpoint (preferred)
    clients = get_clients(base_url="https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1/",
                          api_key="KEY")

    # Method 2: Foundry SDK (DefaultAzureCredential — no API key needed)
    clients = get_clients(project_endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>")

    # Method 3: Azure OpenAI endpoint
    clients = get_clients(azure_endpoint="https://<resource>.openai.azure.com",
                          api_key="KEY")
"""
import argparse
import os
import sys


_AZURE_COGSERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


# ── SFT/DPO Training Cost Estimation ─────────────────────────────────────
# SFT/DPO training prices in USD per 1M trained tokens, **globalStandard tier**.
# "Trained tokens" follows Azure's billing convention:
#     trained_tokens = dataset_tokens × epochs
#
# Sources (verify against the live pricing page before making business decisions):
#   - https://azure.microsoft.com/pricing/details/cognitive-services/openai-service
#   - https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning-cost-management
#   - Skills/references/cost-management.md
#
# Values are from the Azure OpenAI pricing page for the specific model dates.
# Update both this dict and `Skills/references/cost-management.md` when prices
# change. RFT (o4-mini, gpt-5) is time-based, not token-based, so neither is
# listed here.
SFT_TRAINING_PRICE_PER_M_TOKENS = {
    # OpenAI models on Azure (globalStandard tier baseline, USD per 1M trained tokens)
    "gpt-4.1-nano":    1.50,   # gpt-4.1-nano-2025-04-14
    "gpt-4.1-mini":    5.00,   # gpt-4.1-mini-2025-04-14
    "gpt-4.1":        25.00,   # gpt-4.1-2025-04-14
    "gpt-4o-mini":     3.00,   # gpt-4o-mini-2024-07-18
    "gpt-4o":         25.00,   # gpt-4o-2024-08-06
    # OSS models — only globalStandard tier is supported (no developerTier or
    # regional standard). Prices in USD per 1M trained tokens.
    "ministral-3b":    1.00,   # Mistral Ministral 3B  ($0.001 per 1K tokens)
    "qwen3-32b":       3.20,   # Qwen3 32B             ($0.0032 per 1K tokens)
    "llama-3.3-70b":   4.50,   # Llama 3.3 70B         ($0.0045 per 1K tokens)
    "gpt-oss-20b":     3.60,   # GPT OSS 20B           ($0.0036 per 1K tokens)
    # Note: o4-mini and gpt-5 are RFT-only (no SFT or DPO support) and are
    # billed hourly, not per-token. They are intentionally NOT in this dict —
    # estimate_training_cost() returns None for them and callers should use
    # hourly RFT billing instead.
}

# Tier multipliers vs. globalStandard baseline.
#   - developerTier:  50% off globalStandard (per Azure pricing page)
#   - standard:       Regional baseline; globalStandard is 10–30% off regional,
#                     so regional ≈ +10% to +30% on top of globalStandard.
#                     Midpoint (+20%) used for point estimates; show as a range
#                     in user-facing docs (see cost-management.md).
SFT_TIER_MULTIPLIER = {
    "globalstandard":  1.00,   # baseline — Azure published prices
    "developertier":   0.50,   # 50% off globalStandard
    "standard":        1.20,   # regional/standard tier — ranges from 1.10 to 1.30
}

# URL printed alongside cost estimates so users can verify against live pricing.
AZURE_PRICING_URL = (
    "https://azure.microsoft.com/pricing/details/cognitive-services/openai-service"
)


def lookup_training_price(model_id):
    """Look up the globalStandard tier SFT price for a model.

    Strips fine-tune suffixes (e.g. ".ft-foo", ":ft-bar") and version stamps
    (e.g. "-2025-04-14") to find a base model match. Falls back to longest
    prefix match.

    Returns:
        (price_per_M_globalstandard, normalized_key) or (None, None) if unknown.
    """
    if not model_id:
        return None, None
    m = str(model_id).lower()
    # Strip fine-tune suffixes
    for sep in (".ft-", ":ft-", "-ft-"):
        if sep in m:
            m = m.split(sep, 1)[0]
    if m in SFT_TRAINING_PRICE_PER_M_TOKENS:
        return SFT_TRAINING_PRICE_PER_M_TOKENS[m], m
    # Longest-prefix match (e.g. "gpt-4.1-mini-2025-04-14" → "gpt-4.1-mini")
    matches = [(k, v) for k, v in SFT_TRAINING_PRICE_PER_M_TOKENS.items() if m.startswith(k)]
    if matches:
        key, price = max(matches, key=lambda kv: len(kv[0]))
        return price, key
    return None, None


# OSS models only support the globalStandard tier (no developerTier or
# regional/standard support). Used by estimate_training_cost() to enforce
# this constraint regardless of what tier the caller passes.
_OSS_MODELS_GLOBALSTANDARD_ONLY = frozenset({
    "ministral-3b", "qwen3-32b", "llama-3.3-70b", "gpt-oss-20b",
})


def estimate_training_cost(model_id, tier, trained_tokens):
    """Estimate SFT/DPO training cost in USD.

    Args:
        model_id: Base model ID, e.g. 'gpt-4.1-mini' or 'gpt-4.1-mini.ft-foo'
        tier: 'standard', 'globalStandard', or 'developerTier' (case-insensitive).
              Defaults to 'globalStandard' if not provided (matches Azure default).
              For OSS models, this is forced to 'globalStandard' (the only tier
              they support) regardless of what is passed.
        trained_tokens: Total trained tokens (Azure billing: dataset_tokens × epochs).

    Returns:
        dict with keys 'cost', 'price_per_M', 'tier_multiplier', 'matched_model',
        'effective_tier' on success, or None if pricing is unknown for the model
        or token count is zero/missing. 'effective_tier' is the tier actually
        used for the calculation — for OSS models this is always 'globalStandard'
        even if a different tier was requested.
    """
    base, matched = lookup_training_price(model_id)
    if base is None or not trained_tokens:
        return None
    # OSS models only support globalStandard — enforce it here so callers using
    # this function directly (notebooks, ad-hoc scripts) get correct estimates.
    # auto_finetune.py also enforces this at submission time via _resolve_tier(),
    # but estimate_training_cost is a public API and shouldn't trust the input.
    if matched in _OSS_MODELS_GLOBALSTANDARD_ONLY:
        effective_tier = "globalStandard"
    else:
        effective_tier = tier or "globalStandard"
    mult = SFT_TIER_MULTIPLIER.get(effective_tier.lower(), 1.0)
    return {
        "cost": trained_tokens / 1_000_000 * base * mult,
        "price_per_M": base * mult,
        "tier_multiplier": mult,
        "matched_model": matched,
        "effective_tier": effective_tier,
    }


class HelpOnErrorParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help when arguments are invalid.
    
    Standard ArgumentParser only prints a one-line usage summary on error,
    which isn't helpful for first-time users. This prints the full --help.
    """

    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def find_az_cli() -> str:
    """Locate the Azure CLI executable. Returns path string suitable for subprocess.

    Resolution order:
      1. AZ_CLI_PATH env var (explicit override)
      2. shutil.which("az") (PATH lookup)
      3. Common Windows install locations
      4. Fall back to "az" (assumes it's on PATH; subprocess will fail with a useful error)
    """
    import shutil
    explicit = os.environ.get("AZ_CLI_PATH")
    if explicit and os.path.exists(explicit):
        return explicit
    found = shutil.which("az")
    if found:
        return found
    for candidate in (
        r"C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
        r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
    ):
        if os.path.exists(candidate):
            return candidate
    return "az"


def _make_token_provider():
    """Create an auto-refreshing AAD token provider for long-running scripts.
    
    Returns a callable that the OpenAI SDK calls before each request to get
    a fresh token. Tokens are cached and refreshed ~5 min before expiry.
    """
    from azure.identity import DefaultAzureCredential
    credential = DefaultAzureCredential()

    def get_token():
        try:
            token = credential.get_token(_AZURE_COGSERVICES_SCOPE)
            return token.token
        except Exception as e:
            raise RuntimeError(
                f"Azure AD authentication failed: {e}\n"
                "Ensure you're logged in (az login) or have valid "
                "AZURE_CLIENT_ID/AZURE_TENANT_ID/AZURE_CLIENT_SECRET set."
            ) from e

    return get_token


def get_clients(base_url=None, azure_endpoint=None, project_endpoint=None, api_key=None):
    """Initialize and return OpenAI-compatible client.

    Tries in order:
    1. Project /v1/ endpoint with openai.OpenAI() (simplest, preferred)
    2. Foundry SDK with AIProjectClient.get_openai_client() (no API key needed)
    3. Azure OpenAI endpoint with openai.AzureOpenAI() (classic)

    When using DefaultAzureCredential (no API key), tokens are auto-refreshed
    so long-running scripts won't fail with 401 after ~60 min.

    Returns: (openai_client, method_name)
    """
    # Method 1: /v1/ project endpoint
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")

    if base_url:
        import openai
        if not api_key:
            try:
                token_provider = _make_token_provider()
                token_provider()  # verify it works
                # Use a custom httpx auth class that refreshes the token on each request
                import httpx

                class _AzureADAuth(httpx.Auth):
                    def __init__(self, provider):
                        self._provider = provider

                    def auth_flow(self, request):
                        request.headers["Authorization"] = f"Bearer {self._provider()}"
                        yield request

                client = openai.OpenAI(
                    base_url=base_url,
                    api_key="aad",  # required by SDK but overridden by auth
                    http_client=httpx.Client(auth=_AzureADAuth(token_provider)),
                )
                print(f"✅ Connected via /v1/ project endpoint (DefaultAzureCredential, auto-refresh)")
                return client, "project-v1-aad"
            except Exception as e:
                print(f"⚠️ No API key and DefaultAzureCredential failed: {e}")
        else:
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
            print(f"✅ Connected via /v1/ project endpoint")
            return client, "project-v1"

    # Method 2: Foundry SDK
    project_endpoint = project_endpoint or os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    if project_endpoint:
        try:
            from azure.ai.projects import AIProjectClient
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)
            openai_client = project_client.get_openai_client()
            print(f"✅ Connected via Foundry SDK")
            return openai_client, "foundry-sdk"
        except Exception as e:
            print(f"⚠️ Foundry SDK failed: {e}")

    # Method 3: Azure OpenAI endpoint
    azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        import openai
        if api_key:
            client = openai.AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version="2025-04-01-preview",
            )
            print(f"✅ Connected via Azure OpenAI endpoint")
            return client, "azure-openai"
        else:
            # No API key — use DefaultAzureCredential with auto-refresh
            try:
                token_provider = _make_token_provider()
                token_provider()  # verify it works
                client = openai.AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version="2025-04-01-preview",
                )
                print(f"✅ Connected via Azure OpenAI endpoint (DefaultAzureCredential, auto-refresh)")
                return client, "azure-openai-aad"
            except Exception as e:
                print(f"⚠️ DefaultAzureCredential failed for Azure endpoint: {e}")

    print("❌ No valid connection method. Set one of:")
    print("   OPENAI_BASE_URL (preferred)")
    print("   AZURE_AI_PROJECT_ENDPOINT (Foundry SDK)")
    print("   AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY")
    sys.exit(1)
    return None, "none"  # unreachable, satisfies static analysis


def upload_file(openai_client, filepath: str, purpose: str = "fine-tune") -> str:
    """Upload a file to Azure AI Foundry and wait for processing."""
    print(f"📤 Uploading {filepath}...")
    with open(filepath, "rb") as f:
        file_obj = openai_client.files.create(file=f, purpose=purpose)
    print(f"   File ID: {file_obj.id}")
    print(f"   Waiting for processing...")
    openai_client.files.wait_for_processing(file_obj.id)
    print(f"   ✅ File ready")
    return file_obj.id
