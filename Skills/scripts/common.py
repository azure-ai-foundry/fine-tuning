"""
common.py — Shared Azure AI Foundry authentication and client setup.

Adapted from foundry-ft agent's common.py, with our enhancements:
- Supports /v1/ project endpoint (preferred) and Azure endpoint (fallback)
- REST API fallback for OSS models
- No .env dependency — works with az CLI tokens directly

Usage:
    from common import get_clients, upload_file

    # Method 1: Project /v1/ endpoint (preferred)
    clients = get_clients(base_url="https://<resource>.services.ai.azure.com/api/projects/<project>/openai/v1/",
                          api_key="KEY")

    # Method 2: Foundry SDK
    clients = get_clients(project_endpoint="https://<resource>.services.ai.azure.com/api/projects/<project>")

    # Method 3: Azure OpenAI endpoint
    clients = get_clients(azure_endpoint="https://<resource>.openai.azure.com",
                          api_key="KEY")
"""
import os
import sys


def get_clients(base_url=None, azure_endpoint=None, project_endpoint=None, api_key=None):
    """Initialize and return OpenAI-compatible client.

    Tries in order:
    1. Project /v1/ endpoint with openai.OpenAI() (simplest, preferred)
    2. Foundry SDK with AIProjectClient.get_openai_client() (no API key needed)
    3. Azure OpenAI endpoint with openai.AzureOpenAI() (classic)

    Returns: (openai_client, method_name)
    """
    # Method 1: /v1/ project endpoint
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")

    if base_url:
        import openai
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
    if azure_endpoint and api_key:
        import openai
        client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2025-04-01-preview",
        )
        print(f"✅ Connected via Azure OpenAI endpoint")
        return client, "azure-openai"

    print("❌ No valid connection method. Set one of:")
    print("   OPENAI_BASE_URL (preferred)")
    print("   AZURE_AI_PROJECT_ENDPOINT (Foundry SDK)")
    print("   AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY")
    sys.exit(1)


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


def get_env(key: str, required: bool = True) -> str:
    """Get environment variable, exit if required and missing."""
    value = os.environ.get(key)
    if required and not value:
        print(f"❌ Environment variable {key} not set.")
        sys.exit(1)
    return value or ""
