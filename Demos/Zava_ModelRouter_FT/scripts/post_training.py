"""Post-training helpers for the Zava Model Router fine-tuning demo.

Two responsibilities:

1. **Results rendering** — ``render_training_summary`` pivots the wide
   one-row ``results.csv`` Azure returns into a compact 3-row per-model
   summary (Base / Fine-Tuned / Costliest x Cost & Quality) and displays
   it as a minimal HTML table inside a notebook.

2. **Deployment** — ``deploy_finetuned_router`` deploys a fine-tuned
   Model Router via the Azure Management REST API. It is idempotent: if
   a deployment with the same name already exists, it is deleted first
   (re-PUTting an existing FT deployment fails with
   ``ModelUpgradeNotSupported`` because Azure treats it as an in-place
   model upgrade). The new deployment is created with
   ``versionUpgradeOption=NoAutoUpgrade``.
"""

from __future__ import annotations

import time
from urllib.parse import urlparse

import requests


# ------------------------------------------------------------------------------
# Results rendering
# ------------------------------------------------------------------------------

# Wide one-row CSV emitted by Azure: BaseModelRouter_(Cost|Quality),
# FineTunedModelRouter_(Cost|Quality), CostliestModel_(Cost|Quality).
# Map row label -> (cost_column, quality_column) for a clean per-model pivot.
_RESULT_ROWS = {
    "Base Model Router": ("BaseModelRouter_Cost",     "BaseModelRouter_Quality"),
    "Fine-Tuned Router": ("FineTunedModelRouter_Cost", "FineTunedModelRouter_Quality"),
    "Costliest Model":   ("CostliestModel_Cost",       "CostliestModel_Quality"),
}


def render_training_summary(csv_path) -> None:
    """Pivot Azure's wide results.csv into a per-model summary and display it.

    The CSV ships as a single row with cost+quality columns per model. We
    reshape it into 3 rows x (Cost USD, Quality) so the comparison reads
    naturally, then render a minimal white-text HTML table with a delta
    footer (cost % and quality vs. base). Falls back to ``display(df)`` if
    the expected columns aren't present.
    """
    import pandas as pd
    from IPython.display import HTML, display

    df = pd.read_csv(csv_path)
    needed = {c for cols in _RESULT_ROWS.values() for c in cols}
    if not needed.issubset(df.columns) or len(df) < 1:
        print(f"results.csv schema unrecognized — falling back to raw view "
              f"({len(df)} rows x {len(df.columns)} cols).")
        display(df)
        return

    row = df.iloc[-1]
    data = [(label, float(row[c]), float(row[q])) for label, (c, q) in _RESULT_ROWS.items()]
    base_cost, ft_cost = data[0][1], data[1][1]
    base_q,    ft_q    = data[0][2], data[1][2]
    cost_delta_pct = (ft_cost - base_cost) / base_cost * 100.0 if base_cost else 0.0
    quality_delta  = ft_q - base_q

    cell_css = "padding:6px 14px;border:1px solid #fff"
    body_rows = "".join(
        f'<tr>'
        f'<td style="{cell_css};text-align:left">{label}</td>'
        f'<td style="{cell_css};text-align:right">${cost:.4f}</td>'
        f'<td style="{cell_css};text-align:right">{quality:.2f}</td>'
        f'</tr>'
        for label, cost, quality in data
    )

    cost_sign = "\u2193" if cost_delta_pct < 0 else "\u2191"  # down / up arrow
    q_sign    = "+" if quality_delta >= 0 else ""
    footer = (
        f"Fine-Tuned vs Base: {cost_sign} cost {abs(cost_delta_pct):.1f}% "
        f"&middot; {q_sign}{quality_delta:.1f} quality"
    )

    csv_name = getattr(csv_path, "name", str(csv_path))
    display(HTML(
        '<div style="font-family:sans-serif;max-width:520px;color:#fff">'
        '<div style="font-size:14px;margin-bottom:6px">'
        'Training results &mdash; Fine-Tuned Router vs Base &amp; Costliest'
        '</div>'
        '<table style="border-collapse:collapse;font-size:13px;color:#fff">'
        '<thead><tr>'
        f'<th style="{cell_css};text-align:left">Model</th>'
        f'<th style="{cell_css};text-align:right">Cost (USD)</th>'
        f'<th style="{cell_css};text-align:right">Quality</th>'
        '</tr></thead>'
        f'<tbody>{body_rows}</tbody>'
        '</table>'
        f'<div style="margin-top:6px;font-size:12px">{footer}</div>'
        f'<div style="margin-top:2px;font-size:11px">Raw CSV saved to {csv_name} '
        f'({len(df)} row &times; {len(df.columns)} cols)</div>'
        '</div>'
    ))


# ------------------------------------------------------------------------------
# Deployment
# ------------------------------------------------------------------------------

def _deployment_url(
    subscription_id: str,
    resource_group: str,
    resource_name: str,
    deployment_name: str,
) -> str:
    return (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.CognitiveServices/accounts/{resource_name}"
        f"/deployments/{deployment_name}?api-version=2025-06-01"
    )


def _delete_if_exists(
    token: str,
    deployment_name: str,
    *,
    subscription_id: str,
    resource_group: str,
    resource_name: str,
    wait_s: int = 300,
) -> None:
    """Delete ``deployment_name`` if it exists and wait until the delete completes.

    Re-PUTting an existing fine-tuned deployment fails with
    ``ModelUpgradeNotSupported`` because Azure treats the request as an in-place
    model upgrade (not allowed for FT). The clean fix is to delete-then-create.
    """
    url = _deployment_url(subscription_id, resource_group, resource_name, deployment_name)
    auth = {"Authorization": f"Bearer {token}"}
    get_resp = requests.get(url, headers=auth)
    if get_resp.status_code == 404:
        return
    get_resp.raise_for_status()
    existing_model = (get_resp.json().get("properties", {}) or {}).get("model", {}).get("name", "?")
    print(f"   Found existing deployment '{deployment_name}' (model={existing_model}). Deleting...")
    del_resp = requests.delete(url, headers=auth)
    if del_resp.status_code not in (200, 202, 204, 404):
        del_resp.raise_for_status()
    for i in range(wait_s // 5):
        time.sleep(5)
        if requests.get(url, headers=auth).status_code == 404:
            print(f"   Deleted after ~{(i + 1) * 5}s.")
            return
    raise RuntimeError(f"Existing deployment '{deployment_name}' did not delete within {wait_s}s")


def deploy_finetuned_router(
    *,
    token: str,
    fine_tuned_model: str,
    deployment_name: str,
    subscription_id: str,
    resource_group: str,
    resource_name: str,
    project_endpoint: str,
    sku: str = "GlobalStandard",
    capacity: int = 10,
) -> None:
    """Idempotently deploy a fine-tuned Model Router.

    Deletes any existing deployment with the same name first (see
    ``_delete_if_exists`` for rationale), then PUTs a fresh deployment with
    ``versionUpgradeOption=NoAutoUpgrade``. Prints a tidy success summary
    including the inference URL derived from ``project_endpoint``.
    """
    _delete_if_exists(
        token, deployment_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        resource_name=resource_name,
    )
    url = _deployment_url(subscription_id, resource_group, resource_name, deployment_name)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "sku": {"name": sku, "capacity": capacity},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": fine_tuned_model,
                "version": "1",
            },
            "versionUpgradeOption": "NoAutoUpgrade",
        },
    }
    resp = requests.put(url, headers=headers, json=payload)
    if not resp.ok:
        print(f"Deployment failed ({resp.status_code}): {resp.text}")
    resp.raise_for_status()

    body = resp.json()
    props = body.get("properties", {})
    parsed = urlparse(project_endpoint)
    inference_base = f"{parsed.scheme}://{parsed.netloc}"
    print("✅ Deployment created!")
    print(f"   Name:               {body.get('name', deployment_name)}")
    print(f"   Provisioning state: {props.get('provisioningState', 'unknown')}")
    print(f"   SKU / capacity:     {sku} × {capacity}")
    print(f"   Inference URL:      {inference_base}/openai/deployments/{deployment_name}/chat/completions")
