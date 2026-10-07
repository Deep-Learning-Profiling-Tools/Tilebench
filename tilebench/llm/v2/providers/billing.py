"""OPTIONAL provider billing reconciliation (separate from the per-request
experiment metric).

`estimated_list_price_usd` (metrics.cost: request usage x frozen public price)
and `provider_billed_usd` (this module: the provider's AGGREGATED cost report
for a time window) are different quantities and are never merged into one
field. Aggregated billing cannot be attributed to individual requests.

Admin credentials are read only from the environment variables named below;
their values are never logged, written or returned. Without them nothing is
requested and a status file says so; the formal experiment never depends on
this step. Ordinary inference keys (OPENAI_API_KEY / CLAUDE_API_KEY) are never
used here: they do not carry organization-billing permission."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ADMIN_KEY_ENV = {"openai": ("OPENAI_ADMIN_KEY",), "anthropic": ("ANTHROPIC_ADMIN_KEY", "CLAUDE_ADMIN_KEY")}
ENDPOINTS = {"openai": "https://api.openai.com/v1/organization/costs",
             "anthropic": "https://api.anthropic.com/v1/organizations/cost_report"}


def admin_key_env(provider: str) -> str | None:
    for name in ADMIN_KEY_ENV[provider]:
        if os.environ.get(name):
            return name
    return None


def reconcile(campaign_dir: Path, *, start: float, end: float, providers=("openai", "anthropic"), http_get=None) -> dict:
    """Fetch the providers' aggregated cost for [start, end] when an admin key exists; record status otherwise."""
    out_dir = Path(campaign_dir) / "billing_reconciliation"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {"t": time.time(), "window": {"start": start, "end": end}, "providers": {},
              "note": "provider_billed_usd is aggregated over the window and organization/project; it is not per request "
                      "and is never written into the estimated_cost_usd fields"}
    for p in providers:
        env = admin_key_env(p)
        if env is None:
            status["providers"][p] = {"status": "unavailable", "reason": f"no admin credential in {list(ADMIN_KEY_ENV[p])}"}
            continue
        try:
            raw = _fetch(p, os.environ[env], start, end, http_get)
            (out_dir / f"{p}_cost_report_{int(time.time())}.json").write_text(json.dumps(raw, indent=1) + "\n")
            status["providers"][p] = {"status": "ok", "credential_env": env, "provider_billed_usd": _total(p, raw)}
        except Exception as e:  # noqa: BLE001 - optional step; record and continue
            status["providers"][p] = {"status": "error", "credential_env": env, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    (out_dir / "status.json").write_text(json.dumps(status, indent=1) + "\n")
    return status


def _fetch(provider: str, key: str, start: float, end: float, http_get=None) -> dict:
    import httpx
    get = http_get or httpx.get
    if provider == "openai":
        r = get(ENDPOINTS[provider], params={"start_time": int(start), "end_time": int(end), "bucket_width": "1d", "limit": 180},
                headers={"Authorization": f"Bearer {key}"}, timeout=60)
    else:
        iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))  # noqa: E731
        r = get(ENDPOINTS[provider], params={"starting_at": iso(start), "ending_at": iso(end)},
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=60)
    r.raise_for_status()
    return r.json()


def _total(provider: str, raw: dict) -> float | None:
    total = 0.0
    found = False
    for bucket in raw.get("data", []):
        for res in bucket.get("results", []):
            amt = res.get("amount")
            if isinstance(amt, dict) and amt.get("value") is not None:
                total += float(amt["value"])
                found = True
            elif res.get("cost") is not None:
                total += float(res["cost"])
                found = True
    return round(total, 6) if found else None
