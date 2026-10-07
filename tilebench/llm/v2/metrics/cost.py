"""Per-request USD list-price cost from provider-reported raw usage x the
frozen public pricing snapshot (manifests/api_pricing.yaml).

Statuses (usd_cost_status):
  exact_from_usage_and_frozen_price  complete provider usage and exactly one matching pricing rule
  partial_lower_bound                the stream broke after usage events: estimated_cost_usd is None,
                                     estimated_cost_lower_bound_usd prices the partial usage
  unknown                            the request may have been billed and no usage (or no rule) exists:
                                     estimated_cost_usd is None, never 0
  not_sent                           no billable generation: never sent (prompt_too_long), refused by the
                                     provider, or rejected before processing (transport charged "no")

OpenAI: ordinary input = input_tokens - cached_tokens - cache_write_tokens;
reasoning_tokens are part of output_tokens. Anthropic: input_tokens are the
uncached tokens after the last cache breakpoint; cache writes are split 5m /
1h by `cache_creation`; thinking is part of output_tokens. Nothing is added
twice. The USD axis is separate from the logical-token metric."""
from __future__ import annotations

import hashlib
from functools import lru_cache

from tilebench.llm.v2.manifests import schema as ms

PRICING_FILE = "api_pricing.yaml"
USD_STATUSES = ("exact_from_usage_and_frozen_price", "partial_lower_bound", "unknown", "not_sent")
SCHEMA = "tilebench-llm-api-pricing/1"


class PricingError(ValueError):
    pass


def pricing_path():
    return ms.MANIFEST_DIR / PRICING_FILE


def pricing_sha256() -> str:
    return hashlib.sha256(pricing_path().read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def _load(sha: str) -> dict:
    data = ms.load_yaml(PRICING_FILE)
    validate_pricing(data)
    return data


def load_pricing() -> dict:
    return _load(pricing_sha256())


def validate_pricing(data: dict) -> None:
    if data.get("schema") != SCHEMA or data.get("currency") != "USD" or data.get("unit") != "per_1M_tokens":
        raise PricingError(f"{PRICING_FILE}: schema/currency/unit must be {SCHEMA}/USD/per_1M_tokens")
    if not data.get("version") or not data.get("checked_at"):
        raise PricingError(f"{PRICING_FILE}: version and checked_at are required")
    need = {"openai": {"input", "cached_input", "cache_write", "output"},
            "anthropic": {"input", "cache_write_5m", "cache_write_1h", "cache_read", "output"}}
    ids = set()
    for prov, pdata in (data.get("providers") or {}).items():
        if not pdata.get("sources") or not all(s.get("url", "").startswith("https://") for s in pdata["sources"]):
            raise PricingError(f"{PRICING_FILE}: {prov} needs official https sources")
        for model, mdata in (pdata.get("models") or {}).items():
            for rule in mdata.get("rules") or []:
                if rule.get("id") in ids:
                    raise PricingError(f"{PRICING_FILE}: duplicate rule id {rule.get('id')}")
                ids.add(rule.get("id"))
                rates = rule.get("rates") or {}
                if set(rates) != need.get(prov, set(rates)):
                    raise PricingError(f"{PRICING_FILE}: {rule.get('id')} rates must be exactly {sorted(need[prov])}")
                if any(not isinstance(v, (int, float)) or v < 0 for v in rates.values()):
                    raise PricingError(f"{PRICING_FILE}: {rule.get('id')} has a non-numeric or negative rate")


def pricing_binding() -> dict:
    d = load_pricing()
    return {"pricing_snapshot_sha256": pricing_sha256(), "pricing_version": d["version"], "checked_at": d["checked_at"]}


def _rule(provider: str, model_id: str, *, service_tier, inference_geo, input_tokens: int) -> dict | None:
    models = ((load_pricing().get("providers") or {}).get(provider) or {}).get("models") or {}
    rules = (models.get(model_id) or {}).get("rules") or []
    hits = []
    for r in rules:
        w = r.get("applies_when") or {}
        if "service_tier_in" in w and service_tier not in w["service_tier_in"]:
            continue
        if "inference_geo_in" in w and inference_geo not in w["inference_geo_in"]:
            continue
        if "max_input_tokens" in w and input_tokens > w["max_input_tokens"]:
            continue
        if "min_input_tokens" in w and input_tokens < w["min_input_tokens"]:
            continue
        hits.append(r)
    return hits[0] if len(hits) == 1 else None


def _i(x) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


def _price(tokens: dict, rates: dict) -> tuple[dict, float]:
    usd = {f"usd_{k}": round(tokens[k] * rates[r] / 1e6, 10) for k, r in _RATE_OF.items() if k in tokens}
    return usd, round(sum(usd.values()), 10)


_RATE_OF = {"uncached_input": "input", "cached_input": "cached_input", "cache_write": "cache_write",
            "cache_creation_5m": "cache_write_5m", "cache_creation_1h": "cache_write_1h", "cache_read": "cache_read",
            "output": "output"}


def token_components(provider: str, usage_raw: dict) -> dict:
    u = usage_raw or {}
    if provider == "openai":
        det = u.get("input_tokens_details") or {}
        inp, cached, write = _i(u.get("input_tokens")), _i(det.get("cached_tokens")), _i(det.get("cache_write_tokens"))
        return {"uncached_input": max(inp - cached - write, 0), "cached_input": cached, "cache_write": write,
                "output": _i(u.get("output_tokens")),
                "reasoning_output_included": _i((u.get("output_tokens_details") or {}).get("reasoning_tokens"))}
    if provider == "anthropic":
        cc = u.get("cache_creation") or {}
        total_write = _i(u.get("cache_creation_input_tokens"))
        w1h = _i(cc.get("ephemeral_1h_input_tokens"))
        w5m = _i(cc.get("ephemeral_5m_input_tokens")) if cc else total_write
        if cc and w5m + w1h != total_write:
            w5m = max(total_write - w1h, 0)          # breakdown incomplete: the remainder is priced as 5m writes
        return {"uncached_input": _i(u.get("input_tokens")), "cache_creation_5m": w5m, "cache_creation_1h": w1h,
                "cache_read": _i(u.get("cache_read_input_tokens")), "output": _i(u.get("output_tokens"))}
    raise PricingError(f"no usage mapping for provider {provider!r}")


def _total_input(provider: str, comp: dict) -> int:
    if provider == "openai":
        return comp["uncached_input"] + comp["cached_input"] + comp["cache_write"]
    return comp["uncached_input"] + comp["cache_creation_5m"] + comp["cache_creation_1h"] + comp["cache_read"]


def request_cost(provider: str, model_id: str, usage_raw: dict | None, *, service_tier=None, inference_geo=None,
                 partial: bool = False) -> dict:
    """USD record of ONE transport attempt from its raw usage."""
    base = {"pricing_snapshot_sha256": pricing_sha256(), "pricing_rule_id": None, "estimated_cost_usd": None,
            "estimated_cost_lower_bound_usd": None, "usd_cost_status": "unknown", "tokens": None}
    if provider == "mock":
        return {**base, "usd_cost_status": "unknown", "note": "mock provider: no price"}
    if not usage_raw:
        return {**base, "note": "no provider usage"}
    if provider == "anthropic":
        service_tier = usage_raw.get("service_tier", service_tier)
        inference_geo = usage_raw.get("inference_geo", inference_geo)
    comp = token_components(provider, usage_raw)
    rule = _rule(provider, model_id, service_tier=service_tier, inference_geo=inference_geo,
                 input_tokens=_total_input(provider, comp))
    if rule is None:
        return {**base, "tokens": comp, "note": f"no single pricing rule for {provider}/{model_id} "
                                                f"(service_tier={service_tier!r}, inference_geo={inference_geo!r})"}
    usd, total = _price(comp, rule["rates"])
    rec = {**base, "pricing_rule_id": rule["id"], "tokens": comp, **usd}
    if partial:
        rec.update(estimated_cost_lower_bound_usd=total, usd_cost_status="partial_lower_bound")
    else:
        rec.update(estimated_cost_usd=total, usd_cost_status="exact_from_usage_and_frozen_price")
    return rec


def transport_cost(entry: dict, provider: str, model_id: str, *, usage_raw: dict | None = None,
                   service_tier=None, inference_geo=None) -> dict:
    """USD record of one transport row (succeeded / failed / refused / orphaned)."""
    oc = entry.get("outcome") or entry.get("event")
    if oc == "succeeded":
        return request_cost(provider, model_id, usage_raw, service_tier=service_tier, inference_geo=inference_geo)
    if oc == "refused" or entry.get("charged") == "no":
        return {"pricing_snapshot_sha256": pricing_sha256(), "pricing_rule_id": None, "estimated_cost_usd": 0.0,
                "estimated_cost_lower_bound_usd": 0.0, "usd_cost_status": "not_sent", "tokens": None,
                "note": "no billable generation (refused / rejected before processing)"}
    if entry.get("usage_partial"):
        return request_cost(provider, model_id, entry["usage_partial"], service_tier=service_tier,
                            inference_geo=inference_geo, partial=True)
    return {"pricing_snapshot_sha256": pricing_sha256(), "pricing_rule_id": None, "estimated_cost_usd": None,
            "estimated_cost_lower_bound_usd": None, "usd_cost_status": "unknown", "tokens": None,
            "note": "transport failure after sending; charge unknown and no usage reported"}


def combine(records: list[dict]) -> dict:
    """USD of an attempt / round / trajectory from its transport-level records:
    exact only when every record is exact or not_sent; otherwise None, with
    the lower bound summing what is known."""
    if not records:
        return {"estimated_cost_usd": 0.0, "estimated_cost_lower_bound_usd": 0.0, "usd_cost_status": "not_sent",
                "unknown_records": 0}
    exact = all(r.get("usd_cost_status") in ("exact_from_usage_and_frozen_price", "not_sent") for r in records)
    lower = sum((r.get("estimated_cost_usd") if r.get("estimated_cost_usd") is not None
                 else r.get("estimated_cost_lower_bound_usd") or 0.0) for r in records)
    unknown = sum(1 for r in records if r.get("usd_cost_status") in ("unknown", "partial_lower_bound"))
    if all(r.get("usd_cost_status") == "not_sent" for r in records):
        status = "not_sent"
    elif exact:
        status = "exact_from_usage_and_frozen_price"
    elif any(r.get("usd_cost_status") == "unknown" for r in records):
        status = "unknown"
    else:
        status = "partial_lower_bound"
    return {"estimated_cost_usd": round(lower, 10) if exact else None, "estimated_cost_lower_bound_usd": round(lower, 10),
            "usd_cost_status": status, "unknown_records": unknown}
