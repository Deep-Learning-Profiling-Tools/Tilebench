"""Builds a live provider and its request settings from manifests/models.yaml.

Nothing here reads a key: the adapters read the environment variable named
by the role's `api_key_env` at construction time. The factory only decides
which adapter, which model id and which decoding settings are used, and it
refuses any role that is not fully specified."""
from __future__ import annotations

from dataclasses import dataclass

from tilebench.llm.v2.manifests.schema import ManifestError

MODEL_STATUSES = ("unset", "candidate", "approved")


@dataclass(frozen=True)
class GeneratorSpec:
    name: str                 # gpt | claude (role key)
    provider: str             # openai | anthropic
    model_id: str
    api_key_env: str
    settings: dict            # provider-specific decoding settings sent on every request
    status: str               # candidate | approved
    set_by: str | None

    def record(self) -> dict:
        return {"name": self.name, "provider": self.provider, "model_id": self.model_id,
                "api_key_env": self.api_key_env, "settings": self.settings, "status": self.status, "set_by": self.set_by}


def _settings(provider: str, cfg: dict) -> dict:
    if provider == "openai":
        s = {"reasoning_effort": cfg.get("reasoning_effort"), "max_output_tokens": cfg.get("max_output_tokens")}
        if cfg.get("temperature") is not None:
            s["temperature"] = cfg["temperature"]
        if cfg.get("store") is not None:
            s["store"] = cfg["store"]
        missing = [k for k in ("reasoning_effort", "max_output_tokens") if not s.get(k)]
    elif provider == "anthropic":
        s = {"thinking": cfg.get("thinking"), "output_effort": cfg.get("output_effort"),
             "max_output_tokens": cfg.get("max_output_tokens")}
        if cfg.get("temperature") is not None:
            s["temperature"] = cfg["temperature"]
        missing = [k for k in ("thinking", "output_effort", "max_output_tokens") if not s.get(k)]
    else:
        raise ManifestError(f"models.yaml: unknown provider {provider!r}")
    if missing:
        raise ManifestError(f"models.yaml: {provider} settings incomplete: {missing}")
    return s


def generator_spec(models: dict, name: str, *, accept_status: tuple[str, ...] = ("approved",)) -> GeneratorSpec:
    cfg = models.get("roles", {}).get("generator", {}).get(name)
    if cfg is None:
        raise ManifestError(f"models.yaml: generator {name!r} is not defined")
    provider = cfg.get("provider")
    if not cfg.get("model_id") or not provider:
        raise ManifestError(f"models.yaml: generator {name!r} has no model_id/provider")
    status = cfg.get("status")
    if status not in MODEL_STATUSES:
        raise ManifestError(f"models.yaml: generator {name!r} status {status!r} invalid")
    if status not in accept_status:
        raise ManifestError(f"models.yaml: generator {name!r} status is {status!r}; this run type accepts {accept_status}")
    api_key_env = cfg.get("api_key_env") or models.get("providers", {}).get(provider, {}).get("api_key_env")
    if not api_key_env:
        raise ManifestError(f"models.yaml: no api_key_env for provider {provider!r}")
    return GeneratorSpec(name=name, provider=provider, model_id=cfg["model_id"], api_key_env=api_key_env,
                         settings=_settings(provider, cfg), status=status, set_by=cfg.get("set_by"))


def distiller_spec(models: dict, *, accept_status: tuple[str, ...] = ("approved",)) -> GeneratorSpec:
    cfg = models.get("roles", {}).get("distiller")
    if not cfg or not cfg.get("model_id") or not cfg.get("provider"):
        raise ManifestError("models.yaml: distiller model_id/provider is not set")
    status = cfg.get("status")
    if status not in accept_status:
        raise ManifestError(f"models.yaml: distiller status is {status!r}; accepted {accept_status}")
    provider = cfg["provider"]
    api_key_env = cfg.get("api_key_env") or models.get("providers", {}).get(provider, {}).get("api_key_env")
    if not api_key_env:
        raise ManifestError(f"models.yaml: no api_key_env for provider {provider!r}")
    return GeneratorSpec(name="distiller", provider=provider, model_id=cfg["model_id"], api_key_env=api_key_env,
                         settings=_settings(provider, cfg), status=status, set_by=cfg.get("set_by"))


def build_provider(spec: GeneratorSpec, *, timeout_s: float):
    """Instantiate the adapter for a spec. Raises RuntimeError when the key
    environment variable is unset (no fallback to another variable)."""
    if spec.provider == "openai":
        from tilebench.llm.v2.providers.openai_responses import OpenAIResponsesProvider
        return OpenAIResponsesProvider(timeout_s=timeout_s, api_key_env=spec.api_key_env)
    if spec.provider == "anthropic":
        from tilebench.llm.v2.providers.anthropic_messages import AnthropicMessagesProvider
        return AnthropicMessagesProvider(timeout_s=timeout_s, api_key_env=spec.api_key_env)
    raise ManifestError(f"no adapter for provider {spec.provider!r}")
