"""Utility adapters for building LLM clients from configuration dicts.

Usage
-----
::

    from llm_kernelgen.runtime.adapters import client_from_config

    model_cfg = {
        "provider": "openai",
        "api_type": "responses",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_output_tokens": 12000,
        "reasoning_effort": "medium",
    }
    client = client_from_config(model_cfg)
    response = client.generate(
        model=model_cfg["model"],
        instructions="You are a GPU kernel engineer.",
        prompt="Write a Triton softmax kernel.",
    )
"""

from __future__ import annotations

import os
from typing import Any

from llm_kernelgen.clients.base import BaseClient


def _resolve_env(value: str) -> str:
    """Expand ``${ENV_VAR}`` patterns in *value* using current environment."""
    import re

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        resolved = os.environ.get(var, "")
        if not resolved:
            raise EnvironmentError(
                f"Environment variable '{var}' referenced in config is not set."
            )
        return resolved

    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def client_from_config(model_cfg: dict[str, Any]) -> BaseClient:
    """Instantiate the correct :class:`BaseClient` subclass from *model_cfg*.

    The ``api_type`` field (or the provider's ``api_type``) controls which
    client class is used:

    * ``"responses"`` → :class:`~llm_kernelgen.clients.openai_responses.OpenAIResponsesClient`
    * ``"chat_completions"`` → :class:`~llm_kernelgen.clients.openai_chat_compat.OpenAIChatCompatClient`

    Parameters
    ----------
    model_cfg:
        Flat dictionary with at minimum the keys ``api_type``,
        ``api_key_env``, and ``base_url``.

    Returns
    -------
    BaseClient
        Ready-to-use client instance.

    Raises
    ------
    ValueError
        If ``api_type`` is unrecognised.
    """
    api_type: str = model_cfg.get("api_type", "chat_completions")
    api_key_env: str = model_cfg.get("api_key_env", "OPENAI_API_KEY")
    base_url: str | None = model_cfg.get("base_url")
    if base_url:
        base_url = _resolve_env(base_url)

    if api_type == "responses":
        from llm_kernelgen.clients.openai_responses import OpenAIResponsesClient

        return OpenAIResponsesClient(api_key_env=api_key_env, base_url=base_url)

    if api_type == "chat_completions":
        from llm_kernelgen.clients.openai_chat_compat import OpenAIChatCompatClient

        return OpenAIChatCompatClient(
            api_key_env=api_key_env,
            base_url=base_url or "https://api.openai.com/v1",
        )

    raise ValueError(
        f"Unknown api_type '{api_type}'. Expected 'responses' or 'chat_completions'."
    )


def load_model_configs(models_yaml_path: str) -> dict[str, dict[str, Any]]:
    """Load the ``models.yaml`` configuration and flatten provider fields.

    Each model entry in the returned dict is augmented with the fields from
    its provider entry (``api_type``, ``base_url``, ``api_key_env``).

    Parameters
    ----------
    models_yaml_path:
        Path to ``llm_kernelgen/configs/models.yaml``.

    Returns
    -------
    dict
        Mapping of model alias → fully merged config dict.
    """
    import yaml

    with open(models_yaml_path) as fh:
        raw = yaml.safe_load(fh)

    providers: dict[str, dict] = raw.get("providers", {})
    models_raw: dict[str, dict] = raw.get("models", {})

    merged: dict[str, dict[str, Any]] = {}
    for alias, model_def in models_raw.items():
        provider_name = model_def.get("provider", "openai")
        provider_cfg = providers.get(provider_name, {})
        entry: dict[str, Any] = {}
        entry.update(provider_cfg)
        entry.update(model_def)
        merged[alias] = entry

    return merged
