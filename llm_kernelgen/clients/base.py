"""Base types and abstract interface for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Unified response wrapper returned by all LLM clients."""

    # Plain text of the model's reply (extracted from the raw API response).
    text: str

    # Raw API response object (provider-specific).
    raw: Any

    # Token usage statistics (populated when available).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    # Wall-clock time in seconds for the API round-trip.
    latency_s: float = 0.0

    # Model identifier reported by the API.
    model_used: str = ""

    # Extra provider-specific metadata.
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseClient(ABC):
    """Abstract base class for all LLM clients.

    Subclasses must implement :meth:`generate`.
    """

    @abstractmethod
    def generate(
        self,
        *,
        model: str,
        instructions: str,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 12000,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a generation request and return an :class:`LLMResponse`.

        Parameters
        ----------
        model:
            Model identifier string (e.g. ``"gpt-4o"``).
        instructions:
            System-level instructions injected before the user prompt.
        prompt:
            The user-facing prompt content (task description, context, etc.).
        temperature:
            Sampling temperature.
        max_output_tokens:
            Upper bound on generated tokens.
        reasoning_effort:
            Optional reasoning effort level (``"low"``, ``"medium"``, ``"high"``).
            Ignored by providers that do not support it.
        **kwargs:
            Additional provider-specific parameters.
        """
