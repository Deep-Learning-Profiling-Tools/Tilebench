"""Unified LLM client for OpenAI (GPT-5.5) and Anthropic (Claude Opus 4.7).

Both APIs are normalized to: generate(prompt: str) -> str.

OpenAI:    reasoning.effort = "xhigh"
Anthropic: thinking={type: "adaptive"}, output_config.effort = "xhigh"

Env vars expected:
    OPENAI_API_KEY     (for GPT-5.5)
    ANTHROPIC_API_KEY  (for Claude Opus 4.7)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    elapsed_s: float
    raw: object = None  # full SDK response (for debugging)
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    # Cached prompt-token hits — only observable in this response (the API
    # never replays cache stats for prior calls). OpenAI:
    # usage.input_tokens_details.cached_tokens; Anthropic:
    # usage.cache_read_input_tokens.
    cached_input_tokens: int | None = None
    # Time-to-first-token: seconds from request start to the first user-
    # visible output token (post-reasoning). Captured in the stream loop
    # because the final SDK response only retains the total elapsed.
    ttft_s: float | None = None


class LLMClient:
    """Thin wrapper around OpenAI / Anthropic SDKs, normalizing to a single
    `generate(prompt)` call with reasoning effort fixed at `xhigh`.
    """

    def __init__(self, model: str, effort: str = "xhigh"):
        self.model = model
        self.effort = effort
        if model.startswith("gpt"):
            self.provider = "openai"
            self._init_openai()
        elif model.startswith("claude"):
            self.provider = "anthropic"
            self._init_anthropic()
        else:
            raise ValueError(f"Unknown model: {model!r}")

    def _init_openai(self):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # xhigh reasoning can run 10-20 min — default 600s is too tight.
        self._client = OpenAI(api_key=api_key, timeout=1800.0, max_retries=0)

    def _init_anthropic(self):
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY (or CLAUDE_API_KEY) not set")
        self._client = Anthropic(api_key=api_key, timeout=1800.0, max_retries=0)

    def generate(
        self, prompt: str, system: str | None = None,
        retries: int = 3, retry_backoff_s: float = 30.0,
    ) -> LLMResponse:
        t0 = time.time()
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                if self.provider == "openai":
                    text, raw, ttft_s = self._gen_openai(prompt, system, t0)
                else:
                    text, raw, ttft_s = self._gen_anthropic(prompt, system, t0)
                usage = _extract_usage(raw, self.provider)
                return LLMResponse(
                    text=text, model=self.model,
                    elapsed_s=time.time() - t0, raw=raw,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    reasoning_tokens=usage.get("reasoning_tokens"),
                    cached_input_tokens=usage.get("cached_input_tokens"),
                    ttft_s=ttft_s,
                )
            except Exception as e:  # noqa: BLE001 — SDK exceptions vary
                last_exc = e
                if attempt < retries:
                    wait = retry_backoff_s * (2 ** attempt)
                    print(f"  [llm] {type(e).__name__}: {e}; retrying in {wait:.0f}s "
                          f"(attempt {attempt+1}/{retries})", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise last_exc  # unreachable

    # ---- OpenAI (GPT-5.x via Responses API, streaming) ----
    def _gen_openai(
        self, prompt: str, system: str | None, t0: float
    ) -> tuple[str, object, float | None]:
        """Stream the Responses API. The connection stays warm via the SSE
        keepalive event stream, so xhigh effort runs of any length won't trip
        a read-timeout.

        Returns (text, raw_final, ttft_s) where ttft_s is the elapsed time
        between the request and the first user-visible output token (i.e.
        excludes the reasoning phase).
        """
        input_msgs = []
        if system:
            input_msgs.append({"role": "system", "content": system})
        input_msgs.append({"role": "user", "content": prompt})

        chunks: list[str] = []
        ttft_s: float | None = None
        last_print = time.time()
        with self._client.responses.stream(
            model=self.model,
            input=input_msgs,
            reasoning={"effort": self.effort},
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", "")
                if etype == "response.output_text.delta":
                    if ttft_s is None:
                        ttft_s = time.time() - t0
                    chunks.append(event.delta)
                # Periodic progress print for very long reasoning runs.
                now = time.time()
                if now - last_print > 60:
                    print(f"    [llm] streaming … {len(''.join(chunks))} chars so far",
                          flush=True)
                    last_print = now
            final = stream.get_final_response()
        return "".join(chunks), final, ttft_s

    # ---- Anthropic (Claude Opus 4.7, streaming) ----
    def _gen_anthropic(
        self, prompt: str, system: str | None, t0: float
    ) -> tuple[str, object, float | None]:
        kwargs = dict(
            model=self.model,
            max_tokens=128000,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        if system:
            kwargs["system"] = system

        chunks: list[str] = []
        ttft_s: float | None = None
        last_print = time.time()
        with self._client.messages.stream(**kwargs) as stream:
            for text_delta in stream.text_stream:
                if ttft_s is None:
                    ttft_s = time.time() - t0
                chunks.append(text_delta)
                now = time.time()
                if now - last_print > 60:
                    print(f"    [llm] streaming … {len(''.join(chunks))} chars so far",
                          flush=True)
                    last_print = now
            final = stream.get_final_message()
        return "".join(chunks), final, ttft_s


# ----------------------------------------------------------------------------
# Token usage extraction (varies by provider/SDK version)
# ----------------------------------------------------------------------------

def _extract_usage(raw, provider: str) -> dict[str, int | None]:
    """Best-effort token counts from the raw SDK response, including the
    cached-prompt-token count (only observable in this response; the API
    does not replay cache stats for prior calls)."""
    out = {
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "cached_input_tokens": None,
    }
    if raw is None:
        return out
    usage = getattr(raw, "usage", None)
    if usage is None:
        return out
    if provider == "openai":
        # Responses API: usage.input_tokens, usage.output_tokens,
        # usage.output_tokens_details.reasoning_tokens,
        # usage.input_tokens_details.cached_tokens
        out["input_tokens"] = getattr(usage, "input_tokens", None)
        out["output_tokens"] = getattr(usage, "output_tokens", None)
        out_details = getattr(usage, "output_tokens_details", None)
        if out_details is not None:
            out["reasoning_tokens"] = getattr(out_details, "reasoning_tokens", None)
        in_details = getattr(usage, "input_tokens_details", None)
        if in_details is not None:
            out["cached_input_tokens"] = getattr(in_details, "cached_tokens", None)
    else:  # anthropic
        out["input_tokens"] = getattr(usage, "input_tokens", None)
        out["output_tokens"] = getattr(usage, "output_tokens", None)
        # Anthropic: cache_read_input_tokens is the cached-prompt-hit count.
        out["cached_input_tokens"] = getattr(usage, "cache_read_input_tokens", None)
        # adaptive-thinking doesn't expose a separate reasoning-token count.
    return out
