"""
Groq client — uses the OpenAI-compatible REST API via httpx.

WHY no Groq SDK dependency:
  - Groq's SDK is a thin wrapper over their REST endpoint.
  - Avoiding it means one fewer dependency and one fewer way for the SDK
    to change under us. We control exactly what's sent.
  - In an interview: "I went REST-direct so the harness isn't coupled to
    a specific SDK version" — that's a real, defensible engineering call.
"""

import time

import httpx

from harness.clients.base import LLMRequest, LLMResponse
from harness.config.settings import settings
from typing import Any, Protocol, runtime_checkable

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient:
    """Synchronous Groq client implementing the LLMClient protocol."""

    provider_name = "groq"

    def __init__(self, api_key: str | None = None, default_model: str | None = None) -> None:
        # WHY accept api_key as a parameter: testability — tests can inject a fake
        # key, and we don't force a singleton dependency on settings.
        self._api_key = api_key or settings.groq_api_key.get_secret_value()
        self._default_model = default_model or settings.groq_model
        # WHY a long-lived client: httpx.Client reuses TCP connections.
        # Creating a new client per call is wasteful and slow.
        # WHY this conditional: in corporate networks with SSL inspection proxies
        # (and no installed corporate CA), TLS verification fails. The setting
        # is explicit in .env and defaults to False — the workaround must be
        # opted into, never silent.
        verify_tls = not settings.harness_insecure_skip_tls_verify
        if not verify_tls:
            import warnings

            warnings.warn(
                "TLS verification is DISABLED for Groq client. "
                "This should only happen in dev with corporate-proxy SSL inspection. "
                "Set HARNESS_INSECURE_SKIP_TLS_VERIFY=false once the corporate CA is installed.",
                category=UserWarning,
                stacklevel=2,
            )
        self._http = httpx.Client(timeout=60.0, verify=verify_tls)

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self._default_model

        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = request.stop

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # WHY measure latency around the network call only: separates
        # API latency from local processing time. The number we report
        # is what the API actually took.
        start = time.perf_counter()
        response = self._http.post(GROQ_API_URL, json=payload, headers=headers)
        latency_ms = (time.perf_counter() - start) * 1000

        # WHY raise_for_status: fail loudly on HTTP errors.
        # We'll add retry logic on Day 2; for now, errors should surface.
        response.raise_for_status()
        body = response.json()

        choice = body["choices"][0]
        usage = body["usage"]

        return LLMResponse(
            text=choice["message"]["content"],
            model=body["model"],
            provider=self.provider_name,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
            raw=body,
        )

    def __enter__(self) -> "GroqClient":
        return self

    def __exit__(self, *args: object) -> None:
        self._http.close()
