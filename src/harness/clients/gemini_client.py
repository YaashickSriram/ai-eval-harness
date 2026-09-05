"""
Gemini client — Google AI Studio REST API.

WHY this exists:
  Same Protocol as Groq, completely different request/response shape underneath.
  THIS is the proof that the LLMClient abstraction is real — if both providers
  conform to the same interface but call wildly different endpoints, swapping
  them is genuinely a one-line change.

  Gemini's API doesn't use OpenAI-compatible schema. Notice how different
  the payload structure is from groq_client.py — that's the whole point.
"""

import time

import httpx

from harness.clients.base import LLMRequest, LLMResponse
from harness.config.settings import settings
from typing import Any

# WHY URL has the model in it: Gemini's REST endpoint embeds the model in the path,
# unlike OpenAI-compatible APIs that put it in the body. Different shape entirely.
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiClient:
    """Synchronous Gemini client implementing the LLMClient protocol."""

    provider_name = "gemini"

    def __init__(self, api_key: str | None = None, default_model: str | None = None) -> None:
        self._api_key = api_key or settings.gemini_api_key.get_secret_value()
        self._default_model = default_model or settings.gemini_model
        # WHY this conditional: in corporate networks with SSL inspection proxies
        # (and no installed corporate CA), TLS verification fails. The setting
        # is explicit in .env and defaults to False — the workaround must be
        # opted into, never silent.
        verify_tls = not settings.harness_insecure_skip_tls_verify
        if not verify_tls:
            import warnings

            warnings.warn(
                "TLS verification is DISABLED for Gemini client. "
                "This should only happen in dev with corporate-proxy SSL inspection. "
                "Set HARNESS_INSECURE_SKIP_TLS_VERIFY=false once the corporate CA is installed.",
                category=UserWarning,
                stacklevel=2,
            )
        self._http = httpx.Client(timeout=60.0, verify=verify_tls)

    def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self._default_model
        url = f"{GEMINI_API_BASE}/{model}:generateContent"

        # WHY this payload looks weird vs Groq: Gemini uses "contents"
        # (a list of conversation turns) with "parts" inside each.
        # Each turn has a role ("user" or "model" — note: "model" not "assistant").
        contents = [{"role": "user", "parts": [{"text": request.prompt}]}]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }

        # WHY system instruction is separate: Gemini treats system prompts as
        # a top-level field, not a message. Different design from OpenAI/Groq.
        if request.system:
            payload["systemInstruction"] = {"parts": [{"text": request.system}]}

        if request.stop:
            payload["generationConfig"]["stopSequences"] = request.stop

        # WHY key in URL: Google AI Studio uses query-param auth for the public API.
        # In production we'd use Vertex AI with proper IAM; AI Studio is for dev/portfolio.
        params = {"key": self._api_key}

        start = time.perf_counter()
        response = self._http.post(url, json=payload, params=params)
        latency_ms = (time.perf_counter() - start) * 1000

        response.raise_for_status()
        body = response.json()

        # WHY this extraction is more verbose than Groq: Gemini wraps the text
        # in candidates → content → parts → text. The protocol HIDES this mess
        # from every caller.
        candidate = body["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
        finish_reason = candidate.get("finishReason")

        # WHY usageMetadata might be missing on some responses: Gemini's API
        # occasionally omits it. We default to 0 rather than crashing.
        usage = body.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

        return LLMResponse(
            text=text,
            model=model,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            raw=body,
        )

    def __enter__(self) -> "GeminiClient":
        return self

    def __exit__(self, *args: object) -> None:
        self._http.close()
