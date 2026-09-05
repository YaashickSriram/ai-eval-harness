"""
LLM client contract, the provider-agnostic interface.

WHY this matters:
  Every test, every evaluator, every part of the harness talks to LLMs through
  THIS interface, never directly to Groq or Gemini SDKs. Swapping providers,
  adding new ones, or mocking for tests becomes a one-line change.

    - understand abstraction
    - don't couple to vendors
    - design for testability (mocking, tracing)
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMRequest:
    """A normalized request that any provider can fulfill.

    WHY frozen: immutability — once a request is sent, its parameters can't
    be mutated by accident, which matters for trace integrity (Day 5).
    """

    prompt: str
    model: str | None = None  # WHY: None means "use provider default"
    temperature: float = 0.0  # WHY: default 0 for testing — deterministic-as-possible
    max_tokens: int = 1024
    system: str | None = None
    # WHY stop sequences: useful for structured output tests
    stop: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMResponse:
    """A normalized response from any provider.

    Notice we capture EVERYTHING — not just text. Tokens, latency, model
    version, raw response. This is what makes observability (Day 5)
    possible without re-instrumenting every call site.
    """

    text: str
    model: str  # WHY: actual model used (resolved from request or default)
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str | None = None
    # WHY raw: kept for debugging — if a provider returns something weird,
    # we can inspect the original payload without re-running the call.
    raw: dict[str, Any] | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic LLM client interface.

    WHY Protocol (not ABC):
      - Structural typing — anything with these methods IS an LLMClient,
        no inheritance required.
      - Easier to mock in tests — a plain class with these methods works.
      - More Pythonic in 2026; ABC is heavier than needed here.

    WHY runtime_checkable:
      - Lets us do `isinstance(client, LLMClient)` in tests if needed.
    """

    @property
    def provider_name(self) -> str:
        """Short provider identifier, e.g. 'groq', 'gemini'."""
        ...

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a single completion request synchronously.

        Implementations must:
          - Capture latency in milliseconds
          - Return token counts (input + output)
          - Set provider and model fields correctly on the response
          - Preserve the raw response in the `raw` field

        Implementations should NOT:
          - Print or log the prompt/response (caller's responsibility)
          - Modify the request
          - Cache (caching is a separate concern, added later)
        """
        ...
