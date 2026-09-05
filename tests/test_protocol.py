"""Unit tests for the LLM client data classes (no network)."""

from harness.clients import LLMRequest, LLMResponse


def test_llm_request_is_immutable() -> None:
    """LLMRequest is frozen — attempts to mutate should fail.

    WHY this matters: frozen requests mean once we capture a trace,
    the request can't be silently modified between capture and report.
    """
    request = LLMRequest(prompt="hi")
    try:
        request.prompt = "bye"  # type: ignore[misc]
    except Exception:  # noqa: BLE001
        return
    raise AssertionError("LLMRequest should be frozen but mutation succeeded")


def test_llm_request_defaults() -> None:
    request = LLMRequest(prompt="hi")
    assert request.temperature == 0.0
    assert request.max_tokens == 1024
    assert request.model is None
    assert request.system is None
    assert request.stop == []


def test_llm_response_carries_observability_fields() -> None:
    """The fields needed for traces and cost tracking are all there."""
    response = LLMResponse(
        text="ok",
        model="llama-3.3-70b",
        provider="groq",
        input_tokens=10,
        output_tokens=5,
        latency_ms=123.4,
    )
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.latency_ms == 123.4
    assert response.provider == "groq"
