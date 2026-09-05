"""Unit tests for TraceWriter — uses tmp_path, no network, no LLM calls."""

import json
from pathlib import Path

from harness.clients.base import LLMRequest, LLMResponse
from harness.evaluators.base import EvalResult, EvalStatus
from harness.tracing import TraceWriter


def test_trace_writer_creates_directory(tmp_path: Path) -> None:
    """TraceWriter creates its trace_dir if missing."""
    target = tmp_path / "nested" / "traces"
    writer = TraceWriter(trace_dir=target)

    assert target.exists()
    assert writer.trace_dir == target


def test_write_llm_call_produces_readable_json(tmp_path: Path) -> None:
    """An LLM call trace serializes the request and response correctly."""
    writer = TraceWriter(trace_dir=tmp_path)
    request = LLMRequest(prompt="test prompt", system="you are helpful", max_tokens=128)
    response = LLMResponse(
        text="test response",
        model="llama-3.3-70b",
        provider="groq",
        input_tokens=10,
        output_tokens=5,
        latency_ms=42.5,
        finish_reason="stop",
    )

    path = writer.write_llm_call(test_name="my_test", request=request, response=response)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["trace_type"] == "llm_call"
    assert data["test_name"] == "my_test"
    assert data["request"]["prompt"] == "test prompt"
    assert data["request"]["system"] == "you are helpful"
    assert data["response"]["text"] == "test response"
    assert data["response"]["provider"] == "groq"
    assert data["response"]["input_tokens"] == 10
    assert data["response"]["latency_ms"] == 42.5


def test_write_evaluation_serializes_enum_as_string(tmp_path: Path) -> None:
    """EvalStatus enum members write as their string values, not 'EvalStatus.PASS'."""
    writer = TraceWriter(trace_dir=tmp_path)
    result = EvalResult(
        evaluator="judge:relevance",
        status=EvalStatus.PASS,
        score=0.85,
        rationale="answer is on-topic",
        threshold=0.75,
        metadata={"raw_judge_score": 4},
    )

    path = writer.write_evaluation(test_name="my_test", result=result)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "pass"  # not "EvalStatus.PASS"
    assert data["score"] == 0.85
    assert data["threshold"] == 0.75
    assert data["evaluator"] == "judge:relevance"
    assert data["metadata"]["raw_judge_score"] == 4


def test_write_evaluation_with_complex_metadata(tmp_path: Path) -> None:
    """Metadata containing nested dicts, lists, and non-trivial values serializes."""
    writer = TraceWriter(trace_dir=tmp_path)
    result = EvalResult(
        evaluator="schema",
        status=EvalStatus.FAIL,
        score=0.67,
        rationale="missing field",
        metadata={
            "errors": [{"loc": ["summary"], "msg": "required"}],
            "raw_response": '{"category": "billing"}',
            "exception_type": "ValidationError",
        },
    )

    path = writer.write_evaluation(test_name="my_test", result=result)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["errors"][0]["loc"] == ["summary"]
    assert data["metadata"]["raw_response"] == '{"category": "billing"}'


def test_filenames_are_unique_within_same_second(tmp_path: Path) -> None:
    """Two traces written rapidly should not collide on filename."""
    writer = TraceWriter(trace_dir=tmp_path)
    request = LLMRequest(prompt="p")
    response = LLMResponse(
        text="r", model="m", provider="groq",
        input_tokens=1, output_tokens=1, latency_ms=1.0,
    )

    path1 = writer.write_llm_call(test_name="t", request=request, response=response)
    path2 = writer.write_llm_call(test_name="t", request=request, response=response)

    assert path1 != path2
    assert path1.exists() and path2.exists()


def test_evaluator_name_with_colon_produces_safe_filename(tmp_path: Path) -> None:
    """Filenames must be filesystem-safe — 'judge:relevance' has a colon (invalid on Windows)."""
    writer = TraceWriter(trace_dir=tmp_path)
    result = EvalResult(
        evaluator="judge:relevance",
        status=EvalStatus.PASS,
        score=1.0,
        rationale="ok",
    )

    path = writer.write_evaluation(test_name="t", result=result)

    assert ":" not in path.name
    assert path.exists()


def test_unicode_in_prompt_writes_correctly(tmp_path: Path) -> None:
    """Non-ASCII characters should round-trip without encoding errors."""
    writer = TraceWriter(trace_dir=tmp_path)
    request = LLMRequest(prompt="Hello 世界 🌍 Café")
    response = LLMResponse(
        text="Hi 👋", model="m", provider="groq",
        input_tokens=1, output_tokens=1, latency_ms=1.0,
    )

    path = writer.write_llm_call(test_name="t", request=request, response=response)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["request"]["prompt"] == "Hello 世界 🌍 Café"
    assert data["response"]["text"] == "Hi 👋"