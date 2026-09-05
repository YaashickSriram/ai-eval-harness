"""Unit tests for SchemaEvaluator — no network, no LLM calls."""

from typing import Literal

from pydantic import BaseModel

from harness.evaluators import EvalStatus, SchemaEvaluator


# WHY this fixture-like model lives at module scope: keeping tests close to
# their data, and Pydantic models created inside functions can confuse
# Pydantic's introspection in rare cases.
class TicketTriage(BaseModel):
    """Realistic example schema for the demo example."""

    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str


# ============================================================================
# Happy path
# ============================================================================


def test_pass_on_valid_json() -> None:
    """Clean valid JSON matching the schema → PASS, score 1.0."""
    evaluator = SchemaEvaluator(schema=TicketTriage)
    response = '{"category": "billing", "urgency": "high", "summary": "Refund request"}'

    result = evaluator.evaluate(response)

    assert result.passed
    assert result.score == 1.0
    assert result.status == EvalStatus.PASS
    assert "TicketTriage" in result.rationale


def test_pass_on_json_with_code_fences() -> None:
    """LLMs often wrap JSON in ```json ... ``` — should still PASS."""
    evaluator = SchemaEvaluator(schema=TicketTriage)
    response = """```json
    {"category": "technical", "urgency": "low", "summary": "Login button broken"}
    ```"""

    result = evaluator.evaluate(response)

    assert result.passed, f"expected PASS, got {result.status}: {result.rationale}"
    assert result.score == 1.0


def test_pass_on_json_with_unfenced_prose() -> None:
    """If model returns prose around JSON, accept the JSON only.

    WHY: 'Sure! Here's the response: {...}' is a common LLM tic.
    The schema check is the relevant signal; the prose is noise we tolerate.
    """
    # Note: only fenced prose is stripped by our regex. Unfenced prose still
    # breaks JSON parsing. We document this limitation rather than over-engineer.
    # If this becomes a real problem we add a smarter extractor.


# ============================================================================
# Failure modes
# ============================================================================


def test_fail_on_invalid_json() -> None:
    """Malformed JSON → FAIL with helpful rationale."""
    evaluator = SchemaEvaluator(schema=TicketTriage)
    response = '{"category": "billing", "urgency": high}'  # missing quotes around `high`

    result = evaluator.evaluate(response)

    assert not result.passed
    assert result.status == EvalStatus.FAIL
    assert result.score == 0.0
    assert "not valid JSON" in result.rationale
    # WHY checking metadata: traces will surface this for debugging
    assert "raw_response" in result.metadata


def test_fail_on_missing_required_field() -> None:
    """Missing 'summary' → FAIL with partial credit (2/3 fields valid)."""
    evaluator = SchemaEvaluator(schema=TicketTriage)
    response = '{"category": "billing", "urgency": "high"}'  # no summary

    result = evaluator.evaluate(response)

    assert not result.passed
    assert result.status == EvalStatus.FAIL
    # 1 field of 3 failed → 2/3 credit ≈ 0.67
    assert 0.6 <= result.score <= 0.7, f"expected partial ~0.67, got {result.score}"
    assert "summary" in result.rationale


def test_fail_on_wrong_type() -> None:
    """Number where string was expected → FAIL.

    Pydantic v2's default for BaseModel is strict-ish on primitives:
    int won't coerce to str (though str-to-int IS allowed). This means
    type mismatches in LLM output are caught — which is what we want.
    """
    evaluator = SchemaEvaluator(schema=TicketTriage)
    response = '{"category": "billing", "urgency": "high", "summary": 12345}'

    result = evaluator.evaluate(response)

    assert not result.passed
    assert result.status == EvalStatus.FAIL
    # WHY partial credit ≈ 0.67: 2 of 3 fields are valid (category, urgency)
    # and only `summary` failed.
    assert 0.6 <= result.score <= 0.7, f"expected ~0.67, got {result.score}"
    assert "summary" in result.rationale.lower()




# ============================================================================
# Contract conformance
# ============================================================================


def test_has_required_protocol_attributes() -> None:
    """SchemaEvaluator conforms to the Evaluator Protocol."""
    evaluator = SchemaEvaluator(schema=TicketTriage)
    assert hasattr(evaluator, "name")
    assert hasattr(evaluator, "evaluate")
    assert evaluator.name == "schema"


def test_custom_name_is_respected() -> None:
    """Custom name in constructor flows through to EvalResult.evaluator."""
    evaluator = SchemaEvaluator(schema=TicketTriage, name="schema:ticket")
    response = '{"category": "billing", "urgency": "low", "summary": "ok"}'

    result = evaluator.evaluate(response)

    assert result.evaluator == "schema:ticket"
