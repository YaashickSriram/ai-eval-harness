"""Unit tests for the evaluator base contract (no network)."""

from harness.evaluators import EvalResult, EvalStatus


def test_eval_result_is_frozen() -> None:
    """EvalResult is immutable — once produced, can't be silently changed."""
    result = EvalResult(
        evaluator="schema",
        status=EvalStatus.PASS,
        score=1.0,
        rationale="all fields present",
    )
    try:
        result.score = 0.0  # type: ignore[misc]
    except Exception:  # noqa: BLE001
        return
    raise AssertionError("EvalResult should be frozen but mutation succeeded")


def test_eval_result_passed_property() -> None:
    """The .passed convenience property maps cleanly to status."""
    passing = EvalResult(
        evaluator="schema", status=EvalStatus.PASS, score=1.0, rationale="ok"
    )
    failing = EvalResult(
        evaluator="schema", status=EvalStatus.FAIL, score=0.0, rationale="bad"
    )
    erroring = EvalResult(
        evaluator="judge", status=EvalStatus.ERROR, score=0.0, rationale="timeout"
    )

    assert passing.passed is True
    assert failing.passed is False
    assert erroring.passed is False, "ERROR is not PASS"


def test_eval_result_str_is_one_line_summary() -> None:
    """The custom __str__ produces a clean log-friendly line."""
    result = EvalResult(
        evaluator="judge:relevance",
        status=EvalStatus.PASS,
        score=0.85,
        rationale="answer is on-topic",
        threshold=0.7,
    )
    out = str(result)
    assert "judge:relevance" in out
    assert "PASS" in out
    assert "0.85" in out
    assert "answer is on-topic" in out


def test_eval_status_serializes_as_string() -> None:
    """EvalStatus is a str-enum — its value is a plain string.

    WHY this matters: trace files dump EvalResult to JSON. A plain enum
    would serialize as 'EvalStatus.PASS'; str-enum gives us clean 'pass'.
    """
    assert EvalStatus.PASS.value == "pass"
    assert EvalStatus.FAIL.value == "fail"
    assert EvalStatus.ERROR.value == "error"
