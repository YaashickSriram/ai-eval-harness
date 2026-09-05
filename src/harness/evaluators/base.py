"""
Evaluator contract — the interface every evaluator implements.

Same pattern as `clients/base.py`: Protocol + dataclasses, no inheritance.

WHY this file is central:
  Every evaluator — deterministic (schema), semantic (similarity), probabilistic
  (LLM-as-judge), or any future kind (groundedness, toxicity, custom) — conforms
  to THIS interface. Tests don't care which kind of evaluator they're using;
  they just consume EvalResult.

  In an interview: "I designed evaluators around a single contract so the
  test code never needs to know whether it's running a schema check or a
  judge — it just composes them."
"""

from dataclasses import dataclass, field
from enum import StrEnum 
from typing import Any, Protocol, runtime_checkable


class EvalStatus(StrEnum):
    """The verdict of an evaluator.

    WHY str-enum: serializes cleanly to JSON for trace files (Day 2 later).
    Plain Enum would serialize as 'EvalStatus.PASS' which is ugly in logs.

    WHY three states (not just pass/fail):
      - PASS / FAIL: clear verdicts
      - ERROR: the evaluator itself broke (judge LLM timeout, schema lib crash, etc.)
        Without this, you can't tell "the AI did badly" from "your test setup is broken."
        That distinction is critical when debugging flaky AI tests.
    """

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


@dataclass(frozen=True)
class EvalResult:
    """The structured result of one evaluation.

    WHY frozen: once an evaluator produces a result, it can't be silently
    modified before being written to a trace or surfaced in a report.
    Audit-friendly by construction.

    WHY this shape (score + rationale + metadata):
      score:    numeric — sortable, comparable across runs, drift-trackable.
                Always normalized to 0.0-1.0 regardless of source rubric
                (a 4/5 judge score becomes 0.8). Consistency matters.
      rationale: WHY the verdict happened. The single most important field for
                debugging AI tests. Without it, "score: 0.4" is useless.
      metadata: evaluator-specific extras (the raw judge response, the schema
                error, the embedding distance, tokens used by the judge, etc.).
                Kept loose so each evaluator doesn't need its own EvalResult subclass.
    """

    evaluator: str  # WHY: which evaluator produced this — e.g. "schema", "judge:relevance"
    status: EvalStatus
    score: float  # always normalized 0.0–1.0
    rationale: str
    threshold: float | None = None  # pass-fail cutoff used (if applicable)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Convenience for assertions: `assert result.passed`."""
        return self.status == EvalStatus.PASS

    def __str__(self) -> str:
        # WHY a custom __str__: pytest output and trace dumps both benefit
        # from a one-line summary. Default dataclass __repr__ is too noisy.
        return (
            f"[{self.evaluator}] {self.status.value.upper()} "
            f"score={self.score:.2f} threshold={self.threshold} :: {self.rationale}"
        )


@runtime_checkable
class Evaluator(Protocol):
    """Provider-agnostic evaluator interface.

    WHY Protocol (not ABC): same reasoning as LLMClient — structural typing,
    easier mocking, no inheritance hierarchy. Anything with .name and .evaluate
    is an Evaluator.

    Implementations vary widely:
      - Deterministic: validate JSON against a schema (no LLM call)
      - Semantic: embedding similarity (one cheap LLM call)
      - LLM-as-judge: rubric-based scoring by a second LLM (expensive)
      - Heuristic: regex / keyword / length checks (no LLM call)
    """

    @property
    def name(self) -> str:
        """Short identifier for this evaluator instance.

        Used in EvalResult.evaluator and trace files. Should be stable and
        descriptive. Examples: "schema", "judge:relevance", "similarity:cosine".
        """
        ...

    def evaluate(self, response: str, context: dict[str, Any] | None = None) -> EvalResult:
        """Score an LLM response.

        Args:
            response: the raw LLM output text to evaluate.
            context: optional extra info the evaluator may need
                     (the original prompt, the expected output, the source docs,
                      etc.). Loose dict so each evaluator can take what it needs
                      without breaking the contract.

        Returns:
            EvalResult with status, score, rationale.

        Implementations must:
          - Never raise on evaluation failure — return EvalResult with status=FAIL.
          - Only return status=ERROR if the evaluator ITSELF couldn't run
            (e.g. judge LLM timed out, schema parse crashed).
          - Always populate `rationale` — even a one-liner is better than nothing.
        """
        ...
