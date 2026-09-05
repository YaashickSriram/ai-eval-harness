"""
SchemaEvaluator — validates JSON output against a Pydantic model.

THE simplest, cheapest, most reliable evaluator. No LLM calls.

WHY this evaluator exists:
  Most real AI systems are asked to produce STRUCTURED output: a JSON object
  with specific fields ("category", "urgency", "summary"), or a list of items
  with a known shape. When the LLM returns:
    - Invalid JSON (extra commas, missing quotes, prose around the JSON)
    - Wrong field names ("kind" instead of "category")
    - Wrong types ("3" string when "3" int was expected)
    - Missing required fields
    - Extra hallucinated fields
  ...the downstream system breaks. Catching this BEFORE shipping = high ROI.

  In the demo: this is the "deterministic test" that runs first and fast.
  When it fails, you know with certainty something is wrong — no
  interpretation needed.
"""

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from harness.evaluators.base import EvalResult, EvalStatus

# WHY this regex: LLMs frequently wrap JSON in ```json ... ``` code fences,
# or include prose before/after the JSON ("Here is your response: {...}").
# We strip these defensively. If we don't, valid JSON gets rejected as
# unparseable for a purely cosmetic reason.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class SchemaEvaluator:
    """Validates that an LLM response is valid JSON matching a Pydantic schema.

    Usage:
        class TicketTriage(BaseModel):
            category: str
            urgency: Literal["low", "medium", "high"]
            summary: str

        evaluator = SchemaEvaluator(schema=TicketTriage)
        result = evaluator.evaluate(llm_response_text)
    """

    def __init__(self, schema: type[BaseModel], name: str = "schema") -> None:
        # WHY accept the schema in the constructor: each evaluator instance
        # is bound to ONE schema. Composing multiple schema checks = multiple
        # evaluator instances. This keeps each evaluator single-purpose.
        self._schema = schema
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, response: str, context: dict[str, Any] | None = None) -> EvalResult:
        """Score the response against the bound schema."""
        # WHY context is accepted but unused: the Evaluator protocol requires it.
        # Schema validation needs only the response. Future evaluators (judge,
        # similarity) will read context heavily.
        del context

        # ----- Step 1: extract JSON from the response -----
        cleaned = self._strip_code_fences(response)

        # ----- Step 2: parse as JSON -----
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # WHY status=FAIL not ERROR: the LLM did the wrong thing.
            # That's a legitimate test failure, not a harness malfunction.
            return EvalResult(
                evaluator=self._name,
                status=EvalStatus.FAIL,
                score=0.0,
                rationale=f"Response is not valid JSON: {exc.msg} (at line {exc.lineno})",
                metadata={"raw_response": response, "json_error": str(exc)},
            )

        # ----- Step 3: validate against the schema -----
        try:
            validated = self._schema.model_validate(parsed)
        except ValidationError as exc:
            # Pydantic gives us per-field errors. We surface them in the rationale
            # because "which field failed why" is what a developer needs to fix
            # the prompt.
            field_errors = self._format_validation_errors(exc)
            partial_score = self._partial_credit(parsed, exc)
            return EvalResult(
                evaluator=self._name,
                status=EvalStatus.FAIL,
                score=partial_score,
                rationale=f"Schema validation failed: {field_errors}",
                metadata={
                    "raw_response": response,
                    "parsed_json": parsed,
                    "errors": exc.errors(),
                },
            )

        # ----- Step 4: success -----
        return EvalResult(
            evaluator=self._name,
            status=EvalStatus.PASS,
            score=1.0,
            rationale=f"Valid {self._schema.__name__}",
            metadata={"validated": validated.model_dump()},
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove ```json ... ``` wrappers and surrounding prose.

        WHY this is necessary: temperature-0 models still wrap JSON in fences
        ~30% of the time depending on prompt. Stripping them is a free,
        deterministic robustness improvement.
        """
        match = _JSON_FENCE_RE.search(text)
        if match:
            return match.group(1).strip()
        # No fences — return as-is but stripped of leading/trailing whitespace
        return text.strip()

    @staticmethod
    def _format_validation_errors(exc: ValidationError) -> str:
        """Turn Pydantic's error list into a one-line summary for the rationale."""
        # WHY this exists: ValidationError's default str is multi-line and noisy.
        # We collapse to "field1: message1; field2: message2" for the rationale,
        # and keep the full structured errors in metadata.
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            parts.append(f"{loc}: {err['msg']}")
        return "; ".join(parts)

    def _partial_credit(self, parsed: Any, exc: ValidationError) -> float:
        """Award partial credit based on how many fields validated.

        WHY partial credit (instead of just 0.0 on any failure):
          For diagnostic purposes — 0.9 means "almost right, one field off"
          while 0.1 means "completely broken". This nuance is invaluable when
          comparing model versions or prompt variants. A purely binary score
          loses signal.

        Mechanism: count failed fields vs total fields in the schema.
        Crude but effective for the demo. We'll refine later.
        """
        if not isinstance(parsed, dict):
            return 0.0
        total_fields = len(self._schema.model_fields)
        if total_fields == 0:
            return 0.0
        failed_fields = len({err["loc"][0] for err in exc.errors() if err["loc"]})
        passed_fields = max(0, total_fields - failed_fields)
        return round(passed_fields / total_fields, 2)
