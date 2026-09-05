"""
JudgeEvaluator — uses an LLM to score another LLM's response on a rubric.

THE evaluator for open-ended AI quality questions ("is this answer good?",
"is the tone appropriate?", "is this grounded in the context?").

WHY this evaluator is special:
  - Schema/format evaluators answer "is the output well-formed?"
  - Similarity evaluators answer "is the output close to a known good answer?"
  - JUDGE evaluators answer "is the output ACTUALLY GOOD?" — the question that
    matters for real AI quality.

  Cost vs value tradeoff: judge calls are expensive (every test costs a token-
  worth of judge LLM time). Use sparingly — for the dimensions you can't check
  any other way.

WHY it's harder to test than it looks:
  The judge itself is non-deterministic. Same input → different scores (within
  a small range) across runs. Strategies:
    - temperature=0 on the judge (minimizes variance)
    - Use a strong, stable model (Gemini 2.5 Flash, GPT-4-class)
    - Validate the judge's JSON output (catches when the judge breaks the contract)
    - Track score drift over time (Day 5)
"""

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from harness.clients.base import LLMClient, LLMRequest
from harness.evaluators.base import EvalResult, EvalStatus
from harness.prompts import JudgeRubric, build_judge_prompt


# WHY this Pydantic model: the judge MUST return JSON matching this shape.
# If it doesn't, the judge has failed its task — EvalResult.status = ERROR
# (not FAIL — the judge broke, not the response being judged).
#
# Using Pydantic here means we reuse the same schema-validation discipline
# we apply to system-under-test outputs. Meta-testing in action.
class _JudgeResponse(BaseModel):
    """The shape every judge response must conform to."""

    score: int = Field(..., ge=0, le=10)  # accept any score; we check rubric range later
    rationale: str = Field(..., min_length=1)


# WHY this regex: judges sometimes wrap JSON in ```json ... ``` even when
# explicitly told not to. We strip defensively for the same reason as
# SchemaEvaluator. The instruction asks them not to, but real judges are
# imperfect — robust code accepts both.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class JudgeEvaluator:
    """Scores a response against a rubric using a second LLM as the judge.

    Usage:
        from harness.clients import GeminiClient
        from harness.prompts import get_rubric

        judge_client = GeminiClient()
        evaluator = JudgeEvaluator(
            client=judge_client,
            rubric=get_rubric("relevance"),
            threshold=0.75,
        )
        result = evaluator.evaluate(
            response="<the LLM output being scored>",
            context={"user_input": "<the original prompt>"},
        )
    """

    def __init__(
        self,
        client: LLMClient,
        rubric: JudgeRubric,
        threshold: float = 0.75,
    ) -> None:
        # WHY inject the client: same dependency-injection pattern as everywhere
        # else. The evaluator doesn't care WHICH LLM is judging — only that the
        # client satisfies the LLMClient Protocol. Tests pass a mock; production
        # passes Gemini; future-you might pass Claude.
        self._client = client
        self._rubric = rubric
        self._threshold = threshold

    @property
    def name(self) -> str:
        # WHY include the rubric name: when an EvalResult is logged, you can
        # instantly tell which criterion it scored. "judge:relevance" is far
        # more useful in logs than just "judge".
        return f"judge:{self._rubric.name}"

    def evaluate(self, response: str, context: dict[str, Any] | None = None) -> EvalResult:
        """Score `response` against the rubric.

        Args:
            response: the LLM output being judged.
            context: must contain "user_input" — the prompt that produced
                     `response`. Required because rubrics evaluate the
                     response *in relation to* the user's question.

        Returns:
            EvalResult with normalized 0.0-1.0 score, judge rationale,
            and metadata containing the full judge interaction.
        """
        # ----- Step 1: extract required context -----
        user_input = (context or {}).get("user_input")
        if not user_input:
            # WHY ERROR not FAIL: missing context is a TEST SETUP problem, not
            # a problem with the response being judged. Important distinction.
            return EvalResult(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                score=0.0,
                rationale=(
                    "JudgeEvaluator requires context['user_input']. "
                    "Pass the original prompt that produced the response."
                ),
            )

        # ----- Step 2: build the judge prompt -----
        prompt = build_judge_prompt(self._rubric, user_input=user_input, response=response)

        # ----- Step 3: call the judge LLM -----
        # WHY temperature=0 here: judges should be as deterministic as possible.
        # Variance in the judge's scoring is noise on top of variance in the
        # system being tested. Minimize the noise we can control.
        try:
            llm_response = self._client.complete(
                LLMRequest(prompt=prompt, temperature=0.0, max_tokens=2048)
            )
        except Exception as exc:  # noqa: BLE001 — we want to catch ALL LLM-call failures
            # WHY ERROR not FAIL: the judge itself broke (timeout, rate limit,
            # network). The response under test might be fine; we just couldn't
            # evaluate it. This MUST be distinguishable from a real FAIL,
            # otherwise flaky judges look like real quality regressions.
            return EvalResult(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                score=0.0,
                rationale=f"Judge LLM call failed: {type(exc).__name__}: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )

        
        # ----- Step 4: parse the judge's response -----
        cleaned = self._strip_code_fences(llm_response.text)

        # WHY this check before JSON parse: Gemini 2.5's "thinking tokens"
        # can consume the max_tokens budget before the visible response is
        # complete, producing truncated JSON like '{"score": 4, "rationale": "'.
        # We surface this as a distinct, actionable error instead of letting
        # json.loads report a misleading "Unterminated string" message.
        finish_reason = (llm_response.finish_reason or "").upper()
        looks_truncated = finish_reason in {"MAX_TOKENS", "LENGTH"} or not cleaned.rstrip().endswith("}")
        if looks_truncated:
            return EvalResult(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                score=0.0,
                rationale=(
                    f"Judge response was truncated before completion "
                    f"(finish_reason={finish_reason or 'unknown'}). "
                    f"Increase max_tokens for the judge."
                ),
                metadata={
                    "judge_raw_output": llm_response.text,
                    "finish_reason": llm_response.finish_reason,
                },
            )

        try:
            parsed_dict = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return EvalResult(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                score=0.0,
                rationale=f"Judge returned invalid JSON: {exc.msg}",
                metadata={"judge_raw_output": llm_response.text},
            )

        try:
            judge = _JudgeResponse.model_validate(parsed_dict)
        except ValidationError as exc:
            return EvalResult(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                score=0.0,
                rationale=f"Judge JSON didn't match expected shape: {exc.errors()}",
                metadata={"judge_raw_output": llm_response.text},
            )

        # ----- Step 5: validate the score is within the rubric's range -----
        # WHY this check: even if the judge returned valid JSON with an integer,
        # it might have invented a score outside our rubric (e.g., returned 7
        # when our scale is 0-4). That's a judge contract violation.
        if judge.score > self._rubric.max_score:
            return EvalResult(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                score=0.0,
                rationale=(
                    f"Judge returned score {judge.score} outside rubric range "
                    f"(0-{self._rubric.max_score})."
                ),
                metadata={"judge_raw_output": llm_response.text},
            )

        # ----- Step 6: normalize the score and apply the threshold -----
        normalized = judge.score / self._rubric.max_score
        status = EvalStatus.PASS if normalized >= self._threshold else EvalStatus.FAIL

        return EvalResult(
            evaluator=self.name,
            status=status,
            score=round(normalized, 2),
            rationale=judge.rationale,
            threshold=self._threshold,
            metadata={
                # WHY include everything: the trace file (Day 3) will use this
                # to give us full reproducibility of the judge call.
                "rubric": self._rubric.name,
                "raw_judge_score": judge.score,
                "max_score": self._rubric.max_score,
                "judge_model": llm_response.model,
                "judge_provider": llm_response.provider,
                "judge_input_tokens": llm_response.input_tokens,
                "judge_output_tokens": llm_response.output_tokens,
                "judge_latency_ms": llm_response.latency_ms,
            },
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Same defensive strip as SchemaEvaluator — judges also use fences."""
        match = _JSON_FENCE_RE.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()