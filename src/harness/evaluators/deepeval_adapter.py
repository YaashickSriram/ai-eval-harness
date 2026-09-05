from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harness.evaluators.base import EvalResult, EvalStatus


class DeepEvalAdapter:
    """Adapts a DeepEval metric to the harness Evaluator Protocol."""

    def __init__(self, metric: Any, name: str, threshold: float = 0.7) -> None:
        # WHY accept the metric object rather than construct it here:
        # DeepEval metrics have wildly different constructor signatures.
        # Let the caller configure the metric; we only adapt the interface.
        self._metric = metric
        self._name = name
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"deepeval:{self._name}"

    def evaluate(self, response: str, context: dict[str, Any] | None = None) -> EvalResult:
        from deepeval.test_case import LLMTestCase

        ctx = context or {}
        user_input = ctx.get("user_input")
        if not user_input:
            return EvalResult(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                score=0.0,
                rationale="DeepEvalAdapter requires context['user_input'].",
            )

        test_case = LLMTestCase(
            input=user_input,
            actual_output=response,
            # Optional — only populated for RAG-style metrics
            retrieval_context=ctx.get("retrieved_contexts"),
            expected_output=ctx.get("expected_output"),
        )

        # WHY broad except: DeepEval can fail for many reasons (rate limits,
        # judge malformation, network). Same discipline as my own judge:
        # evaluator malfunction is ERROR, never a false PASS.
        try:
            self._measure_with_retry(test_case)
        except Exception as exc:  # noqa: BLE001
            return EvalResult(
                evaluator=self.name,
                status=EvalStatus.ERROR,
                score=0.0,
                rationale=f"DeepEval metric failed after retries: {type(exc).__name__}: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )

        score = float(self._metric.score or 0.0)
        return EvalResult(
            evaluator=self.name,
            status=EvalStatus.PASS if score >= self._threshold else EvalStatus.FAIL,
            score=round(score, 2),
            rationale=self._metric.reason or "DeepEval provided no rationale.",
            threshold=self._threshold,
            metadata={
                "deepeval_metric": type(self._metric).__name__,
                "deepeval_threshold": getattr(self._metric, "threshold", None),
            },
        )

    # WHY retry lives HERE and not inside DeepEval:
    #   DeepEval doesn't expose retry configuration for transient provider
    #   errors (503 rate limits, capacity spikes). Because I wrapped it in an
    #   adapter rather than calling it directly from tests, I can add
    #   resilience around a third-party library without forking it.
    #   This is the adapter pattern earning its keep.
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _measure_with_retry(self, test_case: Any) -> None:
        self._metric.measure(test_case)