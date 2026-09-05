"""
Smoke tests — verifies both LLM clients can complete a basic request.

WHY marked 'integration': these call REAL APIs and use REAL tokens.
They should NOT run on every test invocation. Run explicitly:
    pytest -m integration

In CI we run them on a separate workflow with secrets configured.
"""

import pytest

from harness.clients import GeminiClient, GroqClient, LLMClient, LLMRequest


@pytest.mark.integration
@pytest.mark.smoke
def test_groq_completes_basic_request() -> None:
    """Groq client returns a non-empty response with token counts and latency."""
    with GroqClient() as client:
        response = client.complete(
            LLMRequest(prompt="Reply with exactly the word: PONG", max_tokens=256)
        )

    assert response.text.strip(), "expected non-empty text"
    assert response.provider == "groq"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0
    assert "PONG" in response.text.upper()


@pytest.mark.integration
@pytest.mark.smoke
def test_gemini_completes_basic_request() -> None:
    """Gemini client returns a non-empty response with token counts and latency."""
    with GeminiClient() as client:
        response = client.complete(
            LLMRequest(prompt="Reply with exactly the word: PONG", max_tokens=256)
        )

    assert response.text.strip()
    assert response.provider == "gemini"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0
    assert "PONG" in response.text.upper()


@pytest.mark.integration
def test_both_clients_satisfy_the_protocol() -> None:
    """Structural check — both implementations conform to LLMClient.

    WHY this test matters: if someone changes the Protocol later, this
    test will fail at the SAME instant the implementations diverge.
    It's a contract test, not a behavior test.
    """
    groq: LLMClient = GroqClient()
    gemini: LLMClient = GeminiClient()

    assert groq.provider_name == "groq"
    assert gemini.provider_name == "gemini"

    # WHY this loop: same code path against different providers.
    # Demonstrates the abstraction is real, not theoretical.
    for client in (groq, gemini):
        response = client.complete(LLMRequest(prompt="Say hi", max_tokens=256))
        assert response.text
        assert response.provider in {"groq", "gemini"}

@pytest.mark.integration
def test_real_judge_evaluates_real_llm_response() -> None:
    """End-to-end: Groq produces a response, Gemini judges it on relevance.

    THIS IS THE DEMO MOMENT. When you run this, you can watch the harness
    do AI-judging-AI for real, with real models, on a realistic task.
    """
    from harness.evaluators import EvalStatus, JudgeEvaluator
    from harness.prompts import get_rubric

    user_question = "How do I reset my password?"

    # ----- Step 1: get a response from Groq (system under test) -----
    with GroqClient() as groq:
        sut_response = groq.complete(
            LLMRequest(
                prompt=user_question,
                system=(
                    "You are a helpful customer support agent. "
                    "Answer the user's question directly and concisely."
                ),
                max_tokens=256,
            )
        )

    # ----- Step 2: Gemini judges Groq's response on relevance -----
    with GeminiClient() as gemini:
        evaluator = JudgeEvaluator(
            client=gemini,
            rubric=get_rubric("relevance"),
            threshold=0.75,
        )
        result = evaluator.evaluate(
            response=sut_response.text,
            context={"user_input": user_question},
        )

    # ----- Step 3: print the demo-worthy info -----
    # WHY print (not log): when we run this manually for the PM demo,
    # we WANT this on stdout so the demo audience sees what happened.
    print(f"\n{'=' * 70}")
    print(f"USER:        {user_question}")
    print(f"SUT (Groq):  {sut_response.text}")
    print(f"JUDGE:       {result}")
    print(f"{'=' * 70}\n")

    # We expect a competent model to produce a relevant answer.
    # If the test fails, either: (a) the model genuinely failed, OR
    # (b) the judge was too strict. Either is a teaching moment for the demo.
    assert result.status != EvalStatus.ERROR, (
        f"Judge itself failed: {result.rationale}"
    )
