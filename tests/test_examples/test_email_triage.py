"""
Customer email triage — realistic demo scenario.

THIS FILE IS THE PM DEMO.

What it shows:
  1. A realistic enterprise AI task (categorize + summarize a customer email)
  2. The system-under-test is an LLM prompted to produce structured JSON
  3. The harness tests it on TWO dimensions simultaneously:
       - SchemaEvaluator (deterministic) — is the JSON well-formed?
       - JudgeEvaluator (accuracy) — does the summary faithfully represent the email?
  4. Both results are printed for the demo audience to see, with full rationale.

WHY this example was chosen:
  - Structured output → showcases SchemaEvaluator
  - Subjective field (summary quality) → showcases JudgeEvaluator
  - Maps to MANY real enterprise AI uses cases: ticket routing, intent detection,
    document classification, content moderation. PM can immediately picture
    their own use case in this shape.
  - Failure modes are obvious and educational (wrong category, hallucinated
    fields, sycophantic summary).

WHY we don't fail on judge subjectivity:
  Judge gives a 2/4 doesn't mean "test is broken" — it means "model produced
  mediocre output." For a demo, that's information, not failure. The test
  fails only on schema (deterministic) issues. The judge result is printed
  for the audience.
"""

from typing import Literal

import pytest
from pydantic import BaseModel

from harness.clients import GeminiClient, GroqClient, LLMRequest
from harness.evaluators import EvalStatus, JudgeEvaluator, SchemaEvaluator
from harness.prompts import get_rubric
from harness.tracing import TraceWriter


# =============================================================================
# The contract : what shape we expect the LLM to produce
# =============================================================================


class TicketTriage(BaseModel):
    """Schema the LLM must conform to.

    WHY Literal types for category and urgency: enforces controlled vocabulary.
    Without this, the LLM might invent "kind-of-billing" or "fairly-urgent" 
    plausible but non-routable. Strict enums mean either the output is
    routable, or schema-eval fails fast.
    """

    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str  # one-sentence summary of the customer's issue


# =============================================================================
# The system-under-test prompt
# =============================================================================
# WHY inline (not factored into a module): premature abstraction. When we have
# 3+ example tasks, we'll extract a shared module. Today there's one — keep
# it visible.

_TRIAGE_SYSTEM_PROMPT = """\
You are a customer support triage system. Read the customer email and produce \
a JSON object with EXACTLY these fields:
- category: one of "billing", "technical", "account", "other"
- urgency: one of "low", "medium", "high"
- summary: a one-sentence summary of the customer's issue

Respond with ONLY the JSON object. No prose, no code fences, no explanation.\
"""


# =============================================================================
# The customer email (the test input)
# =============================================================================
# WHY a "realistic" email: PMs immediately recognize the shape. They can
# imagine swapping it for one of their own use cases.

_CUSTOMER_EMAIL = """\
Hi team,

I was charged $79 twice for my subscription this month. I only signed up for \
one plan. This is the second time this has happened — last month I had to \
contact support to get a refund and it took 3 weeks. I need this resolved \
TODAY because I have a board meeting tomorrow and I can't have my company \
card flagged again.

Thanks,
Priya\
"""


# =============================================================================
# The test
# =============================================================================


@pytest.mark.integration
def test_email_triage_with_schema_and_judge() -> None:
    """Run the SUT, evaluate with two evaluators, print everything.

    PM demo notes:
      - Run this with `pytest tests/test_examples/test_email_triage.py -v -s`
      - The -s flag shows all the print output (the demo content)
      - This single test demonstrates the full harness loop:
          SUT call → schema check → judge check → results
    """
    trace_writer = TraceWriter()
    test_name = "email_triage"

    # ----- Step 1: run the system-under-test (Groq) -----
    with GroqClient() as groq:
        sut_request = LLMRequest(
            prompt=_CUSTOMER_EMAIL,
            system=_TRIAGE_SYSTEM_PROMPT,
            max_tokens=256,
        )
        sut_response = groq.complete(sut_request)
    trace_writer.write_llm_call(test_name=test_name, request=sut_request, response=sut_response)

    # ----- Step 2: schema evaluation (deterministic, fast, free) -----
    schema_evaluator = SchemaEvaluator(schema=TicketTriage, name="schema:ticket")
    schema_result = schema_evaluator.evaluate(sut_response.text)
    trace_writer.write_evaluation(test_name=test_name, result=schema_result)

    # ----- Step 3: judge evaluation on accuracy (probabilistic, cheap, LLM call) -----
    with GeminiClient() as gemini:
        judge_evaluator = JudgeEvaluator(
            client=gemini,
            rubric=get_rubric("accuracy"),
            threshold=0.75,
        )
        # The judge needs the original email as context so it can score whether
        # the summary actually reflects the email content.
        judge_result = judge_evaluator.evaluate(
            response=sut_response.text,
            context={"user_input": _CUSTOMER_EMAIL},
        )
        trace_writer.write_evaluation(test_name=test_name, result=judge_result)

    # ----- Step 4: print the demo content -----
    # WHY all this print formatting: this output IS the demo. When the PM is
    # watching, this is what they'll read. Make it legible.
    print(f"\n{'=' * 70}")
    print("CUSTOMER EMAIL TRIAGE — Harness Demo")
    print(f"{'=' * 70}")
    print(f"\nINPUT EMAIL:\n{_CUSTOMER_EMAIL}")
    print(f"\nSUT RESPONSE ({sut_response.provider} / {sut_response.model}):")
    print(f"  {sut_response.text}")
    print(f"  [tokens: in={sut_response.input_tokens}, out={sut_response.output_tokens}, "
          f"latency={sut_response.latency_ms:.0f}ms]")
    print("\nEVALUATION RESULTS:")
    print(f"  {schema_result}")
    print(f"  {judge_result}")
    print(f"{'=' * 70}\n")
    print(f"\nTrace files written to: {trace_writer.trace_dir.resolve()}")

    # ----- Step 5: assertions -----
    # WHY assert on schema but not judge: schema is deterministic — if it fails,
    # there's a real bug in the SUT prompt. Judge is probabilistic — even a
    # 2/4 score is "model gave mediocre output," not a harness failure.
    assert schema_result.passed, (
        f"Schema check failed: {schema_result.rationale}\n"
        f"Raw SUT output: {sut_response.text}"
    )

    # WHY this is asserted: if the judge itself ERRORed (timeout, malformed
    # judge response), the harness is broken — that IS a real test failure.
    assert judge_result.status != EvalStatus.ERROR, (
        f"Judge errored: {judge_result.rationale}"
    )