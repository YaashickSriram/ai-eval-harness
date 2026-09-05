"""
Customer email triage — NEGATIVE cases.

WHY this file exists:
  Passing tests prove a system RUNS. Failing tests prove a system WORKS.
   These tests deliberately break
  the SUT to show the harness flagging the failures.

Two scenarios:
  1. Schema failure: SUT told to produce prose, not JSON.
     → SchemaEvaluator catches it deterministically.
  2. Judge failure: SUT produces valid JSON but with a misleading summary.
     → Schema PASSES, but JudgeEvaluator (accuracy) catches the deception.

WHY both: covers two different failure modes the harness should catch —
deterministic shape issues AND probabilistic quality issues. Each
evaluator type gets to demonstrate its value.
"""

from typing import Literal

import pytest
from pydantic import BaseModel

from harness.clients import GeminiClient, GroqClient, LLMRequest
from harness.evaluators import EvalStatus, JudgeEvaluator, SchemaEvaluator
from harness.prompts import get_rubric
from harness.tracing import TraceWriter


# Same schema as the positive test (deliberate — comparison is the point)
class TicketTriage(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str


# Same customer email as the positive test (deliberate — comparison again)
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
# Scenario 1: Sabotaged prompt → SCHEMA FAILURE
# =============================================================================

# WHY this prompt fails: it tells the SUT to write a friendly paragraph,
# which is exactly what a poorly-engineered prompt might do. The SUT obeys
# faithfully — and the schema evaluator catches that the output isn't JSON.
_BAD_SYSTEM_PROMPT_PROSE = """\
You are a friendly customer support assistant. When a customer email arrives, \
write a warm two-paragraph response acknowledging their concern and explaining \
next steps. Be empathetic and personal.\
"""


@pytest.mark.integration
def test_negative_schema_failure_when_sut_produces_prose() -> None:
    """When the SUT prompt is wrong, the harness catches it via SchemaEvaluator.

    DEMO MOMENT: 'Here's what happens when someone writes a bad prompt.
    The schema evaluator catches it immediately — no LLM judge needed, no
    waiting, no ambiguity. The harness tells you exactly what's wrong.'
    """
    trace_writer = TraceWriter()
    test_name = "email_triage_negative_schema"

    # ----- Run SUT with the SABOTAGED prompt -----
    with GroqClient() as groq:
        sut_request = LLMRequest(
            prompt=_CUSTOMER_EMAIL,
            system=_BAD_SYSTEM_PROMPT_PROSE,  # ← the sabotage
            max_tokens=256,
        )
        sut_response = groq.complete(sut_request)
    trace_writer.write_llm_call(test_name=test_name, request=sut_request, response=sut_response)

    # ----- Schema check should FAIL -----
    schema_evaluator = SchemaEvaluator(schema=TicketTriage, name="schema:ticket")
    schema_result = schema_evaluator.evaluate(sut_response.text)
    trace_writer.write_evaluation(test_name=test_name, result=schema_result)

    # ----- Demo print -----
    print(f"\n{'=' * 70}")
    print("NEGATIVE CASE 1 — Schema Failure (deliberately bad prompt)")
    print(f"{'=' * 70}")
    print(f"\nSABOTAGED SYSTEM PROMPT (note: tells SUT to write prose):")
    print(f"  {_BAD_SYSTEM_PROMPT_PROSE.strip()[:120]}...")
    print(f"\nSUT RESPONSE ({sut_response.provider}):")
    print(f"  {sut_response.text[:200]}{'...' if len(sut_response.text) > 200 else ''}")
    print("\nEVALUATION:")
    print(f"  {schema_result}")
    print(f"\nHarness verdict: caught the failure. Status = {schema_result.status.value.upper()}.")
    print(f"{'=' * 70}\n")

    # WHY assert FAIL (not the usual PASS): we EXPECT failure here.
    # If schema accidentally PASSED, our sabotage didn't work and the demo
    # would lose its punchline. This assertion guards the demo's integrity.
    assert schema_result.status == EvalStatus.FAIL, (
        f"Expected sabotaged prompt to fail schema check, but it passed. "
        f"SUT output: {sut_response.text}"
    )


# =============================================================================
# Scenario 2: Sneaky prompt → JUDGE FAILURE (schema would pass alone)
# =============================================================================

# WHY this prompt is sneaky: it produces VALID JSON matching the schema, but
# the summary is generic and ignores the specific complaint. Schema check
# alone would miss this — the LLM-as-judge catches the quality issue.
_SNEAKY_SYSTEM_PROMPT_GENERIC = """\
You are a customer support triage system. Read the customer email and produce \
a JSON object with EXACTLY these fields:
- category: always "other"
- urgency: always "low"
- summary: write a generic placeholder like "Customer reached out regarding their account."

Respond with ONLY the JSON object. No prose, no code fences.\
"""


@pytest.mark.integration
def test_negative_judge_failure_when_sut_produces_misleading_summary() -> None:
    """SUT produces VALID JSON but with a misleading summary. Schema passes;
    accuracy judge catches it.

    DEMO MOMENT: 'A schema check alone would say this is fine — valid JSON,
    all fields present. But the harness goes further. Gemini reads the
    actual email and the actual summary, and tells you the summary is
    nonsense. THIS is why probabilistic evaluation matters.'
    """
    trace_writer = TraceWriter()
    test_name = "email_triage_negative_judge"

    # ----- Run SUT with the SNEAKY prompt -----
    with GroqClient() as groq:
        sut_request = LLMRequest(
            prompt=_CUSTOMER_EMAIL,
            system=_SNEAKY_SYSTEM_PROMPT_GENERIC,  # ← the sneaky sabotage
            max_tokens=256,
        )
        sut_response = groq.complete(sut_request)
    trace_writer.write_llm_call(test_name=test_name, request=sut_request, response=sut_response)

    # ----- Schema check (should PASS — JSON is valid) -----
    schema_evaluator = SchemaEvaluator(schema=TicketTriage, name="schema:ticket")
    schema_result = schema_evaluator.evaluate(sut_response.text)
    trace_writer.write_evaluation(test_name=test_name, result=schema_result)

    # ----- Judge check (should FAIL — summary is misleading) -----
    with GeminiClient() as gemini:
        judge_evaluator = JudgeEvaluator(
            client=gemini,
            rubric=get_rubric("accuracy"),
            threshold=0.75,
        )
        judge_result = judge_evaluator.evaluate(
            response=sut_response.text,
            context={"user_input": _CUSTOMER_EMAIL},
        )
    trace_writer.write_evaluation(test_name=test_name, result=judge_result)

    # ----- Demo print -----
    print(f"\n{'=' * 70}")
    print("NEGATIVE CASE 2 — Judge Failure (valid JSON, misleading content)")
    print(f"{'=' * 70}")
    print("\nSNEAKY SYSTEM PROMPT (forces generic summaries regardless of email):")
    print("  Always category=other, urgency=low, generic summary.")
    print(f"\nSUT RESPONSE ({sut_response.provider}):")
    print(f"  {sut_response.text}")
    print("\nEVALUATION RESULTS:")
    print(f"  {schema_result}")
    print(f"  {judge_result}")
    print("\nHarness verdict:")
    print(f"  Schema: {schema_result.status.value.upper()} — JSON looks fine.")
    print(f"  Judge:  {judge_result.status.value.upper()} — but the content is wrong.")
    print("  This is exactly why we need BOTH layers of evaluation.")
    print(f"{'=' * 70}\n")

    # The interesting assertions: schema passes, judge fails.
    # This demonstrates the harness's two-layer protection.
    assert schema_result.status == EvalStatus.PASS, (
        "Expected schema to PASS (sneaky prompt produces valid JSON) — "
        "if this assertion fails, our sneaky prompt is too crude."
    )
    # WHY accept FAIL or ERROR: from the harness's perspective, both mean
    # "the judge did NOT approve the bad output." For the demo punchline,
    # what matters is that schema-alone would have rubber-stamped this
    # response — and the judge layer prevented that. ERROR (judge couldn't
    # complete) is acceptable because the harness STILL refuses to mark
    # the response as PASS without a confident judgment.
    assert judge_result.status != EvalStatus.PASS, (
        f"Expected judge to NOT PASS (summary doesn't match email), "
        f"but it passed. Judge rationale: {judge_result.rationale}"
    )