"""
Judge rubric templates.

WHY this file is separate from evaluators/judge.py:
  Rubrics are CONTENT, not logic. Evaluator code stays stable; rubrics evolve
  as we learn what scoring criteria actually work. Keeping them in their own
  module means we can:
    - Add new rubrics without touching evaluator code
    - Version individual rubrics independently
    - Eventually move them to YAML files for non-engineers to edit
    - Diff "what did we ask the judge?" cleanly in code review

WHY rubrics matter so much:
  An LLM-as-judge is only as good as its rubric. A vague rubric ("rate the
  quality 1-10") produces noisy, inconsistent scores. A well-structured
  rubric (discrete levels, explicit criteria, required rationale) produces
  scores that are stable across runs and agree with human judgment.

  The single biggest lever in LLM-as-judge testing is the rubric design.
  Everything else is plumbing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeRubric:
    """A scoring rubric for one evaluation criterion.

    WHY frozen: rubrics shouldn't mutate at runtime. If you want a different
    rubric, you make a new one — never modify an existing one in place.
    This makes traces (Day 3) reproducible: you know exactly which rubric
    version produced which score.

    WHY a dataclass instead of just a string template:
      - Type-safe access: rubric.name, rubric.criterion, rubric.max_score
      - Forces every rubric to declare ALL fields (you can't accidentally
        forget the criterion description)
      - Makes the rubric inventory inspectable: list[JudgeRubric] is
        introspectable in tools, IDEs, and tests
    """

    name: str  # short id, e.g. "relevance" — used in EvalResult.evaluator
    criterion: str  # human-readable name, e.g. "Relevance"
    description: str  # what this criterion actually measures
    scale_definitions: dict[int, str]  # {0: "...", 1: "...", ...}
    # WHY max_score is computed: prevents drift between scale_definitions and
    # a separately-declared max. One source of truth.

    @property
    def max_score(self) -> int:
        """Top of the rubric's scale, derived from scale_definitions."""
        return max(self.scale_definitions.keys())


# =============================================================================
# Rubric inventory
# =============================================================================
# WHY all rubrics live in one module:
#   For now, a flat dict is the simplest registry. Later (Day 4+) we'll move
#   to YAML files and a real loader. The shape of the data won't change.
#
# WHY 0-4 scale (5 levels):
#   Research and practitioner consensus: 5-point scales hit a sweet spot.
#   - 2-3 levels: too coarse — every response either passes or fails, no signal
#     for "almost there"
#   - 6-10 levels: too noisy — judges can't distinguish a "7" from an "8"
#     consistently, and that variability undermines the score
#   - 5 levels: judges can reliably distinguish, AND we get partial-credit nuance
#
# WHY each level has a DESCRIPTION:
#   A bare "0/1/2/3/4" forces the judge to invent meaning, which differs across
#   calls. Explicit descriptions anchor the judge to a consistent definition.

_RELEVANCE = JudgeRubric(
    name="relevance",
    criterion="Relevance",
    description=(
        "Does the response directly address the user's question or request? "
        "Off-topic, tangential, or evasive responses score low."
    ),
    scale_definitions={
        0: "Off-topic; ignores the question entirely.",
        1: "Tangentially related; misses the main point.",
        2: "Partially addresses; touches the topic but missing key info.",
        3: "Mostly addresses; covers the main point with minor gaps.",
        4: "Fully addresses; directly and completely answers the question.",
    },
)

_ACCURACY = JudgeRubric(
    name="accuracy",
    criterion="Accuracy",
    description=(
        "Is the information in the response factually correct given the "
        "user's input and any provided context? Hallucinations, contradictions, "
        "and incorrect details score low."
    ),
    scale_definitions={
        0: "Mostly hallucinated or contradicts the context.",
        1: "Multiple factual errors or significant contradictions.",
        2: "Some factual issues; partial alignment with the source.",
        3: "Mostly accurate; minor factual issues or imprecision.",
        4: "Fully accurate; all claims supported by the context.",
    },
)

_TONE = JudgeRubric(
    name="tone",
    criterion="Tone",
    description=(
        "Is the tone appropriate for a customer support context? "
        "Professional, empathetic, and respectful. Dismissive, robotic, "
        "or overly casual responses score low."
    ),
    scale_definitions={
        0: "Dismissive, rude, or unprofessional.",
        1: "Cold or robotic; lacks empathy.",
        2: "Acceptable but bland; minimal warmth.",
        3: "Polite and professional; appropriate for the context.",
        4: "Warm, empathetic, and professional; exemplary tone.",
    },
)


# WHY a dict registry instead of just module-level constants:
#   - Dynamic lookup by name: get_rubric("relevance") — useful for config-driven
#     tests where the rubric name comes from a YAML file
#   - Iteration: `for name, rubric in RUBRICS.items()` — for tests that exercise
#     all rubrics at once
#   - Single import point: callers just import `get_rubric`, not individual rubrics
_RUBRICS: dict[str, JudgeRubric] = {
    _RELEVANCE.name: _RELEVANCE,
    _ACCURACY.name: _ACCURACY,
    _TONE.name: _TONE,
}


def get_rubric(name: str) -> JudgeRubric:
    """Look up a rubric by name. Raises KeyError for unknown names.

    WHY raise (not return None): callers who pass an unknown rubric name have
    a bug. Failing fast is friendlier than a silent None that crashes elsewhere.
    """
    if name not in _RUBRICS:
        available = ", ".join(sorted(_RUBRICS.keys()))
        raise KeyError(
            f"Unknown rubric: {name!r}. Available rubrics: {available}"
        )
    return _RUBRICS[name]


# =============================================================================
# Prompt builder
# =============================================================================
# WHY a function instead of a string template (f-string at call site):
#   1. Reusability — same builder logic for all rubrics, only the rubric data
#      varies.
#   2. Testability — we can unit-test the builder produces the right prompt
#      structure without ever calling an LLM.
#   3. Future-proofing — when we add few-shot examples, system instructions,
#      or chain-of-thought prompts (Day 4+), the change is here, in ONE place.
#   4. Schema enforcement — the JSON output requirement is baked into the
#      builder, can't be forgotten by mistake.

_PROMPT_TEMPLATE = """\
You are an expert evaluator scoring AI assistant responses on a specific criterion.

CRITERION: {criterion}
DEFINITION: {description}

SCORING SCALE (0-{max_score}):
{scale_lines}

CONTEXT:
The user asked: {user_input}

RESPONSE TO EVALUATE:
{response}

INSTRUCTIONS:
- Score the response on the criterion above.
- Be strict but fair. Default to a lower score when in doubt.
- Provide a one-sentence rationale explaining your score.

Respond with ONLY valid JSON in this exact format:
{{"score": <integer 0-{max_score}>, "rationale": "<one sentence>"}}

Do not include any text outside the JSON object. Do not use markdown code fences.
"""


def build_judge_prompt(
    rubric: JudgeRubric,
    user_input: str,
    response: str,
) -> str:
    """Construct the full prompt sent to the judge LLM.

    Args:
        rubric: which criterion to score on.
        user_input: what the user originally asked / the prompt that produced
                    the response.
        response: the LLM output being judged.

    Returns:
        Plain prompt string ready to send via LLMClient.complete().

    WHY this function is testable in isolation:
      It's pure — no LLM call, no I/O, no global state. Given the same inputs,
      it always returns the same string. That makes it trivially unit-testable
      AND means traces of LLM calls can include the exact prompt sent.
    """
    scale_lines = "\n".join(
        f"  {score} = {definition}"
        for score, definition in sorted(rubric.scale_definitions.items())
    )

    return _PROMPT_TEMPLATE.format(
        criterion=rubric.criterion,
        description=rubric.description,
        max_score=rubric.max_score,
        scale_lines=scale_lines,
        user_input=user_input,
        response=response,
    )