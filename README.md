# AI Evaluation Harness

A pytest-based evaluation harness for testing LLM applications  provider-agnostic, multi-layer evaluation, with full audit traces.

## Why this exists

Testing AI systems is different from testing traditional software:

- **Outputs are non-deterministic.** Same prompt → varying responses. Exact-match assertions don't work.
- **Quality is multi-dimensional.** "Is the response good?" splits into relevance, accuracy, tone, schema compliance, safety : different evaluators for different concerns.
- **Failures need rationale.** A test that says "score: 0.4" is useless. A test that says "the response misclassified urgency as 'low' despite the user stating 'TODAY'" is actionable.

This harness handles all three.

## What it does

- **Provider-agnostic LLM client layer** : swap between Groq, Gemini, or any future provider with a one-line config change.
- **Two evaluator types out of the box:**
  - `SchemaEvaluator` : deterministic JSON / Pydantic validation. Fast, cheap, no LLM calls.
  - `JudgeEvaluator` : LLM-as-judge with structured rubrics (relevance, accuracy, tone). Catches quality issues that schema can't.
- **PASS / FAIL / ERROR distinction** : separates "response was bad" from "evaluator itself failed." Critical when judges are flaky.
- **Trace capture** : every LLM call and evaluation written as a JSON file. Full audit trail, drift-ready.
- **Two realistic examples** : positive case (good triage) and negative cases (sabotaged prompts the harness catches).

## See it work in 60 seconds

```bash
# Install
pip install -e ".[dev]"

# Set up API keys (free tiers : no credit card)
cp .env.example .env
# edit .env with your Groq + Gemini keys

# Run the positive case : Gemini judges Groq's output on a real triage task
pytest tests/test_examples/test_email_triage.py -v -s

# Run the negative cases : watch the harness catch deliberately bad output
pytest tests/test_examples/test_email_triage_negative.py -v -s

# All unit tests (no network)
pytest -m "not integration"
```

The integration tests print full demo output: the input, the SUT response, the evaluator results with rationale, and the path to the trace files.

## Example output

```
======================================================================
CUSTOMER EMAIL TRIAGE : Harness
======================================================================

INPUT EMAIL:
I was charged $79 twice for my subscription this month...

SUT RESPONSE (groq / llama-3.3-70b-versatile):
  {"category": "billing", "urgency": "high",
   "summary": "Customer was double-charged and needs immediate refund."}

EVALUATION RESULTS:
  [schema:ticket]  PASS  score=1.00  :: Valid TicketTriage
  [judge:accuracy] PASS  score=1.00  :: The response accurately captures
                                        the billing category, high urgency,
                                        and double-charge issue.

Trace files written to: ./traces/
```

And when the SUT is wrong:

```
[schema:ticket]  PASS  score=1.00
[judge:accuracy] FAIL  score=0.25  :: Misclassified urgency as 'low' despite
                                       the user explicitly stating 'TODAY',
                                       and provided an unspecific summary
                                       that fails to reflect the billing issue.
```

This is what AI testing looks like when the evaluator explains its reasoning.

## Architecture

```
src/harness/
├── clients/        Provider-agnostic LLM clients (Groq, Gemini, future: Ollama/Anthropic)
├── config/         Pydantic Settings : type-safe .env loading with SecretStr
├── evaluators/     Evaluator Protocol + Schema + Judge implementations
├── prompts/        Rubric registry (relevance, accuracy, tone)
└── tracing/        JSON trace writer for every LLM call and evaluation

tests/
├── test_*.py                 Unit tests (no network)
└── test_examples/            Integration tests on realistic scenarios
    ├── test_email_triage.py             Positive case
    └── test_email_triage_negative.py    Negative cases (harness catches failures)
```

Every layer is provider-agnostic and structured for composition. Adding a third LLM provider (Ollama, Anthropic) is one new file; adding a new evaluator type (similarity, groundedness) is one new file.

## Design principles

- **Deterministic before probabilistic.** Cheap checks (schema, format) run first; expensive checks (LLM judges) only on what passed. Cost discipline by design.
- **Provider-agnostic everything.** No test, evaluator, or prompt is coupled to a specific LLM vendor.
- **Calibrate, don't guess.** Judge thresholds are configurable (default 0.75 = "mostly meets criterion"). Production deployments should calibrate against labeled examples.
- **Traces from day one.** Every LLM call captured in a structured format. Drift detection and audit are additive, not retrofitted.

## What's next

Built so far: core abstractions, two evaluator types, trace capture, working examples.

Designed but not yet built : prioritized for the next phase:

- **Cost & latency dashboards** : aggregate token usage across runs
- **Retry with backoff** : handle rate limits and transient API failures
- **Semantic similarity evaluator** : embedding-based "is this close to expected"
- **Golden datasets + regression suite** : track quality over time
- **Calibration tooling** : measure judge agreement with human grading
- **Ollama as third provider** : fully-local fallback for offline/sensitive data
- **RAG evaluation** : groundedness and citation-faithfulness checks
- **Agent evaluation** : tool-call correctness, multi-turn coherence

The architecture supports all of these as additive modules : none require refactoring existing code.

## A note on the `HARNESS_INSECURE_SKIP_TLS_VERIFY` flag

If you run this in a corporate network with SSL inspection (Zscaler, Netskope, etc.), HTTPS calls to LLM providers may fail with `CERTIFICATE_VERIFY_FAILED` because the corporate root CA isn't in Python's trust store. As a development-only workaround, set:

```
HARNESS_INSECURE_SKIP_TLS_VERIFY=true
```

in `.env`. The harness emits a runtime warning whenever this flag is active. Never set this in production : the proper fix is to install the corporate CA via `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`.

## Stack

Python 3.11+ · pytest · Pydantic · httpx · Groq API · Google Gemini API · GitHub Actions
