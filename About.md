# AI Evaluation Harness — Complete Knowledge Transfer

**Purpose of this document:** everything you need to defend this project in a technical interview, explain any line of it, and credibly discuss extending it to RAG, agents, and industry tooling.

---

# PART 1 — WHY THIS PROJECT EXISTS

## The three-layer "why"

There are three levels at which you can answer "why did you build this?" Use the level that matches the question.

### Level 1 — The 10-second version (for recruiters, opening questions)

> "Traditional test automation asserts equality. AI outputs are non-deterministic, so equality assertions don't work. This harness evaluates AI outputs on multiple quality dimensions — format correctness deterministically, content quality via an LLM judge — with full trace capture for every call."

### Level 2 — The engineering version (for the technical manager)

> "There are three problems that make AI testing structurally different from traditional QA:
>
> **Non-determinism** — the same prompt produces different outputs. You can't assert equality. You need scoring against thresholds.
>
> **Multi-dimensionality** — 'is this output good?' decomposes into format compliance, factual accuracy, relevance, tone, groundedness, safety. Each needs a different evaluation method. Some are deterministic, some require semantic judgment.
>
> **Explainability** — a failing traditional test tells you exactly what differed. A failing AI test that says 'score: 0.4' tells you nothing actionable. You need rationale, not just a number.
>
> This harness addresses all three: a pluggable evaluator system, normalized scoring with configurable thresholds, and rationale captured on every result."

### Level 3 — The architectural version (when they probe deeper)

> "The design constraint I set was: the harness must be agnostic across four axes — LLM provider, use case, evaluation method, and domain. That constraint drove every decision.
>
> Provider-agnostic meant a `Protocol`-based client abstraction — no test, evaluator, or prompt imports a vendor SDK. Use-case-agnostic meant schemas and rubrics are *data*, not code. Evaluator-agnostic meant a single `Evaluator` contract that deterministic checks, semantic checks, and LLM judges all satisfy. Domain-agnostic falls out of the first three.
>
> The result is that onboarding a new AI system — RAG, agentic, content generation — requires writing a schema, choosing rubrics, and composing evaluators. Zero changes to core code."

---

## What problem this solves in *your* career narrative

Your current experience is manual AI validation plus UI/API/DB automation. This POC is the bridge artifact. The story:

> "In my current role I validate AI-driven systems — but manually. I look at outputs, judge them against expectations, and log defects. It doesn't scale, it isn't repeatable, and it produces no historical quality signal.
>
> I built this harness to answer: *what would it take to convert that manual judgment into an automated, auditable system?* The answer turned out to be an evaluator abstraction plus LLM-as-judge plus trace capture. That's what this POC is."

That framing does three things: it's honest, it shows initiative, and it directly maps your existing skill to the new one.

---

# PART 2 — ARCHITECTURE & END-TO-END FLOW

## Package structure

```
src/harness/
├── config/
│   └── settings.py          Pydantic Settings — type-safe .env loading
├── clients/
│   ├── base.py              LLMClient Protocol + LLMRequest/LLMResponse
│   ├── groq_client.py       Groq (OpenAI-compatible REST)
│   └── gemini_client.py     Gemini (Google REST — different wire format)
├── evaluators/
│   ├── base.py              Evaluator Protocol + EvalResult + EvalStatus
│   ├── schema.py            Deterministic — Pydantic validation
│   └── judge.py             Probabilistic — LLM-as-judge
├── prompts/
│   └── judge_prompts.py     JudgeRubric registry + prompt builder
└── tracing/
    └── writer.py            JSON trace persistence

tests/
├── test_protocol.py         Unit: request/response contracts
├── test_evaluator_base.py   Unit: EvalResult contracts
├── test_schema_evaluator.py Unit: schema evaluator (8 cases)
├── test_judge_evaluator.py  Unit: judge with MOCKED LLM (11 cases)
├── test_trace_writer.py     Unit: trace writer with tmp_path (7 cases)
├── test_smoke.py            Integration: real provider calls
└── test_examples/
    ├── test_email_triage.py           Positive scenario
    └── test_email_triage_negative.py  Two failure-detection scenarios
```

**Dependency direction:** `config ← clients ← evaluators ← tests`, with `prompts` and `tracing` as leaf utilities. No cycles. Everything depends on config; nothing depends on tests.

---

## End-to-end execution flow

What actually happens, in order, when you run:

```bash
pytest tests/test_examples/test_email_triage.py -v -s
```

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. PYTEST COLLECTION                                                 │
│    Reads pyproject.toml [tool.pytest.ini_options]                   │
│    → testpaths, markers, addopts                                     │
│    Discovers test_email_triage_with_schema_and_judge                 │
│    -s flag disables stdout capture (demo output visible)             │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. IMPORT CHAIN → CONFIG LOADS (happens once, at import time)        │
│    test file imports harness.clients                                 │
│      → clients/__init__.py imports base, groq_client, gemini_client │
│        → groq_client imports harness.config.settings                 │
│          → Pydantic BaseSettings reads .env                          │
│          → validates types, wraps keys in SecretStr                  │
│          → missing required field = ValidationError HERE, not later  │
│          → module-level `settings` singleton created                 │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. TEST BODY BEGINS                                                  │
│    trace_writer = TraceWriter()                                      │
│      → reads settings.harness_trace_dir                              │
│      → mkdir(parents=True, exist_ok=True)                            │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 4. BUILD SUT REQUEST                                                 │
│    sut_request = LLMRequest(prompt=email, system=triage_instructions,│
│                             max_tokens=256)                          │
│    → frozen dataclass; immutable from here on                        │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 5. SUT INVOCATION — GroqClient                                       │
│    __init__:  resolve api_key + model from settings                  │
│               check harness_insecure_skip_tls_verify                 │
│               emit UserWarning if TLS verification disabled          │
│               create long-lived httpx.Client (connection reuse)      │
│    complete(): build OpenAI-shape payload                            │
│                  {model, messages:[{role,content}...], temp, max}    │
│                perf_counter() ─┐                                     │
│                POST api.groq.com/openai/v1/chat/completions          │
│                perf_counter() ─┘ → latency_ms                        │
│                raise_for_status()  → fail loud on HTTP error         │
│                extract text, usage.prompt_tokens,                    │
│                        usage.completion_tokens, finish_reason        │
│                return frozen LLMResponse                             │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 6. TRACE THE SUT CALL                                                │
│    write_llm_call(test_name, request, response)                      │
│      → payload dict {trace_type, timestamp, request{}, response{}}   │
│      → filename: {ISO_ts}_{test}_llm_{provider}_{uuid6}.json         │
│      → json.dumps(indent=2, ensure_ascii=False)                      │
│      → write_text(encoding="utf-8")   ← Windows cp1252 guard         │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 7. EVALUATOR 1 — SchemaEvaluator  (deterministic, ~1ms, free)       │
│    _strip_code_fences()   regex removes ```json ... ``` wrappers    │
│    json.loads()           → JSONDecodeError ⇒ FAIL score 0.0        │
│    TicketTriage.model_validate()                                     │
│                           → ValidationError ⇒ FAIL + partial credit │
│                             partial = (total_fields − failed) / total│
│                             rationale = "field: message; field: msg" │
│                           → success ⇒ PASS score 1.0                 │
│    returns frozen EvalResult                                         │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 8. TRACE THE SCHEMA RESULT                                           │
│    write_evaluation() → _sanitize(metadata) recursively coerces     │
│      dataclasses / enums / Pydantic models / exceptions → JSON-safe  │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 9. EVALUATOR 2 — JudgeEvaluator  (probabilistic, ~2-4s, LLM call)   │
│    ┌ guard: context["user_input"] present?                           │
│    │        missing ⇒ ERROR (test setup bug, not model failure)      │
│    ├ build_judge_prompt(rubric, user_input, response)                │
│    │        pure function → deterministic prompt string              │
│    ├ GeminiClient.complete(temperature=0.0, max_tokens=2048)         │
│    │        exception ⇒ ERROR (judge broke, verdict unknown)         │
│    ├ truncation guard:                                               │
│    │        finish_reason in {MAX_TOKENS, LENGTH}                    │
│    │        OR text doesn't end with "}"                             │
│    │        ⇒ ERROR with actionable message                          │
│    ├ json.loads() ⇒ ERROR on malformed                               │
│    ├ _JudgeResponse.model_validate()  ← META-TESTING                 │
│    │        judge's own output is Pydantic-validated                 │
│    ├ range guard: score > rubric.max_score ⇒ ERROR                   │
│    ├ normalize: score / max_score  →  0.0–1.0                        │
│    └ threshold: normalized >= 0.75 ? PASS : FAIL                     │
│    returns EvalResult with full judge metadata                       │
│      (rubric, raw score, judge model, tokens, latency)               │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 10. TRACE THE JUDGE RESULT → third JSON file                        │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 11. REPORT — formatted print() block                                 │
│     input / SUT response / tokens+latency / both EvalResults        │
│     EvalResult.__str__ gives one-line summary with rationale         │
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 12. ASSERTIONS                                                       │
│     assert schema_result.passed      ← deterministic, strict         │
│     assert judge.status != ERROR     ← tolerate FAIL, not malfunction│
└────────────────────────────┬─────────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 13. PYTEST AGGREGATION → exit code → CI green/red                    │
└──────────────────────────────────────────────────────────────────────┘
```

**Three artifacts produced per run:** one LLM-call trace, two evaluation traces. All timestamped, all JSON, all on disk.

---

# PART 3 — FILE-BY-FILE LOGIC

## `config/settings.py` — infra layer

**Chunk 1 — The class declaration**
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )
```
`BaseSettings` (from pydantic-settings) reads env vars and `.env` automatically. `case_sensitive=False` means `GROQ_API_KEY` and `groq_api_key` both resolve. `extra="ignore"` means unrecognized env vars don't crash the app.

**Chunk 2 — SecretStr for credentials**
```python
groq_api_key: SecretStr = Field(..., description="Groq API key")
```
`SecretStr` is the important choice. `print(settings.groq_api_key)` outputs `**********`. You must explicitly call `.get_secret_value()` to obtain the real string. This makes accidental key logging structurally difficult rather than merely discouraged.

*Interview point:* "Secret handling is a design property, not a discipline. `SecretStr` makes leaking a key require an explicit, greppable method call."

**Chunk 3 — The singleton**
```python
settings = Settings()  # type: ignore[call-arg]
```
Module-level instantiation means: constructed once on first import, validated at import time, shared everywhere. Missing required env var surfaces immediately, not five LLM calls into a test run.

**If this file didn't exist:** scattered `os.getenv()` calls, no type validation, no fail-fast, keys appearing in logs.

---

## `clients/base.py` — framework layer (most important file)

**Chunk 1 — LLMRequest**
```python
@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    system: str | None = None
    stop: list[str] = field(default_factory=list)
```
`frozen=True` = immutable. Once a request is constructed it cannot be mutated, so the trace record is guaranteed faithful to what was actually sent.

`temperature: float = 0.0` as the *default* is a deliberate testing choice — determinism-first, override when you want variance.

`model: str | None = None` means "use the client's configured default" — lets tests omit the model unless they specifically care.

**Chunk 2 — LLMResponse**
```python
@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None
```
Every field exists for a reason:
- `model` — the *resolved* model, so traces show what actually ran, not what was requested
- `provider` — enables provider-specific analysis across traces
- `input_tokens`/`output_tokens` — cost calculation without re-instrumenting call sites
- `latency_ms` — SLA tracking
- `finish_reason` — truncation detection (this field is what caught the Gemini thinking-token bug)
- `raw` — full original payload preserved for debugging

*Interview point:* "I capture observability data at the boundary. If I'd only captured `text`, adding cost tracking later would mean touching every call site."

**Chunk 3 — The Protocol**
```python
@runtime_checkable
class LLMClient(Protocol):
    @property
    def provider_name(self) -> str: ...
    def complete(self, request: LLMRequest) -> LLMResponse: ...
```

Why `Protocol` and not `ABC`:
- **Structural typing** — anything with those two members *is* an `LLMClient`. No inheritance, no import coupling.
- **Trivial mocking** — the `FakeLLMClient` in the unit tests is a plain dataclass. It doesn't import or subclass anything from the framework.
- **Third-party wrapping** — you can make an external library conform without controlling its class hierarchy.

*Interview point:* "Protocol is Python's answer to interface segregation without inheritance. It's why my judge unit tests never touch the network — the mock satisfies the contract structurally."

---

## `clients/groq_client.py` and `gemini_client.py` — framework layer

These two files exist to prove the abstraction is real. **They have almost nothing in common internally.**

| Concern | Groq | Gemini |
|---|---|---|
| Endpoint | `/openai/v1/chat/completions` | `/v1beta/models/{model}:generateContent` |
| Model location | request body | URL path |
| Auth | `Authorization: Bearer` header | `?key=` query param |
| Messages | `messages: [{role, content}]` | `contents: [{role, parts: [{text}]}]` |
| System prompt | a message with `role: "system"` | top-level `systemInstruction` field |
| Assistant role name | `"assistant"` | `"model"` |
| Token counts | `usage.prompt_tokens` | `usageMetadata.promptTokenCount` |
| Response text | `choices[0].message.content` | `candidates[0].content.parts[0].text` |

**Chunk — the TLS workaround (both clients)**
```python
verify_tls = not settings.harness_insecure_skip_tls_verify
if not verify_tls:
    warnings.warn("TLS verification is DISABLED for X client...",
                  category=UserWarning, stacklevel=2)
self._http = httpx.Client(timeout=60.0, verify=verify_tls)
```
Corporate SSL-inspection proxies present self-signed certs that Python's bundled CA store rejects. The flag is: opt-in, defaults to `False`, named with `INSECURE` in it, and emits a warning every single time it's active.

*Interview point:* "This is a documented, loud, reversible security exception — not a silent hack. When the corporate CA arrives it's a one-line `.env` change, no code change."

**Chunk — defensive Gemini parsing**
```python
parts = content.get("parts")
if not parts:
    raise RuntimeError(
        f"Gemini returned no content parts. finishReason={finish_reason!r}. "
        f"This usually means: (a) maxOutputTokens too small "
        f"(Gemini 2.5 'thinking' tokens consume budget), or "
        f"(b) safety filters blocked the response."
    )
```
This is a real bug I hit. Gemini 2.5 performs internal reasoning that consumes `maxOutputTokens` but produces no visible text. With `max_tokens=10`, all budget went to thinking, `parts` was absent, and the naive `candidate["content"]["parts"][0]["text"]` threw a bare `KeyError`.

*Interview point:* "Errors are documentation. `KeyError: 'parts'` tells you where it broke. This message tells you why and what to do."

---

## `evaluators/base.py` — evaluator layer (core IP)

**Chunk 1 — Three-state enum**
```python
class EvalStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
```
This is the single most defensible design decision in the project.

- `PASS` — output met the criterion
- `FAIL` — output was substantively wrong
- `ERROR` — **the evaluator itself could not produce a verdict**

Collapsing ERROR into FAIL is the common mistake. Do that and a rate-limited judge looks identical to a quality regression. You cannot distinguish "the model is bad" from "my test rig broke."

`StrEnum` (Python 3.11+) means members serialize as `"pass"` not `"EvalStatus.PASS"` — matters for JSON traces.

**Chunk 2 — EvalResult**
```python
@dataclass(frozen=True)
class EvalResult:
    evaluator: str
    status: EvalStatus
    score: float          # ALWAYS normalized 0.0–1.0
    rationale: str
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

The normalization rule matters: a judge's 3-of-4 becomes 0.75, schema partial credit becomes 0.67. Every evaluator reports on the same scale, so scores are comparable across evaluator types, across runs, and over time. That comparability is what makes drift detection possible later.

`rationale` is mandatory, not optional. A score without a reason isn't actionable.

`metadata` is deliberately loose (`dict[str, Any]`) so each evaluator can attach what it needs without requiring an `EvalResult` subclass per evaluator type.

**Chunk 3 — The Evaluator Protocol**
```python
@runtime_checkable
class Evaluator(Protocol):
    @property
    def name(self) -> str: ...
    def evaluate(self, response: str,
                 context: dict[str, Any] | None = None) -> EvalResult: ...
```
The contract that makes everything else pluggable. Notice the documented behavioural rules:
- Never raise on *evaluation* failure — return `FAIL`
- Only return `ERROR` if the evaluator itself couldn't run
- Always populate `rationale`

The loose `context` dict is what lets one signature serve schema checks (needs nothing), judges (need `user_input`), and future groundedness evaluators (need `retrieved_contexts`).

---

## `evaluators/schema.py` — deterministic evaluator

**Chunk 1 — Code-fence stripping**
```python
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
```
LLMs wrap JSON in markdown fences frequently even when told not to. Rejecting valid JSON for a cosmetic wrapper would be a false failure.

**Chunk 2 — Failure ladder**
```python
try:
    parsed = json.loads(cleaned)
except json.JSONDecodeError as exc:
    return EvalResult(status=FAIL, score=0.0,
                      rationale=f"Response is not valid JSON: {exc.msg} (at line {exc.lineno})")
```
`FAIL` not `ERROR` — the model produced bad output; that's a legitimate test failure, not a harness malfunction.

**Chunk 3 — Partial credit**
```python
failed_fields = len({err["loc"][0] for err in exc.errors() if err["loc"]})
passed_fields = max(0, total_fields - failed_fields)
return round(passed_fields / total_fields, 2)
```
Binary scoring loses signal. 0.67 ("two of three fields correct") vs 0.0 ("completely broken") is diagnostically different information, and it matters when comparing prompt variants or model versions.

**Chunk 4 — Per-field rationale**
```python
parts.append(f"{loc}: {err['msg']}")
```
Pydantic's default `str(ValidationError)` is multi-line and noisy. Collapsing to `"summary: Field required; urgency: Input should be 'low','medium','high'"` gives a one-line rationale while the full structured errors stay in `metadata`.

---

## `evaluators/judge.py` — probabilistic evaluator (the centrepiece)

**Chunk 1 — Meta-testing the judge**
```python
class _JudgeResponse(BaseModel):
    score: int = Field(..., ge=0, le=10)
    rationale: str = Field(..., min_length=1)
```
The judge's own output is Pydantic-validated before being trusted. This is the answer to "who validates the validator?" — at minimum, structurally, we do.

**Chunk 2 — Dependency injection**
```python
def __init__(self, client: LLMClient, rubric: JudgeRubric, threshold: float = 0.75):
```
The judge accepts *any* `LLMClient`. Consequences: unit tests inject a mock (no network, no cost, no flakiness); production injects Gemini; you can A/B two judge models by changing one argument.

**Chunk 3 — Guard sequence** (this is the part to walk through slowly in an interview)

```
1. context["user_input"] missing            → ERROR  (test setup bug)
2. client.complete() raises                 → ERROR  (judge unreachable)
3. finish_reason ∈ {MAX_TOKENS, LENGTH}
   OR text doesn't end with "}"             → ERROR  (truncated)
4. json.loads fails                         → ERROR  (malformed)
5. _JudgeResponse validation fails          → ERROR  (wrong shape)
6. score > rubric.max_score                 → ERROR  (out of contract)
7. normalized >= threshold                  → PASS
8. otherwise                                → FAIL
```

**Every path that isn't a confident verdict returns ERROR, never PASS.** The system fails safe. A flaky judge never produces a false green.

**Chunk 4 — Temperature zero**
```python
LLMRequest(prompt=prompt, temperature=0.0, max_tokens=2048)
```
The system under test may be non-deterministic by nature. The *judge* shouldn't add more variance on top. Minimise the noise you control.

**Chunk 5 — Normalization**
```python
normalized = judge.score / self._rubric.max_score
status = PASS if normalized >= self._threshold else FAIL
```
Judge returns 0–4, `EvalResult.score` is 0.0–1.0. Threshold default 0.75 means "judge must give at least 3 of 4" — "mostly meets the criterion."

*Interview point:* "0.75 is a starting point, not a calibrated value. Calibrating it properly means labelling 30–50 examples by hand, measuring judge-human agreement, and choosing the threshold where precision and recall balance for your risk tolerance. That's the next-phase work, and the traces I capture are exactly the dataset you'd use."

---

## `prompts/judge_prompts.py` — hybrid business/framework layer

**Chunk 1 — Rubric as structured data**
```python
@dataclass(frozen=True)
class JudgeRubric:
    name: str
    criterion: str
    description: str
    scale_definitions: dict[int, str]

    @property
    def max_score(self) -> int:
        return max(self.scale_definitions.keys())
```
`max_score` is *derived*, not declared separately — one source of truth, no drift between the scale and the stated maximum.

**Chunk 2 — Why 5 levels (0–4)**

This is the highest-value thing to be able to explain:

- **2–3 levels** — too coarse. No signal for "almost there," no partial credit.
- **6–10 levels** — too noisy. Judges can't reliably distinguish 7 from 8; that variance is pure noise.
- **5 levels** — judges distinguish reliably AND you retain nuance.

And each level has an explicit description. A bare "score 0–4" forces the judge to invent its own meaning per call, which is exactly the inconsistency you're trying to eliminate.

**Chunk 3 — The prompt builder is a pure function**
```python
def build_judge_prompt(rubric, user_input, response) -> str:
```
No I/O, no global state, no LLM call. Same inputs → same output. Consequences: unit-testable without mocking anything; the exact prompt is reproducible from trace data.

**Chunk 4 — Registry pattern**
```python
def get_rubric(name: str) -> JudgeRubric:
    if name not in _RUBRICS:
        raise KeyError(f"Unknown rubric: {name!r}. Available: {available}")
```
Raise, don't return `None`. A caller passing an unknown rubric has a bug; failing fast with the available options listed is friendlier than a `None` that crashes three frames later.

---

## `tracing/writer.py` — observability layer

**Chunk 1 — Recursive sanitizer**
```python
def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)): return value
    if isinstance(value, Enum): return value.value
    if isinstance(value, dict): return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_sanitize(v) for v in value]
    if hasattr(value, "model_dump"): return _sanitize(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type): return _sanitize(asdict(value))
    if isinstance(value, BaseException): return f"{type(value).__name__}: {value}"
    return str(value)
```
`metadata` can contain Pydantic models, enums, exceptions, dataclasses — none of which `json.dumps` handles natively. One recursive coercion point instead of defensive checks at every call site. Final `str()` fallback means it's lossy but never crashes.

**Chunk 2 — Collision-safe filenames**
```python
filename = f"{_filesystem_timestamp()}_{prefix}_{uuid.uuid4().hex[:6]}.json"
```
Second-resolution timestamps collide when multiple traces write within the same second. A 6-char uuid suffix eliminates that without needing a lock or a counter.

**Chunk 3 — Filesystem-safe naming**
```python
_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")
```
Evaluator names like `judge:relevance` contain a colon — illegal in Windows filenames. Normalized to `judge_relevance`.

**Chunk 4 — Explicit encoding**
```python
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
```
Windows defaults to cp1252, which raises on non-ASCII. Any customer email containing an em-dash, accented character, or emoji would crash without the explicit `encoding="utf-8"`. `ensure_ascii=False` keeps the JSON human-readable rather than escaping everything to `\uXXXX`.

---

## `tests/test_judge_evaluator.py` — the mocking pattern

**The FakeLLMClient**
```python
@dataclass
class FakeLLMClient:
    canned_text: str
    raise_on_call: Exception | None = None
    provider_name: str = "fake"
    _last_request: LLMRequest | None = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._last_request = request
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return LLMResponse(text=self.canned_text, ...)
```

Three things to point out in an interview:

1. **No inheritance.** It satisfies `LLMClient` structurally. This only works because the contract is a `Protocol`.
2. **It records the request.** `test_judge_receives_well_formed_prompt` asserts the prompt actually contains the rubric criterion and the response text — a *contract test* between the evaluator and the prompt builder.
3. **It can simulate failure.** `raise_on_call=TimeoutError(...)` lets you unit-test the ERROR path without waiting for a real timeout.

11 unit tests cover every guard in the judge's ladder. Zero network calls. Runs in milliseconds. Costs nothing in CI.

---

# PART 4 — EXTENDING TO OTHER AI SYSTEM TYPES

This is the "is it actually general-purpose?" question. Here's the concrete answer for each.

## RAG systems

**What's needed:** a groundedness/faithfulness evaluator that checks whether the generated answer is supported by the retrieved documents.

**What already exists:** the `Evaluator` Protocol, the `context` dict, the judge machinery, the rubric registry.

**Minimal addition:** one new rubric (`grounding`) plus a variant of `JudgeEvaluator` that formats retrieved chunks into the prompt.

```python
result = groundedness_evaluator.evaluate(
    response=generated_answer,
    context={
        "user_input": question,
        "retrieved_contexts": [chunk1, chunk2, chunk3],
    },
)
```

**Additional RAG dimensions worth naming in an interview:**
- *Context precision* — are the retrieved chunks relevant? (retrieval quality, separate from generation)
- *Context recall* — did retrieval find everything needed?
- *Answer relevance* — does the answer address the question?
- *Faithfulness* — are all claims traceable to context?

Point to make: **RAG failures split into retrieval failures and generation failures, and you must measure them separately.** If the answer is wrong because retrieval returned nothing useful, fixing the generation prompt won't help. Most people conflate them.

## Agentic systems

**What's needed:** tool-call correctness validation.

**What already exists:** `SchemaEvaluator` — a tool call *is* structured output. Validating `{"tool": "search_invoices", "args": {...}}` against a Pydantic model is exactly what the schema evaluator already does.

**Minimal addition:**
```python
class ToolCall(BaseModel):
    tool: Literal["search_invoices", "create_ticket", "escalate"]
    arguments: dict[str, Any]

ToolCallEvaluator = SchemaEvaluator(schema=ToolCall, name="schema:tool_call")
```

Plus a judge rubric for *tool selection appropriateness* — "given the user's request, was this the right tool?" That's a semantic question, so it goes to the judge.

**Agent dimensions to name:**
- Tool selection correctness (right tool for the intent)
- Argument correctness (well-formed and semantically right)
- Trajectory efficiency (did it take 3 steps or 15?)
- Termination behaviour (does it stop, or loop?)
- Multi-turn coherence (does turn 5 contradict turn 2?)

## Content generation systems

**What's needed:** tone, brand voice, factuality, safety.

**What already exists:** the `tone` rubric is already written in the registry — it's just not wired into a demo test yet.

**Minimal addition:** compose existing evaluators differently. Content generation needs relevance + tone + safety rather than schema + accuracy. Same primitives, different composition.

## The general pattern

For **any** new AI system type, the work is:
1. Define the expected output shape (Pydantic model) — if structured
2. Choose or write rubrics for the quality dimensions that matter
3. Compose evaluators in a test
4. Core harness: **unchanged**

---

# PART 5 — INTEGRATING INDUSTRY TOOLING

The JD names Langfuse, LangSmith, DeepEval, RAGAS. Here is a credible, specific answer for each.

## The framing that works

> "I built these primitives from scratch rather than adopting a tool first — deliberately, because I wanted to understand the mechanics before picking an abstraction. That means I can now integrate any of these tools as adapters, and I know exactly what each one is doing internally."

## DeepEval — highest-value, easiest integration

DeepEval is a pytest-native LLM evaluation library. Its `GEval` metric is conceptually the same thing as my `JudgeEvaluator`: LLM-as-judge with a criteria-based rubric.

**Integration approach — adapter conforming to my `Evaluator` Protocol:**

```python
class DeepEvalAdapter:
    """Wraps any DeepEval metric to satisfy the Evaluator Protocol."""

    def __init__(self, metric, name: str, threshold: float = 0.7):
        self._metric = metric
        self._name = name
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"deepeval:{self._name}"

    def evaluate(self, response, context=None):
        from deepeval.test_case import LLMTestCase
        ctx = context or {}
        test_case = LLMTestCase(
            input=ctx.get("user_input", ""),
            actual_output=response,
            retrieval_context=ctx.get("retrieved_contexts"),
        )
        try:
            self._metric.measure(test_case)
        except Exception as exc:
            return EvalResult(evaluator=self.name, status=EvalStatus.ERROR,
                              score=0.0, rationale=f"DeepEval metric failed: {exc}")
        return EvalResult(
            evaluator=self.name,
            status=EvalStatus.PASS if self._metric.score >= self._threshold else EvalStatus.FAIL,
            score=round(self._metric.score, 2),
            rationale=self._metric.reason or "no rationale provided",
            threshold=self._threshold,
            metadata={"deepeval_metric": type(self._metric).__name__},
        )
```

**Effort:** ~50 lines. **Zero changes to core.** DeepEval results flow into the same traces, same reporting, same assertions.

Metrics worth naming: `AnswerRelevancyMetric`, `FaithfulnessMetric`, `HallucinationMetric`, `ContextualPrecisionMetric`, `GEval` (custom criteria).

## RAGAS — RAG-specific evaluation

RAGAS operates on datasets rather than single examples, and returns a dict of metric scores. Slightly different shape, still adaptable.

**Integration approach:**
```python
class RagasEvaluator:
    """Wraps a RAGAS metric. Requires retrieved_contexts in the context dict."""

    def __init__(self, metric_name: str, threshold: float = 0.75):
        self._metric_name = metric_name   # "faithfulness", "answer_relevancy",
                                          # "context_precision", "context_recall"
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"ragas:{self._metric_name}"

    def evaluate(self, response, context=None):
        # Build a single-row RAGAS dataset from the context dict,
        # run ragas.evaluate(), pull out the metric score,
        # map to EvalResult with the same PASS/FAIL/ERROR discipline.
        ...
```

**Effort:** ~80 lines (more, because RAGAS's dataset shape needs marshalling). **Zero changes to core.**

The four RAGAS metrics map directly onto the RAG dimensions listed in Part 4 — that's the point to make: *my architecture already named those dimensions; RAGAS is one implementation of them.*

## Langfuse / LangSmith — observability

**This is the cleanest integration story of all, because `TraceWriter` is already the abstraction point.**

My `TraceWriter` writes JSON to disk. Langfuse and LangSmith write structured traces to a hosted backend. Same responsibility, different sink.

```python
class LangfuseTraceWriter:
    """Same interface as TraceWriter; writes to Langfuse instead of disk."""

    def write_llm_call(self, test_name, request, response) -> None:
        self._client.generation(
            name=test_name,
            model=response.model,
            input=request.prompt,
            output=response.text,
            usage={"input": response.input_tokens, "output": response.output_tokens},
            metadata={"provider": response.provider, "latency_ms": response.latency_ms},
        )

    def write_evaluation(self, test_name, result) -> None:
        self._client.score(
            name=result.evaluator,
            value=result.score,
            comment=result.rationale,
        )
```

**Effort:** ~60 lines. **Zero changes to core** — tests construct whichever writer they want.

*Interview point:* "I wrote flat JSON files deliberately as a first implementation. The trace *format* — prompt, response, model, tokens, latency, scores, rationale — is the same data Langfuse stores. Swapping the sink is an adapter, because I designed the writer as a boundary from the start."

## Promptfoo — different shape, different role

Promptfoo is a **CLI/config-driven test runner**, not a library you import. It's for prompt A/B comparison across models via YAML config.

Honest positioning: it doesn't wrap into the `Evaluator` Protocol because it *is* a harness, not an evaluator. It's a peer, not a component.

**Where it genuinely fits:** prompt experimentation — comparing 5 prompt variants across 3 models on 50 test cases, producing a comparison matrix. That's Promptfoo's strength and it's outside my harness's current scope.

**A credible integration:** generate `promptfoo` YAML configs from my rubric registry, so the same criteria drive both systems.

*Interview point:* "I'd use Promptfoo for prompt experimentation and my harness for regression testing in CI. They solve adjacent problems — one is exploratory, one is a gate."

## The summary table for the interview

| Tool | What it does | How it integrates | Effort |
|---|---|---|---|
| **DeepEval** | Pytest-native LLM metrics incl. GEval | Adapter → `Evaluator` Protocol | ~50 lines |
| **RAGAS** | RAG-specific metrics | Adapter → `Evaluator` Protocol | ~80 lines |
| **Langfuse** | Hosted trace/observability | Swap `TraceWriter` implementation | ~60 lines |
| **LangSmith** | Hosted trace/observability | Swap `TraceWriter` implementation | ~60 lines |
| **Promptfoo** | Prompt A/B via YAML config | Peer tool; generate configs from rubrics | separate workflow |

**The point that lands:** every one of these is an *additive adapter*. Zero core refactoring. That's not an accident — that's what Protocol-based design buys you.

---

# PART 6 — INTERVIEW Q&A BANK

**"Why not just use DeepEval instead of building this?"**
> "For a production team I probably would — and my architecture wraps it in ~50 lines. I built the primitives myself because I wanted to understand the mechanics: how judge rubrics affect score consistency, why the PASS/FAIL/ERROR distinction matters, what a trace actually needs to capture. Now when I use DeepEval I know what it's doing and where it'll fall short, instead of treating it as a black box."

**"The judge is non-deterministic. How is that trustworthy?"**
> "Four mitigations. One: temperature zero on the judge. Two: cross-model — Gemini judges Llama, so shared blind spots are less likely than self-evaluation. Three: the judge's own output is Pydantic-validated, and out-of-range scores, malformed JSON, or truncated responses return ERROR rather than a score. Four: calibration — sample judge outputs, have humans grade the same examples, measure agreement, adjust the rubric or swap models if agreement drops. I haven't built the calibration loop yet, but the traces I capture are exactly the dataset it needs."

**"How do you test non-deterministic output?"**
> "You stop asserting equality and start asserting properties. Three layers, cheapest first: deterministic checks — schema, format, required tokens, forbidden tokens. Then semantic checks — embedding similarity against a reference. Then LLM-as-judge for the dimensions you can't check any other way. Each layer has a threshold, and the thresholds should be calibrated against labelled data, not guessed."

**"What was the hardest bug?"**
> "Gemini 2.5's thinking tokens. The judge kept returning truncated JSON — `{"score": 4, "rationale": "` and nothing more. The naive read is 'malformed JSON', but the raw response showed `finishReason: MAX_TOKENS` and `thoughtsTokenCount: 6` with a 10-token budget. Gemini 2.5 reasons internally before responding and those tokens count against `maxOutputTokens` while producing no visible text. Fix was two parts: raise the budget, and add explicit truncation detection so the error message says 'response was truncated, increase max_tokens' instead of 'unterminated string'. Errors should teach the reader what to do."

**"How would you scale this to 100K evaluations a day?"**
> "Three things, none of which touch the core abstractions. Async clients — `httpx` is already the transport, so an async implementation satisfies the same Protocol. Parallel execution — `pytest-xdist` works out of the box. And swap the trace sink from flat files to a streaming backend like Langfuse. I'd also add response caching keyed on prompt+model hash for temperature-zero runs, and tiered execution: cheap deterministic checks on every commit, full judge suites nightly."

**"What would you do differently if you started over?"**
> "Two things. I'd build the trace writer *before* the evaluators — I wired tracing in afterwards and had to retrofit call sites. And I'd have started the calibration dataset from day one; every judge result I've generated is unlabelled, so I have traces but no ground truth to calibrate against. Both are ordering mistakes, not architecture mistakes."

**"What's this harness bad at?"**
> "Three honest limits. It doesn't evaluate multi-turn conversations yet — every evaluation is single-shot. It has no cost aggregation, so I capture tokens per call but don't roll them up. And LLM-as-judge is unsuitable as the *only* gate for safety-critical decisions; you want human review in the loop there. The architecture supports all three, but they're not built."

---

# PART 7 — THE OPENING PITCH

**10 seconds:**
> "I built an AI evaluation harness in Python that tests LLM outputs on two layers — a schema validator for format correctness and an LLM-as-judge for content quality — with full trace capture on every call. It caught a real failure where valid JSON masked a completely wrong answer that a schema check alone would have shipped."

**60 seconds:**
> "In my current role I validate AI-driven systems manually — I read outputs, judge them against expectations, log defects. It doesn't scale and it produces no historical quality signal. So I built a harness to answer what it would take to automate that judgment.
>
> The architecture is provider-agnostic — a Protocol-based client layer, so no test or evaluator imports a vendor SDK. Evaluators are pluggable behind a single contract: a deterministic schema validator and an LLM-as-judge using structured rubrics. Every result carries a normalized score, a rationale, and a PASS/FAIL/ERROR status — that third state matters, because a flaky judge must never look like a quality regression.
>
> Every LLM call and every evaluation writes a JSON trace, which gives audit trail today and the calibration dataset for tomorrow.
>
> The demo I'd show you is two runs on the same input: one where the AI triages a customer email correctly and both evaluators pass, and one where the AI produces perfectly valid JSON with completely wrong content — schema passes, and the judge catches it, in plain English."