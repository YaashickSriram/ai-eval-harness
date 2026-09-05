"""
TraceWriter — write LLM interactions and evaluator results as JSON trace files.

WHY this exists:
  Without traces, every LLM call vanishes after the test run. Traces turn
  ephemeral test output into auditable evidence:
    - What was the exact prompt sent?
    - What did the LLM return?
    - How long did it take?
    - How many tokens?
    - What did the evaluator score?
    - What was the rationale?

  For PMs: this is the "audit log" view. For engineers: this is the debug log.
  For drift detection (later): this is the historical record.

Design choices kept minimal:
  - Flat JSON files in `traces/` (one file per call)
  - Filename = ISO timestamp + provider + short hash for uniqueness
  - No rotation, no schema versioning, no async writes
  - Opt-in (you construct a TraceWriter; nothing magical)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from harness.clients.base import LLMRequest, LLMResponse
from harness.config.settings import settings
from harness.evaluators.base import EvalResult


# WHY a sanitizer: Pydantic models, frozen dataclasses, enums, exceptions —
# these are all common in our metadata but don't serialize directly with
# json.dumps. We coerce them to plain types in one place.
def _sanitize(value: Any) -> Any:
    """Convert arbitrary Python values to JSON-safe primitives.

    Handles dataclasses, enums, exceptions, Pydantic models (via .model_dump),
    and recursively walks dicts/lists. Unknown types fall back to str().
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if hasattr(value, "model_dump"):  # Pydantic model
        return _sanitize(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize(asdict(value))
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, Path):
        return str(value)
    # Fallback — anything else becomes its repr/str. Lossy but safe.
    return str(value)


class TraceWriter:
    """Writes JSON trace files for LLM calls and evaluations.

    Usage:
        writer = TraceWriter()
        writer.write_llm_call(test_name="email_triage", request=req, response=resp)
        writer.write_evaluation(test_name="email_triage", result=eval_result)
    """

    def __init__(self, trace_dir: Path | None = None) -> None:
        # WHY accept override: tests use a tmp_path; production uses settings default.
        self._dir = Path(trace_dir) if trace_dir else settings.harness_trace_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def trace_dir(self) -> Path:
        return self._dir

    def write_llm_call(
        self,
        test_name: str,
        request: LLMRequest,
        response: LLMResponse,
    ) -> Path:
        """Write a trace for one LLM call. Returns the file path."""
        payload = {
            "trace_type": "llm_call",
            "test_name": test_name,
            "timestamp": _now_iso(),
            "request": {
                "prompt": request.prompt,
                "system": request.system,
                "model": request.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stop": list(request.stop),
            },
            "response": {
                "text": response.text,
                "model": response.model,
                "provider": response.provider,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "finish_reason": response.finish_reason,
            },
        }
        return self._write(payload, prefix=f"{test_name}_llm_{response.provider}")

    def write_evaluation(self, test_name: str, result: EvalResult) -> Path:
        """Write a trace for one evaluator result. Returns the file path."""
        payload = {
            "trace_type": "evaluation",
            "test_name": test_name,
            "timestamp": _now_iso(),
            "evaluator": result.evaluator,
            "status": result.status.value,
            "score": result.score,
            "threshold": result.threshold,
            "rationale": result.rationale,
            "metadata": _sanitize(result.metadata),
        }
        return self._write(payload, prefix=f"{test_name}_eval_{_safe_name(result.evaluator)}")

    def _write(self, payload: dict[str, Any], prefix: str) -> Path:
        """Write the payload as pretty JSON. Returns the file path."""
        # WHY uuid suffix: multiple traces in the same second from the same
        # test+evaluator would collide on filename. uuid avoids that.
        filename = f"{_filesystem_timestamp()}_{prefix}_{uuid.uuid4().hex[:6]}.json"
        path = self._dir / filename
        # WHY encoding='utf-8' explicitly: Windows default cp1252 trips on
        # any non-ASCII character in prompts or responses.
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


# =============================================================================
# Helpers
# =============================================================================


def _now_iso() -> str:
    """ISO 8601 UTC timestamp for the trace contents."""
    return datetime.now(timezone.utc).isoformat()


def _filesystem_timestamp() -> str:
    """Compact timestamp safe for filenames: 20260527T143012Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_name(name: str) -> str:
    """Normalize an evaluator name like 'judge:relevance' to a filename-safe form."""
    return _UNSAFE_CHARS.sub("_", name)