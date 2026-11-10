"""Per-request traces: one JSON line per question, linking every pipeline stage under one trace id.

A trace records what happened to a request and how long each step took, never what was said:

  * stages (spans): Layer 1, retrieval, rerank, Layer 2, context assembly, Layer 3 and its sub-steps, each with
    its start offset, duration and structural attributes (decisions, counts, scores, chunk ids)
  * LLM calls: purpose, time waiting for the model slot, wall time, prompt / output tokens and Ollama's own
    load / prompt-eval / generation durations (load + prompt-eval approximates time to first token)
  * errors: stage and exception type (the message is dropped, since it may echo its input)
  * the outcome (answered / blocked_l1 / no_relevant / blocked_l2 / rejected_l3 / error)

The question, the answer, prompts and document text are never written: only a SHA-256 of the question and its
length. Attribute keys that usually carry text (question, answer, prompt, content, text, snippet ...) are dropped
before writing, so a careless caller cannot leak them.

Settings: TRACE_LOG, the file (default logs/traces.jsonl); TRACE_LOG=off disables writing (traces still build,
so the response keeps its trace id). User feedback on an answer goes to TRACE_FEEDBACK_LOG
(default logs/trace_feedback.jsonl), keyed by trace id.
"""
import contextvars
import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger

DEFAULT_TRACE_LOG = os.path.join("logs", "traces.jsonl")
DEFAULT_FEEDBACK_LOG = os.path.join("logs", "trace_feedback.jsonl")
SCHEMA_VERSION = 1

# Keys whose values are usually user or document text; dropped wherever they appear in span attributes
TEXT_KEYS = {"question", "query", "answer", "prompt", "content", "text", "snippet", "context", "contexts",
             "response", "raw", "draft", "evidence_text", "whole_content", "message"}
FEEDBACK_RATINGS = ("up", "down")
FEEDBACK_REASONS = ("wrong", "incomplete", "not_relevant", "should_have_answered", "too_slow", "other")

_current: contextvars.ContextVar[Optional["Trace"]] = contextvars.ContextVar("atts_trace", default=None)
_write_lock = threading.Lock()


def trace_log_path() -> Optional[str]:
    path = os.getenv("TRACE_LOG", DEFAULT_TRACE_LOG)
    return None if path.strip().lower() in ("", "off", "0", "false", "none") else path


def feedback_log_path() -> str:
    return os.getenv("TRACE_FEEDBACK_LOG", DEFAULT_FEEDBACK_LOG)


def _scrub(value: Any, depth: int = 0) -> Any:
    """Drops text-bearing keys, recursively, and makes the value JSON-safe."""
    if depth > 6:
        return None
    if isinstance(value, dict):
        return {str(k): _scrub(v, depth + 1) for k, v in value.items() if str(k).lower() not in TEXT_KEYS}
    if isinstance(value, (list, tuple, set)):
        return [_scrub(v, depth + 1) for v in value]
    if isinstance(value, float):
        return round(value, 4)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    try:  # numpy scalars and the like
        return round(float(value), 4)
    except (TypeError, ValueError):
        return str(type(value).__name__)


def _append(path: str, record: Dict[str, Any]) -> None:
    with _write_lock:
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"Could not write to {path}: {e}")


class Span:
    __slots__ = ("name", "start_ms", "duration_ms", "attrs", "status")

    def __init__(self, name: str, start_ms: float):
        self.name, self.start_ms, self.duration_ms, self.attrs, self.status = name, start_ms, 0.0, {}, "ok"

    def set(self, **attrs: Any) -> "Span":
        self.attrs.update(attrs)
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "start_ms": round(self.start_ms, 1), "duration_ms": round(self.duration_ms, 1),
                "status": self.status, **({"attrs": _scrub(self.attrs)} if self.attrs else {})}


class _NullSpan:
    """Returned when no trace is active (scripts, tests): accepts attributes and ignores them."""

    def set(self, **attrs: Any) -> "_NullSpan":
        return self


NULL_SPAN = _NullSpan()


class Trace:
    def __init__(self, question: str = "", **attributes: Any):
        self.trace_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._t0 = time.perf_counter()
        self.attributes: Dict[str, Any] = {
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest() if question else None,
            "question_chars": len(question), **attributes}
        self.spans: List[Span] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.outcome: Optional[str] = None
        self._llm_purpose: Optional[str] = None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        s = Span(name, self.elapsed_ms())
        s.set(**attrs)
        self.spans.append(s)
        start = time.perf_counter()
        try:
            yield s
        except Exception as exc:
            s.status = "error"
            self.error(name, exc)
            raise
        finally:
            s.duration_ms = (time.perf_counter() - start) * 1000.0

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def error(self, stage: str, exc: BaseException) -> None:
        self.errors.append({"stage": stage, "type": type(exc).__name__, "at_ms": round(self.elapsed_ms(), 1)})

    @contextmanager
    def llm_purpose(self, purpose: str) -> Iterator[None]:
        previous, self._llm_purpose = self._llm_purpose, purpose
        try:
            yield
        finally:
            self._llm_purpose = previous

    def record_llm(self, model: str, wait_ms: float, wall_ms: float, response: Any = None,
                   error: Optional[BaseException] = None) -> None:
        call: Dict[str, Any] = {"purpose": self._llm_purpose or "llm", "model": model, "at_ms": round(self.elapsed_ms() - wall_ms, 1),
                                "wait_ms": round(wait_ms, 1), "wall_ms": round(wall_ms, 1), "status": "ok"}
        call.update(llm_usage(response))
        if error is not None:
            call["status"] = "error"
            call["error_type"] = type(error).__name__
        self.llm_calls.append(call)

    def finish(self, outcome: str) -> Dict[str, Any]:
        """Closes the trace, writes it (unless TRACE_LOG=off) and returns the record."""
        self.outcome = outcome
        record = self.as_dict()
        path = trace_log_path()
        if path:
            _append(path, record)
        return record

    def summary(self) -> Dict[str, Any]:
        """The compact view returned to the client with the answer."""
        stages: Dict[str, float] = {}
        for s in self.spans:
            if "." not in s.name:
                stages[s.name] = round(stages.get(s.name, 0.0) + s.duration_ms, 1)
        return {"trace_id": self.trace_id, "total_ms": round(self.elapsed_ms(), 1), "stages_ms": stages,
                "llm_calls": len(self.llm_calls), **self._token_totals()}

    def _token_totals(self) -> Dict[str, int]:
        return {"prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in self.llm_calls),
                "output_tokens": sum(c.get("output_tokens") or 0 for c in self.llm_calls)}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION, "trace_id": self.trace_id, "started_at": self.started_at,
            "total_ms": round(self.elapsed_ms(), 1), "outcome": self.outcome,
            "attributes": _scrub(self.attributes),
            "spans": [s.as_dict() for s in self.spans],
            "llm_calls": self.llm_calls, **self._token_totals(),
            "errors": self.errors,
        }


def llm_usage(response: Any) -> Dict[str, Any]:
    """Token counts and Ollama's own timings from a LangChain chat response (None fields when absent)."""
    if response is None:
        return {}
    usage = getattr(response, "usage_metadata", None) or {}
    meta = getattr(response, "response_metadata", None) or {}
    out: Dict[str, Any] = {
        "prompt_tokens": usage.get("input_tokens", meta.get("prompt_eval_count")),
        "output_tokens": usage.get("output_tokens", meta.get("eval_count")),
    }
    ns = {k: meta.get(k) for k in ("load_duration", "prompt_eval_duration", "eval_duration", "total_duration")}
    for key, value in ns.items():
        if isinstance(value, (int, float)):
            out[key.replace("_duration", "_ms")] = round(value / 1e6, 1)
    if "load_ms" in out or "prompt_eval_ms" in out:
        out["ttft_ms_est"] = round(out.get("load_ms", 0.0) + out.get("prompt_eval_ms", 0.0), 1)
    if out.get("output_tokens") and out.get("eval_ms"):
        out["tokens_per_s"] = round(out["output_tokens"] / (out["eval_ms"] / 1000.0), 2)
    if meta.get("done_reason"):
        out["done_reason"] = meta["done_reason"]   # "length" means num_predict cut the answer off
    return out


# ---- context helpers: tracing calls that are no-ops outside a request ---------------------------------------

def current() -> Optional[Trace]:
    return _current.get()


@contextmanager
def activate(trace: Trace) -> Iterator[Trace]:
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    trace = _current.get()
    if trace is None:
        yield NULL_SPAN
        return
    with trace.span(name, **attrs) as s:
        yield s


@contextmanager
def llm_purpose(purpose: str) -> Iterator[None]:
    trace = _current.get()
    if trace is None:
        yield
        return
    with trace.llm_purpose(purpose):
        yield


def file_sha256(path: str) -> Optional[str]:
    """Short content hash of a file (prompt templates, pattern files), so a trace names the exact version used."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        return None


def record_feedback(trace_id: str, user_id: str, tenant_id: str, rating: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """Appends one feedback record. Only a fixed rating and reason code are stored; no free text."""
    if rating not in FEEDBACK_RATINGS:
        raise ValueError(f"rating must be one of {FEEDBACK_RATINGS}")
    if reason is not None and reason not in FEEDBACK_REASONS:
        raise ValueError(f"reason must be one of {FEEDBACK_REASONS}")
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trace_id": trace_id,
              "user_id": user_id, "tenant_id": tenant_id, "rating": rating, "reason": reason}
    _append(feedback_log_path(), record)
    return record
