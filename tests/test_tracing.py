"""Per-request traces (app/rag/tracing.py), the LLM-call recording in the engine, the /feedback endpoint and the
trace report. None of these need Ollama or Chroma."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

from app.rag import tracing
from app.rag.engine import RAGEngine, l3_reason_code

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import trace_report  # noqa: E402

QUESTION = "What is the salary of employee 4471, SSN 123-45-6789?"


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


# ---- trace records ------------------------------------------------------------------------------------------

def test_a_trace_records_stages_and_never_the_question(trace_logs):
    trace = tracing.Trace(QUESTION, user_id="alice", tenant_id="acme")
    with tracing.activate(trace):
        with tracing.span("layer1") as sp:
            sp.set(action="allow", final_risk=0.12)
        with tracing.span("layer3"):
            with tracing.span("layer3.generate") as sp:
                sp.set(n_claims=3)
    record = trace.finish("answered")

    [written] = read(trace_logs["traces"])
    assert written["trace_id"] == record["trace_id"] and written["outcome"] == "answered"
    assert [s["name"] for s in written["spans"]] == ["layer1", "layer3", "layer3.generate"]
    assert written["spans"][0]["attrs"] == {"action": "allow", "final_risk": 0.12}
    assert written["attributes"]["question_chars"] == len(QUESTION)
    raw = open(trace_logs["traces"], encoding="utf-8").read()
    assert "salary" not in raw and "123-45-6789" not in raw


def test_text_bearing_attributes_are_dropped_even_when_nested(trace_logs):
    trace = tracing.Trace("q")
    with tracing.activate(trace), tracing.span("retrieval") as sp:
        sp.set(n=1, candidates=[{"chunk_uid": "c1", "content": "SECRET DOC TEXT", "text": "x"}],
               prompt="SECRET PROMPT", answer="SECRET ANSWER")
    trace.finish("answered")
    raw = open(trace_logs["traces"], encoding="utf-8").read()
    assert "SECRET" not in raw
    [written] = read(trace_logs["traces"])
    assert written["spans"][0]["attrs"] == {"n": 1, "candidates": [{"chunk_uid": "c1"}]}


def test_a_failing_stage_is_marked_and_its_message_is_not_kept(trace_logs):
    trace = tracing.Trace("q")
    with pytest.raises(RuntimeError):
        with tracing.activate(trace), tracing.span("rerank"):
            raise RuntimeError("leaks the query: " + QUESTION)
    trace.finish("error")
    [written] = read(trace_logs["traces"])
    assert written["spans"][0]["status"] == "error"
    assert written["errors"][0]["stage"] == "rerank" and written["errors"][0]["type"] == "RuntimeError"
    assert "salary" not in open(trace_logs["traces"], encoding="utf-8").read()


def test_helpers_do_nothing_outside_a_request(trace_logs):
    with tracing.span("layer1") as sp, tracing.llm_purpose("answer"):
        sp.set(action="allow")
    assert tracing.current() is None
    assert not os.path.exists(trace_logs["traces"])


def test_trace_log_off_disables_writing(trace_logs, monkeypatch):
    monkeypatch.setenv("TRACE_LOG", "off")
    tracing.Trace("q").finish("answered")
    assert not os.path.exists(trace_logs["traces"])


# ---- LLM usage ----------------------------------------------------------------------------------------------

def fake_response(**meta):
    return SimpleNamespace(content="ok", usage_metadata={"input_tokens": 1200, "output_tokens": 150},
                           response_metadata={"load_duration": 50_000_000, "prompt_eval_duration": 950_000_000,
                                              "eval_duration": 12_500_000_000, "total_duration": 13_600_000_000, **meta})


def test_llm_usage_reads_tokens_and_ollama_timings():
    usage = tracing.llm_usage(fake_response(done_reason="length"))
    assert usage["prompt_tokens"] == 1200 and usage["output_tokens"] == 150
    assert usage["ttft_ms_est"] == 1000.0 and usage["eval_ms"] == 12500.0
    assert usage["tokens_per_s"] == 12.0 and usage["done_reason"] == "length"


def test_engine_invoke_records_each_attempt(trace_logs):
    engine = RAGEngine.__new__(RAGEngine)   # no Ollama / Chroma: only _invoke is exercised
    engine.model_name = "llama3"

    class FlakyLLM:
        calls = 0

        def invoke(self, prompt):
            FlakyLLM.calls += 1
            if FlakyLLM.calls == 1:
                raise TimeoutError("ollama timed out")
            return fake_response()

    trace = tracing.Trace("q")
    with tracing.activate(trace), tracing.llm_purpose("answer"):
        assert engine._invoke(FlakyLLM(), "prompt").content == "ok"
    failed, ok = trace.llm_calls
    assert failed["status"] == "error" and failed["error_type"] == "TimeoutError" and failed["purpose"] == "answer"
    assert ok["status"] == "ok" and ok["prompt_tokens"] == 1200 and ok["model"] == "llama3"
    assert trace.summary()["prompt_tokens"] == 1200


def test_layer3_reasons_map_to_fixed_codes():
    assert l3_reason_code("Cross-tenant authorization violation: evidence tenant 'x' ...") == "tenant_violation"
    assert l3_reason_code("No sentence of the draft answer could be verified (INSUFFICIENT).") == "nothing_verified"
    assert l3_reason_code("API key pattern found", safety_flagged=True) == "unsafe_output"
    assert l3_reason_code(None) is None
    assert l3_reason_code("something new") == "other"


# ---- feedback -----------------------------------------------------------------------------------------------

def test_feedback_accepts_only_fixed_values(trace_logs):
    tracing.record_feedback("t1", "alice", "acme", "down", "wrong")
    with pytest.raises(ValueError):
        tracing.record_feedback("t1", "alice", "acme", "meh")
    with pytest.raises(ValueError):
        tracing.record_feedback("t1", "alice", "acme", "down", "free text about the answer")
    [record] = read(trace_logs["feedback"])
    assert record["rating"] == "down" and record["reason"] == "wrong"


def test_feedback_endpoint_only_takes_ratings_from_the_user_who_asked(users, trace_logs, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app import routes

    monkeypatch.setattr(routes.engine, "query", lambda **kw: {
        "answer": "Passwords rotate every 90 days.", "citations": [], "trace_id": "trace-abc", "admin_view": False})
    client = TestClient(app)
    alice, bob = users.headers("alice"), users.headers("bob")

    resp = client.post("/query", json={"question": "How often do passwords rotate?"}, headers=alice)
    assert resp.status_code == 200 and resp.json()["trace_id"] == "trace-abc"

    assert client.post("/feedback", json={"trace_id": "trace-abc", "rating": "up"}, headers=bob).status_code == 404
    assert client.post("/feedback", json={"trace_id": "unknown", "rating": "up"}, headers=alice).status_code == 404
    assert client.post("/feedback", json={"trace_id": "trace-abc", "rating": "great"}, headers=alice).status_code == 400
    assert client.post("/feedback", json={"trace_id": "trace-abc", "rating": "down", "reason": "incomplete"},
                       headers=alice).status_code == 200
    assert client.post("/feedback", json={"trace_id": "trace-abc", "rating": "up"}).status_code == 401
    [record] = read(trace_logs["feedback"])
    assert (record["trace_id"], record["user_id"], record["rating"], record["reason"]) == ("trace-abc", "alice", "down", "incomplete")


def test_a_failed_query_logs_the_trace_id_not_the_question(users, monkeypatch):
    from fastapi.testclient import TestClient
    from loguru import logger
    from app.main import app
    from app import routes

    def boom(**kw):
        exc = RuntimeError("chroma down")
        exc.trace_id = "trace-err"
        raise exc

    monkeypatch.setattr(routes.engine, "query", boom)
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        resp = TestClient(app).post("/query", json={"question": QUESTION}, headers=users.headers("alice"))
    finally:
        logger.remove(sink)
    assert resp.status_code == 500
    assert any("trace-err" in m for m in messages)
    assert not any("salary" in m for m in messages)


# ---- report -------------------------------------------------------------------------------------------------

def synthetic_trace(trace_id, outcome, total, generate_ms, wait_ms, from_rank=1, reason=None):
    spans = [{"name": "layer1", "start_ms": 0, "duration_ms": 300, "status": "ok", "attrs": {"action": "allow"}},
             {"name": "rerank", "start_ms": 300, "duration_ms": 200, "status": "ok",
              "attrs": {"ranked": [{"rank": 1, "chunk_uid": "c", "score": 3.1, "from_rank": from_rank}]}}]
    if outcome in ("answered", "rejected_l3"):
        spans.append({"name": "layer3", "start_ms": 600, "duration_ms": generate_ms + 500, "status": "ok",
                      "attrs": {"decision": "PASS" if outcome == "answered" else "REJECT", "reason_code": reason,
                                "grounding_ratio": 0.8}})
        spans.append({"name": "layer3.generate", "start_ms": 600, "duration_ms": generate_ms, "status": "ok"})
    return {"schema": 1, "trace_id": trace_id, "started_at": "2026-11-10T10:00:00.000+00:00", "total_ms": total,
            "outcome": outcome, "attributes": {}, "spans": spans, "errors": [],
            "llm_calls": [{"purpose": "answer", "wait_ms": wait_ms, "wall_ms": generate_ms, "prompt_tokens": 1000,
                           "output_tokens": 100, "ttft_ms_est": 900, "tokens_per_s": 11.5, "status": "ok"}]}


def test_report_summarises_outcomes_latency_and_feedback():
    traces = [synthetic_trace("a" * 32, "answered", 60000, 50000, 0),
              synthetic_trace("b" * 32, "answered", 90000, 55000, 25000, from_rank=4),
              synthetic_trace("c" * 32, "rejected_l3", 70000, 52000, 0, reason="nothing_verified")]
    feedback = [{"trace_id": "a" * 32, "rating": "up"}, {"trace_id": "c" * 32, "rating": "down"},
                {"trace_id": "c" * 32, "rating": "down", "reason": "should_have_answered"}]
    report = trace_report.build_report(traces, feedback)
    assert "| answered | 2 (66.7%) |" in report
    assert "| rejected_l3 | 1 (33.3%) |" in report
    assert "| nothing_verified | 1 |" in report
    assert "| should_have_answered | 1 |" in report
    assert "not retrieval's top passage in 1 (33.3%)" in report
    assert "| total | 3 | 1.17 min" not in report and "| total | 3 |" in report
    assert report.index("bbbbbbbbbbbb") < report.index("cccccccccccc")   # slowest first


def test_percentile_uses_observed_values():
    assert trace_report.percentile([1, 2, 3, 4, 100], 50) == 3
    assert trace_report.percentile([1, 2, 3, 4, 100], 95) == 100
    assert trace_report.percentile([], 50) is None
