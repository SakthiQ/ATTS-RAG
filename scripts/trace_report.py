#!/usr/bin/env python3
"""Summarises the per-request traces (logs/traces.jsonl) and answer feedback (logs/trace_feedback.jsonl).

    python scripts/trace_report.py                       # all traces, report to stdout
    python scripts/trace_report.py --since 24h --out logs/trace_report.md
    python scripts/trace_report.py --trace 3f9c2a1b      # one request, stage by stage (prefix of the id is enough)

Report sections: outcomes; latency per stage (p50 / p95 / p99); LLM calls (slot wait, estimated time to first
token, tokens/s, token counts, answers cut off by the output limit); Layer 1 actions; Layer 3 reject reasons;
how much the reranker reorders retrieval; errors; user feedback by outcome; the slowest requests.
Reads only the trace files, which hold no question or answer text.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.tracing import DEFAULT_FEEDBACK_LOG, DEFAULT_TRACE_LOG  # noqa: E402


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: {path}:{n} is not valid JSON; skipped", file=sys.stderr)
    return rows


def parse_since(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    units = {"h": "hours", "d": "days", "m": "minutes"}
    if value[-1] in units and value[:-1].isdigit():
        return datetime.now(timezone.utc) - timedelta(**{units[value[-1]]: int(value[:-1])})
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _time(record: Dict[str, Any], key: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(record[key])
    except (KeyError, TypeError, ValueError):
        return None


def percentile(values: List[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (no interpolation), so the value is one that was actually observed."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    rank = max(1, min(len(data), int(-(-q * len(data) // 100))))   # ceil(q/100 * n)
    return data[rank - 1]


def fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


def pct(part: int, whole: int) -> str:
    return f"{part} ({100.0 * part / whole:.1f}%)" if whole else "0"


def table(headers: List[str], rows: Iterable[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def span(trace: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    return next((s for s in trace.get("spans", []) if s.get("name") == name), None)


def latest_feedback(feedback: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The last rating per trace (a thumbs down followed by a reason is two records; the reason one wins)."""
    out: Dict[str, Dict[str, Any]] = {}
    for record in feedback:
        out[record.get("trace_id")] = record
    return out


def build_report(traces: List[Dict[str, Any]], feedback: List[Dict[str, Any]], slowest: int = 10) -> str:
    n = len(traces)
    if not n:
        return "# Trace report\n\nNo traces found.\n"
    parts: List[str] = []
    first, last = min(t.get("started_at", "") for t in traces), max(t.get("started_at", "") for t in traces)
    parts.append(f"# Trace report\n\n{n} request(s) from {first} to {last}.\n")

    # ---- outcomes
    outcomes = Counter(t.get("outcome") or "unknown" for t in traces)
    parts.append("## Outcomes\n\n" + table(["Outcome", "Requests"], [[o, pct(c, n)] for o, c in outcomes.most_common()]))

    # ---- latency per stage
    stage_ms: Dict[str, List[float]] = defaultdict(list)
    for t in traces:
        stage_ms["total"].append(t.get("total_ms"))
        for s in t.get("spans", []):
            stage_ms[s["name"]].append(s.get("duration_ms"))
    order = ["total", "layer1", "retrieval", "rerank", "layer2", "assembly", "layer3",
             "layer3.generate", "layer3.safety", "layer3.verify", "layer3.chart"]
    names = [s for s in order if s in stage_ms] + sorted(set(stage_ms) - set(order))
    parts.append("## Latency by stage\n\n" + table(
        ["Stage", "Requests", "p50", "p95", "p99", "max"],
        [[s, len(stage_ms[s]), fmt_ms(percentile(stage_ms[s], 50)), fmt_ms(percentile(stage_ms[s], 95)),
          fmt_ms(percentile(stage_ms[s], 99)), fmt_ms(max(v for v in stage_ms[s] if v is not None))] for s in names]))

    # ---- LLM calls
    calls = [c for t in traces for c in t.get("llm_calls", [])]
    if calls:
        by_purpose = Counter(c.get("purpose") for c in calls)
        failed = [c for c in calls if c.get("status") == "error"]
        cut = [c for c in calls if c.get("done_reason") == "length"]
        metric = lambda key: [c.get(key) for c in calls if c.get(key) is not None]  # noqa: E731
        rows = [[label, fmt_ms(percentile(metric(key), 50)), fmt_ms(percentile(metric(key), 95))]
                for label, key in (("Wait for model slot", "wait_ms"), ("Wall time", "wall_ms"),
                                   ("Time to first token (est.)", "ttft_ms_est"), ("Model load", "load_ms"))]
        tok = lambda key: percentile(metric(key), 50)  # noqa: E731
        parts.append(
            "## LLM calls\n\n"
            f"{len(calls)} call(s), {len(calls) / n:.2f} per request; by purpose: "
            + ", ".join(f"{p} {c}" for p, c in by_purpose.most_common()) + ".\n"
            f"Failed attempts: {pct(len(failed), len(calls))}"
            + (f" ({', '.join(f'{k} {v}' for k, v in Counter(c.get('error_type') for c in failed).items())})" if failed else "")
            + f". Cut off by the output-token limit: {pct(len(cut), len(calls))}.\n"
            f"Median tokens per call: {tok('prompt_tokens')} prompt, {tok('output_tokens')} output; "
            f"median generation speed {tok('tokens_per_s')} tokens/s.\n\n"
            + table(["Timing", "p50", "p95"], rows))

    # ---- Layer 1
    actions = Counter((span(t, "layer1") or {}).get("attrs", {}).get("action") for t in traces if span(t, "layer1"))
    if actions:
        parts.append("## Layer 1 actions\n\n" + table(["Action", "Requests"],
                                                      [[a, pct(c, sum(actions.values()))] for a, c in actions.most_common()]))

    # ---- Layer 3 reasons
    l3 = [span(t, "layer3")["attrs"] for t in traces if span(t, "layer3") and span(t, "layer3").get("attrs")]
    if l3:
        rejects = Counter(a.get("reason_code") for a in l3 if a.get("decision") == "REJECT")
        ratios = [a.get("grounding_ratio") for a in l3 if a.get("decision") == "PASS" and a.get("grounding_ratio") is not None]
        parts.append(
            "## Layer 3\n\n"
            f"{len(l3)} request(s) reached Layer 3; rejected {pct(sum(rejects.values()), len(l3))}. "
            f"Median grounding ratio of released answers: {percentile(ratios, 50)}.\n\n"
            + (table(["Reject reason", "Requests"], [[r, c] for r, c in rejects.most_common()]) if rejects else ""))

    # ---- rerank movement
    moved, top_from = [], []
    for t in traces:
        ranked = ((span(t, "rerank") or {}).get("attrs") or {}).get("ranked") or []
        if ranked:
            top = ranked[0].get("from_rank")
            if top is not None:
                top_from.append(top)
                moved.append(top != 1)
    if top_from:
        parts.append("## Reranking\n\n"
                     f"The reranker's top passage was not retrieval's top passage in {pct(sum(moved), len(moved))} requests; "
                     f"median retrieval rank of the final top passage: {percentile(top_from, 50)}, p95: {percentile(top_from, 95)}.")

    # ---- errors
    errors = Counter((e.get("stage"), e.get("type")) for t in traces for e in t.get("errors", []))
    parts.append("## Errors\n\n" + (table(["Stage", "Type", "Count"], [[s, ty, c] for (s, ty), c in errors.most_common()])
                                    if errors else "None recorded."))

    # ---- feedback
    fb = latest_feedback(feedback)
    by_id = {t["trace_id"]: t for t in traces}
    rated = {tid: r for tid, r in fb.items() if tid in by_id}
    if rated:
        per_outcome: Dict[str, Counter] = defaultdict(Counter)
        for tid, r in rated.items():
            per_outcome[by_id[tid].get("outcome") or "unknown"][r.get("rating")] += 1
        reasons = Counter(r.get("reason") for r in rated.values() if r.get("rating") == "down" and r.get("reason"))
        ups = sum(1 for r in rated.values() if r.get("rating") == "up")
        parts.append(
            "## User feedback\n\n"
            f"{len(rated)} rated answer(s) of {n} ({100.0 * len(rated) / n:.1f}%); thumbs up {pct(ups, len(rated))}.\n\n"
            + table(["Outcome", "Up", "Down"], [[o, c["up"], c["down"]] for o, c in sorted(per_outcome.items())])
            + ("\n\n" + table(["Thumbs-down reason", "Count"], [[r, c] for r, c in reasons.most_common()]) if reasons else ""))
    else:
        parts.append("## User feedback\n\nNo feedback on these requests.")

    # ---- slowest
    worst = sorted(traces, key=lambda t: t.get("total_ms") or 0, reverse=True)[:slowest]
    rows = []
    for t in worst:
        top_spans = sorted((s for s in t.get("spans", []) if "." not in s["name"]), key=lambda s: s.get("duration_ms", 0), reverse=True)
        wait = sum(c.get("wait_ms") or 0 for c in t.get("llm_calls", []))
        rows.append([t["trace_id"][:12], t.get("started_at", "")[:19], fmt_ms(t.get("total_ms")), t.get("outcome"),
                     top_spans[0]["name"] if top_spans else "n/a", fmt_ms(wait)])
    parts.append(f"## Slowest {len(worst)} requests\n\n"
                 + table(["Trace", "Started", "Total", "Outcome", "Slowest stage", "Waited for model"], rows)
                 + "\n\nInspect one with `python scripts/trace_report.py --trace <id>`.")
    return "\n\n".join(parts) + "\n"


def show_trace(trace: Dict[str, Any], feedback: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"Trace {trace['trace_id']}  {trace.get('started_at')}  outcome={trace.get('outcome')}  total={fmt_ms(trace.get('total_ms'))}"]
    attrs = trace.get("attributes", {})
    lines.append("  " + ", ".join(f"{k}={attrs.get(k)}" for k in ("user_id", "tenant_id", "session_id", "role", "clearance",
                                                                 "retrieval_mode", "question_chars") if k in attrs))
    if attrs.get("versions"):
        lines.append("  versions: " + ", ".join(f"{k}={v}" for k, v in attrs["versions"].items()))
    for s in trace.get("spans", []):
        indent = "    " if "." in s["name"] else "  "
        a = dict(s.get("attrs") or {})
        detail = ", ".join(f"{k}={v}" for k, v in a.items() if k not in ("candidates", "ranked", "evidence"))
        lines.append(f"{indent}+{s['start_ms']:>9.0f} ms  {s['name']:<16} {fmt_ms(s['duration_ms']):>9}  [{s['status']}]  {detail}")
        if "ranked" in a:
            lines.append(indent + "    ranked: " + "; ".join(
                f"#{r['rank']} {r.get('chunk_uid')} score={r.get('score')} (was #{r.get('from_rank')})" for r in a["ranked"]))
    for c in trace.get("llm_calls", []):
        lines.append(f"  LLM {c.get('purpose')}: wait {fmt_ms(c.get('wait_ms'))}, wall {fmt_ms(c.get('wall_ms'))}, "
                     f"ttft~{fmt_ms(c.get('ttft_ms_est'))}, tokens {c.get('prompt_tokens')} in / {c.get('output_tokens')} out, "
                     f"{c.get('tokens_per_s')} tok/s, {c.get('status')}" + (f" (cut off)" if c.get("done_reason") == "length" else ""))
    for e in trace.get("errors", []):
        lines.append(f"  ERROR at {e.get('at_ms')} ms in {e.get('stage')}: {e.get('type')}")
    if trace["trace_id"] in feedback:
        r = feedback[trace["trace_id"]]
        lines.append(f"  feedback: {r.get('rating')}" + (f" ({r.get('reason')})" if r.get("reason") else ""))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default=os.getenv("TRACE_LOG", DEFAULT_TRACE_LOG))
    parser.add_argument("--feedback", default=os.getenv("TRACE_FEEDBACK_LOG", DEFAULT_FEEDBACK_LOG))
    parser.add_argument("--since", help="only traces after this: 24h, 7d, 30m or an ISO date")
    parser.add_argument("--trace", help="print one trace (id or id prefix)")
    parser.add_argument("--slowest", type=int, default=10)
    parser.add_argument("--out", help="write the report here instead of stdout")
    args = parser.parse_args()

    traces = read_jsonl(args.log)
    feedback = read_jsonl(args.feedback)
    since = parse_since(args.since)
    if since:
        traces = [t for t in traces if (_time(t, "started_at") or since) >= since]

    if args.trace:
        matches = [t for t in traces if t.get("trace_id", "").startswith(args.trace)]
        if not matches:
            print(f"No trace starting with {args.trace} in {args.log}", file=sys.stderr)
            return 1
        print("\n\n".join(show_trace(t, latest_feedback(feedback)) for t in matches))
        return 0

    report = build_report(traces, feedback, slowest=args.slowest)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote {args.out} ({len(traces)} traces)")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
