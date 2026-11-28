"""Health details and metrics for monitoring (Uptime Kuma, Prometheus or a person with a token).

  GET /health            public, for the app and Docker: "healthy" or "degraded" and one ok flag per subsystem,
                         no details (no model names, no index size)
  GET /health/details    every check below with its detail; HTTP 503 when any check fails, so a plain HTTP
                         monitor alerts on it
  GET /metrics           the same checks plus request outcomes and latency from logs/traces.jsonl, in the
                         Prometheus text format

The two protected endpoints accept `Authorization: Bearer <MONITOR_TOKEN>` (a read-only secret for monitoring
tools, set in .env) or a super admin's log-in token. Nothing they return contains document text, questions,
answers or user names: only counts, durations, sizes and ok flags.

Checks (each {"ok": bool, "detail": str}):
  ollama        Ollama answers and OLLAMA_MODEL is installed
  index         the three index stores (Chroma, BM25, registry) agree
  disk          free space where data/, chroma_db/ and logs/ live: at least DISK_MIN_FREE_GB (default 5) and
                DISK_MIN_FREE_PCT (default 10) percent
  writable      the app can create files in data/, chroma_db/ and logs/
  accounts_db   data/users.db passes SQLite's quick_check
  audit_chain   the log-in audit log's hash chain, archives included (cached for AUDIT_CHECK_SECONDS, default 600)
  backup        the last successful backup (logs/backup_status.json, written by scripts/backup.py) is at most
                BACKUP_MAX_AGE_HOURS old (default 26); BACKUP_CHECK=off turns this check off
"""
import hmac
import json
import os
import shutil
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app import audit
from app.rag import llm_gate, tracing

DATA_DIRS = {"data": "data", "chroma_db": "chroma_db", "logs": "logs"}
BACKUP_STATUS_FILE = os.path.join("logs", "backup_status.json")
TRACE_TAIL_BYTES = 8 * 1024 * 1024   # /metrics reads at most this much of the end of traces.jsonl
STAGES = ("layer1", "retrieval", "rerank", "layer2", "assembly", "layer3", "layer3.generate", "layer3.verify")
OUTCOMES = ("answered", "blocked_l1", "no_relevant", "blocked_l2", "rejected_l3", "error")

_audit_cache: Dict[str, Any] = {"at": 0.0, "result": None}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

async def check_ollama() -> Dict[str, Any]:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    wanted = os.getenv("OLLAMA_MODEL", "llama3")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{host}/api/tags")
        if r.status_code != 200:
            return {"ok": False, "detail": f"HTTP {r.status_code}"}
        models = [m["name"] for m in r.json().get("models", [])]
        if wanted in models or f"{wanted}:latest" in models:
            return {"ok": True, "detail": f"{wanted} available ({len(models)} model(s) installed)"}
        return {"ok": False, "detail": f"model '{wanted}' is not installed; run: ollama pull {wanted}"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)[:120] or type(exc).__name__}


def check_index(vsm) -> Dict[str, Any]:
    try:
        report = vsm.consistency_report()
    except Exception as exc:
        return {"ok": False, "detail": str(exc)[:120], "chunks": None}
    detail = f"{report['chroma_chunks']} chunks indexed"
    if not report["consistent"]:
        return {"ok": False, "detail": detail + "; the stores disagree: run scripts/check_index.py",
                "chunks": report["chroma_chunks"]}
    return {"ok": True, "detail": detail, "chunks": report["chroma_chunks"]}


def disk_usage() -> Dict[str, Dict[str, float]]:
    out = {}
    for name, path in DATA_DIRS.items():
        target = path if os.path.exists(path) else "."
        usage = shutil.disk_usage(target)
        out[name] = {"free_bytes": usage.free, "total_bytes": usage.total,
                     "free_pct": round(100.0 * usage.free / usage.total, 1) if usage.total else 0.0}
    return out


def check_disk(usage: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    usage = usage if usage is not None else disk_usage()
    min_gb, min_pct = _env_float("DISK_MIN_FREE_GB", 5.0), _env_float("DISK_MIN_FREE_PCT", 10.0)
    low = [f"{name} ({u['free_bytes'] / 1e9:.1f} GB, {u['free_pct']}% free)" for name, u in usage.items()
           if u["free_bytes"] < min_gb * 1e9 or u["free_pct"] < min_pct]
    if low:
        return {"ok": False, "detail": f"low disk space: {', '.join(low)}; minimum {min_gb:g} GB and {min_pct:g}%"}
    worst = min(usage.values(), key=lambda u: u["free_bytes"])
    return {"ok": True, "detail": f"lowest free space {worst['free_bytes'] / 1e9:.1f} GB ({worst['free_pct']}%)"}


def check_writable() -> Dict[str, Any]:
    failed = []
    for name, path in DATA_DIRS.items():
        probe = os.path.join(path, f".write-check-{uuid.uuid4().hex[:8]}")
        try:
            os.makedirs(path, exist_ok=True)
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
        except OSError as exc:
            failed.append(f"{name} ({type(exc).__name__})")
    return {"ok": not failed, "detail": "cannot write to " + ", ".join(failed) if failed else "data, chroma_db, logs"}


def check_accounts_db() -> Dict[str, Any]:
    path = os.getenv("USER_DB_PATH", os.path.join("data", "users.db"))
    if not os.path.exists(path):
        return {"ok": False, "detail": "the account database does not exist"}
    try:
        con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True, timeout=5)
        try:
            result = con.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as exc:
        return {"ok": False, "detail": f"cannot read the account database: {type(exc).__name__}"}
    return {"ok": result == "ok", "detail": "quick_check ok" if result == "ok" else f"quick_check: {result}"[:120]}


def check_audit_chain() -> Dict[str, Any]:
    ttl = _env_float("AUDIT_CHECK_SECONDS", 600.0)
    now = time.time()
    if _audit_cache["result"] is None or now - _audit_cache["at"] > ttl:
        _audit_cache["result"], _audit_cache["at"] = audit.verify_all(), now
    chain = _audit_cache["result"]
    if chain["intact"]:
        return {"ok": True, "detail": f"{chain['records']} records in {chain['files']} file(s), chain intact"}
    where = chain["broken"]
    return {"ok": False, "detail": f"chain broken in {os.path.basename(where['file'])} at line {where['line']}"}


def backup_status() -> Optional[Dict[str, Any]]:
    path = os.getenv("BACKUP_STATUS_FILE", BACKUP_STATUS_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check_backup(now: Optional[datetime] = None) -> Dict[str, Any]:
    if os.getenv("BACKUP_CHECK", "on").strip().lower() in ("off", "0", "false", "no"):
        return {"ok": True, "detail": "backup check turned off (BACKUP_CHECK=off)", "age_seconds": None}
    st = backup_status()
    max_hours = _env_float("BACKUP_MAX_AGE_HOURS", 26.0)
    if not st or not st.get("last_success"):
        return {"ok": False, "detail": "no successful backup recorded yet (run scripts/backup.py)", "age_seconds": None}
    try:
        last = datetime.fromisoformat(st["last_success"])
    except ValueError:
        return {"ok": False, "detail": "unreadable backup status file", "age_seconds": None}
    age = ((now or datetime.now(timezone.utc)) - last).total_seconds()
    failed_after = st.get("last_failure") and st["last_failure"] > st["last_success"]
    if age > max_hours * 3600:
        return {"ok": False, "detail": f"last successful backup {age / 3600:.1f} h ago (limit {max_hours:g} h)",
                "age_seconds": age}
    if failed_after:
        return {"ok": False, "detail": f"the last backup run failed: {st.get('last_error', 'see the backup log')}"[:160],
                "age_seconds": age}
    return {"ok": True, "detail": f"last successful backup {age / 3600:.1f} h ago", "age_seconds": age}


async def run_checks(vsm) -> Dict[str, Dict[str, Any]]:
    return {
        "ollama": await check_ollama(),
        "index": check_index(vsm),
        "disk": check_disk(),
        "writable": check_writable(),
        "accounts_db": check_accounts_db(),
        "audit_chain": check_audit_chain(),
        "backup": check_backup(),
    }


# ---------------------------------------------------------------------------
# Metrics from the traces
# ---------------------------------------------------------------------------

def _tail_jsonl(path: Optional[str], max_bytes: int = TRACE_TAIL_BYTES) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read()
    lines = data.split(b"\n")
    if size > max_bytes:
        lines = lines[1:]   # the first line may be cut
    out = []
    for line in lines:
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _since(records: List[Dict[str, Any]], key: str, cutoff: datetime) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        try:
            if datetime.fromisoformat(r.get(key, "")) >= cutoff:
                out.append(r)
        except (TypeError, ValueError):
            continue
    return out


def percentile(values: List[float], q: float) -> Optional[float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def trace_summary(window_minutes: Optional[float] = None, now: Optional[datetime] = None) -> Dict[str, Any]:
    window = window_minutes if window_minutes is not None else _env_float("METRICS_WINDOW_MINUTES", 60.0)
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(minutes=window)
    traces = _since(_tail_jsonl(tracing.trace_log_path()), "started_at", cutoff)
    feedback = _since(_tail_jsonl(tracing.feedback_log_path(), 1024 * 1024), "time", cutoff)
    stage_ms: Dict[str, List[float]] = defaultdict(list)
    waits: List[float] = []
    for t in traces:
        stage_ms["total"].append(t.get("total_ms"))
        for s in t.get("spans", []):
            stage_ms[s.get("name")].append(s.get("duration_ms"))
        waits.extend(c.get("wait_ms") for c in t.get("llm_calls", []) if c.get("wait_ms") is not None)
    return {
        "window_minutes": window,
        "requests": len(traces),
        "outcomes": Counter(t.get("outcome") or "unknown" for t in traces),
        "latency_ms": {name: (percentile(v, 50), percentile(v, 95)) for name, v in stage_ms.items() if name},
        "llm_wait_ms": (percentile(waits, 50), percentile(waits, 95)),
        "feedback": Counter(f.get("rating") for f in feedback),
    }


def _fmt(value: Optional[float]) -> str:
    return "NaN" if value is None else (f"{value:.3f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value))


def render_metrics(checks: Dict[str, Dict[str, Any]], usage: Dict[str, Dict[str, float]],
                   summary: Dict[str, Any], queue: Dict[str, int]) -> str:
    lines: List[str] = []

    def metric(name: str, kind: str, help_text: str, samples: List[Tuple[str, Optional[float]]]):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            lines.append(f"{name}{labels} {_fmt(value)}")

    metric("atts_up", "gauge", "1 while the API answers.", [("", 1)])
    metric("atts_check_ok", "gauge", "1 if the health check passes, else 0.",
           [(f'{{check="{name}"}}', 1 if c["ok"] else 0) for name, c in checks.items()])
    metric("atts_disk_free_bytes", "gauge", "Free disk space where each data folder lives.",
           [(f'{{folder="{name}"}}', u["free_bytes"]) for name, u in usage.items()])
    metric("atts_index_chunks", "gauge", "Chunks in the vector index.", [("", checks["index"].get("chunks"))])
    metric("atts_backup_age_seconds", "gauge", "Seconds since the last successful backup (NaN: none recorded).",
           [("", checks["backup"].get("age_seconds"))])
    metric("atts_llm_calls", "gauge", "Model calls now waiting for a slot or running.",
           [('{state="waiting"}', queue.get("waiting", 0)), ('{state="running"}', queue.get("running", 0))])
    window = f'{summary["window_minutes"]:g}m'
    metric("atts_requests", "gauge", "Questions answered or stopped in the last window, by outcome.",
           [(f'{{outcome="{o}",window="{window}"}}', summary["outcomes"].get(o, 0))
            for o in OUTCOMES + tuple(sorted(set(summary["outcomes"]) - set(OUTCOMES)))])
    latency = []
    for stage in ("total",) + STAGES:
        if stage in summary["latency_ms"]:
            p50, p95 = summary["latency_ms"][stage]
            latency += [(f'{{stage="{stage}",quantile="0.5",window="{window}"}}', p50),
                        (f'{{stage="{stage}",quantile="0.95",window="{window}"}}', p95)]
    metric("atts_latency_ms", "gauge", "Median and 95th-percentile duration per stage in the last window.", latency)
    p50, p95 = summary["llm_wait_ms"]
    metric("atts_llm_wait_ms", "gauge", "Time a model call waited for a slot in the last window.",
           [(f'{{quantile="0.5",window="{window}"}}', p50), (f'{{quantile="0.95",window="{window}"}}', p95)])
    metric("atts_feedback", "gauge", "Answer ratings given in the last window.",
           [(f'{{rating="{r}",window="{window}"}}', summary["feedback"].get(r, 0)) for r in tracing.FEEDBACK_RATINGS])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Access and routes
# ---------------------------------------------------------------------------

def require_monitor(request: Request) -> str:
    """A MONITOR_TOKEN bearer, or a super admin's log-in token."""
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    expected = os.getenv("MONITOR_TOKEN", "")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Send 'Authorization: Bearer <MONITOR_TOKEN>'.",
                            {"WWW-Authenticate": "Bearer"})
    if len(expected) >= 24 and hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
        return "monitor"
    from app import auth   # imported here: app.auth imports the account store at module load
    user = auth.get_current_user(auth.get_current_user_enrolling(token))
    if not user.is_super_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only a super admin or the monitoring token can read this.")
    return user.username


def build_router(vsm) -> APIRouter:
    router = APIRouter(tags=["System"])

    @router.get("/health/details")
    async def health_details(_: str = Depends(require_monitor)):
        """Every health check with its detail. HTTP 503 when any check fails."""
        checks = await run_checks(vsm)
        ok = all(c["ok"] for c in checks.values())
        body = {"ok": ok, "status": "healthy" if ok else "degraded",
                "failing": [name for name, c in checks.items() if not c["ok"]],
                "checks": {name: {"ok": c["ok"], "detail": c["detail"]} for name, c in checks.items()},
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        return JSONResponse(body, status_code=200 if ok else 503)

    @router.get("/metrics", response_class=PlainTextResponse)
    async def metrics(_: str = Depends(require_monitor)):
        """Prometheus text format: health checks, disk, backup age, model queue, outcomes and latency."""
        checks = await run_checks(vsm)
        return PlainTextResponse(render_metrics(checks, disk_usage(), trace_summary(), llm_gate.queue_state()),
                                 media_type="text/plain; version=0.0.4")

    return router
