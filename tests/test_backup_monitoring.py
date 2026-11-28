"""Backups (scripts/backup.py, scripts/restore.py), log rotation (app/log_files.py) and the monitoring endpoints
(app/monitoring.py).

The encrypted round trip needs the `age` program (AGE_BIN, or on PATH); CI installs it, and the test is skipped
where it is missing.
"""
import importlib.util
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import audit, log_files, monitoring
from app.main import app

ROOT = pathlib.Path(__file__).resolve().parents[1]
client = TestClient(app)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backup = load("backup")
AGE = backup.find_program(None, "AGE_BIN", "age")
needs_age = pytest.mark.skipif(AGE is None, reason="age is not installed (set AGE_BIN)")


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def names_every_6_hours(days, end):
    t, out = end - timedelta(days=days), []
    while t <= end:
        out.append(f"atts-{t:%Y%m%d-%H%M%S}.tar.gz.age")
        t += timedelta(hours=6)
    return out


def test_retention_keeps_7_daily_4_weekly_6_monthly():
    now = datetime(2026, 11, 28, 3, 0, 0)
    names = names_every_6_hours(400, now)
    keep = backup.select_keep(names + ["notes.txt"], now, 7, 4, 6)
    assert "notes.txt" in keep, "files that are not backups are never deleted"
    kept = sorted(n for n in keep if n != "notes.txt")
    assert len(kept) <= 7 + 4 + 6
    assert names[-1] in kept, "the newest backup is always kept"
    days = {backup.parse_stamp(n).date() for n in kept}
    assert all((now - timedelta(days=i)).date() in days for i in range(7)), "one per day for the last 7 days"
    months = {(backup.parse_stamp(n).year, backup.parse_stamp(n).month) for n in kept}
    assert len(months) == 6 and (2026, 11) in months and (2026, 6) in months
    assert backup.parse_stamp(kept[0]) > now - timedelta(days=190), "nothing older than the 6 monthly backups"


def test_retention_with_few_backups_keeps_them_all():
    now = datetime(2026, 11, 28, 3, 0, 0)
    names = [f"atts-202611{d:02d}-020000.tar.gz.age" for d in (25, 26, 27, 28)]
    assert backup.select_keep(names, now) == set(names)


# ---------------------------------------------------------------------------
# Snapshot of a database that is being written to
# ---------------------------------------------------------------------------

def test_sqlite_copy_is_consistent_while_the_database_is_written(tmp_path):
    db = tmp_path / "live.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (i INTEGER, pad TEXT)")
    con.commit()
    stop = threading.Event()

    def writer():
        w = sqlite3.connect(db)
        i = 0
        while not stop.is_set():
            w.execute("INSERT INTO t VALUES (?, ?)", (i, "x" * 500))
            w.commit()
            i += 1
        w.close()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for n in range(5):
            copy = tmp_path / f"copy{n}.db"
            backup.copy_sqlite(str(db), str(copy))
            check = sqlite3.connect(copy)
            assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            check.close()
    finally:
        stop.set()
        thread.join()
        con.close()


# ---------------------------------------------------------------------------
# A small installation to back up
# ---------------------------------------------------------------------------

@pytest.fixture()
def install(tmp_path, monkeypatch):
    """An ATTS-RAG folder with accounts, an indexed document, an audit log and a .env."""
    from app import auth
    from app.rag.chunker import DocumentChunker
    from app.rag.vectorstore import VectorStoreManager

    root = tmp_path / "atts"
    (root / "data" / "uploads" / "default_tenant" / "f1").mkdir(parents=True)
    (root / "logs").mkdir()
    store = auth.UserStore(str(root / "data" / "users.db"))
    store.create_user("ana", "password-123456", role="user")
    store.create_user("ben", "password-123456", role="super_admin")
    doc = root / "data" / "uploads" / "default_tenant" / "f1" / "policy.txt"
    doc.write_text("\n\n".join(f"Policy clause {i}: staff travelling to site {i} book rooms through the portal "
                               f"at least {i + 3} days ahead and keep receipts for {i + 30} days." for i in range(12)),
                   encoding="utf-8")
    chunks = DocumentChunker().chunk_documents([{"content": doc.read_text(encoding="utf-8"),
                                                "metadata": {"source": "policy.txt", "type": "text"}}])
    for c in chunks:
        c["metadata"].update({"scan_status": "clean", "anomaly_score": 0.0})
    built = tmp_path / "built_index"
    VectorStoreManager(persist_directory=str(built)).add_chunks(chunks, document_id="travel_policy")
    # this process keeps the index it built open, so the installation gets a copy, as a stopped app would leave it
    shutil.copytree(built, root / "chroma_db")
    monkeypatch.setenv("AUTH_AUDIT_LOG", str(root / "logs" / "auth_audit.jsonl"))
    for i in range(3):
        audit.audit("login", actor="ana", tenant="default_tenant", attempt=i)
    (root / ".env").write_text("OLLAMA_MODEL=llama3\n", encoding="utf-8")
    monkeypatch.delenv("BACKUP_AGE_RECIPIENT", raising=False)
    monkeypatch.delenv("BACKUP_REMOTE", raising=False)
    monkeypatch.delenv("BACKUP_HEARTBEAT_URL", raising=False)
    monkeypatch.delenv("BACKUP_STATUS_FILE", raising=False)
    return root


def make_key(path):
    """(private key file, public key) from age-keygen, which sits next to age."""
    keygen = shutil.which("age-keygen") or str(pathlib.Path(AGE).with_name("age-keygen" + pathlib.Path(AGE).suffix))
    subprocess.run([keygen, "-o", str(path)], check=True, capture_output=True)
    recipient = next(l.split(": ", 1)[1] for l in path.read_text().splitlines() if l.startswith("# public key: "))
    return str(path), recipient


@pytest.fixture()
def keypair(tmp_path):
    return make_key(tmp_path / "key.txt")


def test_an_unencrypted_backup_is_refused_unless_asked_for(install, tmp_path, capsys):
    assert backup.run(["--root", str(install), "--out", str(tmp_path / "out")]) == 1
    assert "refusing to write an unencrypted backup" in capsys.readouterr().err
    status = json.loads((install / "logs" / "backup_status.json").read_text())
    assert "last_failure" in status and "last_success" not in status


@needs_age
def test_encrypted_backup_and_restore_test_round_trip(install, keypair, tmp_path, capsys):
    key, recipient = keypair
    out = tmp_path / "out"
    assert backup.run(["--root", str(install), "--out", str(out), "--recipient", recipient, "--age", AGE]) == 0
    files = list(out.glob("atts-*.tar.gz.age"))
    assert len(files) == 1 and not list(out.glob(".staging-*")), "one archive, no snapshot left behind"
    assert files[0].read_bytes().startswith(b"age-encryption.org/v1")
    assert b"password-123456" not in files[0].read_bytes() and b"llama3" not in files[0].read_bytes()
    status = json.loads((install / "logs" / "backup_status.json").read_text())
    assert status["encrypted"] and status["last_sha256"] and status["last_success"]

    restore = load("restore")
    capsys.readouterr()
    assert restore.run([str(files[0]), "--identity", key, "--age", AGE]) == 0
    report = capsys.readouterr().out
    for name in ("files", "databases", "accounts", "index", "search", "audit"):
        assert f"PASS  {name}" in report, report
    assert "2 account(s)" in report and "Restore test passed" in report


@needs_age
def test_a_damaged_or_wrong_key_backup_fails_the_restore_test(install, keypair, tmp_path):
    key, recipient = keypair
    out = tmp_path / "out"
    backup.run(["--root", str(install), "--out", str(out), "--recipient", recipient, "--age", AGE])
    archive = next(out.glob("atts-*.tar.gz.age"))
    restore = load("restore")
    other, _ = make_key(tmp_path / "other.txt")
    assert restore.run([str(archive), "--identity", other, "--age", AGE]) == 1, "a different key cannot open it"
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 0xFF
    damaged = tmp_path / archive.name
    damaged.write_bytes(bytes(data))
    assert restore.run([str(damaged), "--identity", key, "--age", AGE]) == 1, "a flipped byte is detected"


def test_the_restore_check_finds_a_changed_file(install, tmp_path):
    out = tmp_path / "out"
    assert backup.run(["--root", str(install), "--out", str(out), "--allow-unencrypted"]) == 0
    archive = next(out.glob("atts-*.tar.gz"))
    restore = load("restore")
    scratch = tmp_path / "scratch"
    restore.unpack(str(archive), str(scratch), None, None)
    assert all(r["ok"] for r in restore.check(str(scratch)))
    # Opening the index (the index and search checks) lets Chroma rewrite its own files, so check() hashes the
    # files first; the second check below therefore needs a freshly unpacked copy
    scratch = tmp_path / "scratch2"
    restore.unpack(str(archive), str(scratch), None, None)
    doc = scratch / "data" / "uploads" / "default_tenant" / "f1" / "policy.txt"
    doc.write_text("edited", encoding="utf-8")
    results = {r["name"]: r for r in restore.check(str(scratch))}
    assert not results["files"]["ok"] and "1 changed" in results["files"]["detail"]


def test_restore_apply_keeps_the_previous_folders(install, tmp_path):
    out = tmp_path / "out"
    backup.run(["--root", str(install), "--out", str(out), "--allow-unencrypted"])
    archive = next(out.glob("atts-*.tar.gz"))
    target = tmp_path / "fresh"
    (target / "data").mkdir(parents=True)
    (target / "data" / "old.txt").write_text("previous", encoding="utf-8")
    restore = load("restore")
    assert restore.run([str(archive), "--root", str(target), "--apply", "--yes"]) == 0
    assert (target / "data" / "users.db").exists() and (target / ".env").exists()
    aside = list(target.glob("data.before-restore-*"))
    assert len(aside) == 1 and (aside[0] / "old.txt").read_text() == "previous"


# ---------------------------------------------------------------------------
# Log rotation and the audit chain
# ---------------------------------------------------------------------------

def test_plain_logs_rotate_into_the_archive(tmp_path):
    log = tmp_path / "logs" / "traces.jsonl"
    log.parent.mkdir()
    log.write_text("x" * 100, encoding="utf-8")
    assert log_files.rotate_if_needed(str(log), limit=1000) is None
    archived = log_files.rotate_if_needed(str(log), limit=50)
    assert archived and not log.exists() and pathlib.Path(archived).read_text() == "x" * 100
    assert log_files.archives(str(log)) == [archived]


def test_the_audit_chain_runs_on_across_rotated_files(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "auth_audit.jsonl"
    monkeypatch.setenv("AUTH_AUDIT_LOG", str(path))
    monkeypatch.setenv("LOG_MAX_MB", str(600 / (1024 * 1024)))   # rotate after about 600 bytes
    for i in range(12):
        audit.audit("login", actor=f"user{i}", tenant="t1")
    archives = log_files.archives(str(path))
    assert len(archives) >= 2, "the log rotated"
    first_line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first_line["event"] == audit.ROTATED_EVENT
    assert audit.verify(str(path))[0], "the current file verifies on its own"
    chain = audit.verify_all(str(path))
    assert chain["intact"] and chain["records"] == 12 + len(archives) and chain["files"] == len(archives) + 1

    os.remove(archives[1])   # a missing archive breaks the link into the next file
    broken = audit.verify_all(str(path))
    assert not broken["intact"] and broken["broken"]["line"] == 1


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

TOKEN = "monitor-token-for-tests-0123456789"


def test_public_health_gives_no_details():
    body = client.get("/health").json()
    assert set(body["details"]) == {"api", "ollama", "vector_db"}
    assert set(body["details"]["vector_db"]) == {"ok"} and set(body["details"]["ollama"]) == {"ok"}


def test_health_details_and_metrics_need_the_monitoring_token(monkeypatch):
    monkeypatch.setenv("MONITOR_TOKEN", TOKEN)
    assert client.get("/health/details").status_code == 401
    assert client.get("/health/details", headers={"Authorization": "Bearer wrong-token"}).status_code == 401
    assert client.get("/metrics").status_code == 401
    r = client.get("/health/details", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code in (200, 503)
    body = r.json()
    assert set(body["checks"]) == {"ollama", "index", "disk", "writable", "accounts_db", "audit_chain", "backup"}
    assert (r.status_code == 200) == body["ok"] and body["failing"] == [n for n, c in body["checks"].items() if not c["ok"]]
    m = client.get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert m.status_code == 200 and "atts_up 1" in m.text and 'atts_check_ok{check="disk"}' in m.text


def test_a_short_monitoring_token_is_never_accepted(monkeypatch):
    monkeypatch.setenv("MONITOR_TOKEN", "short")
    assert client.get("/metrics", headers={"Authorization": "Bearer short"}).status_code == 401


def test_super_admin_can_read_health_details(users, monkeypatch):
    monkeypatch.delenv("MONITOR_TOKEN", raising=False)
    assert client.get("/health/details", headers=users.headers("root", role="super_admin")).status_code in (200, 503)
    assert client.get("/health/details", headers=users.headers("ana")).status_code == 403


def test_disk_check_fails_below_either_limit(monkeypatch):
    monkeypatch.setenv("DISK_MIN_FREE_GB", "5")
    monkeypatch.setenv("DISK_MIN_FREE_PCT", "10")
    roomy = {"data": {"free_bytes": 50e9, "total_bytes": 250e9, "free_pct": 20.0}}
    assert monitoring.check_disk(roomy)["ok"]
    assert not monitoring.check_disk({"data": {"free_bytes": 4e9, "total_bytes": 10e9, "free_pct": 40.0}})["ok"]
    assert not monitoring.check_disk({"data": {"free_bytes": 20e9, "total_bytes": 400e9, "free_pct": 5.0}})["ok"]


def test_backup_check_reads_the_status_file(tmp_path, monkeypatch):
    status = tmp_path / "backup_status.json"
    monkeypatch.setenv("BACKUP_STATUS_FILE", str(status))
    monkeypatch.setenv("BACKUP_MAX_AGE_HOURS", "26")
    monkeypatch.delenv("BACKUP_CHECK", raising=False)
    now = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
    assert not monitoring.check_backup(now)["ok"], "no backup yet"
    status.write_text(json.dumps({"last_success": (now - timedelta(hours=3)).isoformat()}))
    assert monitoring.check_backup(now)["ok"]
    status.write_text(json.dumps({"last_success": (now - timedelta(hours=30)).isoformat()}))
    assert "30.0 h ago" in monitoring.check_backup(now)["detail"]
    status.write_text(json.dumps({"last_success": (now - timedelta(hours=3)).isoformat(),
                                  "last_failure": (now - timedelta(hours=1)).isoformat(), "last_error": "disk full"}))
    result = monitoring.check_backup(now)
    assert not result["ok"] and "disk full" in result["detail"]
    monkeypatch.setenv("BACKUP_CHECK", "off")
    assert monitoring.check_backup(now)["ok"]


def test_metrics_summarise_the_traces_window(tmp_path, monkeypatch):
    traces = tmp_path / "traces.jsonl"
    now = datetime.now(timezone.utc)
    rows = [{"started_at": (now - timedelta(minutes=m)).isoformat(), "outcome": o, "total_ms": t,
             "spans": [{"name": "retrieval", "duration_ms": t / 10}], "llm_calls": [{"wait_ms": 5.0}]}
            for m, o, t in ((1, "answered", 1000.0), (2, "answered", 3000.0), (3, "blocked_l1", 50.0),
                            (500, "answered", 99999.0))]
    traces.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("TRACE_LOG", str(traces))
    summary = monitoring.trace_summary(window_minutes=60)
    assert summary["requests"] == 3, "the request outside the window is left out"
    assert summary["outcomes"]["answered"] == 2 and summary["outcomes"]["blocked_l1"] == 1
    assert summary["latency_ms"]["total"][0] == 1000.0
    checks = {name: {"ok": True, "detail": ""} for name in ("ollama", "disk", "writable", "accounts_db", "audit_chain")}
    checks["index"] = {"ok": True, "detail": "", "chunks": 7}
    checks["backup"] = {"ok": False, "detail": "", "age_seconds": None}
    text = monitoring.render_metrics(checks, {"data": {"free_bytes": 1, "total_bytes": 2, "free_pct": 50.0}},
                                     summary, {"waiting": 2, "running": 1})
    assert 'atts_requests{outcome="answered",window="60m"} 2' in text
    assert 'atts_check_ok{check="backup"} 0' in text and "atts_backup_age_seconds NaN" in text
    assert 'atts_llm_calls{state="waiting"} 2' in text and "atts_index_chunks 7" in text
