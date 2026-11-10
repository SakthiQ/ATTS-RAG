"""Model-call queue, the model check in /health, and upload progress (/ingestion/{document_id})."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import main, routes
from app.main import app
from app.rag import llm_gate

client = TestClient(app)
FILE = {"file": ("policy.txt", b"Refunds are accepted within 30 days.", "text/plain")}


# ---- model-call queue ----

def test_model_calls_take_turns():
    running, peak, lock = [0], [0], threading.Lock()

    def call():
        with llm_gate.llm_slot():
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1, "with the default limit of 1, calls must not overlap"


# ---- /health checks that the configured model is installed ----

class _FakeResponse:
    status_code = 200

    def __init__(self, names):
        self._names = names

    def json(self):
        return {"models": [{"name": n} for n in self._names]}


def _fake_client(names):
    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            return _FakeResponse(names)
    return Client


@pytest.mark.parametrize("installed, ok", [(["llama3:latest", "qwen2.5:14b"], True), (["qwen2.5:14b"], False)])
def test_health_reports_whether_the_model_is_installed(monkeypatch, installed, ok):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setattr(main.httpx, "AsyncClient", _fake_client(installed))
    ollama = client.get("/health").json()["details"]["ollama"]
    assert ollama["ok"] is ok
    if not ok:
        assert "ollama pull llama3" in ollama["detail"]


# ---- upload progress ----

@pytest.fixture()
def uploads(tmp_path, monkeypatch):
    """Uploads are saved to a temporary folder; the background job is run by the test."""
    monkeypatch.setattr(routes, "UPLOAD_DIR", str(tmp_path))
    jobs = []
    monkeypatch.setattr(routes, "process_document_background", lambda *args: jobs.append(args))
    monkeypatch.setattr(routes, "_ingestion_status", type(routes._ingestion_status)())
    return jobs


def test_an_upload_is_processing_until_its_scan_finishes(users, uploads, monkeypatch):
    h = users.headers("ann", tenant="acme")
    assert client.post("/upload", files=FILE, data={"document_id": "refunds"}, headers=h).status_code == 200
    assert client.get("/ingestion/refunds", headers=h).json()["state"] == "processing"

    monkeypatch.setattr(routes, "ingest_file", lambda *a, **kw: {"status": "indexed", "indexed": 1, "flagged": 0, "quarantined": 0})
    PROCESS(*uploads[0])   # the real worker, with indexing stubbed
    status = client.get("/ingestion/refunds", headers=h).json()
    assert status["state"] == "indexed" and status["indexed"] == 1 and "finished_at" in status


def test_a_failed_scan_is_reported(users, uploads, monkeypatch):
    h = users.headers("ann", tenant="acme")
    client.post("/upload", files=FILE, data={"document_id": "broken"}, headers=h)

    def boom(*a, **kw):
        raise RuntimeError("could not read the file")
    monkeypatch.setattr(routes, "ingest_file", boom)
    PROCESS(*uploads[0])
    status = client.get("/ingestion/broken", headers=h).json()
    assert status["state"] == "failed" and "could not read" in status["error"]


def test_progress_is_private_to_the_tenant_and_the_clearance(users, uploads):
    tara = users.headers("tara", role="tenant_admin", tenant="acme")
    client.post("/upload", files=FILE, data={"document_id": "secret_plan", "acl_tier": "3"}, headers=tara)
    assert client.get("/ingestion/secret_plan", headers=tara).status_code == 200
    assert client.get("/ingestion/secret_plan", headers=users.headers("ann", tenant="acme")).status_code == 404, \
        "above the user's clearance"
    assert client.get("/ingestion/secret_plan", headers=users.headers("bob", tenant="globex")).status_code == 404, \
        "another tenant's upload"
    other = {"tenant_id": "acme"}
    assert client.get("/ingestion/secret_plan", params=other, headers=users.headers("bob", tenant="globex")).status_code == 403
    root = users.headers("root", role="super_admin", tenant="globex")
    assert client.get("/ingestion/secret_plan", params=other, headers=root).status_code == 200
    assert client.get("/ingestion/secret_plan").status_code == 401


def test_only_the_newest_uploads_are_remembered(monkeypatch):
    monkeypatch.setattr(routes, "_ingestion_status", type(routes._ingestion_status)())
    monkeypatch.setattr(routes, "INGESTION_STATUS_LIMIT", 3)
    for i in range(5):
        routes._set_ingestion_status("t", f"doc{i}", state="processing")
    assert [k[1] for k in routes._ingestion_status] == ["doc2", "doc3", "doc4"]


PROCESS = routes.process_document_background
