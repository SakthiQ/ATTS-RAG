"""Upload permission checks. Every request here is rejected before the file is saved, so nothing is written."""
from fastapi.testclient import TestClient

from app.main import app
from app import routes

client = TestClient(app)
FILE = {"file": ("policy.txt", b"Refunds are accepted within 30 days.", "text/plain")}


def test_elevated_tier_requires_admin_token(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-token")
    resp = client.post("/upload", files=FILE, data={"source_tier": "official"})
    assert resp.status_code == 403


def test_wrong_admin_token_is_rejected(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-token")
    resp = client.post("/upload", files=FILE, data={"source_tier": "official"}, headers={"X-Admin-Token": "wrong-token"})
    assert resp.status_code == 403


def test_elevated_tiers_are_disabled_when_no_admin_token_is_configured(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    resp = client.post("/upload", files=FILE, data={"source_tier": "official"}, headers={"X-Admin-Token": "anything"})
    assert resp.status_code == 403


def test_unknown_tier_value_is_rejected():
    resp = client.post("/upload", files=FILE, data={"source_tier": "super_official"})
    assert resp.status_code == 400


def test_invalid_document_id_is_rejected():
    resp = client.post("/upload", files=FILE, data={"document_id": "../../etc/passwd"})
    assert resp.status_code == 400


def test_anonymous_upload_cannot_add_a_version_to_an_existing_document(monkeypatch):
    monkeypatch.setitem(routes.vsm.registry, "test-hash", {"document_id": "refund_policy"})
    resp = client.post("/upload", files=FILE, data={"document_id": "refund_policy"})
    assert resp.status_code == 409


def test_is_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert routes._is_admin("s3cret")
    assert not routes._is_admin("wrong")
    assert not routes._is_admin(None)
    monkeypatch.delenv("ADMIN_TOKEN")
    assert not routes._is_admin("s3cret")
