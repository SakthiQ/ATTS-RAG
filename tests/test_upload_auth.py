"""Upload and delete permissions. Every refused request is rejected before the file is saved, so nothing is written."""
from fastapi.testclient import TestClient

from app.main import app
from app import routes

client = TestClient(app)
FILE = {"file": ("policy.txt", b"Refunds are accepted within 30 days.", "text/plain")}


def test_upload_needs_a_login(users):
    assert client.post("/upload", files=FILE).status_code == 401


def test_elevated_tier_requires_an_admin(users):
    resp = client.post("/upload", files=FILE, data={"source_tier": "official"}, headers=users.headers("alice"))
    assert resp.status_code == 403


def test_a_forged_token_is_rejected(users):
    resp = client.post("/upload", files=FILE, data={"source_tier": "official"},
                       headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_unknown_tier_value_is_rejected(users):
    resp = client.post("/upload", files=FILE, data={"source_tier": "super_official"}, headers=users.headers("alice"))
    assert resp.status_code == 400


def test_invalid_document_id_is_rejected(users):
    resp = client.post("/upload", files=FILE, data={"document_id": "../../etc/passwd"}, headers=users.headers("alice"))
    assert resp.status_code == 400


def test_a_user_cannot_add_a_version_to_an_existing_document(users, monkeypatch):
    monkeypatch.setitem(routes.vsm.registry, "test-hash", {"document_id": "refund_policy"})
    resp = client.post("/upload", files=FILE, data={"document_id": "refund_policy"}, headers=users.headers("alice"))
    assert resp.status_code == 409


def test_delete_without_a_login_is_rejected_and_keeps_the_document(users, monkeypatch):
    monkeypatch.setitem(routes.vsm.registry, "delete-test-hash", {"document_id": "refund_policy", "ids": []})
    deleted = []
    monkeypatch.setattr(routes.vsm, "delete_document", lambda h: deleted.append(h))
    assert client.delete("/documents/delete-test-hash").status_code == 401
    assert deleted == [] and "delete-test-hash" in routes.vsm.registry


def test_delete_by_an_ordinary_user_is_rejected(users, monkeypatch):
    monkeypatch.setitem(routes.vsm.registry, "delete-test-hash", {"document_id": "refund_policy", "ids": []})
    deleted = []
    monkeypatch.setattr(routes.vsm, "delete_document", lambda h: deleted.append(h))
    assert client.delete("/documents/delete-test-hash", headers=users.headers("alice")).status_code == 403
    assert deleted == []


def test_delete_by_an_admin_reaches_the_document_lookup(users, monkeypatch):
    deleted = []
    monkeypatch.setattr(routes.vsm, "delete_document", lambda h: deleted.append(h))
    monkeypatch.setitem(routes.vsm.registry, "admin-delete-hash", {"document_id": "refund_policy", "ids": []})
    admin = users.headers("root", role="admin")
    assert client.delete("/documents/no-such-hash", headers=admin).status_code == 404
    assert client.delete("/documents/admin-delete-hash", headers=admin).status_code == 200
    assert deleted == ["admin-delete-hash"]
