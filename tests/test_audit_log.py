"""The account audit log: what is recorded, what never is, the hash chain, and who may read it."""
import json

import pytest
from fastapi.testclient import TestClient

from app import audit
from app.main import app
from conftest import TEST_PASSWORD

client = TestClient(app)


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_chain_is_intact_and_detects_an_edited_line(audit_log):
    for i in range(3):
        audit.audit("test_event", actor=f"u{i}", tenant="t")
    assert audit.verify() == (True, 3, None)
    lines = audit_log.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1]) | {"actor": "someone_else"}
    lines[1] = json.dumps(tampered)
    audit_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert audit.verify() == (False, 1, 2)


def test_a_deleted_line_breaks_the_chain(audit_log):
    for i in range(3):
        audit.audit("test_event", actor=f"u{i}")
    lines = audit_log.read_text(encoding="utf-8").splitlines()
    audit_log.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    assert audit.verify()[0] is False


def test_secrets_are_refused():
    with pytest.raises(ValueError):
        audit.audit("login", actor="ann", password="hunter2")


def test_log_ins_are_recorded_without_passwords_or_unknown_names(users, audit_log):
    users.create("ann", tenant="tenant_a")
    client.post("/auth/login", json={"username": "ann", "password": TEST_PASSWORD})
    client.post("/auth/login", json={"username": "ann", "password": "wrong password!!"})
    client.post("/auth/login", json={"username": "my-secret-pass", "password": "x" * 12})
    logins = [r for r in records(audit_log) if r["event"] == "login"]
    assert [(r["actor"], r["outcome"]) for r in logins] == [("ann", "ok"), ("ann", "bad_credentials"),
                                                           (None, "bad_credentials")]
    text = audit_log.read_text(encoding="utf-8")
    assert TEST_PASSWORD not in text and "wrong password" not in text and "my-secret-pass" not in text
    assert logins[0]["tenant"] == "tenant_a"


def test_a_lockout_is_recorded(users, audit_log):
    users.create("ann")
    for _ in range(5):
        client.post("/auth/login", json={"username": "ann", "password": "wrong password!!"})
    assert [r["target"] for r in records(audit_log) if r["event"] == "account_locked"] == ["ann"]


def test_admin_actions_are_recorded_and_reset_codes_are_not(users, audit_log):
    users.create("ann")
    root = users.headers("root", role="super_admin")
    client.patch("/auth/users/ann", json={"role": "viewer"}, headers=root)
    code = client.post("/auth/users/ann/reset", headers=root).json()["reset_code"]
    events = [(r["event"], r["actor"], r.get("target")) for r in records(audit_log)]
    assert ("account_updated", "root", "ann") in events and ("reset_code_issued", "root", "ann") in events
    assert code not in audit_log.read_text(encoding="utf-8")


def test_a_tenant_admin_reads_only_their_tenants_records(users):
    audit.audit("test_event", actor="x", tenant="tenant_a")
    audit.audit("test_event", actor="y", tenant="tenant_b")
    tara = users.headers("tara", role="tenant_admin", tenant="tenant_a")
    root = users.headers("root", role="super_admin")
    seen = client.get("/auth/audit", params={"event": "test_event"}, headers=tara).json()
    assert [r["actor"] for r in seen["records"]] == ["x"] and seen["chain_intact"] is True
    assert len(client.get("/auth/audit", params={"event": "test_event"}, headers=root).json()["records"]) == 2
    assert client.get("/auth/audit", headers=users.headers("ann")).status_code == 403
