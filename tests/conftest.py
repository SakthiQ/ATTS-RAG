"""Shared fixtures. `users` gives every test its own empty account database and a quick way to log in."""
import os
import tempfile

# Test runs must never write to the real logs/ folder: made-up accounts and admin views in logs/backend.log,
# or test answers in logs/layer3_audit.jsonl (which scripts/sample_audit_logs.py samples for audits).
# These are set here, before any test module is imported, because some objects fix their log path when they
# are built: app.main adds the backend log on import, and a Layer3Gate built in a module-scoped fixture picks
# its audit path before the per-test fixtures below run. The per-test fixtures then give each test its own file.
_TEST_LOG_DIR = tempfile.mkdtemp(prefix="atts-test-logs-")
os.environ["BACKEND_LOG"] = "off"
for _name, _file in (("LAYER3_AUDIT_LOG", "layer3_audit.jsonl"), ("ADMIN_AUDIT_LOG", "admin_view_audit.jsonl"),
                     ("AUTH_AUDIT_LOG", "auth_audit.jsonl"), ("TRACE_LOG", "traces.jsonl"),
                     ("TRACE_FEEDBACK_LOG", "trace_feedback.jsonl")):
    os.environ[_name] = os.path.join(_TEST_LOG_DIR, _file)

import pytest  # noqa: E402

from app import auth  # noqa: E402

TEST_PASSWORD = "test-password-123"


class Users:
    """Creates accounts directly in the store (no sign-up limits) and returns Authorization headers for them."""

    def __init__(self, store: auth.UserStore):
        self.store = store

    def create(self, username: str, role: str = "user", tenant: str = auth.DEFAULT_TENANT,
               password: str = TEST_PASSWORD) -> auth.User:
        role = auth.normalize_role(role)   # "admin" is the old name of super_admin
        existing = self.store.get_by_username(username)
        if existing is not None:
            assert (existing.role, existing.tenant_id) == (role, tenant), \
                f"{username} already exists as {existing.role} in {existing.tenant_id}"
            return existing
        return self.store.create_user(username, password, tenant_id=tenant, role=role)

    def headers(self, username: str, role: str = "user", tenant: str = auth.DEFAULT_TENANT) -> dict:
        user = self.create(username, role=role, tenant=tenant)
        token, _ = auth.issue_token(self.store.get(user.id))
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def audit_log(tmp_path, monkeypatch):
    """Every test writes audit records to its own file, never to logs/auth_audit.jsonl."""
    path = tmp_path / "auth_audit.jsonl"
    monkeypatch.setenv("AUTH_AUDIT_LOG", str(path))
    return path


@pytest.fixture(autouse=True)
def trace_logs(tmp_path, monkeypatch):
    """Per-request traces and answer feedback go to this test's own files, never to logs/."""
    paths = {"traces": tmp_path / "traces.jsonl", "feedback": tmp_path / "trace_feedback.jsonl"}
    monkeypatch.setenv("TRACE_LOG", str(paths["traces"]))
    monkeypatch.setenv("TRACE_FEEDBACK_LOG", str(paths["feedback"]))
    return paths


@pytest.fixture(autouse=True)
def layer3_and_admin_logs(tmp_path, monkeypatch):
    """Layer 3 audit records and admin-view records go to this test's own files, never to logs/.

    logs/layer3_audit.jsonl is what scripts/sample_audit_logs.py samples real answers from, so a test record
    there would end up in an audit."""
    paths = {"layer3": tmp_path / "layer3_audit.jsonl", "admin_view": tmp_path / "admin_view_audit.jsonl"}
    monkeypatch.setenv("LAYER3_AUDIT_LOG", str(paths["layer3"]))
    monkeypatch.setenv("ADMIN_AUDIT_LOG", str(paths["admin_view"]))
    return paths


@pytest.fixture()
def users(tmp_path, monkeypatch):
    monkeypatch.setenv("BCRYPT_ROUNDS", "4")   # the minimum cost: keeps tests fast
    monkeypatch.setenv("JWT_SECRET", "test-only-secret-" + "x" * 40)
    monkeypatch.setenv("SIGNUP_MODE", "open")
    monkeypatch.setenv("SIGNUPS_PER_SUBNET_PER_HOUR", "1000")
    monkeypatch.delenv("INVITE_CODES", raising=False)
    store = auth.UserStore(str(tmp_path / "users.db"))
    auth.set_store(store)
    yield Users(store)
    auth.set_store(None)
