"""Shared fixtures. `users` gives every test its own empty account database and a quick way to log in."""
import pytest

from app import auth

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
