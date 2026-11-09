"""Account administration and self-service: who may manage whom, password change and admin-issued reset codes."""
import pytest
from fastapi.testclient import TestClient

from app import auth
from app.main import app
from conftest import TEST_PASSWORD

client = TestClient(app)
NEW_PASSWORD = "a brand new password"


def login(username, password=TEST_PASSWORD):
    return client.post("/auth/login", json={"username": username, "password": password})


# ---- creating and listing ----

def test_a_tenant_admin_creates_accounts_in_their_own_tenant_only(users):
    h = users.headers("tara", role="tenant_admin", tenant="tenant_a")
    r = client.post("/auth/users", json={"username": "ann", "password": TEST_PASSWORD}, headers=h)
    assert r.status_code == 201 and r.json()["tenant_id"] == "tenant_a" and r.json()["role"] == "user"
    other = {"username": "bob", "password": TEST_PASSWORD, "tenant_id": "tenant_b"}
    assert client.post("/auth/users", json=other, headers=h).status_code == 403
    boss = {"username": "boss", "password": TEST_PASSWORD, "role": "super_admin"}
    assert client.post("/auth/users", json=boss, headers=h).status_code == 403
    peer = {"username": "tom", "password": TEST_PASSWORD, "role": "tenant_admin"}
    assert client.post("/auth/users", json=peer, headers=h).status_code == 201


def test_ordinary_users_cannot_administer_accounts(users):
    h = users.headers("ann")
    assert client.post("/auth/users", json={"username": "x-user", "password": TEST_PASSWORD}, headers=h).status_code == 403
    assert client.get("/auth/users", headers=h).status_code == 403
    assert client.patch("/auth/users/ann", json={"role": "super_admin"}, headers=h).status_code == 403


def test_the_account_list_is_scoped_to_the_tenant(users):
    users.create("ann", tenant="tenant_a")
    users.create("bob", tenant="tenant_b")
    tara = users.headers("tara", role="tenant_admin", tenant="tenant_a")
    root = users.headers("root", role="super_admin")
    assert {u["username"] for u in client.get("/auth/users", headers=tara).json()} == {"ann", "tara"}
    assert {u["username"] for u in client.get("/auth/users", headers=root).json()} == {"ann", "bob", "tara", "root"}


# ---- changing role and status ----

def test_another_tenants_account_is_reported_as_missing(users):
    users.create("bob", tenant="tenant_b")
    tara = users.headers("tara", role="tenant_admin", tenant="tenant_a")
    for call in (client.patch("/auth/users/bob", json={"status": "disabled"}, headers=tara),
                 client.delete("/auth/users/bob", headers=tara),
                 client.post("/auth/users/bob/reset", headers=tara)):
        assert call.status_code == 404
    assert client.patch("/auth/users/nobody", json={"status": "disabled"}, headers=tara).status_code == 404


def test_a_tenant_admin_cannot_touch_a_super_admin_or_promote_to_one(users):
    users.create("root", role="super_admin", tenant="tenant_a")
    users.create("ann", tenant="tenant_a")
    tara = users.headers("tara", role="tenant_admin", tenant="tenant_a")
    assert client.patch("/auth/users/root", json={"status": "disabled"}, headers=tara).status_code == 403
    assert client.patch("/auth/users/ann", json={"role": "super_admin"}, headers=tara).status_code == 403
    assert client.patch("/auth/users/ann", json={"role": "tenant_admin"}, headers=tara).status_code == 200


def test_nobody_changes_their_own_role_or_status_or_deletes_themselves(users):
    tara = users.headers("tara", role="tenant_admin")
    assert client.patch("/auth/users/tara", json={"role": "super_admin"}, headers=tara).status_code == 400
    assert client.patch("/auth/users/tara", json={"status": "disabled"}, headers=tara).status_code == 400
    assert client.delete("/auth/users/tara", headers=tara).status_code == 400


def test_a_role_change_takes_effect_at_once(users):
    old_token = users.headers("ann")
    root = users.headers("root", role="super_admin")
    r = client.patch("/auth/users/ann", json={"role": "viewer"}, headers=root)
    assert r.status_code == 200 and r.json()["role"] == "viewer"
    assert client.get("/auth/me", headers=old_token).status_code == 401, "the old session must end"
    assert login("ann").json()["user"]["role"] == "viewer"


def test_a_disabled_account_cannot_log_in_until_enabled_again(users):
    users.create("ann")
    root = users.headers("root", role="super_admin")
    assert client.patch("/auth/users/ann", json={"status": "disabled"}, headers=root).json()["status"] == "disabled"
    assert login("ann").status_code == 403
    assert client.patch("/auth/users/ann", json={"status": "active"}, headers=root).status_code == 200
    assert login("ann").status_code == 200


def test_unknown_role_or_status_is_rejected(users):
    users.create("ann")
    root = users.headers("root", role="super_admin")
    assert client.patch("/auth/users/ann", json={"role": "owner"}, headers=root).status_code == 400
    assert client.patch("/auth/users/ann", json={"status": "frozen"}, headers=root).status_code == 400
    assert client.patch("/auth/users/ann", json={}, headers=root).status_code == 400


def test_a_deleted_account_is_gone_and_its_sessions_end(users):
    ann = users.headers("ann")
    root = users.headers("root", role="super_admin")
    assert client.delete("/auth/users/ann", headers=root).status_code == 204
    assert client.get("/auth/me", headers=ann).status_code == 401
    assert users.store.get_by_username("ann") is None


def test_the_last_super_admin_cannot_be_removed(users):
    root = users.create("root", role="super_admin")
    with pytest.raises(auth.AuthError) as e:
        auth.check_not_last_super_admin(users.store, root, removing=True)
    assert e.value.status_code == 409
    users.create("root2", role="super_admin")
    auth.check_not_last_super_admin(users.store, root, removing=True)   # another one exists now
    h = users.headers("root2", role="super_admin")
    assert client.patch("/auth/users/root", json={"role": "user"}, headers=h).status_code == 200


# ---- passwords ----

def test_changing_your_password_ends_your_other_sessions(users):
    old = users.headers("ann")
    wrong = client.post("/auth/password", json={"old_password": "not my password", "new_password": NEW_PASSWORD}, headers=old)
    assert wrong.status_code == 400
    r = client.post("/auth/password", json={"old_password": TEST_PASSWORD, "new_password": NEW_PASSWORD}, headers=old)
    assert r.status_code == 200
    assert client.get("/auth/me", headers=old).status_code == 401
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).status_code == 200
    assert login("ann").status_code == 401 and login("ann", NEW_PASSWORD).status_code == 200


def test_a_new_password_must_meet_the_rules(users):
    h = users.headers("ann")
    assert client.post("/auth/password", json={"old_password": TEST_PASSWORD, "new_password": "short"}, headers=h).status_code == 400


def _reset(username, code, password=NEW_PASSWORD):
    return client.post("/auth/password/reset", json={"username": username, "code": code, "new_password": password})


def test_an_admin_issued_reset_code_works_once(users):
    users.create("ann", tenant="tenant_a")
    tara = users.headers("tara", role="tenant_admin", tenant="tenant_a")
    r = client.post("/auth/users/ann/reset", headers=tara)
    assert r.status_code == 200
    code = r.json()["reset_code"]
    assert len(code) == 14 and code.count("-") == 2
    assert _reset("ann", code.lower().replace("-", " ")).status_code == 200, "case and separators do not matter"
    assert login("ann", NEW_PASSWORD).status_code == 200
    assert _reset("ann", code, "another new password").status_code == 400, "a code is single-use"


def test_an_expired_or_wrong_reset_code_is_refused(users, monkeypatch):
    users.create("ann")
    root = users.headers("root", role="super_admin")
    monkeypatch.setenv("RESET_CODE_MINUTES", "0")
    code = client.post("/auth/users/ann/reset", headers=root).json()["reset_code"]
    assert _reset("ann", code).status_code == 400
    assert _reset("nobody", "AAAA-BBBB-CCCC").status_code == 400


def test_reset_codes_cannot_be_guessed_faster_than_passwords(users):
    users.create("ann")
    root = users.headers("root", role="super_admin")
    code = client.post("/auth/users/ann/reset", headers=root).json()["reset_code"]
    for _ in range(auth.MAX_FAILED_LOGINS):
        assert _reset("ann", "AAAA-BBBB-CCCC").status_code == 400
    assert _reset("ann", code).status_code == 429


def test_you_cannot_issue_a_reset_code_for_yourself(users):
    h = users.headers("root", role="super_admin")
    assert client.post("/auth/users/root/reset", headers=h).status_code == 400


# ---- command line ----

def test_the_command_line_manages_accounts_and_keeps_one_super_admin(users, capsys):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "manage_users", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "manage_users.py")
    manage_users = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manage_users)
    users.create("root", role="super_admin")
    users.create("ann")
    manage_users.main(["set-role", "--username", "ann", "--role", "tenant_admin"])
    assert users.store.get_by_username("ann").role == "tenant_admin"
    manage_users.main(["disable", "--username", "ann"])
    assert users.store.get_by_username("ann").status == "disabled"
    manage_users.main(["reset-password", "--username", "ann"])
    assert "Reset code for 'ann'" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        manage_users.main(["set-role", "--username", "root", "--role", "user"])
    assert "last active super admin" in str(e.value)
    with pytest.raises(SystemExit) as e:
        manage_users.main(["audit-verify"])
    assert e.value.code == 0
