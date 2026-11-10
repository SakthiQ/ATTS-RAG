"""Invite codes kept in the database: they expire, have a use limit, can be revoked and are scoped to a tenant."""
from fastapi.testclient import TestClient

from app import auth
from app.main import app
from conftest import TEST_PASSWORD

client = TestClient(app)


def signup(username, code):
    return client.post("/auth/signup", json={"username": username, "password": TEST_PASSWORD, "invite_code": code})


def make(users, admin_headers, **body):
    return client.post("/auth/invites", json=body, headers=admin_headers)


def test_an_invite_decides_tenant_and_role(users, monkeypatch):
    monkeypatch.setenv("SIGNUP_MODE", "invite")
    tara = users.headers("tara", role="tenant_admin", tenant="acme")
    r = make(users, tara, role="viewer", max_uses=2)
    assert r.status_code == 201 and r.json()["code"].startswith("INV-")
    body = signup("vera", r.json()["code"]).json()
    assert body["user"]["tenant_id"] == "acme" and body["user"]["role"] == "viewer"


def test_an_invite_works_only_as_often_as_allowed(users):
    root = users.headers("root", role="super_admin")
    code = make(users, root, tenant_id="acme", max_uses=2).json()["code"]
    assert signup("ann1", code).status_code == 201
    assert signup("ann2", code.lower()).status_code == 201, "case does not matter"
    assert signup("ann3", code).status_code == 400
    assert users.store.get_by_username("ann3") is None


def test_an_expired_or_revoked_invite_is_refused(users):
    root = users.headers("root", role="super_admin")
    invite = users.store.create_invite("acme", expires_in_hours=0.0001)
    import time
    time.sleep(0.5)
    assert signup("ann", invite["code"]).status_code == 400
    fresh = make(users, root, tenant_id="acme").json()
    assert client.delete(f"/auth/invites/{fresh['id']}", headers=root).status_code == 204
    assert signup("bob", fresh["code"]).status_code == 400


def test_a_failed_signup_does_not_use_up_the_invite(users):
    users.create("taken")
    code = users.store.create_invite("acme")["code"]
    assert signup("taken", code).status_code == 409
    assert signup("ann", code).status_code == 201


def test_invites_never_create_admins_and_limits_are_checked(users):
    root = users.headers("root", role="super_admin")
    assert make(users, root, role="tenant_admin").status_code == 400
    assert make(users, root, max_uses=0).status_code == 400
    assert make(users, root, expires_in_hours=10_000).status_code == 400


def test_a_tenant_admin_manages_invites_of_their_own_tenant_only(users):
    tara = users.headers("tara", role="tenant_admin", tenant="acme")
    root = users.headers("root", role="super_admin")
    assert make(users, tara, tenant_id="globex").status_code == 403
    mine = make(users, tara).json()
    theirs = make(users, root, tenant_id="globex").json()
    assert [i["id"] for i in client.get("/auth/invites", headers=tara).json()] == [mine["id"]]
    assert all("code" not in i and "code_hash" not in i for i in client.get("/auth/invites", headers=root).json())
    assert client.delete(f"/auth/invites/{theirs['id']}", headers=tara).status_code == 404
    assert client.post("/auth/invites", json={}, headers=users.headers("ann")).status_code == 403


def test_codes_are_stored_only_as_hashes(users):
    code = users.store.create_invite("acme")["code"]
    with users.store._db() as db:
        row = db.execute("SELECT * FROM invites").fetchone()
    assert code not in str(tuple(row))


def test_old_invite_codes_from_the_environment_still_work(users, monkeypatch):
    monkeypatch.setenv("SIGNUP_MODE", "invite")
    monkeypatch.setenv("INVITE_CODES", "legacy-2026:tenant_old")
    r = signup("ann", "legacy-2026")
    assert r.status_code == 201 and r.json()["user"]["tenant_id"] == "tenant_old" and r.json()["user"]["role"] == "user"
    assert auth.INVITE_ROLES == ("viewer", "user")


def test_the_command_line_creates_invites(users, capsys):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "manage_users", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "manage_users.py")
    manage_users = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manage_users)
    manage_users.main(["create-invite", "--tenant", "acme", "--uses", "3", "--hours", "24"])
    code = capsys.readouterr().out.strip().rsplit(" ", 1)[-1]
    assert signup("ann", code).json()["user"]["tenant_id"] == "acme"
