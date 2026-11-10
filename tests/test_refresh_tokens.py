"""Short access tokens renewed with single-use refresh tokens, and what happens when a refresh token is stolen."""
import json
import time

from fastapi.testclient import TestClient

from app import auth
from app.main import app
from conftest import TEST_PASSWORD

client = TestClient(app)


def login(username="ann"):
    return client.post("/auth/login", json={"username": username, "password": TEST_PASSWORD}).json()


def refresh(token):
    return client.post("/auth/refresh", json={"refresh_token": token})


def me(body):
    return client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})


def test_login_returns_a_short_access_token_and_a_refresh_token(users):
    users.create("ann")
    body = login()
    assert body["expires_in"] == 15 * 60 and body["refresh_expires_in"] == 7 * 86400
    assert body["refresh_token"] and body["refresh_token"] != body["access_token"]


def test_refresh_tokens_are_stored_only_as_hashes(users):
    users.create("ann")
    token = login()["refresh_token"]
    with users.store._db() as db:
        stored = [r[0] for r in db.execute("SELECT token_hash FROM refresh_tokens")]
    assert token not in stored and len(stored) == 1


def test_a_refresh_gives_new_tokens_and_the_old_refresh_token_stops_working(users):
    users.create("ann")
    first = login()
    second = refresh(first["refresh_token"])
    assert second.status_code == 200
    second = second.json()
    assert second["refresh_token"] != first["refresh_token"] and me(second).status_code == 200
    third = refresh(second["refresh_token"])
    assert third.status_code == 200, "the new refresh token works"


def test_reusing_a_refresh_token_ends_every_session_of_that_login(users, audit_log):
    users.create("ann")
    stolen = login()["refresh_token"]
    legit = refresh(stolen).json()                     # the real client refreshes first
    reuse = refresh(stolen)                            # the thief replays the copy
    assert reuse.status_code == 401 and "already used" in reuse.json()["detail"]
    assert refresh(legit["refresh_token"]).status_code == 401, "the whole family is revoked"
    other_login = login()
    assert refresh(other_login["refresh_token"]).status_code == 200, "other log-ins are not affected"
    events = [json.loads(line)["event"] for line in audit_log.read_text(encoding="utf-8").splitlines()]
    assert "refresh_token_reused" in events


def test_logout_and_account_changes_end_refresh_tokens_too(users):
    users.create("ann")
    body = login()
    assert client.post("/auth/logout", headers={"Authorization": f"Bearer {body['access_token']}"}).status_code == 204
    assert refresh(body["refresh_token"]).status_code == 401
    body = login()
    ann = users.store.get_by_username("ann")
    users.store.update_user(ann.id, role="viewer")
    assert refresh(body["refresh_token"]).status_code == 401


def test_an_expired_or_unknown_refresh_token_is_refused(users, monkeypatch):
    users.create("ann")
    monkeypatch.setenv("REFRESH_EXPIRE_DAYS", "0")
    assert refresh(login()["refresh_token"]).status_code == 401
    assert refresh("not-a-token").status_code == 401


def test_an_expired_access_token_is_renewed_by_refresh(users, monkeypatch):
    users.create("ann")
    monkeypatch.setenv("JWT_EXPIRE_MINUTES", str(1 / 60))   # one second
    body = login()
    time.sleep(1.2)
    assert me(body).status_code == 401
    renewed = refresh(body["refresh_token"]).json()
    monkeypatch.setenv("JWT_EXPIRE_MINUTES", "15")
    assert me(refresh(renewed["refresh_token"]).json()).status_code == 200


def test_a_disabled_account_cannot_refresh(users):
    users.create("ann")
    body = login()
    users.store.update_user(users.store.get_by_username("ann").id, account_status="disabled")
    assert refresh(body["refresh_token"]).status_code == 401
