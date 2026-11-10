"""Two-factor log-in with authenticator-app codes: set-up, log-in, replay refusal, recovery codes, admin reset."""
import pytest
from fastapi.testclient import TestClient

from app import auth, totp
from app.main import app
from conftest import TEST_PASSWORD

client = TestClient(app)


@pytest.fixture(autouse=True)
def totp_key(monkeypatch):
    from cryptography.fernet import Fernet
    monkeypatch.setenv("TOTP_KEY", Fernet.generate_key().decode())


def bearer(body):
    return {"Authorization": f"Bearer {body['access_token']}"}


def login(username="ann", otp=None, password=TEST_PASSWORD):
    body = {"username": username, "password": password}
    if otp is not None:
        body["otp"] = otp
    return client.post("/auth/login", json=body)


def enrol(users, username="ann", role="user"):
    """Sets up 2FA for an account; returns (secret, recovery codes)."""
    h = users.headers(username, role=role)
    secret = client.post("/auth/2fa/setup", headers=h).json()["secret"]
    r = client.post("/auth/2fa/enable", json={"code": totp.current_code(secret)}, headers=h)
    assert r.status_code == 200, r.text
    return secret, r.json()["recovery_codes"]


def test_setup_needs_a_matching_code_before_anything_changes(users):
    h = users.headers("ann")
    setup = client.post("/auth/2fa/setup", headers=h).json()
    assert setup["otpauth_uri"].startswith("otpauth://totp/") and "ATTS-RAG" in setup["otpauth_uri"]
    assert client.post("/auth/2fa/enable", json={"code": "000000"}, headers=h).status_code == 400
    assert login().status_code == 200, "2FA is not on until a code was confirmed"
    with users.store._db() as db:
        stored = db.execute("SELECT totp_pending FROM users WHERE username = 'ann'").fetchone()[0]
    assert setup["secret"] not in stored, "the secret is stored encrypted"


def test_after_enrolment_a_login_needs_the_code(users):
    secret, _ = enrol(users)
    r = login()
    assert r.status_code == 401 and r.headers.get("X-2FA-Required") == "true"
    assert login(otp="123456").status_code == 401
    ok = login(otp=totp.current_code(secret, at=__import__("time").time() + 30))   # enrolling used the current one
    assert ok.status_code == 200 and ok.json()["two_factor"] == {"enabled": True, "required": False}


def test_the_code_is_asked_for_only_after_a_correct_password(users):
    enrol(users)
    r = login(password="wrong password!!")
    assert r.status_code == 401 and "X-2FA-Required" not in r.headers


def test_a_code_cannot_be_replayed(users):
    secret, _ = enrol(users)
    code = totp.current_code(secret, at=__import__("time").time() + 30)            # the next step's code
    assert login(otp=code).status_code == 200
    assert login(otp=code).status_code == 401, "the same code twice"


def test_wrong_codes_count_as_failed_logins(users):
    enrol(users)
    for _ in range(auth.MAX_FAILED_LOGINS):
        assert login(otp="000000").status_code == 401
    assert login(otp="000000").status_code == 429


def test_a_recovery_code_works_once(users):
    _, codes = enrol(users)
    assert len(codes) == totp.RECOVERY_CODE_COUNT
    assert login(otp=codes[0].lower()).status_code == 200
    assert login(otp=codes[0]).status_code == 401
    assert users.store.recovery_codes_left(users.store.get_by_username("ann").id) == totp.RECOVERY_CODE_COUNT - 1


def test_disabling_needs_password_and_code(users):
    secret, codes = enrol(users)
    h = bearer(login(otp=codes[0]).json())
    assert client.post("/auth/2fa/disable", json={"password": "wrong password!!", "code": "000000"}, headers=h).status_code == 400
    code = totp.current_code(secret, at=__import__("time").time() + 30)   # newer than the code used to enrol
    assert client.post("/auth/2fa/disable", json={"password": TEST_PASSWORD, "code": code}, headers=h).status_code == 200
    assert login().status_code == 200


def test_an_admin_resets_a_lost_phone_and_ends_the_sessions(users):
    enrol(users)
    ann_session = users.headers("ann")
    root = users.headers("root", role="super_admin")
    assert client.delete("/auth/users/ann/2fa", headers=root).status_code == 204
    assert client.get("/auth/me", headers=ann_session).status_code == 401
    assert login().status_code == 200
    other = users.headers("otto", role="tenant_admin", tenant="other")
    assert client.delete("/auth/users/ann/2fa", headers=other).status_code == 404


def test_a_required_role_can_only_set_up_2fa_until_it_has(users, monkeypatch):
    monkeypatch.setenv("REQUIRE_2FA_ROLES", "tenant_admin,admin")
    body = client.post("/auth/login", json={"username": users.create("tara", role="tenant_admin").username,
                                            "password": TEST_PASSWORD}).json()
    assert body["two_factor"] == {"enabled": False, "required": True}
    h = bearer(body)
    blocked = client.get("/auth/users", headers=h)
    assert blocked.status_code == 403 and blocked.headers.get("X-2FA-Enrolment-Required") == "true"
    assert client.get("/auth/me", headers=h).status_code == 200
    secret = client.post("/auth/2fa/setup", headers=h).json()["secret"]
    assert client.post("/auth/2fa/enable", json={"code": totp.current_code(secret)}, headers=h).status_code == 200
    assert client.get("/auth/users", headers=h).status_code == 200
    code = totp.current_code(secret, at=__import__("time").time() + 30)
    assert client.post("/auth/2fa/disable", json={"password": TEST_PASSWORD, "code": code}, headers=h).status_code == 403
    # ordinary users are not affected
    assert client.get("/documents", headers=users.headers("ann")).status_code == 200


def test_changing_password_does_not_ask_for_a_code(users):
    enrol(users)
    h = users.headers("ann")
    r = client.post("/auth/password", json={"old_password": TEST_PASSWORD, "new_password": "a brand new password"}, headers=h)
    assert r.status_code == 200


def test_two_factor_events_are_audited(users, audit_log):
    import json
    _, codes = enrol(users)
    login(otp=codes[0])
    events = [json.loads(line)["event"] for line in audit_log.read_text(encoding="utf-8").splitlines()]
    assert "two_factor_enabled" in events and "recovery_code_used" in events
    text = audit_log.read_text(encoding="utf-8")
    assert all(code not in text for code in codes)
