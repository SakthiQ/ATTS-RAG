"""Accounts and authentication: a SQLite user store, bcrypt password hashes and signed JWT access tokens.

Every API call except /health and the /auth sign-up, log-in and password-reset endpoints needs a token. The
token names the user, so a client can no longer claim another identity with a header: Layer 1's per-user
suspicion and the tenant filters of Layers 2-3 all key on the authenticated user. Role, tenant and status are
read from the database on every request, never trusted from the token.

Roles (lowest to highest):
  viewer        asks questions; reads their own tenant's documents
  user          also uploads documents into their own tenant, at the 'unknown' source tier
  tenant_admin  administers their own tenant: source tiers, new versions, deletion, unmasked personal data,
                and the tenant's accounts (roles up to tenant_admin)
  super_admin   all of the above in every tenant, and super_admin accounts
The old single 'admin' role is now super_admin: existing accounts are migrated when the store opens, and
'admin' is still accepted as another name for it.

Clearance (1 general, 2 confidential, 3 restricted): an account only retrieves, lists and previews documents
whose clearance tier (acl_tier) is at most its own. Accounts start at 1; both admin roles always have 3.

Settings (environment variables, read at use time so tests can change them):
  JWT_SECRET                  HMAC key for tokens. If unset, a random key is generated once and kept in
                              data/.jwt_secret, so tokens survive a restart on the same machine.
  JWT_EXPIRE_MINUTES          access-token lifetime (default 15); clients renew it with the refresh token
  REFRESH_EXPIRE_DAYS         refresh-token lifetime (default 7)
  USER_DB_PATH                SQLite file (default data/users.db)
  SIGNUP_MODE                 invite (default): an invite code is required and decides the tenant;
                              open: anyone may also sign up without a code, into the shared default tenant
                              (which holds every document indexed before tenants existed); closed: only an
                              admin creates accounts.
  INVITE_CODES                deprecated: "code:tenant,code2:tenant2" codes that never expire. Still accepted;
                              use invites made with POST /auth/invites (or manage_users.py create-invite)
                              instead, which expire, have a use limit and can be revoked
  SIGNUPS_PER_SUBNET_PER_HOUR new accounts allowed per client /24 network per hour (default 5)
  LOGIN_FAILURES_PER_SUBNET_PER_HOUR
                              failed log-ins and password-reset attempts (over all usernames) allowed per
                              client /24 network per hour before that network is refused (default 30)
  RESET_CODE_MINUTES          lifetime of an admin-issued password-reset code (default 30)
  BCRYPT_ROUNDS               bcrypt cost (default 12)
  ATTS_ADMIN_USERNAME / ATTS_ADMIN_PASSWORD
                              if both are set, this super_admin account is created at start-up when missing

Logout is enforced on the server: each user has a token version that is written into every token and bumped
on logout, password change, and any change of role or status, so earlier tokens stop working immediately.

Sessions: a log-in returns a short-lived access token (JWT) and a refresh token (random, stored only as a
SHA-256 hash). POST /auth/refresh swaps a refresh token for a new pair; each refresh token works once. A refresh
token presented a second time means it was copied, so every token descended from that log-in is revoked.

Failed log-ins are counted per (username, client /24 network). Five failures within LOCKOUT_SECONDS lock the
account for that network only, so someone who knows a username (the admin's, say) can no longer lock its owner
out by typing wrong passwords from elsewhere. Guessing across many usernames is capped separately per network.
Behind a reverse proxy, run uvicorn with --proxy-headers so the client address is the real one.

Account events (log-ins, lock-outs, sign-ups, log-outs, account changes) go to the audit log (app/audit.py).
"""
import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from loguru import logger
from pydantic import BaseModel

from .audit import audit

DEFAULT_TENANT = "default_tenant"
ROLES = ("viewer", "user", "tenant_admin", "super_admin")
ADMIN_ROLES = ("tenant_admin", "super_admin")
ROLE_ALIASES = {"admin": "super_admin"}   # the former single admin role
ACCOUNT_STATUSES = ("active", "disabled")
INVITE_ROLES = ("viewer", "user")        # an invite never creates an admin
MAX_INVITE_USES = 1000
MAX_INVITE_HOURS = 90 * 24
CLEARANCE_LEVELS = (1, 2, 3)
MAX_CLEARANCE = max(CLEARANCE_LEVELS)
USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,39}$")
TENANT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 256
MAX_FAILED_LOGINS = 5          # failures from one network within LOCKOUT_SECONDS that lock the account there
LOCKOUT_SECONDS = 15 * 60
NETWORK_WINDOW_SECONDS = 60 * 60
RESET_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O or 1/I, so a code read aloud survives
RESET_CODE_LENGTH = 12                                    # 60 bits
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "atts-rag"


class AuthError(Exception):
    """A request that must be refused; `status_code` is what the API returns, `reason` a short machine label."""

    def __init__(self, message: str, status_code: int = 400, reason: str = "invalid"):
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason


def validate_clearance(clearance: Any) -> int:
    if isinstance(clearance, bool) or not isinstance(clearance, int) or clearance not in CLEARANCE_LEVELS:
        raise AuthError(f"Clearance must be one of {', '.join(map(str, CLEARANCE_LEVELS))}.")
    return clearance


def normalize_role(role: Optional[str]) -> str:
    name = (role or "").strip().lower()
    name = ROLE_ALIASES.get(name, name)
    if name not in ROLES:
        raise AuthError(f"Unknown role '{role}'. Allowed: {', '.join(ROLES)}.")
    return name


@dataclass(frozen=True)
class User:
    id: str
    username: str
    tenant_id: str
    role: str
    status: str = "active"
    token_version: int = 0
    clearance: int = 1

    @property
    def effective_clearance(self) -> int:
        """The clearance used for access: admins administer every document of their tenant, so they see all tiers."""
        return MAX_CLEARANCE if self.is_admin else self.clearance

    @property
    def is_super_admin(self) -> bool:
        return self.role == "super_admin"

    @property
    def is_admin(self) -> bool:
        """An admin of at least their own tenant (tenant_admin or super_admin)."""
        return self.role in ADMIN_ROLES

    @property
    def can_upload(self) -> bool:
        return self.role != "viewer"

    def manages(self, tenant_id: str) -> bool:
        """May administer documents and accounts of `tenant_id`."""
        return self.is_super_admin or (self.role == "tenant_admin" and tenant_id == self.tenant_id)

    def can_assign(self, role: str) -> bool:
        """May give an account this role: a tenant_admin up to tenant_admin, a super_admin any."""
        return self.is_super_admin or (self.role == "tenant_admin" and role != "super_admin")

    def public(self) -> Dict[str, Any]:
        return {"id": self.id, "username": self.username, "tenant_id": self.tenant_id, "role": self.role,
                "clearance": self.effective_clearance}


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def _prehash(password: str) -> bytes:
    # bcrypt reads at most 72 bytes; hashing first lets long passwords count in full
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    rounds = int(os.getenv("BCRYPT_ROUNDS", "12"))
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt(rounds=rounds)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode("ascii"))
    except ValueError:
        return False


def validate_password(password: str) -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")


def normalize_username(username: str) -> str:
    name = (username or "").strip().lower()
    if not USERNAME_PATTERN.match(name):
        raise AuthError("Username must be 3-40 characters: lowercase letters, digits, '.', '_' or '-', starting with a letter or digit.")
    return name


def normalize_tenant(tenant_id: str) -> str:
    tenant = (tenant_id or "").strip().lower()
    if not TENANT_PATTERN.match(tenant):
        raise AuthError("tenant_id may contain only lowercase letters, digits, '_' and '-' (max 100 characters).")
    return tenant


def subnet_of(client_ip: str) -> str:
    return ".".join(client_ip.split(".")[:3]) + ".0/24" if "." in client_ip else client_ip


def _code_digest_raw(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _code_digest(code: str) -> str:
    """Reset codes are random and high-entropy, so a plain SHA-256 is enough to keep them unreadable at rest."""
    return hashlib.sha256(re.sub(r"[\s-]", "", code or "").upper().encode("ascii", "ignore")).hexdigest()


# ---------------------------------------------------------------------------
# User store
# ---------------------------------------------------------------------------

class UserStore:
    """SQLite-backed accounts. Safe to share between request threads (one connection per call, writes serialised)."""

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        # Compared against when the username does not exist, so a miss costs as much time as a wrong password
        self._dummy_hash = hash_password(secrets.token_hex(16))
        with self._db() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS users (
                       id TEXT PRIMARY KEY,
                       username TEXT UNIQUE NOT NULL,
                       password_hash TEXT NOT NULL,
                       tenant_id TEXT NOT NULL,
                       role TEXT NOT NULL,
                       status TEXT NOT NULL DEFAULT 'active',
                       token_version INTEGER NOT NULL DEFAULT 0,
                       failed_logins INTEGER NOT NULL DEFAULT 0,
                       locked_until REAL NOT NULL DEFAULT 0,
                       signup_subnet TEXT,
                       created_at REAL NOT NULL)""")
            # failed_logins / locked_until above are no longer used (a lock is per network now); kept so existing
            # databases need no migration
            db.execute("""CREATE TABLE IF NOT EXISTS login_failures (
                       username TEXT NOT NULL,
                       subnet TEXT NOT NULL,
                       at REAL NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS login_failures_by_user ON login_failures (username, subnet, at)")
            db.execute("CREATE INDEX IF NOT EXISTS login_failures_by_subnet ON login_failures (subnet, at)")
            db.execute("""CREATE TABLE IF NOT EXISTS refresh_tokens (
                       token_hash TEXT PRIMARY KEY,
                       user_id TEXT NOT NULL,
                       family TEXT NOT NULL,
                       token_version INTEGER NOT NULL,
                       issued_at REAL NOT NULL,
                       expires_at REAL NOT NULL,
                       used_at REAL,
                       revoked INTEGER NOT NULL DEFAULT 0)""")
            db.execute("CREATE INDEX IF NOT EXISTS refresh_tokens_by_family ON refresh_tokens (family)")
            db.execute("CREATE INDEX IF NOT EXISTS refresh_tokens_by_user ON refresh_tokens (user_id)")
            db.execute("""CREATE TABLE IF NOT EXISTS invites (
                       id TEXT PRIMARY KEY,
                       code_hash TEXT UNIQUE NOT NULL,
                       tenant_id TEXT NOT NULL,
                       role TEXT NOT NULL,
                       max_uses INTEGER NOT NULL,
                       uses INTEGER NOT NULL DEFAULT 0,
                       expires_at REAL NOT NULL,
                       created_by TEXT,
                       created_at REAL NOT NULL,
                       revoked INTEGER NOT NULL DEFAULT 0)""")
            self._migrate(db)

    @staticmethod
    def _add_column(db: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        if column not in {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _migrate(self, db: sqlite3.Connection) -> None:
        """Idempotent, non-destructive upgrades of an existing database."""
        self._add_column(db, "users", "reset_code_hash", "TEXT")
        self._add_column(db, "users", "reset_expires", "REAL NOT NULL DEFAULT 0")
        self._add_column(db, "users", "clearance", "INTEGER NOT NULL DEFAULT 1")
        migrated = db.execute("UPDATE users SET role = 'super_admin' WHERE role = 'admin'").rowcount
        if migrated:
            logger.info(f"Migrated {migrated} 'admin' account(s) to role 'super_admin'.")

    @contextmanager
    def _db(self):
        """A connection that commits on success, rolls back on error and is always closed (Windows keeps open files locked)."""
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _user(row: sqlite3.Row) -> User:
        return User(id=row["id"], username=row["username"], tenant_id=row["tenant_id"], role=row["role"],
                    status=row["status"], token_version=row["token_version"], clearance=row["clearance"])

    # ---- accounts ---------------------------------------------------------------------------------------

    def create_user(self, username: str, password: str, tenant_id: str = DEFAULT_TENANT, role: str = "user",
                    signup_subnet: Optional[str] = None, clearance: int = 1) -> User:
        name = normalize_username(username)
        validate_password(password)
        tenant = normalize_tenant(tenant_id)
        role = normalize_role(role)
        clearance = validate_clearance(clearance)
        password_hash = hash_password(password)
        user_id = uuid.uuid4().hex
        with self._lock, self._db() as db:
            try:
                db.execute("INSERT INTO users (id, username, password_hash, tenant_id, role, signup_subnet, created_at, "
                           "clearance) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (user_id, name, password_hash, tenant, role, signup_subnet, time.time(), clearance))
            except sqlite3.IntegrityError:
                raise AuthError("That username is already taken.", status.HTTP_409_CONFLICT, "taken")
        logger.info(f"Account created: {name} (tenant {tenant}, role {role})")
        return User(id=user_id, username=name, tenant_id=tenant, role=role, clearance=clearance)

    def get(self, user_id: str) -> Optional[User]:
        with self._db() as db:
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._user(row) if row else None

    def get_by_username(self, username: str) -> Optional[User]:
        with self._db() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", ((username or "").strip().lower(),)).fetchone()
        return self._user(row) if row else None

    def list_users(self, tenant_id: Optional[str] = None) -> List[User]:
        with self._db() as db:
            if tenant_id is None:
                rows = db.execute("SELECT * FROM users ORDER BY created_at")
            else:
                rows = db.execute("SELECT * FROM users WHERE tenant_id = ? ORDER BY created_at", (tenant_id,))
            return [self._user(r) for r in rows]

    def count_active(self, role: str) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM users WHERE role = ? AND status = 'active'", (role,)).fetchone()[0]

    def update_user(self, user_id: str, role: Optional[str] = None, account_status: Optional[str] = None,
                    clearance: Optional[int] = None) -> User:
        """Changes role, status and/or clearance. Bumps the token version, so the account's sessions end at once."""
        sets, values = [], []
        if clearance is not None:
            sets.append("clearance = ?"); values.append(validate_clearance(clearance))
        if role is not None:
            sets.append("role = ?"); values.append(normalize_role(role))
        if account_status is not None:
            if account_status not in ACCOUNT_STATUSES:
                raise AuthError(f"Unknown status '{account_status}'. Allowed: {', '.join(ACCOUNT_STATUSES)}.")
            sets.append("status = ?"); values.append(account_status)
        if not sets:
            raise AuthError("Nothing to change.")
        with self._lock, self._db() as db:
            db.execute(f"UPDATE users SET {', '.join(sets)}, token_version = token_version + 1 WHERE id = ?",
                       (*values, user_id))
        user = self.get(user_id)
        if user is None:
            raise AuthError("No such account.", status.HTTP_404_NOT_FOUND, "not_found")
        return user

    def set_password(self, user_id: str, password: str) -> User:
        """Sets a new password, withdraws any reset code and ends every session of the account."""
        validate_password(password)
        password_hash = hash_password(password)
        with self._lock, self._db() as db:
            db.execute("UPDATE users SET password_hash = ?, reset_code_hash = NULL, reset_expires = 0, "
                       "token_version = token_version + 1 WHERE id = ?", (password_hash, user_id))
        return self.get(user_id)

    def delete_user(self, user_id: str) -> None:
        with self._lock, self._db() as db:
            db.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def bump_token_version(self, user_id: str) -> None:
        with self._lock, self._db() as db:
            db.execute("UPDATE users SET token_version = token_version + 1 WHERE id = ?", (user_id,))

    def recent_signups(self, subnet: str, seconds: float = 3600) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM users WHERE signup_subnet = ? AND created_at > ?",
                              (subnet, time.time() - seconds)).fetchone()[0]

    # ---- log-in throttling --------------------------------------------------------------------------------

    @staticmethod
    def _throttled(db: sqlite3.Connection, name: str, subnet: str, now: float) -> Optional[AuthError]:
        """The refusal for a network or (account, network) that failed too often, or None."""
        network_limit = int(os.getenv("LOGIN_FAILURES_PER_SUBNET_PER_HOUR", "30"))
        network_failures = db.execute("SELECT COUNT(*) FROM login_failures WHERE subnet = ? AND at > ?",
                                      (subnet, now - NETWORK_WINDOW_SECONDS)).fetchone()[0]
        if network_failures >= network_limit:
            return AuthError("Too many failed log-ins from this network. Try again later.",
                             status.HTTP_429_TOO_MANY_REQUESTS, "network_throttled")
        recent = [r["at"] for r in db.execute(
            "SELECT at FROM login_failures WHERE username = ? AND subnet = ? AND at > ? ORDER BY at DESC",
            (name, subnet, now - LOCKOUT_SECONDS))]
        if len(recent) >= MAX_FAILED_LOGINS:
            # Locked until the MAX_FAILED_LOGINS-th most recent failure leaves the window
            minutes = int((recent[MAX_FAILED_LOGINS - 1] + LOCKOUT_SECONDS - now) // 60) + 1
            return AuthError(f"Too many failed log-ins for this account. Try again in {minutes} minute(s).",
                             status.HTTP_429_TOO_MANY_REQUESTS, "locked")
        return None

    @staticmethod
    def _record_failure(db: sqlite3.Connection, name: str, subnet: str, now: float) -> bool:
        """Records a failure; True when it is the one that locks the account for this network."""
        db.execute("INSERT INTO login_failures (username, subnet, at) VALUES (?, ?, ?)", (name, subnet, now))
        db.execute("DELETE FROM login_failures WHERE at < ?", (now - max(LOCKOUT_SECONDS, NETWORK_WINDOW_SECONDS),))
        count = db.execute("SELECT COUNT(*) FROM login_failures WHERE username = ? AND subnet = ? AND at > ?",
                           (name, subnet, now - LOCKOUT_SECONDS)).fetchone()[0]
        return count == MAX_FAILED_LOGINS

    def authenticate(self, username: str, password: str, client_ip: Optional[str] = None) -> User:
        """Returns the user for a correct username and password; raises AuthError otherwise.

        Unknown username and wrong password give the same message. After MAX_FAILED_LOGINS failures from one
        client network within LOCKOUT_SECONDS, that network cannot log in to the account (even with the correct
        password) until the oldest of those failures is LOCKOUT_SECONDS old; other networks are not affected.
        A network with LOGIN_FAILURES_PER_SUBNET_PER_HOUR failures over all usernames is refused for the hour.
        """
        name = (username or "").strip().lower()
        subnet = subnet_of(client_ip or "unknown")
        now = time.time()
        error: Optional[AuthError] = None
        locked_now = False
        # Errors are raised only after the block: raising inside it would roll back the recorded failure
        with self._lock, self._db() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", (name,)).fetchone()
            error = self._throttled(db, name, subnet, now)
            if error is None:
                # An unknown username costs a bcrypt check too, so it takes as long as a wrong password
                correct = verify_password(password or "", row["password_hash"] if row is not None else self._dummy_hash)
                if row is None or not correct:
                    # Unknown usernames are recorded too, so guessing names counts against the network
                    locked_now = self._record_failure(db, name, subnet, now) and row is not None
                    error = AuthError("Incorrect username or password.", status.HTTP_401_UNAUTHORIZED, "bad_credentials")
                elif row["status"] != "active":
                    error = AuthError("This account is disabled.", status.HTTP_403_FORBIDDEN, "disabled")
                else:
                    # A correct log-in clears this network's failures for the account (not the network-wide count)
                    db.execute("DELETE FROM login_failures WHERE username = ? AND subnet = ?", (name, subnet))
        if locked_now:
            logger.warning(f"Account locked for network {subnet} after {MAX_FAILED_LOGINS} failed log-ins: {name}")
            audit("account_locked", actor=None, tenant=row["tenant_id"], target=name, subnet=subnet)
        if error is not None:
            raise error
        return self._user(row)

    # ---- refresh tokens ------------------------------------------------------------------------------------

    def issue_refresh_token(self, user: User, family: Optional[str] = None) -> Tuple[str, int]:
        """(token, lifetime in seconds). `family` groups the tokens descended from one log-in."""
        lifetime = int(float(os.getenv("REFRESH_EXPIRE_DAYS", "7")) * 86400)
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock, self._db() as db:
            db.execute("DELETE FROM refresh_tokens WHERE expires_at < ?", (now - 86400,))
            db.execute("INSERT INTO refresh_tokens (token_hash, user_id, family, token_version, issued_at, expires_at) "
                       "VALUES (?, ?, ?, ?, ?, ?)",
                       (_code_digest_raw(token), user.id, family or uuid.uuid4().hex, user.token_version, now, now + lifetime))
        return token, lifetime

    def rotate_refresh_token(self, token: str) -> Tuple[User, str, int]:
        """Spends a refresh token: returns (user, new refresh token, its lifetime) or raises AuthError (401).

        A token that was already spent revokes its whole family (the reason is "reused"): either the client or a
        thief holds a copy, and the server cannot tell which, so both must log in again."""
        digest = _code_digest_raw(token or "")
        now = time.time()
        error: Optional[AuthError] = None
        with self._lock, self._db() as db:
            row = db.execute("SELECT * FROM refresh_tokens WHERE token_hash = ?", (digest,)).fetchone()
            if row is None or row["revoked"]:
                error = AuthError("Invalid refresh token. Please log in again.", status.HTTP_401_UNAUTHORIZED, "invalid")
            elif row["used_at"] is not None:
                db.execute("UPDATE refresh_tokens SET revoked = 1 WHERE family = ?", (row["family"],))
                error = AuthError("This refresh token was already used; all sessions from that log-in have ended. "
                                  "Please log in again.", status.HTTP_401_UNAUTHORIZED, "reused")
            elif row["expires_at"] <= now:
                error = AuthError("Session expired. Please log in again.", status.HTTP_401_UNAUTHORIZED, "expired")
            else:
                db.execute("UPDATE refresh_tokens SET used_at = ? WHERE token_hash = ?", (now, digest))
        if error is not None:
            error.user_id = row["user_id"] if row is not None else None
            raise error
        user = self.get(row["user_id"])
        # Logged out, password changed, role or status changed, disabled or deleted since this token was issued
        if user is None or user.status != "active" or user.token_version != row["token_version"]:
            raise AuthError("Session is no longer valid. Please log in again.", status.HTTP_401_UNAUTHORIZED, "stale")
        new_token, lifetime = self.issue_refresh_token(user, family=row["family"])
        return user, new_token, lifetime

    def revoke_refresh_tokens(self, user_id: str) -> None:
        with self._lock, self._db() as db:
            db.execute("UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ?", (user_id,))

    # ---- invites -----------------------------------------------------------------------------------------

    def create_invite(self, tenant_id: str, role: str = "user", max_uses: int = 1, expires_in_hours: float = 168,
                      created_by: Optional[str] = None) -> Dict[str, Any]:
        """A new invite; the returned dict holds the code, which is stored only hashed and shown only now."""
        tenant = normalize_tenant(tenant_id)
        role = normalize_role(role)
        if role not in INVITE_ROLES:
            raise AuthError(f"An invite can only create {' or '.join(INVITE_ROLES)} accounts.")
        if not isinstance(max_uses, int) or not 1 <= max_uses <= MAX_INVITE_USES:
            raise AuthError(f"max_uses must be between 1 and {MAX_INVITE_USES}.")
        if not 0 < float(expires_in_hours) <= MAX_INVITE_HOURS:
            raise AuthError(f"expires_in_hours must be more than 0 and at most {MAX_INVITE_HOURS}.")
        raw = "".join(secrets.choice(RESET_CODE_ALPHABET) for _ in range(RESET_CODE_LENGTH))
        code = "INV-" + "-".join(raw[i:i + 4] for i in range(0, RESET_CODE_LENGTH, 4))
        invite = {"id": uuid.uuid4().hex[:12], "tenant_id": tenant, "role": role, "max_uses": max_uses, "uses": 0,
                  "expires_at": time.time() + float(expires_in_hours) * 3600, "created_by": created_by,
                  "created_at": time.time(), "revoked": False}
        with self._lock, self._db() as db:
            db.execute("INSERT INTO invites (id, code_hash, tenant_id, role, max_uses, expires_at, created_by, created_at) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (invite["id"], _code_digest(code), tenant, role, max_uses, invite["expires_at"], created_by,
                        invite["created_at"]))
        return invite | {"code": code}

    @staticmethod
    def _invite(row: sqlite3.Row) -> Dict[str, Any]:
        out = {k: row[k] for k in ("id", "tenant_id", "role", "max_uses", "uses", "expires_at", "created_by", "created_at")}
        out["revoked"] = bool(row["revoked"])
        out["usable"] = not out["revoked"] and out["uses"] < out["max_uses"] and out["expires_at"] > time.time()
        return out

    def list_invites(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._db() as db:
            if tenant_id is None:
                rows = db.execute("SELECT * FROM invites ORDER BY created_at")
            else:
                rows = db.execute("SELECT * FROM invites WHERE tenant_id = ? ORDER BY created_at", (tenant_id,))
            return [self._invite(r) for r in rows]

    def get_invite(self, invite_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as db:
            row = db.execute("SELECT * FROM invites WHERE id = ?", (invite_id,)).fetchone()
        return self._invite(row) if row else None

    def find_usable_invite(self, code: str) -> Optional[Dict[str, Any]]:
        """The invite for `code` if it is unrevoked, unexpired and has uses left."""
        with self._db() as db:
            row = db.execute("SELECT * FROM invites WHERE code_hash = ?", (_code_digest(code),)).fetchone()
        invite = self._invite(row) if row else None
        return invite if invite and invite["usable"] else None

    def consume_invite(self, invite_id: str) -> bool:
        """Uses up one use; False if it ran out (or expired, or was revoked) in the meantime."""
        with self._lock, self._db() as db:
            return db.execute("UPDATE invites SET uses = uses + 1 WHERE id = ? AND revoked = 0 AND uses < max_uses "
                              "AND expires_at > ?", (invite_id, time.time())).rowcount == 1

    def revoke_invite(self, invite_id: str) -> None:
        with self._lock, self._db() as db:
            db.execute("UPDATE invites SET revoked = 1 WHERE id = ?", (invite_id,))

    # ---- password reset codes ------------------------------------------------------------------------------

    def issue_reset_code(self, user_id: str) -> Tuple[str, int]:
        """(code, lifetime in seconds). The code is single-use, replaces any earlier one and is stored hashed."""
        lifetime = int(float(os.getenv("RESET_CODE_MINUTES", "30")) * 60)
        raw = "".join(secrets.choice(RESET_CODE_ALPHABET) for _ in range(RESET_CODE_LENGTH))
        code = "-".join(raw[i:i + 4] for i in range(0, RESET_CODE_LENGTH, 4))
        with self._lock, self._db() as db:
            db.execute("UPDATE users SET reset_code_hash = ?, reset_expires = ? WHERE id = ?",
                       (_code_digest(code), time.time() + lifetime, user_id))
        return code, lifetime

    def reset_password_with_code(self, username: str, code: str, new_password: str,
                                 client_ip: Optional[str] = None) -> User:
        """Sets a new password if `code` is the account's current, unexpired reset code. Wrong codes count as
        failed log-ins, so codes cannot be guessed faster than passwords."""
        validate_password(new_password)
        name = (username or "").strip().lower()
        subnet = subnet_of(client_ip or "unknown")
        now = time.time()
        with self._lock, self._db() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", (name,)).fetchone()
            error = self._throttled(db, name, subnet, now)
            if error is None:
                valid = (row is not None and row["reset_code_hash"] and row["reset_expires"] > now
                         and hmac.compare_digest(row["reset_code_hash"], _code_digest(code)))
                if not valid:
                    self._record_failure(db, name, subnet, now)
                    error = AuthError("Invalid or expired reset code.", status.HTTP_400_BAD_REQUEST, "bad_code")
                elif row["status"] != "active":
                    error = AuthError("This account is disabled.", status.HTTP_403_FORBIDDEN, "disabled")
        if error is not None:
            raise error
        user = self.set_password(row["id"], new_password)
        with self._lock, self._db() as db:
            db.execute("DELETE FROM login_failures WHERE username = ?", (name,))
        return user


def check_not_last_super_admin(store: "UserStore", target: User, removing: bool) -> None:
    """Refuses to demote, disable or delete the last active super_admin (nobody could administer the system)."""
    if removing and target.is_super_admin and target.status == "active" and store.count_active("super_admin") <= 1:
        raise AuthError("This is the last active super admin; create another one first.",
                        status.HTTP_409_CONFLICT, "last_super_admin")


_store: Optional[UserStore] = None
_store_lock = threading.Lock()


def get_store() -> UserStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = UserStore(os.getenv("USER_DB_PATH", os.path.join("data", "users.db")))
        return _store


def set_store(store: Optional[UserStore]) -> None:
    """Replaces the process-wide store (tests use a temporary database)."""
    global _store
    with _store_lock:
        _store = store


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

_secret_cache: Optional[str] = None


def _jwt_secret() -> str:
    global _secret_cache
    configured = os.getenv("JWT_SECRET")
    if configured:
        if len(configured) < 32:
            logger.warning("JWT_SECRET is shorter than 32 characters; use a long random value.")
        return configured
    if _secret_cache is None:
        path = os.path.join("data", ".jwt_secret")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _secret_cache = f.read().strip()
        if not _secret_cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _secret_cache = secrets.token_urlsafe(48)
            with open(path, "w", encoding="utf-8") as f:
                f.write(_secret_cache)
            logger.warning(f"JWT_SECRET not set; generated one and stored it in {path}.")
    return _secret_cache


def issue_token(user: User) -> Tuple[str, int]:
    """(token, lifetime in seconds)."""
    lifetime = int(float(os.getenv("JWT_EXPIRE_MINUTES", "15")) * 60)
    now = int(time.time())
    claims = {"sub": user.id, "username": user.username, "tenant": user.tenant_id, "role": user.role,
              "ver": user.token_version, "iss": JWT_ISSUER, "iat": now, "exp": now + lifetime,
              "jti": secrets.token_hex(8)}
    return jwt.encode(claims, _jwt_secret(), algorithm=JWT_ALGORITHM), lifetime


def decode_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM], issuer=JWT_ISSUER,
                      options={"require": ["exp", "iat", "sub", "iss"]})


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)
_UNAUTHORIZED = {"WWW-Authenticate": "Bearer"}


def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> User:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in. Send 'Authorization: Bearer <token>'.", _UNAUTHORIZED)
    try:
        claims = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired. Please log in again.", _UNAUTHORIZED)
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token.", _UNAUTHORIZED)
    user = get_store().get(claims["sub"])
    # A revoked (logged out), disabled or deleted account invalidates every token issued before
    if user is None or user.status != "active" or user.token_version != claims.get("ver"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session is no longer valid. Please log in again.", _UNAUTHORIZED)
    return user


def require_tenant_admin(user: User = Depends(get_current_user)) -> User:
    """A tenant_admin or super_admin. Endpoints must still check `user.manages(tenant)` for the tenant concerned."""
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only an admin can do this.")
    return user


def require_super_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_super_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only a super admin can do this.")
    return user


# ---------------------------------------------------------------------------
# Sign-up policy and start-up bootstrap
# ---------------------------------------------------------------------------

def _invite_codes() -> Dict[str, str]:
    codes = {}
    for item in os.getenv("INVITE_CODES", "").split(","):
        if ":" in item:
            code, tenant = item.split(":", 1)
            if code.strip() and tenant.strip():
                codes[code.strip()] = tenant.strip().lower()
    return codes


_warned_legacy_invites = False


def signup_policy(invite_code: Optional[str]) -> Tuple[str, str, Optional[str]]:
    """(tenant, role, invite id) for a self-service sign-up, or AuthError if sign-up is not allowed.

    Invites in the database come first; a code from the deprecated INVITE_CODES setting still works (role user)."""
    global _warned_legacy_invites
    mode = os.getenv("SIGNUP_MODE", "invite").strip().lower()
    if mode == "closed":
        raise AuthError("Sign-up is disabled. Ask an administrator for an account.", status.HTTP_403_FORBIDDEN)
    if invite_code:
        invite = get_store().find_usable_invite(invite_code)
        if invite is not None:
            return invite["tenant_id"], invite["role"], invite["id"]
        tenant = _invite_codes().get(invite_code.strip())
        if tenant is not None:
            if not _warned_legacy_invites:
                logger.warning("A sign-up used an INVITE_CODES code. These never expire; create invites with "
                               "POST /auth/invites instead and empty INVITE_CODES.")
                _warned_legacy_invites = True
            return tenant, "user", None
        raise AuthError("Invalid or expired invite code.", status.HTTP_400_BAD_REQUEST, "bad_invite")
    if mode == "invite":
        raise AuthError("An invite code is required to sign up.", status.HTTP_403_FORBIDDEN)
    return DEFAULT_TENANT, "user", None


def bootstrap_admin() -> None:
    """Creates the super_admin named by ATTS_ADMIN_USERNAME / ATTS_ADMIN_PASSWORD if it does not exist yet."""
    username, password = os.getenv("ATTS_ADMIN_USERNAME"), os.getenv("ATTS_ADMIN_PASSWORD")
    if not username or not password:
        return
    store = get_store()
    if store.get_by_username(username) is None:
        user = store.create_user(username, password, tenant_id=os.getenv("ATTS_ADMIN_TENANT", DEFAULT_TENANT),
                                 role="super_admin")
        audit("account_created", actor="bootstrap", tenant=user.tenant_id, target=user.username, role=user.role)


# ---------------------------------------------------------------------------
# Endpoints: sign-up, log-in, log-out (account administration is in app/auth_admin.py)
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["Auth"])


class SignupRequest(BaseModel):
    username: str
    password: str
    invite_code: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


def token_response(user: User, refresh_token: Optional[str] = None, refresh_lifetime: Optional[int] = None) -> Dict[str, Any]:
    """An access token plus a refresh token: a new log-in family unless a rotated `refresh_token` is passed."""
    token, lifetime = issue_token(user)
    if refresh_token is None:
        refresh_token, refresh_lifetime = get_store().issue_refresh_token(user)
    return {"access_token": token, "token_type": "bearer", "expires_in": lifetime, "refresh_token": refresh_token,
            "refresh_expires_in": refresh_lifetime, "user": user.public()}


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "127.0.0.1"


def http_error(exc: AuthError) -> HTTPException:
    return HTTPException(exc.status_code, str(exc), _UNAUTHORIZED if exc.status_code == 401 else None)


def _login(username: str, password: str, ip: str) -> Dict[str, Any]:
    store = get_store()
    try:
        user = store.authenticate(username, password, client_ip=ip)
    except AuthError as exc:
        # The attempted name is recorded only if such an account exists: a password typed into the username
        # field must not end up in the audit log
        known = store.get_by_username(username)
        audit("login", actor=known.username if known else None, tenant=known.tenant_id if known else None,
              outcome=exc.reason, subnet=subnet_of(ip))
        raise http_error(exc)
    audit("login", actor=user.username, tenant=user.tenant_id, subnet=subnet_of(ip))
    return token_response(user)


@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest, request: Request):
    """Creates an account and logs it in. Tenant and role come from the invite (or the sign-up policy), never
    from the caller."""
    subnet = subnet_of(client_ip(request))
    limit = int(os.getenv("SIGNUPS_PER_SUBNET_PER_HOUR", "5"))
    store = get_store()
    try:
        tenant, role, invite_id = signup_policy(body.invite_code)
        if store.recent_signups(subnet) >= limit:
            raise AuthError("Too many accounts were created from this network recently. Try again later.",
                            status.HTTP_429_TOO_MANY_REQUESTS)
        user = store.create_user(body.username, body.password, tenant_id=tenant, role=role, signup_subnet=subnet)
        if invite_id is not None and not store.consume_invite(invite_id):
            store.delete_user(user.id)   # the last use went to someone else a moment ago
            raise AuthError("Invalid or expired invite code.", status.HTTP_400_BAD_REQUEST, "bad_invite")
    except AuthError as exc:
        raise http_error(exc)
    audit("signup", actor=user.username, tenant=user.tenant_id, subnet=subnet, role=user.role, invite_id=invite_id,
          legacy_invite=bool(body.invite_code) and invite_id is None)
    return token_response(user)


@router.post("/login")
def login(body: LoginRequest, request: Request):
    """Logs in with a JSON body and returns a bearer token."""
    return _login(body.username, body.password, client_ip(request))


@router.post("/token", include_in_schema=True)
def token(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow (form fields), used by the Swagger UI 'Authorize' button."""
    return _login(form.username, form.password, client_ip(request))


@router.post("/refresh")
def refresh(body: RefreshRequest):
    """Swaps a refresh token for a new access token and a new refresh token (the old one stops working)."""
    store = get_store()
    try:
        user, new_refresh, lifetime = store.rotate_refresh_token(body.refresh_token)
    except AuthError as exc:
        if exc.reason == "reused":
            owner = store.get(getattr(exc, "user_id", None) or "")
            logger.warning(f"Refresh token reuse for {owner.username if owner else 'a deleted account'}; family revoked.")
            audit("refresh_token_reused", actor=owner.username if owner else None,
                  tenant=owner.tenant_id if owner else None, outcome="family_revoked")
        raise http_error(exc)
    return token_response(user, new_refresh, lifetime)


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return user.public()


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(user: User = Depends(get_current_user)):
    """Invalidates every token issued to this user so far, refresh tokens included."""
    get_store().bump_token_version(user.id)
    get_store().revoke_refresh_tokens(user.id)
    audit("logout", actor=user.username, tenant=user.tenant_id)
