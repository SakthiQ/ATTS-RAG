"""Accounts and authentication: a SQLite user store, bcrypt password hashes and signed JWT access tokens.

Every API call except /health and the /auth sign-up and log-in endpoints needs a token. The token names the
user, their tenant and their role, so a client can no longer claim another identity with a header: Layer 1's
per-user suspicion and the tenant filters of Layers 2-3 all key on the authenticated user.

Settings (environment variables, read at use time so tests can change them):
  JWT_SECRET                  HMAC key for tokens. If unset, a random key is generated once and kept in
                              data/.jwt_secret, so tokens survive a restart on the same machine.
  JWT_EXPIRE_MINUTES          token lifetime (default 60)
  USER_DB_PATH                SQLite file (default data/users.db)
  SIGNUP_MODE                 invite (default): an invite code is required and decides the tenant;
                              open: anyone may also sign up without a code, into the shared default tenant
                              (which holds every document indexed before tenants existed); closed: only an
                              admin creates accounts.
  INVITE_CODES                "code:tenant,code2:tenant2" - invite codes and the tenant each one joins
  SIGNUPS_PER_SUBNET_PER_HOUR new accounts allowed per client /24 network per hour (default 5)
  LOGIN_FAILURES_PER_SUBNET_PER_HOUR
                              failed log-ins (over all usernames) allowed per client /24 network per hour
                              before that network is refused for the rest of the hour (default 30)
  BCRYPT_ROUNDS               bcrypt cost (default 12)
  ATTS_ADMIN_USERNAME / ATTS_ADMIN_PASSWORD
                              if both are set, this admin account is created at start-up when missing

Logout is enforced on the server: each user has a token version that is written into every token and bumped
on logout, so earlier tokens stop working immediately.

Failed log-ins are counted per (username, client /24 network). Five failures within LOCKOUT_SECONDS lock the
account for that network only, so someone who knows a username (the admin's, say) can no longer lock its owner
out by typing wrong passwords from elsewhere. Guessing across many usernames is capped separately per network.
Behind a reverse proxy, run uvicorn with --proxy-headers so the client address is the real one.
"""
import base64
import hashlib
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

DEFAULT_TENANT = "default_tenant"
ROLES = ("user", "admin")
USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,39}$")
TENANT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 256
MAX_FAILED_LOGINS = 5          # failures from one network within LOCKOUT_SECONDS that lock the account there
LOCKOUT_SECONDS = 15 * 60
NETWORK_WINDOW_SECONDS = 60 * 60
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "atts-rag"


class AuthError(Exception):
    """A sign-up or log-in that must be refused; `status_code` is what the API returns."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class User:
    id: str
    username: str
    tenant_id: str
    role: str
    status: str = "active"
    token_version: int = 0

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def public(self) -> Dict[str, Any]:
        return {"id": self.id, "username": self.username, "tenant_id": self.tenant_id, "role": self.role}


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


def subnet_of(client_ip: str) -> str:
    return ".".join(client_ip.split(".")[:3]) + ".0/24" if "." in client_ip else client_ip


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
                    status=row["status"], token_version=row["token_version"])

    def create_user(self, username: str, password: str, tenant_id: str = DEFAULT_TENANT, role: str = "user",
                    signup_subnet: Optional[str] = None) -> User:
        name = normalize_username(username)
        validate_password(password)
        tenant = (tenant_id or "").strip().lower()
        if not TENANT_PATTERN.match(tenant):
            raise AuthError("tenant_id may contain only lowercase letters, digits, '_' and '-' (max 100 characters).")
        if role not in ROLES:
            raise AuthError(f"Unknown role '{role}'. Allowed: {', '.join(ROLES)}.")
        password_hash = hash_password(password)
        user_id = uuid.uuid4().hex
        with self._lock, self._db() as db:
            try:
                db.execute("INSERT INTO users (id, username, password_hash, tenant_id, role, signup_subnet, created_at) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (user_id, name, password_hash, tenant, role, signup_subnet, time.time()))
            except sqlite3.IntegrityError:
                raise AuthError("That username is already taken.", status.HTTP_409_CONFLICT)
        logger.info(f"Account created: {name} (tenant {tenant}, role {role})")
        return User(id=user_id, username=name, tenant_id=tenant, role=role)

    def get(self, user_id: str) -> Optional[User]:
        with self._db() as db:
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._user(row) if row else None

    def get_by_username(self, username: str) -> Optional[User]:
        with self._db() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", ((username or "").strip().lower(),)).fetchone()
        return self._user(row) if row else None

    def list_users(self) -> List[User]:
        with self._db() as db:
            return [self._user(r) for r in db.execute("SELECT * FROM users ORDER BY created_at")]

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
        network_limit = int(os.getenv("LOGIN_FAILURES_PER_SUBNET_PER_HOUR", "30"))
        error: Optional[AuthError] = None
        # Errors are raised only after the block: raising inside it would roll back the recorded failure
        with self._lock, self._db() as db:
            network_failures = db.execute("SELECT COUNT(*) FROM login_failures WHERE subnet = ? AND at > ?",
                                          (subnet, now - NETWORK_WINDOW_SECONDS)).fetchone()[0]
            recent = [r["at"] for r in db.execute(
                "SELECT at FROM login_failures WHERE username = ? AND subnet = ? AND at > ? ORDER BY at DESC",
                (name, subnet, now - LOCKOUT_SECONDS))]
            row = db.execute("SELECT * FROM users WHERE username = ?", (name,)).fetchone()
            if network_failures >= network_limit:
                error = AuthError("Too many failed log-ins from this network. Try again later.",
                                  status.HTTP_429_TOO_MANY_REQUESTS)
            elif len(recent) >= MAX_FAILED_LOGINS:
                # Locked until the MAX_FAILED_LOGINS-th most recent failure leaves the window
                minutes = int((recent[MAX_FAILED_LOGINS - 1] + LOCKOUT_SECONDS - now) // 60) + 1
                error = AuthError(f"Too many failed log-ins for this account. Try again in {minutes} minute(s).",
                                  status.HTTP_429_TOO_MANY_REQUESTS)
            else:
                # An unknown username costs a bcrypt check too, so it takes as long as a wrong password
                correct = verify_password(password or "", row["password_hash"] if row is not None else self._dummy_hash)
                if row is None or not correct:
                    # Unknown usernames are recorded too, so guessing names counts against the network
                    db.execute("INSERT INTO login_failures (username, subnet, at) VALUES (?, ?, ?)", (name, subnet, now))
                    db.execute("DELETE FROM login_failures WHERE at < ?",
                               (now - max(LOCKOUT_SECONDS, NETWORK_WINDOW_SECONDS),))
                    if row is not None and len(recent) + 1 >= MAX_FAILED_LOGINS:
                        logger.warning(f"Account locked for network {subnet} after {len(recent) + 1} failed log-ins: {name}")
                    error = AuthError("Incorrect username or password.", status.HTTP_401_UNAUTHORIZED)
                elif row["status"] != "active":
                    error = AuthError("This account is disabled.", status.HTTP_403_FORBIDDEN)
                else:
                    # A correct log-in clears this network's failures for the account (not the network-wide count)
                    db.execute("DELETE FROM login_failures WHERE username = ? AND subnet = ?", (name, subnet))
        if error is not None:
            raise error
        return self._user(row)

    def bump_token_version(self, user_id: str) -> None:
        with self._lock, self._db() as db:
            db.execute("UPDATE users SET token_version = token_version + 1 WHERE id = ?", (user_id,))

    def recent_signups(self, subnet: str, seconds: float = 3600) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM users WHERE signup_subnet = ? AND created_at > ?",
                              (subnet, time.time() - seconds)).fetchone()[0]


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
    lifetime = int(float(os.getenv("JWT_EXPIRE_MINUTES", "60")) * 60)
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


def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only an admin can do this.")
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


def signup_tenant(invite_code: Optional[str]) -> str:
    """The tenant a self-service sign-up joins, or AuthError if sign-up is not allowed."""
    mode = os.getenv("SIGNUP_MODE", "invite").strip().lower()
    if mode == "closed":
        raise AuthError("Sign-up is disabled. Ask an administrator for an account.", status.HTTP_403_FORBIDDEN)
    if invite_code:
        tenant = _invite_codes().get(invite_code.strip())
        if tenant is None:
            raise AuthError("Invalid invite code.", status.HTTP_400_BAD_REQUEST)
        return tenant
    if mode == "invite":
        raise AuthError("An invite code is required to sign up.", status.HTTP_403_FORBIDDEN)
    return DEFAULT_TENANT


def bootstrap_admin() -> None:
    """Creates the admin named by ATTS_ADMIN_USERNAME / ATTS_ADMIN_PASSWORD if it does not exist yet."""
    username, password = os.getenv("ATTS_ADMIN_USERNAME"), os.getenv("ATTS_ADMIN_PASSWORD")
    if not username or not password:
        return
    store = get_store()
    if store.get_by_username(username) is None:
        store.create_user(username, password, tenant_id=os.getenv("ATTS_ADMIN_TENANT", DEFAULT_TENANT), role="admin")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["Auth"])


class SignupRequest(BaseModel):
    username: str
    password: str
    invite_code: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    tenant_id: str = DEFAULT_TENANT
    role: str = "user"


def _token_response(user: User) -> Dict[str, Any]:
    token, lifetime = issue_token(user)
    return {"access_token": token, "token_type": "bearer", "expires_in": lifetime, "user": user.public()}


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "127.0.0.1"


def _login(username: str, password: str, client_ip: str) -> Dict[str, Any]:
    try:
        user = get_store().authenticate(username, password, client_ip=client_ip)
    except AuthError as exc:
        raise HTTPException(exc.status_code, str(exc), _UNAUTHORIZED if exc.status_code == 401 else None)
    return _token_response(user)


@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest, request: Request):
    """Creates a user account (role `user`) and logs it in. The tenant comes from the sign-up policy, never from the caller."""
    subnet = subnet_of(_client_ip(request))
    limit = int(os.getenv("SIGNUPS_PER_SUBNET_PER_HOUR", "5"))
    store = get_store()
    try:
        tenant = signup_tenant(body.invite_code)
        if store.recent_signups(subnet) >= limit:
            raise AuthError("Too many accounts were created from this network recently. Try again later.",
                            status.HTTP_429_TOO_MANY_REQUESTS)
        user = store.create_user(body.username, body.password, tenant_id=tenant, role="user", signup_subnet=subnet)
    except AuthError as exc:
        raise HTTPException(exc.status_code, str(exc))
    return _token_response(user)


@router.post("/login")
def login(body: LoginRequest, request: Request):
    """Logs in with a JSON body and returns a bearer token."""
    return _login(body.username, body.password, _client_ip(request))


@router.post("/token", include_in_schema=True)
def token(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow (form fields), used by the Swagger UI 'Authorize' button."""
    return _login(form.username, form.password, _client_ip(request))


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return user.public()


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(user: User = Depends(get_current_user)):
    """Invalidates every token issued to this user so far."""
    get_store().bump_token_version(user.id)


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(body: CreateUserRequest, admin: User = Depends(require_admin)):
    """Admin only: creates an account in any tenant, with role `user` or `admin`."""
    try:
        user = get_store().create_user(body.username, body.password, tenant_id=body.tenant_id, role=body.role)
    except AuthError as exc:
        raise HTTPException(exc.status_code, str(exc))
    logger.info(f"Admin {admin.username} created account {user.username}")
    return user.public()


@router.get("/users")
def list_users(admin: User = Depends(require_admin)):
    """Admin only: all accounts (no password data)."""
    return [u.public() | {"status": u.status} for u in get_store().list_users()]
