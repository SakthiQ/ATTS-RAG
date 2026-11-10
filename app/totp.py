"""Two-factor log-in with authenticator-app codes (TOTP, RFC 6238: Google Authenticator, Microsoft Authenticator,
Aegis, 1Password and the like). Free and local: no SMS or e-mail service is involved.

* The shared secret is encrypted at rest with Fernet (AES-128-CBC + HMAC). The key comes from TOTP_KEY, or is
  generated once and kept in data/.totp_key. Losing the key makes every enrolled authenticator unusable (an
  admin can reset 2FA for an account).
* A code is accepted for its own 30-second step and one step either side (clock drift), and never twice: the
  last step used is stored, so a code seen over someone's shoulder cannot be replayed.
* Enabling 2FA returns RECOVERY_CODE_COUNT one-time recovery codes (stored hashed) for a lost phone.
"""
import hashlib
import hmac
import os
import secrets
import time
from typing import List, Optional
from urllib.parse import quote

import pyotp
from cryptography.fernet import Fernet, InvalidToken
from loguru import logger

ISSUER = "ATTS-RAG"
STEP_SECONDS = 30
DRIFT_STEPS = 1
RECOVERY_CODE_COUNT = 8
_RECOVERY_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_key_cache: Optional[bytes] = None


def _key() -> bytes:
    global _key_cache
    configured = os.getenv("TOTP_KEY")
    if configured:
        return configured.encode("ascii")
    if _key_cache is None:
        path = os.path.join("data", ".totp_key")
        if os.path.exists(path):
            with open(path, "rb") as f:
                _key_cache = f.read().strip()
        if not _key_cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _key_cache = Fernet.generate_key()
            with open(path, "wb") as f:
                f.write(_key_cache)
            logger.warning(f"TOTP_KEY not set; generated one and stored it in {path}. Back it up with the user database.")
    return _key_cache


def new_secret() -> str:
    return pyotp.random_base32()


def encrypt(secret: str) -> str:
    return Fernet(_key()).encrypt(secret.encode("ascii")).decode("ascii")


def decrypt(token: str) -> Optional[str]:
    try:
        return Fernet(_key()).decrypt(token.encode("ascii")).decode("ascii")
    except (InvalidToken, ValueError):
        logger.error("A stored 2FA secret could not be decrypted (wrong or lost TOTP_KEY?).")
        return None


def provisioning_uri(secret: str, username: str) -> str:
    """The otpauth:// link an authenticator app reads (usually shown as a QR code)."""
    return pyotp.TOTP(secret, interval=STEP_SECONDS).provisioning_uri(name=quote(username), issuer_name=ISSUER)


def current_code(secret: str, at: Optional[float] = None) -> str:
    """The code an authenticator would show now (used by tests and the live check)."""
    return pyotp.TOTP(secret, interval=STEP_SECONDS).at(int(at if at is not None else time.time()))


def verify(secret: str, code: str, last_step: int = 0, at: Optional[float] = None) -> Optional[int]:
    """The time step `code` belongs to if it is valid and newer than `last_step`, else None."""
    code = "".join(ch for ch in (code or "") if ch.isdigit())
    if len(code) != 6:
        return None
    totp = pyotp.TOTP(secret, interval=STEP_SECONDS)
    now_step = int((at if at is not None else time.time()) // STEP_SECONDS)
    for step in range(now_step - DRIFT_STEPS, now_step + DRIFT_STEPS + 1):
        if step > last_step and hmac.compare_digest(totp.at(step * STEP_SECONDS), code):
            return step
    return None


def looks_like_recovery_code(code: str) -> bool:
    return len(_normalise(code)) == 10 and not _normalise(code).isdigit()


def _normalise(code: str) -> str:
    return "".join(ch for ch in (code or "").upper() if ch.isalnum())


def recovery_digest(code: str) -> str:
    return hashlib.sha256(_normalise(code).encode("ascii")).hexdigest()


def new_recovery_codes() -> List[str]:
    codes = []
    for _ in range(RECOVERY_CODE_COUNT):
        raw = "".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(10))   # 50 bits
        codes.append(f"{raw[:5]}-{raw[5:]}")
    return codes
