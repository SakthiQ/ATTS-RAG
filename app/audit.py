"""Audit log of account and access events: an append-only, hash-chained JSON-lines file.

Every record carries the SHA-256 of the record before it ("prev") and of itself ("hash"), so editing, deleting
or reordering a line breaks the chain and `verify()` reports the first broken line. Passwords, codes and tokens
are never written: `audit()` refuses fields with those names.

Settings: AUTH_AUDIT_LOG, the file (default logs/auth_audit.jsonl). Writes are serialised within the process;
run one API process per log file.
"""
import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

DEFAULT_PATH = os.path.join("logs", "auth_audit.jsonl")
GENESIS = "0" * 64
SECRET_FIELDS = {"password", "old_password", "new_password", "code", "reset_code", "otp", "token", "access_token",
                 "refresh_token", "secret", "recovery_code"}
_lock = threading.Lock()


def log_path() -> str:
    return os.getenv("AUTH_AUDIT_LOG", DEFAULT_PATH)


def _digest(record: Dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _last_hash(path: str) -> str:
    if not os.path.exists(path):
        return GENESIS
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last)["hash"] if last else GENESIS


def audit(event: str, actor: Optional[str] = None, tenant: Optional[str] = None, outcome: str = "ok",
          **fields: Any) -> Dict[str, Any]:
    """Appends one record. `actor` is who acted (username, or None for an anonymous caller), `tenant` the tenant
    the event concerns (used to show a tenant admin only their own tenant's events)."""
    secret = SECRET_FIELDS & set(fields)
    if secret:
        raise ValueError(f"audit() must not record secrets: {sorted(secret)}")
    record: Dict[str, Any] = {"ts": round(time.time(), 3), "event": event, "actor": actor, "tenant": tenant,
                              "outcome": outcome, **fields}
    path = log_path()
    with _lock:
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            record["prev"] = _last_hash(path)
            record["hash"] = _digest(record)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:   # an audit failure must be visible, but must not take the API down
            logger.error(f"Could not write the audit record for {event}: {exc}")
    return record


def read(tenant: Optional[str] = None, event: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """The newest `limit` records, oldest first; only `tenant`'s when given."""
    path = log_path()
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if (tenant is None or rec.get("tenant") == tenant) and (event is None or rec.get("event") == event):
                out.append(rec)
    return out[-limit:] if limit else out


def verify(path: Optional[str] = None) -> Tuple[bool, int, Optional[int]]:
    """(chain intact, records checked, first broken line number or None)."""
    path = path or log_path()
    if not os.path.exists(path):
        return True, 0, None
    prev, count = GENESIS, 0
    with open(path, "r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return False, count, number
            if rec.get("prev") != prev or rec.get("hash") != _digest(rec):
                return False, count, number
            prev, count = rec["hash"], count + 1
    return True, count, None
