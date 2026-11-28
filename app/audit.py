"""Audit log of account and access events: an append-only, hash-chained JSON-lines file.

Every record carries the SHA-256 of the record before it ("prev") and of itself ("hash"), so editing, deleting
or reordering a line breaks the chain and `verify()` reports the first broken line. Passwords, codes and tokens
are never written: `audit()` refuses fields with those names.

Settings: AUTH_AUDIT_LOG, the file (default logs/auth_audit.jsonl). Writes are serialised within the process;
run one API process per log file.

Rotation (app/log_files.py): when the file reaches LOG_MAX_MB it is moved to logs/archive/ and the new file starts
with a "log_rotated" record whose "prev" is the archive's last hash, so the chain runs on across files.
`verify()` checks one file; `verify_all()` checks the archives and the current file, and every link between them.
Archives of this log are never deleted automatically.
"""
import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app import log_files

DEFAULT_PATH = os.path.join("logs", "auth_audit.jsonl")
GENESIS = "0" * 64
ROTATED_EVENT = "log_rotated"   # first record of a file started by rotation
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
            prev = _last_hash(path)
            archived = log_files.rotate_if_needed(path, keep_days=0)   # 0: never prune audit archives
            lines = []
            if archived:
                link = {"ts": record["ts"], "event": ROTATED_EVENT, "actor": None, "tenant": None, "outcome": "ok",
                        "archive": os.path.basename(archived), "prev": prev}
                link["hash"] = _digest(link)
                lines.append(link)
                prev = link["hash"]
            record["prev"] = prev
            record["hash"] = _digest(record)
            lines.append(record)
            with open(path, "a", encoding="utf-8") as f:
                f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines))
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


def _check_file(path: str) -> Tuple[bool, int, Optional[int], Optional[str], Optional[str]]:
    """(intact, records, first broken line, first record's prev, last hash). A file started by rotation may begin
    with any prev (its "log_rotated" record links it to the archive); any other file must begin at GENESIS."""
    prev, count, first_prev = None, 0, None
    with open(path, "r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return False, count, number, first_prev, prev
            if prev is None:
                first_prev = rec.get("prev")
                expected = first_prev if rec.get("event") == ROTATED_EVENT else GENESIS
            else:
                expected = prev
            if rec.get("prev") != expected or rec.get("hash") != _digest(rec):
                return False, count, number, first_prev, prev
            prev, count = rec["hash"], count + 1
    return True, count, None, first_prev, prev


def verify(path: Optional[str] = None) -> Tuple[bool, int, Optional[int]]:
    """(chain intact, records checked, first broken line number or None) for one file."""
    path = path or log_path()
    if not os.path.exists(path):
        return True, 0, None
    intact, count, broken, _, _ = _check_file(path)
    return intact, count, broken


def verify_all(path: Optional[str] = None) -> Dict[str, Any]:
    """Checks the archives (oldest first) and the current file, and that each file continues the one before it.

    Returns {"intact", "records", "files", "broken": None or {"file", "line"}}. The oldest file must begin at
    GENESIS; a missing or edited archive shows up as a broken link at line 1 of the file after it."""
    path = path or log_path()
    files = log_files.archives(path) + ([path] if os.path.exists(path) else [])
    total, last = 0, GENESIS
    for name in files:
        intact, count, broken, first_prev, end = _check_file(name)
        total += count
        if not intact:
            return {"intact": False, "records": total, "files": len(files), "broken": {"file": name, "line": broken}}
        if count and first_prev != last:
            return {"intact": False, "records": total, "files": len(files), "broken": {"file": name, "line": 1}}
        last = end or last
    return {"intact": True, "records": total, "files": len(files), "broken": None}
