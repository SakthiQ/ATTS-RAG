"""Size-based rotation for the JSON-lines logs (traces, answer feedback, Layer 3 audit, admin-view audit, log-in audit).

Each writer opens its file, appends one line and closes it, so a file can be renamed between two writes. When a file
reaches LOG_MAX_MB (default 20), the next write first moves it to logs/archive/<name>-<UTC time>.jsonl; the write then
starts a new file. Archives are kept: traces and the Layer 3 audit feed the offline evaluation, and the log-in audit
is evidence. LOG_ARCHIVE_DAYS (default: keep) deletes plain-log archives older than that many days; archives of the
hash-chained log-in audit are never deleted automatically.

Rotation also keeps appends cheap: the log-in audit reads its whole file to find the previous record's hash.
"""
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

ARCHIVE_DIR = "archive"
DEFAULT_MAX_MB = 20.0


def max_bytes() -> int:
    try:
        return int(float(os.getenv("LOG_MAX_MB", str(DEFAULT_MAX_MB))) * 1024 * 1024)
    except ValueError:
        return int(DEFAULT_MAX_MB * 1024 * 1024)


def archive_dir(path: str) -> str:
    return os.path.join(os.path.dirname(path) or ".", ARCHIVE_DIR)


def archives(path: str) -> List[str]:
    """This log's archives, oldest first (the UTC time in the name sorts them)."""
    stem, ext = os.path.splitext(os.path.basename(path))
    folder = archive_dir(path)
    if not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder) if n.startswith(stem + "-") and n.endswith(ext)]
    return [os.path.join(folder, n) for n in sorted(names)]


def rotate_if_needed(path: str, limit: Optional[int] = None, keep_days: Optional[float] = None) -> Optional[str]:
    """Moves `path` into the archive folder if it has reached the size limit. Returns the archive path, else None.

    Call it while holding the writer's own lock, right before appending. A failure is logged, never raised: a log
    that cannot rotate must not stop the request that writes it."""
    limit = max_bytes() if limit is None else limit
    try:
        if limit <= 0 or not os.path.exists(path) or os.path.getsize(path) < limit:
            return None
        folder = archive_dir(path)
        os.makedirs(folder, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(path))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = os.path.join(folder, f"{stem}-{stamp}{ext}")
        os.replace(path, target)
        logger.info(f"Rotated {path} ({limit // (1024 * 1024)} MB limit) to {target}")
        _prune(path, keep_days)
        return target
    except OSError as exc:
        logger.error(f"Could not rotate {path}: {exc}")
        return None


def _prune(path: str, keep_days: Optional[float]) -> None:
    if keep_days is None:
        raw = os.getenv("LOG_ARCHIVE_DAYS", "")
        try:
            keep_days = float(raw) if raw.strip() else None
        except ValueError:
            keep_days = None
    if not keep_days or keep_days <= 0:
        return
    cutoff = time.time() - keep_days * 86400
    for old in archives(path):
        try:
            if os.path.getmtime(old) < cutoff:
                os.remove(old)
        except OSError:
            pass
