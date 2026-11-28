"""Docker health check for the backend that also restarts a backend which has stopped answering.

    python -m app.healthcheck_probe          (the backend service's healthcheck in deploy/docker-compose.prod.yml)

Docker's restart policy restarts a container whose process exits, not one that is running but hung. This probe
asks GET /health; after HEALTH_KILL_AFTER failures in a row (default 3) it kills the API process, so the container
exits and Docker starts it again. It needs `init: true` on the service: the API then runs under a small init
process, and kill(-1) can reach it (signals sent from inside a container to its own PID 1 are ignored).

/health answers 200 even while Ollama is down, so a missing model never causes a restart loop: only a backend that
does not answer at all is restarted. Exit code 0 healthy, 1 unhealthy (what Docker expects).
"""
import os
import signal
import sys
import tempfile
import urllib.request

URL = os.getenv("HEALTH_URL", "http://127.0.0.1:8000/health")
STATE = os.path.join(tempfile.gettempdir(), "atts-health-failures")


def _failures(update=None) -> int:
    if update is not None:
        with open(STATE, "w", encoding="utf-8") as f:
            f.write(str(update))
        return update
    try:
        with open(STATE, encoding="utf-8") as f:
            return int(f.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=float(os.getenv("HEALTH_TIMEOUT", "10"))) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    if ok:
        _failures(0)
        return 0
    count = _failures(_failures() + 1)
    limit = int(os.getenv("HEALTH_KILL_AFTER", "3"))
    if limit > 0 and count >= limit and os.getenv("HEALTH_KILL", "on") != "off":
        _failures(0)
        print(f"backend did not answer {count} health checks in a row: restarting it", file=sys.stderr)
        try:
            os.kill(-1, signal.SIGKILL)   # every process this user may signal, except PID 1 (init) and this one
        except (OSError, AttributeError):
            pass
    return 1


if __name__ == "__main__":
    sys.exit(main())
