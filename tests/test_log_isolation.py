"""Tests must never write to the real logs/ folder (see tests/conftest.py)."""
import os

import pytest
from loguru import logger

from app.rag.layer3.gate import Layer3Gate
from app.rag.layer3.telemetry import Layer3TelemetryStore

REAL_LOGS = os.path.abspath("logs")


def _inside_real_logs(path) -> bool:
    return os.path.abspath(str(path)).startswith(REAL_LOGS + os.sep)


def test_layer3_audit_log_is_redirected(layer3_and_admin_logs):
    path = Layer3TelemetryStore()._log_path
    assert not _inside_real_logs(path)
    assert os.path.abspath(path) == os.path.abspath(str(layer3_and_admin_logs["layer3"]))


def test_admin_view_audit_log_is_redirected():
    assert not _inside_real_logs(os.environ["ADMIN_AUDIT_LOG"])


def test_backend_log_file_is_not_added():
    import app.main  # noqa: F401  (adds the file sink on import unless BACKEND_LOG=off)
    files = [getattr(h._sink, "_path", None) for h in logger._core.handlers.values()]
    assert not any(f and _inside_real_logs(f) for f in files), files


@pytest.fixture(scope="module")
def module_gate():
    """Built once per module, before any per-test fixture runs: the case that used to reach logs/."""
    return Layer3Gate(llm_invoker=lambda prompt: "")


def test_a_gate_built_in_a_module_fixture_is_redirected(module_gate):
    assert not _inside_real_logs(module_gate.telemetry_store._log_path)


@pytest.mark.parametrize("name", ["AUTH_AUDIT_LOG", "TRACE_LOG", "TRACE_FEEDBACK_LOG"])
def test_other_logs_are_redirected(name):
    assert not _inside_real_logs(os.environ[name])
