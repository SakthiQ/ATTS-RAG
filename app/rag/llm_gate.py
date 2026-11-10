"""Queues calls to the local language model.

One Ollama instance on a CPU does not get faster by running requests in parallel: two concurrent questions
each ran 2-4x slower than alone, so the first user waited as long as the last. Taking turns lets each call
run at full speed. OLLAMA_MAX_CONCURRENCY (default 1) raises the limit for a GPU host.
"""
import os
import threading
from contextlib import contextmanager

_slots = threading.BoundedSemaphore(max(1, int(os.getenv("OLLAMA_MAX_CONCURRENCY", "1"))))


@contextmanager
def llm_slot():
    """Holds one model slot for the duration of the block."""
    with _slots:
        yield
