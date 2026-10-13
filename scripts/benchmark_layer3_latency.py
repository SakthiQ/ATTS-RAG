"""Latency benchmark for Layer 3's verification stages (the LLM call is replaced by a canned reply,
so these numbers are the gate's own overhead, not generation time).

Writes logs/latency_benchmark.json in the format scripts/check_latency_regression.py reads.
"""
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from app.rag.layer3.claim_verify import DirectClaimVerifier
from app.rag.layer3.gate import Layer3Gate
from app.rag.layer3.generation import AnswerGenerator
from app.rag.layer3.reconstruction import VerifiedAnswerReconstructor
from app.rag.layer3.safety import FastFailSafetyScanner

EVIDENCE = [
    {"content": "Official IT Policy: Employees must rotate passwords every 90 days. Multi-factor authentication is "
                "required for all remote access. Contractors may request an exception from the security office.",
     "metadata": {"chunk_uid": "chk_1", "source": "policy.pdf", "page": 1}},
    {"content": "Data retention: Audit logs are stored for seven years. Backups are encrypted with AES-256 and "
                "tested every quarter. Incident reports must be filed within 24 hours.",
     "metadata": {"chunk_uid": "chk_2", "source": "policy.pdf", "page": 2}},
    {"content": "Remote work: Employees may work remotely up to three days per week with manager approval. "
                "Company laptops must use full-disk encryption.",
     "metadata": {"chunk_uid": "chk_3", "source": "hr.pdf", "page": 7}},
]
REPLY = (
    "INTENT: The user wants the security rules.\nANSWER:\n"
    "Employees must rotate passwords every 90 days. [E1]\n"
    "Multi-factor authentication is required for all remote access. [E1]\n"
    "Audit logs are stored for seven years. [E2]\n"
    "Backups are encrypted with AES-256 and tested every quarter. [E2]\n"
    "Company laptops must use full-disk encryption. [E3]"
)
QUESTION = "What are the IT security policies?"


def percentiles(ms: List[float]) -> Dict[str, float]:
    arr = np.array(ms)
    return {
        "p50": round(float(np.percentile(arr, 50)), 3), "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3), "mean": round(float(arr.mean()), 3),
        "stddev": round(float(arr.std()), 3),
    }


def timed(fn, iterations: int) -> List[float]:
    out = []
    for _ in range(iterations):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000.0)
    return out


def run(iterations: int = 30, warmups: int = 3) -> Dict[str, Any]:
    verifier = DirectClaimVerifier()
    gate = Layer3Gate(llm_invoker=lambda prompt: REPLY, verifier=verifier)
    generator = AnswerGenerator(llm_invoker=lambda prompt: REPLY)
    scanner, reconstructor = FastFailSafetyScanner(), VerifiedAnswerReconstructor()

    draft = generator.generate(QUESTION, EVIDENCE)
    text = " ".join(s.text for line in draft.lines for s in line.sentences)
    results = verifier.verify_claims(draft.claims, draft.evidence_lookup)

    for _ in range(warmups):
        gate.process(QUESTION, EVIDENCE)

    return {
        "metadata": {
            "hardware": {"system": platform.system(), "machine": platform.machine(),
                         "processor": platform.processor() or "CPU", "python_version": platform.python_version()},
            "iterations": iterations, "warmup_iterations": warmups,
            "claim_count": len(draft.claims), "evidence_chunk_count": len(EVIDENCE),
            "nli_model": verifier.NLI_MODEL_NAME,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "component_latency_ms": {
            "draft_parsing": percentiles(timed(lambda: generator.generate(QUESTION, EVIDENCE), iterations)),
            "safety_scan": percentiles(timed(lambda: scanner.scan(text), iterations)),
            "nli_verification": percentiles(timed(lambda: verifier.verify_claims(draft.claims, draft.evidence_lookup), iterations)),
            "answer_reconstruction": percentiles(timed(lambda: reconstructor.reconstruct(draft, results), iterations)),
            "complete_layer3_gate": percentiles(timed(lambda: gate.process(QUESTION, EVIDENCE), iterations)),
        },
    }


def main():
    data = run()
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "latency_benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Results written to {path}\n")
    for name, s in data["component_latency_ms"].items():
        print(f"  {name:24s} p50 {s['p50']:8.2f} ms | p95 {s['p95']:8.2f} ms | p99 {s['p99']:8.2f} ms")


if __name__ == "__main__":
    main()
