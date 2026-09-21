import os
import sys
import time
import json
import numpy as np
import platform
import asyncio
from typing import List, Dict, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from app.rag.layer3.gate import Layer3Gate
from app.rag.layer3.contracts import GenerationContract, ClaimObject
from app.rag.layer3.safety import FastFailSafetyScanner
from app.rag.layer3.schema import SchemaValidator
from app.rag.layer3.relevance import RelevanceEvaluator
from app.rag.layer3.claim_verify import DirectClaimVerifier
from app.rag.layer3.failure_router import FailureTypeRouter
from app.rag.layer3.reconstruction import VerifiedAnswerReconstructor


def get_hardware_info() -> Dict[str, str]:
    """Retrieves system hardware and environment metadata."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "x86_64 CPU",
        "python_version": platform.python_version(),
        "version": "layer3-v1.1-nli-deberta"
    }


def compute_percentiles(durations_ms: List[float]) -> Dict[str, float]:
    """Computes p50, p95, p99, mean, and stddev metrics in milliseconds."""
    arr = np.array(durations_ms)
    return {
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "mean": round(float(np.mean(arr)), 3),
        "stddev": round(float(np.std(arr)), 3)
    }


def run_latency_benchmark(iterations: int = 50, warmups: int = 5) -> Dict[str, Any]:
    """Executes benchmark measuring each Layer 3 component and full gate execution."""
    print("Initializing Layer 3 Benchmark Suite...")
    
    # 1. Setup Test Fixtures ($N=4$ claims, $M=4$ evidence chunks)
    evidence_package = [
        {"content": "Official IT Policy: Employees must rotate passwords every 90 days.", "metadata": {"chunk_uid": "E1"}},
        {"content": "MFA is mandatory for all employee accounts.", "metadata": {"chunk_uid": "E2"}},
        {"content": "Privileged accounts require hardware security keys.", "metadata": {"chunk_uid": "E3"}},
        {"content": "Remote access requires active VPN connection.", "metadata": {"chunk_uid": "E4"}}
    ]

    evidence_lookup = {item["metadata"]["chunk_uid"]: item["content"] for item in evidence_package}
    allowed_ids = set(evidence_lookup.keys())

    contract = GenerationContract(
        claims=[
            ClaimObject(claim_id="C1", order=1, text="Passwords must be rotated every 90 days.", evidence_ids=["E1"]),
            ClaimObject(claim_id="C2", order=2, text="MFA is mandatory for all employee accounts.", evidence_ids=["E2"]),
            ClaimObject(claim_id="C3", order=3, text="Privileged accounts require hardware security keys.", evidence_ids=["E3"]),
            ClaimObject(claim_id="C4", order=4, text="Remote access requires active VPN connection.", evidence_ids=["E4"])
        ]
    )

    safety_scanner = FastFailSafetyScanner()
    schema_validator = SchemaValidator()
    relevance_evaluator = RelevanceEvaluator()
    claim_verifier = DirectClaimVerifier()
    failure_router = FailureTypeRouter()
    reconstructor = VerifiedAnswerReconstructor()
    gate = Layer3Gate()

    # 2. Warm-up iterations (excluded from stats)
    print(f"Running {warmups} warm-up iterations...")
    for _ in range(warmups):
        safety_scanner.scan(contract)
        schema_validator.validate(contract, allowed_ids)
        relevance_evaluator.evaluate("What are the IT security policies?", contract)
        claim_verifier.verify_claims(contract.claims, evidence_lookup)
        asyncio.run(gate.process("What are the IT security policies?", evidence_package))

    # 3. Component Benchmarking
    print(f"Running {iterations} benchmark iterations per component...")

    t_safety = []
    t_relevance = []
    t_nli_direct = []
    t_nli_baseline_pairwise = []
    t_router = []
    t_reconstruction = []
    t_full_gate = []

    for _ in range(iterations):
        # Component 1: Safety & Schema Precheck
        s_start = time.perf_counter()
        schema_res = schema_validator.validate(contract, allowed_ids)
        safety_res = safety_scanner.scan(contract)
        t_safety.append((time.perf_counter() - s_start) * 1000.0)

        # Component 2: Relevance Evaluation
        rel_start = time.perf_counter()
        rel_res = relevance_evaluator.evaluate("What are the IT security policies?", contract)
        t_relevance.append((time.perf_counter() - rel_start) * 1000.0)

        # Component 3A: Redesigned Direct NLI Verification O(N)
        dir_start = time.perf_counter()
        claim_res = claim_verifier.verify_claims(contract.claims, evidence_lookup)
        t_nli_direct.append((time.perf_counter() - dir_start) * 1000.0)

        # Component 3B: Baseline Pairwise Matching O(N x M)
        pair_start = time.perf_counter()
        pairwise_evals = 0
        for claim in contract.claims:
            for eid, etext in evidence_lookup.items():
                claim_verifier._evaluate_nli(claim.text, etext)
                pairwise_evals += 1
        t_nli_baseline_pairwise.append((time.perf_counter() - pair_start) * 1000.0)

        # Component 4: Failure Router
        r_start = time.perf_counter()
        failure_router.route_failure(safety_res, rel_res, claim_res, retry_count=0, has_usable_evidence=True)
        t_router.append((time.perf_counter() - r_start) * 1000.0)

        # Component 5: Answer Reconstruction
        rec_start = time.perf_counter()
        reconstructor.reconstruct(contract.claims, claim_res)
        t_reconstruction.append((time.perf_counter() - rec_start) * 1000.0)

        # Component 6: Full End-to-End Gate Execution
        gate_start = time.perf_counter()
        asyncio.run(gate.process("What are the IT security policies?", evidence_package))
        t_full_gate.append((time.perf_counter() - gate_start) * 1000.0)

    # 4. Assemble Benchmark Report
    results = {
        "metadata": {
            "hardware": get_hardware_info(),
            "iterations": iterations,
            "warmup_iterations": warmups,
            "claim_count": len(contract.claims),
            "evidence_chunk_count": len(evidence_package),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "component_latency_ms": {
            "safety_prechecks": compute_percentiles(t_safety),
            "relevance_evaluator": compute_percentiles(t_relevance),
            "batched_direct_nli_O_N": compute_percentiles(t_nli_direct),
            "baseline_pairwise_nli_O_NxM": compute_percentiles(t_nli_baseline_pairwise),
            "failure_router": compute_percentiles(t_router),
            "answer_reconstruction": compute_percentiles(t_reconstruction),
            "complete_layer3_gate": compute_percentiles(t_full_gate)
        }
    }

    # Calculate speedup percentage
    direct_p50 = results["component_latency_ms"]["batched_direct_nli_O_N"]["p50"]
    baseline_p50 = results["component_latency_ms"]["baseline_pairwise_nli_O_NxM"]["p50"]
    speedup = round((1.0 - (direct_p50 / max(baseline_p50, 0.001))) * 100.0, 1)
    results["metadata"]["nli_speedup_percentage_p50"] = speedup

    return results


def main():
    benchmark_data = run_latency_benchmark(iterations=50, warmups=5)

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, "latency_benchmark.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n[OK] Latency Benchmark Complete! Results exported to: {json_path}")
    print("\n--- Summary Benchmark Results (ms) ---")
    for comp, stats in benchmark_data["component_latency_ms"].items():
        print(f"  {comp:30s} -> p50: {stats['p50']:6.3f} ms | p95: {stats['p95']:6.3f} ms | p99: {stats['p99']:6.3f} ms")

    print(f"\nDirect Lookup O(N) Speedup over Baseline O(N x M): {benchmark_data['metadata']['nli_speedup_percentage_p50']}% at p50")


if __name__ == "__main__":
    main()
