#!/usr/bin/env python3
"""Latency Regression Gate for ATTS-RAG Layer 3 pipeline.
Parses benchmark JSON results and enforces latency SLAs.
"""

import sys
import json
import argparse
from typing import Dict, Any


DEFAULT_MAX_P50_MS = 3500.0
DEFAULT_MAX_P95_MS = 7000.0
DEFAULT_MAX_P99_MS = 10000.0


def check_latency_regression(
    benchmark_file: str,
    component: str = "complete_layer3_gate",
    max_p50: float = DEFAULT_MAX_P50_MS,
    max_p95: float = DEFAULT_MAX_P95_MS,
    max_p99: float = DEFAULT_MAX_P99_MS
) -> bool:
    """Validates benchmark results against SLA thresholds. Returns True if passed, False if regression."""
    try:
        with open(benchmark_file, "r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[FAIL] Could not read benchmark file {benchmark_file}: {exc}")
        return False

    comp_data = data.get("component_latency_ms", {}).get(component, {})
    if not comp_data:
        # Fallback to top-level percentiles_ms if flat
        comp_data = data.get("percentiles_ms", {})

    p50 = comp_data.get("p50", 0.0)
    p95 = comp_data.get("p95", 0.0)
    p99 = comp_data.get("p99", 0.0)

    print(f"=== ATTS-RAG Layer 3 Latency Regression Check [{component}] ===")
    print(f"p50: {p50:.2f} ms (max allowed: {max_p50:.2f} ms)")
    print(f"p95: {p95:.2f} ms (max allowed: {max_p95:.2f} ms)")
    print(f"p99: {p99:.2f} ms (max allowed: {max_p99:.2f} ms)")

    violations = []
    if p50 > max_p50:
        violations.append(f"p50 latency {p50:.2f}ms exceeds threshold {max_p50:.2f}ms")
    if p95 > max_p95:
        violations.append(f"p95 latency {p95:.2f}ms exceeds threshold {max_p95:.2f}ms")
    if p99 > max_p99:
        violations.append(f"p99 latency {p99:.2f}ms exceeds threshold {max_p99:.2f}ms")

    if violations:
        print("\n[FAIL] LATENCY REGRESSION DETECTED:")
        for v in violations:
            print(f"  - {v}")
        return False

    print("\n[SUCCESS] All latency metrics within SLA budget.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check Layer 3 Latency Regression")
    parser.add_argument("--file", required=True, help="Path to benchmark JSON results file")
    parser.add_argument("--max-p50", type=float, default=DEFAULT_MAX_P50_MS, help="Max allowed p50 in ms")
    parser.add_argument("--max-p95", type=float, default=DEFAULT_MAX_P95_MS, help="Max allowed p95 in ms")
    parser.add_argument("--max-p99", type=float, default=DEFAULT_MAX_P99_MS, help="Max allowed p99 in ms")

    args = parser.parse_args()
    success = check_latency_regression(
        benchmark_file=args.file,
        max_p50=args.max_p50,
        max_p95=args.max_p95,
        max_p99=args.max_p99
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
