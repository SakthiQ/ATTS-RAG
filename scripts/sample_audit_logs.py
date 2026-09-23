#!/usr/bin/env python3
"""Offline Improvement Flywheel — Audit Log Sampler.

Step 1 of the ATTS-RAG offline flywheel pipeline:

  Telemetry logs  →  THIS SCRIPT  →  sampled_audit_<date>.jsonl
                                           │
                                      Human / LLM-as-Judge review
                                           │
                                   Threshold calibration & prompt tuning

The sampler reads the Layer 3 JSON audit log (written by Layer3TelemetryStore)
and selects records that deserve offline review:

  * REJECT decisions (always)
  * PASS decisions where retry_count > 0  (near-miss recoveries)
  * Claims with status INSUFFICIENT or CONTRADICTION
  * Claims with NLI confidence near the decision boundary (configurable)

Usage::

    python scripts/sample_audit_logs.py \\
        --log  logs/layer3_audit.jsonl \\
        --out  logs/sampled_audit_$(date +%Y%m%d).jsonl \\
        --near-threshold 0.10 \\
        --max-samples 500
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Sampling predicates
# ---------------------------------------------------------------------------

def _is_rejection(record: Dict[str, Any]) -> bool:
    return record.get("decision") == "REJECT"


def _is_retry(record: Dict[str, Any]) -> bool:
    return record.get("retry_count", 0) > 0


def _has_failed_claim(record: Dict[str, Any]) -> bool:
    """Any claim that did NOT achieve ENTAILMENT."""
    claims = record.get("claim_results", [])
    return any(c.get("status") in ("INSUFFICIENT", "CONTRADICTION", "UNKNOWN") for c in claims)


def _is_near_threshold(record: Dict[str, Any], margin: float) -> bool:
    """True when any claim's NLI confidence is within `margin` of the decision
    boundary thresholds (ENTAILMENT ≥ 0.70, CONTRADICTION ≥ 0.50)."""
    ENTAILMENT_BOUNDARY = 0.70
    CONTRADICTION_BOUNDARY = 0.50

    claims = record.get("claim_results", [])
    for c in claims:
        conf = c.get("confidence", None)
        if conf is None:
            continue
        status = c.get("status", "")
        if status == "ENTAILMENT" and abs(conf - ENTAILMENT_BOUNDARY) <= margin:
            return True
        if status == "CONTRADICTION" and abs(conf - CONTRADICTION_BOUNDARY) <= margin:
            return True
    return False


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

def sample_audit_log(
    log_path: str,
    output_path: str,
    near_threshold_margin: float = 0.10,
    max_samples: int = 500,
) -> Dict[str, int]:
    """Reads `log_path` (JSONL), applies sampling predicates, writes results."""

    if not os.path.exists(log_path):
        print(f"[ERROR] Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    stats = {
        "total_records": 0,
        "sampled_rejection": 0,
        "sampled_retry": 0,
        "sampled_failed_claim": 0,
        "sampled_near_threshold": 0,
        "total_sampled": 0,
        "truncated": False,
    }

    seen_query_ids: set = set()
    sampled: List[Dict[str, Any]] = []

    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line: {exc}", file=sys.stderr)
                continue

            stats["total_records"] += 1
            query_id = record.get("query_id", "")

            # Deduplicate by query_id
            if query_id and query_id in seen_query_ids:
                continue

            reasons: List[str] = []

            if _is_rejection(record):
                reasons.append("REJECTION")
                stats["sampled_rejection"] += 1
            if _is_retry(record):
                reasons.append("RETRY")
                stats["sampled_retry"] += 1
            if _has_failed_claim(record):
                reasons.append("FAILED_CLAIM")
                stats["sampled_failed_claim"] += 1
            if _is_near_threshold(record, near_threshold_margin):
                reasons.append("NEAR_THRESHOLD")
                stats["sampled_near_threshold"] += 1

            if not reasons:
                continue  # clean PASS with no edge-case signals → skip

            if query_id:
                seen_query_ids.add(query_id)

            sampled.append({
                "sampled_at": datetime.now(_UTC).isoformat(),
                "sample_reasons": reasons,
                "record": record,
            })

            if len(sampled) >= max_samples:
                stats["truncated"] = True
                break

    stats["total_sampled"] = len(sampled)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for entry in sampled:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Generate Executive Markdown Audit Summary Report
    report_md_path = "docs/reports/Audit_Flywheel_Summary.md"
    os.makedirs("docs/reports", exist_ok=True)
    
    rej_rate = (stats["sampled_rejection"] / stats["total_records"] * 100) if stats["total_records"] > 0 else 0.0
    
    md_content = f"""# ATTS-RAG Executive Audit & Flywheel Report

**Generated At**: `{datetime.now(_UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}`  
**Log Path**: `{log_path}`  
**Output Sample**: `{output_path}`  

---

## 📊 Summary Metrics

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Total Logged Queries** | `{stats['total_records']}` | `100.0%` |
| **Total Sampled for Offline Review** | `{stats['total_sampled']}` | `{(stats['total_sampled'] / stats['total_records'] * 100) if stats['total_records'] > 0 else 0.0:.1f}%` |
| **Rejections (REJECT)** | `{stats['sampled_rejection']}` | `{rej_rate:.1f}%` |
| **Retried Query Recoveries** | `{stats['sampled_retry']}` | `{(stats['sampled_retry'] / stats['total_records'] * 100) if stats['total_records'] > 0 else 0.0:.1f}%` |
| **Failed Claim Assertions** | `{stats['sampled_failed_claim']}` | `{(stats['sampled_failed_claim'] / stats['total_records'] * 100) if stats['total_records'] > 0 else 0.0:.1f}%` |
| **Near-Threshold Decision Boundary** | `{stats['sampled_near_threshold']}` | `{(stats['sampled_near_threshold'] / stats['total_records'] * 100) if stats['total_records'] > 0 else 0.0:.1f}%` |

---

## 🛡️ Key Audit Takeaways

1. **Rejection Rate**: **{rej_rate:.1f}%** of queries failed Layer 3 verification or safety checks and were safely blocked.
2. **Flywheel Recommendations**:
   - Review `{stats['total_sampled']}` sampled edge cases in `{output_path}` for model tuning and prompt calibration.
   - Investigate `{stats['sampled_near_threshold']}` near-boundary queries to fine-tune `ENTAILMENT_THRESHOLD` (0.70).
"""
    with open(report_md_path, "w", encoding="utf-8") as rmd:
        rmd.write(md_content)

    print(f"[INFO] Executive Markdown report written to: {report_md_path}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ATTS-RAG Offline Flywheel — Layer 3 Audit Log Sampler"
    )
    parser.add_argument(
        "--log",
        default="logs/layer3_audit.jsonl",
        help="Path to the Layer 3 JSON-lines audit log (default: logs/layer3_audit.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=f"logs/sampled_audit_{datetime.now(_UTC).strftime('%Y%m%d')}.jsonl",
        help="Output path for the sampled JSONL report",
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.10,
        metavar="MARGIN",
        help=(
            "Include claims whose NLI confidence is within MARGIN of the "
            "ENTAILMENT(0.70) or CONTRADICTION(0.50) decision boundary "
            "(default: 0.10)"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of records to include in the output (default: 500)",
    )

    args = parser.parse_args()

    print(f"[INFO] Reading audit log:  {args.log}")
    print(f"[INFO] Writing sample to:  {args.out}")
    print(f"[INFO] Near-threshold margin: ±{args.near_threshold}")

    stats = sample_audit_log(
        log_path=args.log,
        output_path=args.out,
        near_threshold_margin=args.near_threshold,
        max_samples=args.max_samples,
    )

    print("\n=== Sampling Summary ===")
    print(f"  Total records read  : {stats['total_records']}")
    print(f"  REJECTION samples   : {stats['sampled_rejection']}")
    print(f"  RETRY samples       : {stats['sampled_retry']}")
    print(f"  FAILED_CLAIM samples: {stats['sampled_failed_claim']}")
    print(f"  NEAR_THRESHOLD      : {stats['sampled_near_threshold']}")
    print(f"  Total written       : {stats['total_sampled']}")
    if stats["truncated"]:
        print(f"  [!] Output truncated at {args.max_samples} records (increase --max-samples if needed)")

    print(f"\n[DONE] Sampled audit log written to: {args.out}")


if __name__ == "__main__":
    main()
