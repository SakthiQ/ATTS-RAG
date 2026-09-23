# ATTS-RAG Executive Audit & Flywheel Report

**Generated At**: `2026-09-23 08:26:31 UTC`  
**Log Path**: `logs/_test_audit.jsonl`  
**Output Sample**: `logs/_test_sampled.jsonl`  

---

## 📊 Summary Metrics

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Total Logged Queries** | `4` | `100.0%` |
| **Total Sampled for Offline Review** | `3` | `75.0%` |
| **Rejections (REJECT)** | `1` | `25.0%` |
| **Retried Query Recoveries** | `1` | `25.0%` |
| **Failed Claim Assertions** | `1` | `25.0%` |
| **Near-Threshold Decision Boundary** | `1` | `25.0%` |

---

## 🛡️ Key Audit Takeaways

1. **Rejection Rate**: **25.0%** of queries failed Layer 3 verification or safety checks and were safely blocked.
2. **Flywheel Recommendations**:
   - Review `3` sampled edge cases in `logs/_test_sampled.jsonl` for model tuning and prompt calibration.
   - Investigate `1` near-boundary queries to fine-tune `ENTAILMENT_THRESHOLD` (0.70).
