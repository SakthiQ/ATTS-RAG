# Offline Improvement Flywheel — Operations Guide

**System:** ATTS-RAG  
**Component:** Offline Quality & Safety Improvement Pipeline  
**Status:** Operational  

---

## Overview

The ATTS-RAG Offline Flywheel is the asynchronous companion to the real-time Layer 3 gate.
The real-time pipeline uses **fixed, deterministic thresholds** (ENTAILMENT ≥ 0.70, CONTRADICTION ≥ 0.50)
to guarantee sub-2.5s latency with zero runtime threshold drift.

The flywheel reads the telemetry that the live pipeline emits, surfaces failures and near-misses
for review, and feeds calibrated improvements back in through CI-gated model/threshold updates.

```
User Query ──► [Layer 3 Gate] ──► User Response
                    │
                    ▼ (every decision)
         logs/layer3_audit.jsonl
                    │
         ┌──────────▼──────────┐
         │  sample_audit_logs  │  (Step 1 — this guide)
         └──────────┬──────────┘
                    │ sampled_audit_<date>.jsonl
                    ▼
         Human / LLM-as-Judge review
                    │
                    ▼
         Threshold calibration & prompt tuning
                    │
                    ▼
         CI regression gate (check_latency_regression.py)
                    │
                    ▼
         Promoted to production
```

---

## Step 1 — Audit Log Collection

The `Layer3TelemetryStore` automatically appends a JSON record to
`logs/layer3_audit.jsonl` after every query decision.  
**No raw prompt or answer text is ever written** — only structural metadata.

### Audit Record Schema

```json
{
  "query_id":             "Q-session_xyz",
  "model_version":        "generator-v1",
  "decision":             "REJECT | PASS",
  "evidence_ids_used":    ["chk_001", "chk_002"],
  "total_claims":         3,
  "verified_claim_count": 2,
  "failed_claim_count":   1,
  "failure_reason":       "Claim C2 not supported by cited evidence.",
  "retry_count":          1,
  "execution_time_ms":    847.3,
  "timestamp":            "2026-09-20T15:00:00Z",
  "claim_results": [
    {
      "claim_id":    "C1",
      "status":      "ENTAILMENT",
      "confidence":  0.91
    },
    {
      "claim_id":    "C2",
      "status":      "INSUFFICIENT",
      "confidence":  0.62
    }
  ]
}
```

Configure the log path with the environment variable:
```bash
export LAYER3_AUDIT_LOG=logs/layer3_audit.jsonl  # default
```

---

## Step 2 — Sampling Edge Cases

Run the sampler to extract records worth reviewing:

```powershell
# Basic usage — reads logs/layer3_audit.jsonl, writes to logs/sampled_audit_YYYYMMDD.jsonl
python scripts/sample_audit_logs.py

# Custom paths and thresholds
python scripts/sample_audit_logs.py `
    --log  logs/layer3_audit.jsonl `
    --out  logs/sampled_audit_2026-09-20.jsonl `
    --near-threshold 0.08 `
    --max-samples 1000
```

### Sampling Predicates

| Predicate | Trigger |
|-----------|---------|
| `REJECTION` | `decision == "REJECT"` |
| `RETRY` | `retry_count > 0` (near-miss recovery) |
| `FAILED_CLAIM` | Any claim with status `INSUFFICIENT` or `CONTRADICTION` |
| `NEAR_THRESHOLD` | Any claim confidence within ±margin of the decision boundary |

Clean PASS decisions with no edge-case signals are skipped to avoid flooding the review queue.

---

## Step 3 — Human / LLM-as-Judge Review

Open the sampled JSONL in your annotation tool. For each record, label:

- `correct_reject` — the gate was right to reject
- `false_reject` — the answer was valid but got blocked (false negative)
- `correct_pass` — legitimate pass
- `false_pass` — should have been blocked (false positive)

LLM-as-Judge prompt template (example):

```
You are a grounding quality judge. Given:
- Claim text: {claim_text}
- Cited evidence: {evidence_text}
- Gate decision: {status} (confidence: {confidence})

Was the gate decision correct? Respond: CORRECT / INCORRECT / UNCERTAIN.
Reason: <one sentence>
```

---

## Step 4 — Threshold Calibration

After collecting enough labeled records, recalibrate the NLI thresholds:

```python
# Pseudo-code calibration workflow
from sklearn.metrics import precision_recall_curve

# Load labeled claim records
labels = [...]        # 1 = should_entail, 0 = should_not
confidences = [...]   # model output confidence

precision, recall, thresholds = precision_recall_curve(labels, confidences)
# Choose threshold maximising F1 on the domain eval set
best_threshold = thresholds[argmax(2*precision*recall / (precision + recall))]
```

Update `app/rag/layer3/claim_verify.py`:
```python
ENTAILMENT_THRESHOLD = 0.70   # ← replace with calibrated value
CONTRADICTION_THRESHOLD = 0.50
```

---

## Step 5 — CI Regression Gate

Before deploying a threshold or model change, validate latency SLAs:

```powershell
# 1. Run the Layer 3 benchmark
python scripts/benchmark_layer3_latency.py --output benchmark_results.json

# 2. Check against SLA thresholds (fails CI if violated)
python scripts/check_latency_regression.py `
    --file benchmark_results.json `
    --max-p50 3500 `
    --max-p95 7000 `
    --max-p99 10000
```

Exit code 0 = passed; exit code 1 = regression detected.

---

## Flywheel Cadence Recommendation

| Frequency | Action |
|-----------|--------|
| Daily | Run `sample_audit_logs.py` — review new REJECTs and retries |
| Weekly | Run LLM-as-Judge batch evaluation on sampled records |
| Monthly | Recalibrate thresholds against updated domain evaluation set |
| Per-release | Run full latency regression gate before deploying model updates |
