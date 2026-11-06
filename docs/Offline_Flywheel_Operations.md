# Offline Improvement Flywheel — Operations Guide

**System:** ATTS-RAG  
**Component:** Offline Quality & Safety Improvement Pipeline  
**Status:** Partial — Steps 1, 2 and 5 are implemented; Steps 3 and 4 are manual, with no tooling yet  

---

## Overview

The ATTS-RAG Offline Flywheel is the asynchronous companion to the real-time Layer 3 gate.
The real-time pipeline uses **fixed, deterministic thresholds** (ENTAILMENT ≥ 0.60, CONTRADICTION ≥ 0.60),
so its behaviour never drifts at run time.

The flywheel reads the telemetry that the live pipeline emits and surfaces failures and near-misses
for review. The live pipeline never reads its output: improvements reach production only when someone
edits the thresholds in code and the change passes CI.

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
**No raw prompt or answer text is ever written** — only structural metadata. Claim text stays in the
telemetry returned to the live UI and is removed before the record reaches the log; verifier reasons that
would quote evidence, a table row or a claim's numbers are replaced by fixed wording, and `UNKNOWN`
reasons (raw exception text) are dropped. Logs written before this rule was enforced may still contain
claim text.

### Audit Record Schema

```json
{
  "query_id":             "Q-session_xyz",
  "model_version":        "generator-v2",
  "decision":             "REJECT | PASS",
  "evidence_ids_used":    ["chk_001", "chk_002"],
  "total_claims":         3,
  "verified_claim_count": 2,
  "failed_claim_count":   1,
  "failure_reason":       "Claim C2 not supported by cited evidence.",
  "retry_count":          0,
  "execution_time_ms":    847.3,
  "timestamp":            "2026-09-20T15:00:00Z",
  "claim_results": [
    {
      "claim_id":               "C1",
      "cited_evidence_ids":     ["chk_001"],
      "supporting_evidence_id": "chk_001",
      "status":                 "ENTAILMENT",
      "confidence":             0.91,
      "reason":                 "Evidence supports claim."
    },
    {
      "claim_id":               "C2",
      "cited_evidence_ids":     ["chk_002"],
      "supporting_evidence_id": "chk_002",
      "status":                 "INSUFFICIENT",
      "confidence":             0.31,
      "reason":                 "Insufficient evidence support (best entailment 0.31)."
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
| `RETRY` | `retry_count > 0` (legacy records only; Layer 3 no longer retries) |
| `FAILED_CLAIM` | Any claim with status `INSUFFICIENT` or `CONTRADICTION` |
| `NEAR_THRESHOLD` | Any claim confidence within ±margin of the decision boundary |

Clean PASS decisions with no edge-case signals are skipped to avoid flooding the review queue.

---

## Step 3 — Human / LLM-as-Judge Review

> **Not implemented.** This repository has no annotation tool, label store or judge script for audit
> records; the process below is the intended procedure. The audit log holds no claim or evidence text,
> so reviewers must look up evidence by ID and reproduce a claim to fill the template below.

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

> **Not implemented.** The code below is pseudo-code; there is no calibration script for audit records.

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
ENTAILMENT_THRESHOLD = 0.60   # ← replace with calibrated value
CONTRADICTION_THRESHOLD = 0.60
```

---

## Step 5 — CI Regression Gate

Before deploying a threshold or model change, validate latency SLAs. This gate checks speed only, so it
will not catch a threshold change that hurts accuracy:

```powershell
# 1. Run the Layer 3 benchmark
python scripts/benchmark_layer3_latency.py --runs 30 --json-out benchmark_results.json

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
