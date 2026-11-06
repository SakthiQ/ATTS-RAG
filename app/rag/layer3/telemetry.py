import hashlib
import json
import os
import time
from typing import Dict, Any, List, Optional
from app.rag.layer3.contracts import ClaimObject

_AUDIT_LOG_PATH = os.getenv("LAYER3_AUDIT_LOG", "logs/layer3_audit.jsonl")

# Per-claim fields safe to persist. Claim text is model output, so it stays out of the log.
_PERSISTED_CLAIM_FIELDS = ("claim_id", "cited_evidence_ids", "supporting_evidence_id", "status", "confidence", "reason")

# Verifier reasons that embed document or claim text, replaced by fixed wording before they reach the log
_TEXT_BEARING_REASONS = (
    ("Second opinion confirmed", "Second opinion confirmed the claim with a verbatim quote (quote omitted)."),
    ("Table check:", "Table check confirmed the claim (row omitted)."),
    ("Number(s)", "Number(s) in the claim do not appear in the evidence (values omitted)."),
)


def _minimize_claim_result(claim_result: Dict[str, Any]) -> Dict[str, Any]:
    """Reduces a claim result to structural metadata for the audit log."""
    persisted = {k: claim_result[k] for k in _PERSISTED_CLAIM_FIELDS if k in claim_result}
    reason = persisted.get("reason")
    # UNKNOWN reasons wrap raw verifier exception strings, which may echo their input
    if persisted.get("status") == "UNKNOWN" or not reason:
        persisted.pop("reason", None)
        return persisted
    for prefix, replacement in _TEXT_BEARING_REASONS:
        if reason.startswith(prefix):
            persisted["reason"] = replacement
            break
    return persisted


class Layer3TelemetryStore:
    """Stores minimized decision metadata and audit telemetry.

    Every verification decision is appended to a persistent JSONL audit log
    so the offline flywheel can sample failures and near-misses for review.
    Raw prompt / answer text is NEVER written to the log — only structural
    metadata. The returned record keeps claim text for the live per-claim UI.
    """

    def __init__(self, audit_log_path: Optional[str] = None):
        self._log_path = audit_log_path or _AUDIT_LOG_PATH
        log_dir = os.path.dirname(self._log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def build_telemetry(
        self,
        query_id: str,
        model_version: str,
        decision: str,
        verified_claims: List[ClaimObject],
        failed_claims: List[ClaimObject],
        failure_reason: str = None,
        execution_time_ms: float = 0.0,
        claim_results: Optional[List[Dict[str, Any]]] = None,
        retry_count: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assembles a privacy-preserving auditable verification record and
        appends it to the persistent JSONL audit log for the offline flywheel."""
        all_evidence_ids = list(set([
            eid for c in verified_claims + failed_claims for eid in c.evidence_ids
        ]))

        record: Dict[str, Any] = {
            "query_id": query_id,
            "model_version": model_version,
            "decision": decision,
            "evidence_ids_used": all_evidence_ids,
            "total_claims": len(verified_claims) + len(failed_claims),
            "verified_claim_count": len(verified_claims),
            "failed_claim_count": len(failed_claims),
            "failure_reason": failure_reason,
            "retry_count": retry_count,
            "execution_time_ms": round(execution_time_ms, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # claim_results carries per-claim NLI confidence for near-threshold sampling
            "claim_results": claim_results or [],
        }
        if extra:
            record.update(extra)

        # Persist to audit log (best-effort — never raise into the calling pipeline)
        try:
            persisted_record = {
                **record,
                "claim_results": [_minimize_claim_result(c) for c in record["claim_results"]],
            }
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(persisted_record, ensure_ascii=False) + "\n")
        except Exception:  # pragma: no cover
            pass  # telemetry must not crash the live query path

        return record
