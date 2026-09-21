import hashlib
import json
import os
import time
from typing import Dict, Any, List, Optional
from app.rag.layer3.contracts import ClaimObject

_AUDIT_LOG_PATH = os.getenv("LAYER3_AUDIT_LOG", "logs/layer3_audit.jsonl")


class Layer3TelemetryStore:
    """Stores minimized decision metadata and audit telemetry.

    Every verification decision is appended to a persistent JSONL audit log
    so the offline flywheel can sample failures and near-misses for review.
    Raw prompt / answer text is NEVER written — only structural metadata.
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

        # Persist to audit log (best-effort — never raise into the calling pipeline)
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:  # pragma: no cover
            pass  # telemetry must not crash the live query path

        return record
