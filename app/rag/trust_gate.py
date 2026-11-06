import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .field_redactor import FieldRedactor


@dataclass
class EvaluatedCandidate:
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    is_authorized: bool
    is_intact: bool
    has_provenance: bool
    is_safe: bool
    trust_tier: str  # "TIER_1", "TIER_2", "QUARANTINE", "REJECT"
    reasons: List[str] = field(default_factory=list)


@dataclass
class Layer2TrustResult:
    allowed: bool
    status: str  # "TRUSTED_EVIDENCE_READY" or "INSUFFICIENT_TRUSTED_EVIDENCE"
    evidence_package: List[Dict[str, Any]]
    quarantined_items: List[Dict[str, Any]]
    rejected_count: int
    tier_1_count: int
    tier_2_count: int
    post_assembly_safe: bool
    reasons: List[str]
    execution_time_ms: float
    revealed: Dict[str, int] = field(default_factory=dict)   # personal data left unmasked for an admin view (counts per class)
    withheld_count: int = 0   # tier-2 passages held back because the user's suspicion is high
    fallback_unverified: bool = False   # no trusted evidence existed, so tier-2 evidence was used under stricter checks


class Layer2TrustGate:
    """Layer 2 — Knowledge Trust & Retrieval Gate Orchestrator.
    
    Evaluates candidate retrieval hits using a Discrete Decision Tree & Tiered Bucket
    Sort model (avoiding continuous float algebra vulnerabilities).
    """

    MIN_PROVENANCE_SCORE = 0.60
    MAX_POISONING_RISK = 0.30

    def __init__(self, vsm: Optional[Any] = None):
        self.vsm = vsm
        self.redactor = FieldRedactor()
        
        # Fast regex threat patterns for Tier 1 poisoning checks
        self.poison_patterns = [
            re.compile(r"ignore\s+(all\s+)?(previous\s+)?instructions", re.I),
            re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
            re.compile(r"you\s+are\s+now\s+in\s+DAN\s+mode", re.I),
            re.compile(r"<\|im_start\|>", re.I),
            re.compile(r"\[SYSTEM\s+INSTRUCTION", re.I),
        ]

    def check_authorization(self, metadata: Dict[str, Any], tenant_id: str, user_clearance: int = 1) -> bool:
        """Rule 1: Pre-retrieval and post-retrieval Tenant & ACL Authorization check."""
        doc_tenant = metadata.get("tenant_id", "default_tenant")
        doc_clearance = int(metadata.get("acl_tier", 1))

        if doc_tenant != tenant_id and doc_tenant != "global":
            return False
        if user_clearance < doc_clearance:
            return False
        return True

    def check_integrity(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Rule 2: Cryptographic Content Hash Integrity check against stored registry reference."""
        chunk_uid = metadata.get("chunk_uid")
        if not chunk_uid:
            return True  # If no chunk_uid, fallback to content presence

        current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if self.vsm is not None and hasattr(self.vsm, "registry"):
            registry = self.vsm.registry
            for entry in registry.values():
                chunk_hashes = entry.get("chunk_hashes", {})
                if chunk_uid in chunk_hashes:
                    expected_hash = chunk_hashes[chunk_uid]
                    if current_hash != expected_hash:
                        logger.warning(f"Integrity check failed for chunk {chunk_uid}: current={current_hash[:8]} vs expected={expected_hash[:8]}")
                        return False
                    return True

        return True

    def check_provenance(self, metadata: Dict[str, Any]) -> bool:
        """Rule 3: Provenance Lineage Verification."""
        doc_id = metadata.get("document_id")
        version = metadata.get("version")
        source_score = float(metadata.get("source_score", 0.80))

        if not doc_id or version is None:
            return False
        if source_score < self.MIN_PROVENANCE_SCORE:
            return False
        return True

    def check_safety(self, content: str, metadata: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """Rule 4: Tiered Poisoning & Indirect Injection Scan."""
        reasons = []
        
        # 1. Check metadata scan_status from ingestion
        ingestion_status = metadata.get("scan_status", "clean")
        if ingestion_status == "quarantined":
            reasons.append("Chunk flagged as quarantined during ingestion")
            return False, 1.0, reasons

        anomaly_score = float(metadata.get("anomaly_score", 0.0))
        if anomaly_score > self.MAX_POISONING_RISK:
            reasons.append(f"Ingestion anomaly score ({anomaly_score:.2f}) exceeds threshold ({self.MAX_POISONING_RISK})")

        # 2. Tier 1 Fast Regex Threat Scan
        for pattern in self.poison_patterns:
            if pattern.search(content):
                reasons.append(f"Matched threat pattern: {pattern.pattern}")
                return False, 0.90, reasons

        if anomaly_score > self.MAX_POISONING_RISK:
            return False, anomaly_score, reasons

        return True, 0.0, reasons

    def assign_tier(self, is_auth: bool, is_intact: bool, has_prov: bool, is_safe: bool, source_tier: str) -> str:
        """Assigns chunk to a discrete Trust Tier."""
        if not is_auth or not is_intact:
            return "REJECT"
        if not is_safe:
            return "QUARANTINE"
        if not has_prov or source_tier in ("untrusted", "unknown"):
            return "TIER_2"
        return "TIER_1"

    def post_assembly_scan(self, assembled_text: str) -> bool:
        """Post-Assembly Concatenated Payload Scan.
        
        Inspects the concatenated 1,800-token evidence payload before passing to Layer 3
        to defend against split-payload indirect prompt injection attacks.
        """
        for pattern in self.poison_patterns:
            if pattern.search(assembled_text):
                logger.warning(f"Post-assembly scan caught split-payload threat pattern: {pattern.pattern}")
                return False
        return True

    def process_candidates(
        self,
        candidates: List[Dict[str, Any]],
        tenant_id: str = "default_tenant",
        user_clearance: int = 1,
        policy: Optional[Any] = None,
        reveal_pii: bool = False,
    ) -> Layer2TrustResult:
        """Processes candidate hits through Layer 2 Discrete Decision Tree & Tiered Bucket Sorting.

        `policy` (an AdaptivePolicy) lets the user's suspicion tighten the gate: when it says tier-2 evidence is not
        allowed, passages from unverified or untrusted sources are withheld and only tier-1 evidence is used.
        """
        start_time = time.perf_counter()
        
        evaluated: List[EvaluatedCandidate] = []
        reasons: List[str] = []

        # Step 1: Discrete Evaluation of Candidates
        for candidate in candidates:
            content = candidate.get("content", "")
            metadata = candidate.get("metadata", {})
            rel_score = float(candidate.get("relevance_score", 0.0))

            is_auth = self.check_authorization(metadata, tenant_id, user_clearance)
            is_intact = self.check_integrity(content, metadata)
            has_prov = self.check_provenance(metadata)
            is_safe, poison_risk, safety_reasons = self.check_safety(content, metadata)

            source_tier = metadata.get("source_tier", "approved_external")
            tier = self.assign_tier(is_auth, is_intact, has_prov, is_safe, source_tier)

            cand_reasons = []
            if not is_auth:
                cand_reasons.append("Unauthorized document access")
            if not is_intact:
                cand_reasons.append("Hash integrity mismatch")
            if not has_prov:
                cand_reasons.append("Unverified or missing provenance")
            cand_reasons.extend(safety_reasons)

            evaluated.append(EvaluatedCandidate(
                content=content,
                metadata=metadata,
                relevance_score=rel_score,
                is_authorized=is_auth,
                is_intact=is_intact,
                has_provenance=has_prov,
                is_safe=is_safe,
                trust_tier=tier,
                reasons=cand_reasons
            ))

        # Step 2: Separate into Tier Buckets
        tier_1 = [c for c in evaluated if c.trust_tier == "TIER_1"]
        tier_2 = [c for c in evaluated if c.trust_tier == "TIER_2"]
        quarantined = [c for c in evaluated if c.trust_tier == "QUARANTINE"]
        rejected = [c for c in evaluated if c.trust_tier == "REJECT"]

        # Step 3: Bucket Sort by Cross-Encoder Relevance Score
        tier_1.sort(key=lambda x: x.relevance_score, reverse=True)
        tier_2.sort(key=lambda x: x.relevance_score, reverse=True)

        # An adaptive policy can withhold evidence from unverified sources (tier 2) for a suspicious user
        withheld, fallback = 0, False
        if policy is not None and not getattr(policy, "allow_tier2", True) and tier_2:
            if tier_1 or not getattr(policy, "tier2_fallback", True):
                withheld, tier_2 = len(tier_2), []
                reasons.append(f"Session suspicion {getattr(policy, 'suspicion', 0.0):.2f}: {withheld} passage(s) from "
                               "unverified sources were withheld.")
            else:  # nothing trusted to fall back on: answer from the unverified sources, but Layer 3 checks harder
                fallback = True
                reasons.append(f"Session suspicion {getattr(policy, 'suspicion', 0.0):.2f}: no trusted evidence exists, so "
                               "unverified sources are used under stricter verification.")

        # Priority Ranking: Tier 1 > Tier 2
        selected_candidates = tier_1 + tier_2

        if not selected_candidates:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return Layer2TrustResult(
                allowed=False,
                status="INSUFFICIENT_TRUSTED_EVIDENCE",
                evidence_package=[],
                quarantined_items=[{"content": c.content[:100], "reasons": c.reasons} for c in quarantined],
                rejected_count=len(rejected),
                tier_1_count=0,
                tier_2_count=0,
                post_assembly_safe=True,
                reasons=["No verifiable evidence satisfies trust and policy requirements."] + reasons,
                execution_time_ms=round(elapsed_ms, 2),
                withheld_count=withheld,
                fallback_unverified=fallback,
            )

        # Step 4: Package Evidence with PII Redaction
        evidence_package = []
        revealed_total: Dict[str, int] = {}
        for cand in selected_candidates:
            redacted_content, redaction_counts, revealed = self.redactor.redact_detailed(
                cand.content, reveal=self.redactor.REVEALABLE if reveal_pii else None)
            for cls, n in revealed.items():
                revealed_total[cls] = revealed_total.get(cls, 0) + n
            evidence_package.append({
                "content": redacted_content,
                "metadata": cand.metadata,
                "relevance_score": cand.relevance_score,
                "trust_tier": cand.trust_tier,
                "redactions": redaction_counts
            })

        # Step 5: Post-Assembly Payload Scan
        concatenated_text = " ".join(item["content"] for item in evidence_package)
        assembly_safe = self.post_assembly_scan(concatenated_text)

        if not assembly_safe:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            reasons.append("Post-assembly scan detected split-payload injection threat")
            return Layer2TrustResult(
                allowed=False,
                status="INSUFFICIENT_TRUSTED_EVIDENCE",
                evidence_package=[],
                quarantined_items=[{"content": c.content[:100], "reasons": c.reasons} for c in quarantined],
                rejected_count=len(rejected),
                tier_1_count=len(tier_1),
                tier_2_count=len(tier_2),
                post_assembly_safe=False,
                reasons=reasons,
                execution_time_ms=round(elapsed_ms, 2)
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return Layer2TrustResult(
            allowed=True,
            status="TRUSTED_EVIDENCE_READY",
            evidence_package=evidence_package,
            quarantined_items=[{"content": c.content[:100], "reasons": c.reasons} for c in quarantined],
            rejected_count=len(rejected),
            tier_1_count=len(tier_1),
            tier_2_count=len(tier_2),
            post_assembly_safe=True,
            reasons=["Evidence successfully verified across Layer 2 gates."] + reasons,
            execution_time_ms=round(elapsed_ms, 2),
            revealed=revealed_total,
            withheld_count=withheld,
            fallback_unverified=fallback,
        )
