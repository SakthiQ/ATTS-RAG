import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from . import trust_policy
from .field_redactor import FieldRedactor
from .text_scan import load_patterns, normalize_for_scan
from .trust_policy import FLAG_THRESHOLD, QUERY_SCAN_MIN_SEVERITY


@dataclass
class EvaluatedCandidate:
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    is_authorized: bool
    is_intact: bool
    has_provenance: bool
    is_safe: bool
    trust_tier: str  # "TIER_1", "TIER_2", "TIER_3", "QUARANTINE", "REJECT"
    reasons: List[str] = field(default_factory=list)
    uploaded_by: Optional[str] = None


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
    fallback_unverified: bool = False   # no trusted evidence existed, so unverified (or untrusted) evidence was used
    tier_3_count: int = 0          # passages from "untrusted" sources that were used (only when nothing better existed)
    conflict_count: int = 0        # unverified passages withheld because they contradict a trusted passage
    unauthorized_count: int = 0    # candidates outside the user's tenant or clearance
    integrity_failures: int = 0    # candidates whose text no longer matches the registry (not the user's doing)


_WORD = re.compile(r"[a-z]+")
_NUMBER = re.compile(r"\d[\d,]*\.?\d*")
_NEGATION = re.compile(r"\b(?:not|no|never|cannot|without|neither|nor|none)\b|n't", re.I)
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were", "for", "on", "that", "with", "as",
         "by", "it", "this", "be", "from", "at", "their", "its", "can", "has", "have", "will", "must", "may", "should"}
_FEED_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _feed(path: str) -> List[Dict[str, Any]]:
    if path not in _FEED_CACHE:
        try:
            _FEED_CACHE[path] = [p for p in load_patterns(path) if p["severity"] >= QUERY_SCAN_MIN_SEVERITY]
        except OSError as exc:   # the built-in patterns still run; the upload scan refuses to start without the feed
            logger.error(f"Layer 2 could not load the threat feed {path}: {exc}")
            _FEED_CACHE[path] = []
    return _FEED_CACHE[path]


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _words(text: str) -> Set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _numbers(text: str) -> Set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


class Layer2TrustGate:
    """Layer 2 — Knowledge Trust & Retrieval Gate Orchestrator.

    Evaluates candidate retrieval hits using a Discrete Decision Tree & Tiered Bucket
    Sort model (avoiding continuous float algebra vulnerabilities).
    """

    MIN_PROVENANCE_SCORE = 0.60
    MAX_POISONING_RISK = 0.30
    # An unverified sentence contradicts a trusted one when it says nearly the same thing (most of its words are in the
    # trusted sentence) but with other numbers or the opposite polarity: "Refunds are accepted within 365 days."
    # against "Refunds are accepted within 30 days of purchase."
    CONFLICT_SHARE_OWN = 0.75
    CONFLICT_SHARE_TRUSTED = 0.5
    CONFLICT_MIN_WORDS = 3

    def __init__(self, vsm: Optional[Any] = None, patterns_path: str = "config/injection_patterns.yaml"):
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
        # Strict mode adds the threat feed's high-severity patterns (the ones with almost no legitimate use)
        self.feed_patterns = _feed(patterns_path) if trust_policy.L2_STRICT else []

    # ------------------------------------------------------------------ registry
    def _registry_entry(self, chunk_uid: Optional[str]) -> Optional[Dict[str, Any]]:
        if not chunk_uid or self.vsm is None:
            return None
        if hasattr(self.vsm, "registry_entry_for"):
            return self.vsm.registry_entry_for(chunk_uid)
        for entry in getattr(self.vsm, "registry", {}).values():
            if chunk_uid in entry.get("chunk_hashes", {}):
                return entry
        return None

    def _has_registry(self) -> bool:
        return self.vsm is not None and hasattr(self.vsm, "registry")

    def uploader(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Who uploaded the document this chunk belongs to (from the registry, not from the chunk's own metadata)."""
        entry = self._registry_entry(metadata.get("chunk_uid"))
        return entry.get("uploaded_by") if entry else None

    # ------------------------------------------------------------------ rules
    def check_authorization(self, metadata: Dict[str, Any], tenant_id: str, user_clearance: int = 1) -> bool:
        """Rule 1: Pre-retrieval and post-retrieval Tenant & ACL Authorization check."""
        if not trust_policy.L2_STRICT:
            doc_tenant = metadata.get("tenant_id", "default_tenant")
            if doc_tenant != tenant_id and doc_tenant != "global":
                return False
            return user_clearance >= int(metadata.get("acl_tier", 1))
        # Strict: a chunk without a tenant belongs to nobody, and there is no shared "global" tenant (a tenant may be
        # called "global"); an unreadable clearance tier counts as the highest
        doc_tenant = metadata.get("tenant_id")
        if not doc_tenant or doc_tenant != tenant_id:
            return False
        try:
            doc_clearance = int(metadata.get("acl_tier", 1))
        except (TypeError, ValueError):
            doc_clearance = 3
        return user_clearance >= doc_clearance

    def check_integrity(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Rule 2: Cryptographic Content Hash Integrity check against stored registry reference.

        Strict mode fails closed: when a registry is available, a chunk it does not know (or a chunk without an id)
        is not intact. Without a registry (unit tests, the evaluation's bare candidates) there is nothing to check."""
        chunk_uid = metadata.get("chunk_uid")
        if not trust_policy.L2_STRICT:
            if not chunk_uid or not self._has_registry():
                return True
            entry = self._registry_entry(chunk_uid)
            return entry is None or entry["chunk_hashes"][chunk_uid] == hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not self._has_registry():
            return True
        entry = self._registry_entry(chunk_uid)
        if entry is None:
            logger.warning(f"Integrity check failed: chunk {chunk_uid} is not in the registry")
            return False
        current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        expected_hash = entry["chunk_hashes"][chunk_uid]
        if current_hash != expected_hash:
            logger.warning(f"Integrity check failed for chunk {chunk_uid}: current={current_hash[:8]} vs expected={expected_hash[:8]}")
            return False
        return True

    def whole_content_intact(self, metadata: Dict[str, Any]) -> bool:
        """A table's full text (which replaces the matched row before generation) must match its registry hash too."""
        whole = metadata.get("whole_content")
        if not whole or not trust_policy.L2_STRICT or not self._has_registry():
            return bool(whole)
        entry = self._registry_entry(metadata.get("chunk_uid"))
        expected = (entry or {}).get("whole_hashes", {}).get(metadata.get("chunk_uid"))
        return expected is not None and expected == hashlib.sha256(whole.encode("utf-8")).hexdigest()

    def check_provenance(self, metadata: Dict[str, Any]) -> bool:
        """Rule 3: Provenance Lineage Verification."""
        doc_id = metadata.get("document_id")
        version = metadata.get("version")
        if trust_policy.L2_STRICT and metadata.get("source_score") is None:
            return False   # no recorded source reputation: unverified, never assumed trusted
        source_score = float(metadata.get("source_score", 0.80))

        if not doc_id or version is None:
            return False
        if source_score < self.MIN_PROVENANCE_SCORE:
            return False
        return True

    def scan_hits(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Set[str]:
        """Injection patterns found in the text: the built-in ones on the raw text, and in strict mode the built-in and
        high-severity feed patterns on the normalised text (no zero-width characters or look-alike letters).

        A feed pattern that the upload scan already saw in this chunk and its judge cleared is not counted again."""
        hits = {p.pattern for p in self.poison_patterns if p.search(content)}
        if trust_policy.L2_STRICT:
            plain = normalize_for_scan(content)
            hits |= {p.pattern for p in self.poison_patterns if p.search(plain)}
            notes = (metadata or {}).get("scan_notes") or ""
            cleared = "cleared by judge" in notes
            hits |= {p["id"] for p in self.feed_patterns
                     if p["regex"].search(plain) and not (cleared and p["id"] in notes)}
        return hits

    def check_safety(self, content: str, metadata: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """Rule 4: Tiered Poisoning & Indirect Injection Scan."""
        reasons = []

        # 1. Check metadata scan_status from ingestion
        ingestion_status = metadata.get("scan_status", "clean")
        if ingestion_status == "quarantined":
            reasons.append("Chunk flagged as quarantined during ingestion")
            return False, 1.0, reasons

        anomaly_score = float(metadata.get("anomaly_score", 0.0))
        # Strict: the same threshold as the upload scan's "flagged" (at or above 0.3)
        too_risky = anomaly_score >= FLAG_THRESHOLD if trust_policy.L2_STRICT else anomaly_score > self.MAX_POISONING_RISK
        if too_risky:
            reasons.append(f"Ingestion anomaly score ({anomaly_score:.2f}) exceeds threshold ({self.MAX_POISONING_RISK})")

        # 2. Tier 1 Fast Regex Threat Scan
        hits = self.scan_hits(content, metadata)
        if hits:
            reasons.append(f"Matched threat pattern: {sorted(hits)[0]}")
            return False, 0.90, reasons

        if too_risky:
            return False, anomaly_score, reasons

        return True, 0.0, reasons

    def assign_tier(self, is_auth: bool, is_intact: bool, has_prov: bool, is_safe: bool, source_tier: str) -> str:
        """Assigns chunk to a discrete Trust Tier."""
        if not is_auth or not is_intact:
            return "REJECT"
        if not is_safe:
            return "QUARANTINE"
        if trust_policy.L2_STRICT and source_tier == "untrusted":
            return "TIER_3"   # used only when nothing trusted or merely unverified was found
        if not has_prov or source_tier in ("untrusted", "unknown"):
            return "TIER_2"
        return "TIER_1"

    def post_assembly_scan(self, assembled_text: str) -> bool:
        """Post-Assembly Concatenated Payload Scan.

        Inspects the concatenated 1,800-token evidence payload before passing to Layer 3
        to defend against split-payload indirect prompt injection attacks.
        """
        hits = self.scan_hits(assembled_text)
        if hits:
            logger.warning(f"Post-assembly scan caught split-payload threat pattern: {sorted(hits)[0]}")
            return False
        return True

    def introduces_threat(self, existing: List[str], new_text: str, new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """True if adding `new_text` to the passages already accepted completes an injection that neither part holds
        on its own (a payload split across passages). A pattern the new passage holds by itself was judged already."""
        if not existing:
            return False
        before = self.scan_hits(" ".join(existing))
        alone = self.scan_hits(new_text) | self.scan_hits(new_text, new_metadata)
        return bool(self.scan_hits(" ".join(existing + [new_text])) - before - alone)

    def conflicts_with(self, content: str, trusted: List[str]) -> Optional[str]:
        """The trusted sentence this unverified passage contradicts, if any (see CONFLICT_* above)."""
        trusted_sents = [(s, _words(s), _numbers(s), len(_NEGATION.findall(s))) for t in trusted for s in _sentences(t)]
        for sent in _sentences(content):
            words, nums, negs = _words(sent), _numbers(sent), len(_NEGATION.findall(sent))
            if len(words) < self.CONFLICT_MIN_WORDS:
                continue
            for t_sent, t_words, t_nums, t_negs in trusted_sents:
                shared = words & t_words
                if (len(t_words) < self.CONFLICT_MIN_WORDS or len(shared) / len(words) < self.CONFLICT_SHARE_OWN
                        or len(shared) / len(t_words) < self.CONFLICT_SHARE_TRUSTED):
                    continue
                if (nums and t_nums and not nums <= t_nums) or negs != t_negs:
                    return t_sent
        return None

    # ------------------------------------------------------------------ orchestration
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
        strict = trust_policy.L2_STRICT

        evaluated: List[EvaluatedCandidate] = []
        reasons: List[str] = []

        # Step 1: Discrete Evaluation of Candidates
        for candidate in candidates:
            content = candidate.get("content", "")
            metadata = candidate.get("metadata", {})
            # The reranker writes rerank_score; relevance_score is accepted for callers that pass it directly
            rel_score = float(candidate.get("rerank_score", candidate.get("relevance_score", 0.0)))

            is_auth = self.check_authorization(metadata, tenant_id, user_clearance)
            is_intact = self.check_integrity(content, metadata)
            has_prov = self.check_provenance(metadata)
            is_safe, poison_risk, safety_reasons = self.check_safety(content, metadata)

            # Strict: a chunk that does not say where it came from is treated as an unknown source, never as trusted
            source_tier = metadata.get("source_tier") or ("unknown" if strict else "approved_external")
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
                reasons=cand_reasons,
                uploaded_by=self.uploader(metadata) if tier == "QUARANTINE" else None,
            ))

        # Step 2: Separate into Tier Buckets
        tier_1 = [c for c in evaluated if c.trust_tier == "TIER_1"]
        tier_2 = [c for c in evaluated if c.trust_tier == "TIER_2"]
        tier_3 = [c for c in evaluated if c.trust_tier == "TIER_3"]
        quarantined = [c for c in evaluated if c.trust_tier == "QUARANTINE"]
        rejected = [c for c in evaluated if c.trust_tier == "REJECT"]
        unauthorized = sum(1 for c in rejected if not c.is_authorized)
        integrity_failures = sum(1 for c in rejected if c.is_authorized and not c.is_intact)

        # Step 3: Bucket Sort by Cross-Encoder Relevance Score
        for bucket in (tier_1, tier_2, tier_3):
            bucket.sort(key=lambda x: x.relevance_score, reverse=True)

        # An unverified passage that contradicts a trusted one on the same point is withheld (strict)
        conflicts = 0
        if strict and tier_1 and tier_2:
            trusted_text = [c.content for c in tier_1]
            kept = []
            for c in tier_2:
                clash = self.conflicts_with(c.content, trusted_text)
                if clash is None:
                    kept.append(c)
                else:
                    conflicts += 1
                    logger.info(f"Layer 2 withheld an unverified passage that contradicts a trusted one: {clash[:80]!r}")
            if conflicts:
                reasons.append(f"{conflicts} passage(s) from unverified sources were withheld because they contradict "
                               "a trusted source.")
            tier_2 = kept

        # An adaptive policy can withhold evidence from unverified sources (tier 2 and 3) for a suspicious user
        withheld, fallback = 0, False
        if policy is not None and not getattr(policy, "allow_tier2", True) and (tier_2 or tier_3):
            if tier_1 or not getattr(policy, "tier2_fallback", True):
                withheld, tier_2, tier_3 = len(tier_2) + len(tier_3), [], []
                reasons.append(f"Session suspicion {getattr(policy, 'suspicion', 0.0):.2f}: {withheld} passage(s) from "
                               "unverified sources were withheld.")
            else:  # nothing trusted to fall back on: answer from the unverified sources, but Layer 3 checks harder
                fallback = True
                reasons.append(f"Session suspicion {getattr(policy, 'suspicion', 0.0):.2f}: no trusted evidence exists, so "
                               "unverified sources are used under stricter verification.")
        # Untrusted sources are a last resort: only when no trusted or unverified passage was found
        if tier_3 and (tier_1 or tier_2):
            tier_3 = []
        elif tier_3:
            fallback = True
            reasons.append("Only untrusted sources matched the question; they are used as a last resort.")

        # Priority Ranking: Tier 1 > Tier 2 > Tier 3
        selected_candidates = tier_1 + tier_2 + tier_3
        quarantined_items = [{"content": c.content[:100], "reasons": c.reasons, "uploaded_by": c.uploaded_by}
                             for c in quarantined]

        def refuse(why: List[str], post_safe: bool = True) -> Layer2TrustResult:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return Layer2TrustResult(
                allowed=False,
                status="INSUFFICIENT_TRUSTED_EVIDENCE",
                evidence_package=[],
                quarantined_items=quarantined_items,
                rejected_count=len(rejected),
                tier_1_count=len(tier_1) if not post_safe else 0,
                tier_2_count=len(tier_2) if not post_safe else 0,
                post_assembly_safe=post_safe,
                reasons=why,
                execution_time_ms=round(elapsed_ms, 2),
                withheld_count=withheld,
                fallback_unverified=fallback,
                conflict_count=conflicts,
                unauthorized_count=unauthorized,
                integrity_failures=integrity_failures,
            )

        if not selected_candidates:
            return refuse(["No verifiable evidence satisfies trust and policy requirements."] + reasons)

        # Step 4: Package Evidence with PII Redaction
        evidence_package = []
        revealed_total: Dict[str, int] = {}
        for cand in selected_candidates:
            redacted_content, redaction_counts, revealed = self.redactor.redact_detailed(
                cand.content, reveal=self.redactor.REVEALABLE if reveal_pii else None)
            evidence_package.append(({
                "content": redacted_content,
                "metadata": cand.metadata,
                "relevance_score": cand.relevance_score,
                "trust_tier": cand.trust_tier,
                "redactions": redaction_counts
            }, revealed))

        # Step 5: Post-Assembly Payload Scan
        if strict:
            # Passages are added in rank order; one that would complete an injection split across passages is left out
            # (and reported as quarantined) instead of refusing the whole question
            accepted, texts = [], []
            for item, revealed in evidence_package:
                if self.introduces_threat(texts, item["content"], item["metadata"]):
                    reasons.append("Post-assembly scan left out a passage that completed a split-payload injection.")
                    quarantined_items.append({"content": item["content"][:100], "uploaded_by": self.uploader(item["metadata"]),
                                              "reasons": ["Completes an injection split across passages"]})
                    continue
                accepted.append((item, revealed))
                texts.append(item["content"])
            evidence_package = accepted
        else:
            concatenated_text = " ".join(item["content"] for item, _ in evidence_package)
            if not self.post_assembly_scan(concatenated_text):
                reasons.append("Post-assembly scan detected split-payload injection threat")
                return refuse(reasons, post_safe=False)

        for _, revealed in evidence_package:
            for cls, n in revealed.items():
                revealed_total[cls] = revealed_total.get(cls, 0) + n
        evidence_package = [item for item, _ in evidence_package]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return Layer2TrustResult(
            allowed=True,
            status="TRUSTED_EVIDENCE_READY",
            evidence_package=evidence_package,
            quarantined_items=quarantined_items,
            rejected_count=len(rejected),
            tier_1_count=len(tier_1),
            tier_2_count=len(tier_2),
            post_assembly_safe=True,
            reasons=["Evidence successfully verified across Layer 2 gates."] + reasons,
            execution_time_ms=round(elapsed_ms, 2),
            revealed=revealed_total,
            withheld_count=withheld,
            fallback_unverified=fallback,
            tier_3_count=len(tier_3),
            conflict_count=conflicts,
            unauthorized_count=unauthorized,
            integrity_failures=integrity_failures,
        )
