import asyncio
import hashlib
import hmac
import html
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger

# Import existing trust policy thresholds or defaults
try:
    from .trust_policy import FLAG_THRESHOLD, QUARANTINE_THRESHOLD
except ImportError:
    FLAG_THRESHOLD = 0.30
    QUARANTINE_THRESHOLD = 0.70

LAYER1_POLICY_VERSION = "1.1.0"
DEFAULT_MODEL_VERSION = "llama3-onnx-int8-v1"
DEFAULT_CORPUS_VERSION = "attack-vector-corpus-v1"


# ============================================================================
# Data Classes & Interfaces
# ============================================================================

@dataclass
class ThreatResult:
    allowed: bool
    action: str  # "PASS", "ISOLATE", "BLOCK"
    final_risk: float
    base_risk: float
    session_risk: float
    disagreement: float
    detector_scores: Dict[str, float]
    original_query: str
    processed_query: str
    encoding_anomaly: bool
    reasons: List[str]
    execution_time_ms: float
    policy_version: str = LAYER1_POLICY_VERSION
    model_version: str = DEFAULT_MODEL_VERSION
    corpus_version: str = DEFAULT_CORPUS_VERSION
    degraded_mode: bool = False
    isolated_payload: Optional[Dict[str, Any]] = None


@dataclass
class SessionRecord:
    timestamp: float
    base_risk: float
    final_risk: float
    decision: str
    detector_scores: Dict[str, float]
    signature_ids: List[str]
    policy_version: str = LAYER1_POLICY_VERSION


# ============================================================================
# Node 0: Request Admission & Abuse Control
# ============================================================================

class TokenBucketRateLimiter:
    """Per-client IP subnet Token Bucket Rate Limiter."""

    def __init__(self, capacity: float = 60.0, refill_rate: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, Tuple[float, float]] = {}  # client_key -> (tokens, last_update)

    def check_and_consume(self, client_key: str, tokens_requested: float = 1.0) -> bool:
        now = time.time()
        if client_key not in self.buckets:
            tokens, last_update = self.capacity, now
        else:
            tokens, last_update = self.buckets[client_key]
            elapsed = now - last_update
            tokens = min(self.capacity, tokens + elapsed * self.refill_rate)

        if tokens >= tokens_requested:
            self.buckets[client_key] = (tokens - tokens_requested, now)
            return True
        else:
            self.buckets[client_key] = (tokens, now)
            return False


class RequestAdmission:
    """Node 0: Admission & Abuse Control."""

    MAX_BYTES = 32 * 1024  # 32 KB
    MAX_TOKENS = 1024
    GLOBAL_DEADLINE_MS = 60.0

    def __init__(self, capacity: float = 60.0, refill_rate: float = 1.0):
        self.rate_limiter = TokenBucketRateLimiter(capacity, refill_rate)

    def validate(self, query: str, client_ip: str) -> Tuple[bool, Optional[str]]:
        # 1. Rate Limiting Check
        subnet = ".".join(client_ip.split(".")[:3]) + ".0/24" if "." in client_ip else client_ip
        if not self.rate_limiter.check_and_consume(subnet, 1.0):
            return False, f"Rate limit exceeded for subnet {subnet}"

        # 2. Size Validation
        raw_bytes = query.encode("utf-8")
        if len(raw_bytes) > self.MAX_BYTES:
            return False, f"Request payload size ({len(raw_bytes)} bytes) exceeds max limit ({self.MAX_BYTES} bytes)"

        # 3. Token Count Check (Approximate token count via whitespace split)
        approx_tokens = len(query.split())
        if approx_tokens > self.MAX_TOKENS:
            return False, f"Request token count ({approx_tokens}) exceeds max limit ({self.MAX_TOKENS})"

        return True, None


# ============================================================================
# Node 1: Ingestion & Payload Normalization
# ============================================================================

class PayloadNormalizer:
    """Node 1: Unicode NFKC Normalization & Bounded Multi-pass Decoding."""

    MAX_DECODE_PASSES = 3
    TIME_BUDGET_MS = 8.0

    @staticmethod
    def _strip_control_chars(text: str) -> str:
        """Strips control, zero-width, and bidirectional override characters."""
        return "".join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\r", "\t")
        )

    @staticmethod
    def _attempt_decode(text: str) -> Tuple[str, bool]:
        """Attempts Base64, Hex, HTML entities, or URL decoding if payload appears encoded."""
        stripped = text.strip()

        # 1. HTML Entity Unescaping
        if "&" in stripped and ";" in stripped:
            unescaped = html.unescape(stripped)
            if unescaped != stripped:
                return unescaped, True

        # 2. URL Percent Decoding
        if "%" in stripped:
            try:
                import urllib.parse
                decoded = urllib.parse.unquote(stripped)
                if decoded != stripped:
                    return decoded, True
            except Exception:
                pass

        # 3. Base64 Detection & Decoding
        if len(stripped) >= 16 and len(stripped) % 4 == 0 and re.match(r"^[A-Za-z0-9+/=]+$", stripped):
            try:
                import base64
                decoded = base64.b64decode(stripped).decode("utf-8", errors="ignore")
                if decoded and len(decoded) > 3 and decoded != stripped:
                    return decoded, True
            except Exception:
                pass

        # 4. Hex Encoding Detection (\x41 or 414243)
        if re.search(r"(\\x[0-9a-fA-F]{2})+", stripped):
            try:
                decoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), stripped)
                if decoded != stripped:
                    return decoded, True
            except Exception:
                pass

        return text, False

    def normalize(self, raw_text: str) -> Tuple[str, bool, float]:
        start_time = time.perf_counter()
        current_text = raw_text
        encoding_anomaly = False

        for pass_idx in range(self.MAX_DECODE_PASSES):
            # NFKC Unicode Normalization
            current_text = unicodedata.normalize("NFKC", current_text)
            # Control / zero-width character filtering
            current_text = self._strip_control_chars(current_text)

            # Bounded Decoding check
            decoded_text, was_decoded = self._attempt_decode(current_text)
            if not was_decoded:
                break
            current_text = decoded_text

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms > self.TIME_BUDGET_MS:
                encoding_anomaly = True
                logger.warning(f"Normalization exceeded time budget ({elapsed_ms:.2f}ms). Halting decoding.")
                break
        else:
            # If loop finished without breaking, max passes were reached
            encoding_anomaly = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return current_text, encoding_anomaly, elapsed_ms


# ============================================================================
# Node 2: Stateful Session Intelligence
# ============================================================================

class SessionRiskTracker:
    """Node 2: Multi-Factor Cryptographic Session Identity & Time-Decayed Risk."""

    def __init__(self, secret_key: str = "atts_rag_secret_key", half_life_sec: float = 1800.0, c_i: float = 0.20, s_max: float = 1.0):
        self.secret_key = secret_key
        self.half_life_sec = half_life_sec
        self.c_i = c_i
        self.s_max = s_max
        self.sessions: Dict[str, List[SessionRecord]] = {}

    def generate_bound_session_key(self, tenant_id: str, user_id: str, client_ip: str, session_id: str) -> str:
        ip_subnet = ".".join(client_ip.split(".")[:3]) + ".0/24" if "." in client_ip else client_ip
        raw_identity = f"{tenant_id}:{user_id}:{ip_subnet}:{session_id}"
        return hmac.new(self.secret_key.encode(), raw_identity.encode(), hashlib.sha256).hexdigest()

    def get_session_risk(self, session_key: str) -> float:
        history = self.sessions.get(session_key, [])
        if not history:
            return 0.0

        now = time.time()
        accumulated_risk = 0.0

        for record in history:
            delta_t = now - record.timestamp
            if delta_t < 0:
                delta_t = 0
            # Time decay formula: 2 ^ (-delta_t / h)
            decay_factor = math.pow(2.0, -delta_t / self.half_life_sec)
            accumulated_risk += self.c_i * record.base_risk * decay_factor

        return min(self.s_max, accumulated_risk)

    def record_turn(self, session_key: str, base_risk: float, final_risk: float, decision: str, detector_scores: Dict[str, float], signature_ids: List[str]):
        if session_key not in self.sessions:
            self.sessions[session_key] = []
        
        record = SessionRecord(
            timestamp=time.time(),
            base_risk=base_risk,
            final_risk=final_risk,
            decision=decision,
            detector_scores=detector_scores,
            signature_ids=signature_ids,
            policy_version=LAYER1_POLICY_VERSION
        )
        self.sessions[session_key].append(record)
        # Keep bounded sliding window of last 50 turns per session
        if len(self.sessions[session_key]) > 50:
            self.sessions[session_key] = self.sessions[session_key][-50:]


# ============================================================================
# Node 3: Concurrent Threat Ensemble (H_t, V_t, C_t)
# ============================================================================

class HeuristicDetector:
    """Heuristic Regex Pattern Engine ($H_t$)."""

    def __init__(self, patterns_path: str = "config/injection_patterns.yaml"):
        self.patterns = self._load_patterns(patterns_path)

    @staticmethod
    def _load_patterns(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            logger.warning(f"Pattern file {path} not found. Creating default threat patterns.")
            return [
                {"id": "RULE_001", "severity": 1.00, "regex": re.compile(r"ignore\s+(all\s+)?(previous\s+)?instructions", re.I)},
                {"id": "RULE_002", "severity": 0.90, "regex": re.compile(r"you\s+are\s+now\s+in\s+DAN\s+mode", re.I)},
                {"id": "RULE_003", "severity": 0.85, "regex": re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I)},
                {"id": "RULE_004", "severity": 0.80, "regex": re.compile(r"<\|im_start\|>", re.I)},
            ]

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        flags = re.IGNORECASE | re.DOTALL | re.MULTILINE
        patterns = []
        for p in config.get("patterns", []):
            patterns.append({
                "id": p.get("id", "UNKNOWN"),
                "severity": float(p.get("severity", 0.50)),
                "regex": re.compile(p["regex"], flags)
            })
        return patterns

    def evaluate(self, text: str, encoding_anomaly: bool = False) -> Tuple[float, List[str]]:
        matched_ids = []
        max_severity = 0.0

        for p in self.patterns:
            if p["regex"].search(text):
                matched_ids.append(p["id"])
                if p["severity"] > max_severity:
                    max_severity = p["severity"]

        if encoding_anomaly:
            matched_ids.append("ENCODING_ANOMALY")
            max_severity = min(1.0, max_severity + 0.35)

        return max_severity, matched_ids


class SemanticGuard:
    """Semantic Vector Similarity Guard ($V_t$)."""

    def __init__(self, embedder_fn: Optional[Callable[[str], List[float]]] = None, corpus_version: str = DEFAULT_CORPUS_VERSION):
        self.embedder_fn = embedder_fn
        self.corpus_version = corpus_version
        self.attack_vectors: List[np.ndarray] = []

    def evaluate(self, text: str) -> float:
        if self.embedder_fn is None or not self.attack_vectors:
            # Baseline heuristic fallback if embedder not active
            keywords = ["override", "bypass", "jailbreak", "unrestricted", "system rules", "exfiltrate"]
            count = sum(1 for kw in keywords if kw in text.lower())
            return min(1.0, count * 0.25)

        try:
            vec = np.array(self.embedder_fn(text), dtype=float)
            vec_norm = vec / (np.linalg.norm(vec) + 1e-12)
            max_sim = 0.0
            for att_vec in self.attack_vectors:
                sim = float(np.dot(vec_norm, att_vec))
                if sim > max_sim:
                    max_sim = sim
            return max(0.0, min(1.0, max_sim))
        except Exception as e:
            logger.warning(f"Semantic guard evaluation failed: {e}")
            return 0.0


class LocalSLMClassifier:
    """Local Quantized SLM / LLM Classifier ($C_t$)."""

    def __init__(self, judge_fn: Optional[Callable[[str], float]] = None, model_version: str = DEFAULT_MODEL_VERSION):
        self.judge_fn = judge_fn
        self.model_version = model_version

    def evaluate(self, text: str) -> float:
        # Cap input tokens to max 256 for fast latency budget
        truncated_text = " ".join(text.split()[:256])

        if self.judge_fn is not None:
            try:
                return min(1.0, max(0.0, float(self.judge_fn(truncated_text))))
            except Exception as e:
                logger.warning(f"SLM Classifier error: {e}")
                return 0.0

        # Heuristic fallback simulator for standalone test execution
        suspicious_terms = ["ignore", "system prompt", "developer mode", "override", "dan", "jailbreak"]
        hits = sum(1 for term in suspicious_terms if term in truncated_text.lower())
        if hits >= 2:
            return 0.85
        elif hits == 1:
            return 0.40
        return 0.02


# ============================================================================
# Main Orchestrator: Layer 1 ThreatGate
# ============================================================================

class ThreatGate:
    """Layer 1 — Adaptive Threat Intelligence Gate Orchestrator."""

    def __init__(
        self,
        patterns_path: str = "config/injection_patterns.yaml",
        secret_key: str = "atts_rag_secret_key",
        judge_fn: Optional[Callable[[str], float]] = None,
        embedder_fn: Optional[Callable[[str], List[float]]] = None,
        w_h: float = 0.20,
        w_v: float = 0.30,
        w_c: float = 0.50,
        w_b: float = 0.70,
        w_s: float = 0.30,
        policy_version: str = LAYER1_POLICY_VERSION,
        model_version: str = DEFAULT_MODEL_VERSION,
        corpus_version: str = DEFAULT_CORPUS_VERSION
    ):
        self.admission = RequestAdmission()
        self.normalizer = PayloadNormalizer()
        self.session_tracker = SessionRiskTracker(secret_key=secret_key)
        self.heuristic = HeuristicDetector(patterns_path=patterns_path)
        self.semantic = SemanticGuard(embedder_fn=embedder_fn, corpus_version=corpus_version)
        self.slm = LocalSLMClassifier(judge_fn=judge_fn, model_version=model_version)

        # Versioning metadata
        self.policy_version = policy_version
        self.model_version = model_version
        self.corpus_version = corpus_version

        # Weights
        self.w_h = w_h
        self.w_v = w_v
        self.w_c = w_c
        self.w_b = w_b
        self.w_s = w_s

    def _compute_base_risk(self, h: float, v: float, c: float) -> float:
        """Dominant-Signal Max-Weighted Base Risk calculation."""
        max_score = max(h, v, c)
        weighted_score = (self.w_h * h) + (self.w_v * v) + (self.w_c * c)
        # Dominant-Signal Rule: If any single detector hit >= 0.70, max score dominates
        if max_score >= 0.70:
            return max_score
        return max(max_score * 0.9, weighted_score)

    def _compute_disagreement(self, h: float, v: float, c: float) -> float:
        """Calculates detector disagreement (Standard Deviation)."""
        return float(np.std([h, v, c]))

    def screen(
        self,
        query: str,
        session_id: str = "default_session",
        tenant_id: str = "default_tenant",
        user_id: str = "default_user",
        client_ip: str = "127.0.0.1"
    ) -> ThreatResult:
        start_time = time.perf_counter()
        reasons = []

        # --------------------------------------------------------------------
        # Node 0: Request Admission
        # --------------------------------------------------------------------
        valid, admission_err = self.admission.validate(query, client_ip)
        if not valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ThreatResult(
                allowed=False,
                action="BLOCK",
                final_risk=1.0,
                base_risk=1.0,
                session_risk=0.0,
                disagreement=0.0,
                detector_scores={"H": 1.0, "V": 1.0, "C": 1.0},
                original_query=query,
                processed_query=query,
                encoding_anomaly=False,
                reasons=[f"Node 0 Admission Failure: {admission_err}"],
                execution_time_ms=elapsed_ms,
                policy_version=self.policy_version,
                model_version=self.model_version,
                corpus_version=self.corpus_version
            )

        # --------------------------------------------------------------------
        # Node 1: Normalization & Decoding
        # --------------------------------------------------------------------
        norm_query, encoding_anomaly, norm_time = self.normalizer.normalize(query)
        if encoding_anomaly:
            reasons.append("Payload encoding anomaly / decoding timeout detected")

        # --------------------------------------------------------------------
        # Node 2: Session Intelligence
        # --------------------------------------------------------------------
        try:
            session_key = self.session_tracker.generate_bound_session_key(tenant_id, user_id, client_ip, session_id)
            session_risk = self.session_tracker.get_session_risk(session_key)
            session_degraded = False
        except Exception as e:
            logger.warning(f"Session store failure: {e}. Using BaseRisk-only degraded mode.")
            session_key = f"degraded_{session_id}"
            session_risk = 0.0
            session_degraded = True
            reasons.append("Session store unavailable: Operating in BaseRisk-only degraded mode")

        # --------------------------------------------------------------------
        # Node 3: Concurrent Threat Ensemble (H_t, V_t, C_t)
        # --------------------------------------------------------------------
        h_score, sig_ids = self.heuristic.evaluate(norm_query, encoding_anomaly)
        v_score = self.semantic.evaluate(norm_query)
        c_score = self.slm.evaluate(norm_query)

        detector_scores = {"H": round(h_score, 4), "V": round(v_score, 4), "C": round(c_score, 4)}

        # --------------------------------------------------------------------
        # Node 4: Adaptive Risk Aggregation
        # --------------------------------------------------------------------
        base_risk = self._compute_base_risk(h_score, v_score, c_score)
        # Security Invariant 1: Clean session history (SessionRisk=0) must never discount BaseRisk below itself
        session_weighted_risk = (self.w_b * base_risk) + (self.w_s * session_risk)
        final_risk = min(1.0, max(base_risk, session_weighted_risk))
        disagreement = self._compute_disagreement(h_score, v_score, c_score)

        if sig_ids:
            reasons.append(f"Matched Threat Signatures: {', '.join(sig_ids)}")

        # --------------------------------------------------------------------
        # Node 5: Policy Enforcement & Isolation Semantics
        # --------------------------------------------------------------------
        action = "PASS"
        if final_risk >= QUARANTINE_THRESHOLD:
            action = "BLOCK"
        elif final_risk >= FLAG_THRESHOLD:
            action = "ISOLATE"
        else:
            # Disagreement Escalation Policy Fix: High uncertainty forces ISOLATE
            if disagreement >= 0.35:
                action = "ISOLATE"
                reasons.append(f"High detector disagreement ({disagreement:.2f}) escalated decision to ISOLATE")

        allowed = action != "BLOCK"
        isolated_payload = None

        if action == "ISOLATE":
            isolated_payload = {
                "query": norm_query,
                "security_metadata": {
                    "trust_level": "untrusted_user_input",
                    "final_risk": round(final_risk, 4),
                    "base_risk": round(base_risk, 4),
                    "session_risk": round(session_risk, 4),
                    "action": "ISOLATE",
                    "isolation_policy": "STRICT_CONTAINMENT"
                }
            }

        # --------------------------------------------------------------------
        # Node 6: Atomic Session Update
        # --------------------------------------------------------------------
        if not session_degraded:
            self.session_tracker.record_turn(
                session_key=session_key,
                base_risk=base_risk,
                final_risk=final_risk,
                decision=action,
                detector_scores=detector_scores,
                signature_ids=sig_ids
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ThreatResult(
            allowed=allowed,
            action=action,
            final_risk=round(final_risk, 4),
            base_risk=round(base_risk, 4),
            session_risk=round(session_risk, 4),
            disagreement=round(disagreement, 4),
            detector_scores=detector_scores,
            original_query=query,
            processed_query=norm_query,
            encoding_anomaly=encoding_anomaly,
            reasons=reasons,
            execution_time_ms=round(elapsed_ms, 2),
            policy_version=self.policy_version,
            model_version=self.model_version,
            corpus_version=self.corpus_version,
            degraded_mode=session_degraded,
            isolated_payload=isolated_payload
        )
