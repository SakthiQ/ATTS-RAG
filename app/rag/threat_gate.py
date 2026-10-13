"""Layer 1 - query threat gate.

Pipeline: admission (rate / size limits) -> normalization (Unicode + bounded decoding) ->
detectors (regex rules H, embedding similarity to known attacks V) -> session-aware risk fusion -> decision.

Design notes
- Blocking is driven by deterministic evidence (regex rules, repeated suspicious turns). The embedding
  detector is a soft signal: MiniLM similarity does not separate attacks from benign queries cleanly
  (a benign "list all users" scores 0.60, a paraphrased attack can score 0.35), so V is capped below the
  block threshold and can flag a query but never block it alone.
- Session risk is time-decayed and bound to tenant/user/IP-subnet/session, so a clean history never
  lowers a query's own risk and an attacker cannot reset it by changing session id alone.
"""
import hashlib
import hmac
import html
import math
import os
import re
import secrets
import time
import unicodedata
import urllib.parse
import base64
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger

from .trust_policy import FLAG_THRESHOLD, QUARANTINE_THRESHOLD

LAYER1_POLICY_VERSION = "2.0.0"
DEFAULT_CORPUS_VERSION = "attack-corpus-v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ThreatResult:
    allowed: bool
    action: str  # "PASS", "FLAG" (allowed, logged, raises session risk) or "BLOCK"
    final_risk: float
    base_risk: float
    session_risk: float
    detector_scores: Dict[str, float]
    original_query: str
    processed_query: str
    encoding_anomaly: bool
    reasons: List[str]
    execution_time_ms: float
    policy_version: str = LAYER1_POLICY_VERSION
    corpus_version: str = DEFAULT_CORPUS_VERSION


@dataclass
class SessionRecord:
    timestamp: float
    base_risk: float


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

def _subnet(client_ip: str) -> str:
    return ".".join(client_ip.split(".")[:3]) + ".0/24" if "." in client_ip else client_ip


class TokenBucketRateLimiter:
    """Per-client-subnet token bucket."""

    def __init__(self, capacity: float = 60.0, refill_rate: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last_update)

    def check_and_consume(self, client_key: str, tokens_requested: float = 1.0) -> bool:
        now = time.time()
        tokens, last = self.buckets.get(client_key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_rate)
        allowed = tokens >= tokens_requested
        self.buckets[client_key] = (tokens - tokens_requested if allowed else tokens, now)
        return allowed


class RequestAdmission:
    """Rejects abusive requests before any analysis is spent on them."""

    MAX_BYTES = 32 * 1024
    MAX_TOKENS = 1024

    def __init__(self, capacity: float = 60.0, refill_rate: float = 1.0):
        self.rate_limiter = TokenBucketRateLimiter(capacity, refill_rate)

    def validate(self, query: str, client_ip: str) -> Tuple[bool, Optional[str]]:
        subnet = _subnet(client_ip)
        if not self.rate_limiter.check_and_consume(subnet):
            return False, f"Rate limit exceeded for subnet {subnet}"
        size = len(query.encode("utf-8"))
        if size > self.MAX_BYTES:
            return False, f"Request payload size ({size} bytes) exceeds max limit ({self.MAX_BYTES} bytes)"
        approx_tokens = len(query.split())
        if approx_tokens > self.MAX_TOKENS:
            return False, f"Request token count ({approx_tokens}) exceeds max limit ({self.MAX_TOKENS})"
        return True, None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class PayloadNormalizer:
    """NFKC normalization, control/zero-width stripping and bounded multi-pass decoding.

    Attackers hide instructions behind HTML entities, percent-encoding, base64 or hex escapes; the
    detectors must see the decoded text. Needing the maximum number of passes is itself an anomaly.
    """

    MAX_DECODE_PASSES = 3
    TIME_BUDGET_MS = 8.0

    @staticmethod
    def _strip_control_chars(text: str) -> str:
        return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\r", "\t"))

    @staticmethod
    def _attempt_decode(text: str) -> Tuple[str, bool]:
        stripped = text.strip()
        if "&" in stripped and ";" in stripped:
            unescaped = html.unescape(stripped)
            if unescaped != stripped:
                return unescaped, True
        if "%" in stripped:
            decoded = urllib.parse.unquote(stripped)
            if decoded != stripped:
                return decoded, True
        if len(stripped) >= 16 and len(stripped) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/=]+", stripped):
            try:
                decoded = base64.b64decode(stripped).decode("utf-8", errors="ignore")
                if len(decoded) > 3 and decoded != stripped:
                    return decoded, True
            except Exception:
                pass
        if re.search(r"(\\x[0-9a-fA-F]{2})+", stripped):
            decoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), stripped)
            if decoded != stripped:
                return decoded, True
        return text, False

    def normalize(self, raw_text: str) -> Tuple[str, bool, float]:
        start = time.perf_counter()
        text, anomaly = raw_text, False
        for _ in range(self.MAX_DECODE_PASSES):
            text = self._strip_control_chars(unicodedata.normalize("NFKC", text))
            text, was_decoded = self._attempt_decode(text)
            if not was_decoded:
                break
            if (time.perf_counter() - start) * 1000.0 > self.TIME_BUDGET_MS:
                anomaly = True
                break
        else:
            anomaly = True  # still decoding after the maximum number of passes
        return text, anomaly, (time.perf_counter() - start) * 1000.0


# ---------------------------------------------------------------------------
# Session risk
# ---------------------------------------------------------------------------

class SessionRiskTracker:
    """Time-decayed risk accumulated over a session's earlier turns."""

    def __init__(self, secret_key: Optional[str] = None, half_life_sec: float = 1800.0,
                 c_i: float = 0.20, s_max: float = 1.0):
        # Per-process random key unless configured: a guessable key lets a client forge session identities
        self.secret_key = secret_key or os.getenv("SESSION_SECRET") or secrets.token_hex(16)
        self.half_life_sec = half_life_sec
        self.c_i = c_i
        self.s_max = s_max
        self.sessions: Dict[str, List[SessionRecord]] = {}

    def generate_bound_session_key(self, tenant_id: str, user_id: str, client_ip: str, session_id: str) -> str:
        identity = f"{tenant_id}:{user_id}:{_subnet(client_ip)}:{session_id}"
        return hmac.new(self.secret_key.encode(), identity.encode(), hashlib.sha256).hexdigest()

    def get_session_risk(self, session_key: str) -> float:
        now = time.time()
        total = sum(
            self.c_i * r.base_risk * math.pow(2.0, -max(0.0, now - r.timestamp) / self.half_life_sec)
            for r in self.sessions.get(session_key, [])
        )
        return min(self.s_max, total)

    def record_turn(self, session_key: str, base_risk: float, **_ignored):
        history = self.sessions.setdefault(session_key, [])
        history.append(SessionRecord(time.time(), base_risk))
        del history[:-50]  # bounded sliding window


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

class HeuristicDetector:
    """Regex rule engine (H): the deterministic, auditable signal."""

    def __init__(self, patterns_path: str = "config/injection_patterns.yaml"):
        self.patterns = self._load_patterns(patterns_path)

    @staticmethod
    def _load_patterns(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            logger.warning(f"Pattern file {path} not found. Using built-in threat patterns.")
            return [
                {"id": "ignore_instructions", "severity": 1.00, "regex": re.compile(r"ignore\s+(all\s+)?(previous\s+)?instructions", re.I)},
                {"id": "dan_mode", "severity": 0.90, "regex": re.compile(r"you\s+are\s+now\s+in\s+DAN\s+mode", re.I)},
                {"id": "reveal_prompt", "severity": 0.85, "regex": re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I)},
                {"id": "chat_markup", "severity": 0.80, "regex": re.compile(r"<\|im_start\|>", re.I)},
            ]
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        flags = re.IGNORECASE | re.DOTALL | re.MULTILINE
        return [
            {"id": p.get("id", "UNKNOWN"), "severity": float(p.get("severity", 0.50)), "regex": re.compile(p["regex"], flags)}
            for p in config.get("patterns", [])
        ]

    def evaluate(self, text: str, encoding_anomaly: bool = False) -> Tuple[float, List[str]]:
        matched = [p for p in self.patterns if p["regex"].search(text)]
        ids = [p["id"] for p in matched]
        score = max((p["severity"] for p in matched), default=0.0)
        if encoding_anomaly:
            ids.append("ENCODING_ANOMALY")
            score = min(1.0, score + 0.35)
        return score, ids


class SemanticGuard:
    """Embedding similarity to a corpus of known attack phrasings (V): a capped, soft signal.

    Similarity below SIM_FLOOR scores 0; the score rises linearly to SCORE_CAP at SIM_CEIL. SCORE_CAP is
    kept under the block threshold on purpose (see module docstring).
    """

    SIM_FLOOR = 0.60
    SIM_CEIL = 0.85
    SCORE_CAP = 0.60

    def __init__(self, embedder_fn: Optional[Callable[[str], List[float]]] = None,
                 corpus_path: str = "config/attack_corpus.yaml", corpus_version: str = DEFAULT_CORPUS_VERSION):
        self.corpus_version = corpus_version
        self.embedder_fn = embedder_fn
        self.attack_vectors: Optional[np.ndarray] = None
        self._corpus_path = corpus_path
        self._model = None
        self._loaded = False

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.embedder_fn is not None:
            vecs = np.array([self.embedder_fn(t) for t in texts], dtype=float)
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(EMBEDDING_MODEL)
            vecs = np.asarray(self._model.encode(texts), dtype=float)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    def _load(self):
        self._loaded = True
        try:
            with open(self._corpus_path, "r", encoding="utf-8") as f:
                phrases = (yaml.safe_load(f) or {}).get("phrases", [])
            if phrases:
                self.attack_vectors = self._embed(phrases)
        except Exception as exc:
            logger.warning(f"Semantic guard disabled (could not build attack corpus): {exc}")

    def evaluate(self, text: str) -> float:
        if not self._loaded:
            self._load()
        if self.attack_vectors is None or not text.strip():
            return 0.0
        try:
            sim = float(np.max(self.attack_vectors @ self._embed([text])[0]))
        except Exception as exc:
            logger.warning(f"Semantic guard evaluation failed: {exc}")
            return 0.0
        span = self.SIM_CEIL - self.SIM_FLOOR
        return round(min(self.SCORE_CAP, max(0.0, (sim - self.SIM_FLOOR) / span * self.SCORE_CAP)), 4)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ThreatGate:
    """Layer 1 orchestrator."""

    def __init__(
        self,
        patterns_path: str = "config/injection_patterns.yaml",
        secret_key: Optional[str] = None,
        embedder_fn: Optional[Callable[[str], List[float]]] = None,
        w_b: float = 0.70,
        w_s: float = 0.30,
        policy_version: str = LAYER1_POLICY_VERSION,
        corpus_version: str = DEFAULT_CORPUS_VERSION,
    ):
        self.admission = RequestAdmission()
        self.normalizer = PayloadNormalizer()
        self.session_tracker = SessionRiskTracker(secret_key=secret_key)
        self.heuristic = HeuristicDetector(patterns_path=patterns_path)
        self.semantic = SemanticGuard(embedder_fn=embedder_fn, corpus_version=corpus_version)
        self.policy_version = policy_version
        self.corpus_version = corpus_version
        self.w_b = w_b
        self.w_s = w_s

    @staticmethod
    def _compute_base_risk(h: float, v: float) -> float:
        """The strongest detector decides: a clear rule hit must never be diluted by quiet detectors."""
        return max(h, v)

    def _blocked(self, query: str, reason: str, start: float) -> ThreatResult:
        return ThreatResult(
            allowed=False, action="BLOCK", final_risk=1.0, base_risk=1.0, session_risk=0.0,
            detector_scores={"H": 1.0, "V": 1.0}, original_query=query, processed_query=query,
            encoding_anomaly=False, reasons=[reason],
            execution_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
            policy_version=self.policy_version, corpus_version=self.corpus_version,
        )

    def screen(
        self,
        query: str,
        session_id: str = "default_session",
        tenant_id: str = "default_tenant",
        user_id: str = "default_user",
        client_ip: str = "127.0.0.1",
    ) -> ThreatResult:
        start = time.perf_counter()
        reasons: List[str] = []

        valid, admission_err = self.admission.validate(query, client_ip)
        if not valid:
            return self._blocked(query, f"Admission failure: {admission_err}", start)

        norm_query, encoding_anomaly, _ = self.normalizer.normalize(query)
        if encoding_anomaly:
            reasons.append("Payload encoding anomaly / decoding timeout detected")

        session_key = self.session_tracker.generate_bound_session_key(tenant_id, user_id, client_ip, session_id)
        session_risk = self.session_tracker.get_session_risk(session_key)

        h_score, signature_ids = self.heuristic.evaluate(norm_query, encoding_anomaly)
        v_score = self.semantic.evaluate(norm_query)
        detector_scores = {"H": round(h_score, 4), "V": round(v_score, 4)}

        base_risk = self._compute_base_risk(h_score, v_score)
        # A clean session never discounts the query's own risk; a risky one raises it
        final_risk = min(1.0, max(base_risk, self.w_b * base_risk + self.w_s * session_risk))
        if signature_ids:
            reasons.append(f"Matched Threat Signatures: {', '.join(signature_ids)}")
        if v_score >= FLAG_THRESHOLD:
            reasons.append(f"Query resembles known attack phrasing (semantic score {v_score:.2f})")

        if final_risk >= QUARANTINE_THRESHOLD:
            action = "BLOCK"
        elif final_risk >= FLAG_THRESHOLD:
            action = "FLAG"
        else:
            action = "PASS"

        self.session_tracker.record_turn(session_key, base_risk)
        return ThreatResult(
            allowed=action != "BLOCK", action=action, final_risk=round(final_risk, 4),
            base_risk=round(base_risk, 4), session_risk=round(session_risk, 4),
            detector_scores=detector_scores, original_query=query, processed_query=norm_query,
            encoding_anomaly=encoding_anomaly, reasons=reasons,
            execution_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
            policy_version=self.policy_version, corpus_version=self.corpus_version,
        )
