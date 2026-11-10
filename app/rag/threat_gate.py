"""Layer 1 - query threat gate.

Pipeline: admission (rate / size limits) -> normalization (Unicode + bounded decoding) ->
detectors (regex rules H, embedding similarity to known attacks V) -> adaptive thresholds -> decision.

Design notes
- Blocking is driven by deterministic evidence (regex rules, repeated suspicious turns). The embedding
  detector is a soft signal: MiniLM similarity does not separate attacks from benign queries cleanly
  (a benign "list all users" scores 0.60, a paraphrased attack can score 0.35), so V is capped below the
  block threshold and can flag a query but never block it alone.
- Adaptivity: a per-user suspicion value (app/rag/session_trust.py), fed by Layers 1-3, lowers the flag and
  block thresholds, so a user who keeps probing is blocked sooner. With suspicion 0 the thresholds are the
  static ones. The value is keyed on tenant + user + IP subnet, so a new session id does not reset it.
"""
import html
import os
import re
import time
import unicodedata
import urllib.parse
import base64
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger

from .session_trust import CLASSIFIER_ONLY_EVIDENCE, STATIC_POLICY, AdaptivePolicy, SessionTrustStore
from .threat_classifier import ThreatClassifier, resolve_model
from .trust_policy import FLAG_THRESHOLD, QUARANTINE_THRESHOLD

LAYER1_POLICY_VERSION = "2.0.0"
DEFAULT_CORPUS_VERSION = "attack-corpus-v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Account roles whose authority claims are true (see app/auth.py); "admin" is the former name of super_admin
ADMIN_ROLES = ("admin", "tenant_admin", "super_admin")


@dataclass
class ThreatResult:
    allowed: bool
    action: str  # "PASS", "FLAG" (allowed, logged, raises session risk) or "BLOCK"
    final_risk: float
    base_risk: float
    session_risk: float   # the user's suspicion before this query (0 for a clean history)
    detector_scores: Dict[str, float]
    original_query: str
    processed_query: str
    encoding_anomaly: bool
    reasons: List[str]
    execution_time_ms: float
    policy_version: str = LAYER1_POLICY_VERSION
    corpus_version: str = DEFAULT_CORPUS_VERSION
    session_key: str = ""   # lets later layers report evidence about this user
    policy: AdaptivePolicy = STATIC_POLICY   # the thresholds this query was judged with


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

    # Cyrillic/Greek letters that render like Latin ones. NFKC leaves them alone, so
    # "Ignоre" with a Cyrillic "о" would otherwise slip past every Latin regex.
    HOMOGLYPHS = str.maketrans({
        # Cyrillic
        "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
        "с": "c", "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d",
        "ԛ": "q", "ԝ": "w", "ү": "y", "һ": "h",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
        "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J",
        # Greek
        "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
        "υ": "u", "χ": "x",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
        "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    })

    @classmethod
    def _fold_homoglyphs(cls, text: str) -> str:
        """Maps look-alike Cyrillic/Greek letters to Latin where they are disguising Latin text.

        Folds everything when every non-ASCII letter is a look-alike (the foreign script is
        pure disguise); otherwise folds only words that mix ASCII letters with look-alikes.
        Real Cyrillic/Greek text almost always has letters with no Latin twin (ч, л, π, ή...),
        so genuine non-Latin queries are left untouched.
        """
        foreign = [ch for ch in text if ch.isalpha() and not ch.isascii()]
        if not foreign:
            return text
        if all(ord(ch) in cls.HOMOGLYPHS for ch in foreign):
            return text.translate(cls.HOMOGLYPHS)
        return re.sub(
            r"\w+",
            lambda m: m.group(0).translate(cls.HOMOGLYPHS)
            if any(ch.isascii() and ch.isalpha() for ch in m.group(0)) else m.group(0),
            text,
        )

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
        embedded = PayloadNormalizer._decode_embedded_base64(stripped)
        if embedded != stripped:
            return embedded, True
        if re.search(r"(\\x[0-9a-fA-F]{2})+", stripped):
            decoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), stripped)
            if decoded != stripped:
                return decoded, True
        return text, False

    _BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])")

    @staticmethod
    def _decode_embedded_base64(text: str) -> str:
        """Replaces Base64 tokens inside a longer message by what they decode to.

        Decoding only a message that is entirely Base64 let an attacker defeat it by adding a word after it
        ("<base64> (please decode this)"). A token is replaced only when it decodes to mostly readable text, so
        hashes, identifiers and ordinary long words, which decode to bytes, are left alone.
        """
        def replace(match):
            token = match.group(0)
            padded = token + "=" * (-len(token) % 4)
            try:
                decoded = base64.b64decode(padded, validate=True).decode("utf-8")
            except Exception:
                return token
            letters = sum(ch.isalpha() or ch.isspace() for ch in decoded)
            if len(decoded) >= 8 and decoded.isprintable() and letters / len(decoded) >= 0.7:
                return decoded
            return token
        return PayloadNormalizer._BASE64_TOKEN.sub(replace, text)

    def normalize(self, raw_text: str) -> Tuple[str, bool, float]:
        start = time.perf_counter()
        text, anomaly = raw_text, False
        for _ in range(self.MAX_DECODE_PASSES):
            text = self._fold_homoglyphs(self._strip_control_chars(unicodedata.normalize("NFKC", text)))
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
                 corpus_path: str = "config/attack_corpus.yaml", corpus_version: str = DEFAULT_CORPUS_VERSION,
                 model_name: str = EMBEDDING_MODEL, sim_floor: Optional[float] = None, sim_ceil: Optional[float] = None):
        self.corpus_version = corpus_version
        self.embedder_fn = embedder_fn
        self.model_name = model_name or EMBEDDING_MODEL
        # Similarity levels depend on the embedding model, so a model other than MiniLM brings its own calibration
        self.sim_floor = self.SIM_FLOOR if sim_floor is None else float(sim_floor)
        self.sim_ceil = self.SIM_CEIL if sim_ceil is None else float(sim_ceil)
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
                self._model = SentenceTransformer(resolve_model(self.model_name))
            vecs = np.asarray(self._model.encode(texts, batch_size=32), dtype=float)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Normalised embeddings of several texts in one batch (loads the attack corpus on first use)."""
        if not self._loaded:
            self._load()
        return self._embed(texts)

    def score_vectors(self, vecs: np.ndarray) -> np.ndarray:
        """Score V for each embedding: similarity to the closest known attack phrasing, mapped to [0, SCORE_CAP]."""
        if self.attack_vectors is None or len(vecs) == 0:
            return np.zeros(len(vecs))
        sims = np.max(np.asarray(vecs) @ self.attack_vectors.T, axis=1)
        span = max(self.sim_ceil - self.sim_floor, 1e-9)
        return np.round(np.clip((sims - self.sim_floor) / span * self.SCORE_CAP, 0.0, self.SCORE_CAP), 4)

    def _load(self):
        self._loaded = True
        try:
            with open(self._corpus_path, "r", encoding="utf-8") as f:
                phrases = (yaml.safe_load(f) or {}).get("phrases", [])
            if phrases:
                self.attack_vectors = self._embed(phrases)
        except Exception as exc:
            logger.warning(f"Semantic guard disabled (could not build attack corpus): {exc}")

    def evaluate_full(self, text: str) -> Tuple[float, Optional[np.ndarray]]:
        """(score V, query embedding). The embedding is returned so the classifier signal can reuse it."""
        if not self._loaded:
            self._load()
        if not text.strip():
            return 0.0, None
        try:
            vec = self._embed([text])[0]
        except Exception as exc:
            logger.warning(f"Semantic guard evaluation failed: {exc}")
            return 0.0, None
        if self.attack_vectors is None:
            return 0.0, vec
        return float(self.score_vectors(vec[None, :])[0]), vec

    def evaluate(self, text: str) -> float:
        return self.evaluate_full(text)[0]


class AuthorityClaimDetector:
    """Social-engineering signal (A): a claim of authority, urgency or a pretext COMBINED WITH a request for something
    protected (switching checks off, the hidden instructions, every user's / other departments' data).

    Either half alone is ordinary ("As the HR lead, how many leave days do we get?", "List all documents about travel"),
    so only the combination scores. Since identity now comes from the login token, a message that claims an admin role
    from an account that is not an admin is a stronger signal still. Admin accounts are not checked: their claim is true
    and their access is enforced by Layer 2 anyway. A flags; it never blocks on its own (its scores stay under the block
    threshold), but it counts as a second signal backing the classifier, and repeated hits raise the user's suspicion.
    English only, like the rules."""

    CLAIM_SCORE = 0.45
    MISMATCH_SCORE = 0.65

    _ROLE = (r"(?:[\w-]+\s+){0,2}"   # up to two words before the role noun: "chief information officer", "IT security lead"
             r"(?:admin(?:istrator)?|sysadmin|superuser|root(?:\s+user)?|owner|developer|engineer|officer|lead|manager|"
             r"director|supervisor|auditor|executive|ceo|cto|cfo|cio|ciso|president|moderator|operator|head\s+of\s+\w+)")
    CLAIM = re.compile("|".join([
        rf"\b(?:i\s*['’]?\s*m|i\s+am|this\s+is|speaking\s+as|writing\s+as|as|i\s+work\s+as)\s+(?:the\s+|your\s+|an?\s+|our\s+)?{_ROLE}\b",
        r"\b(?:authori[sz](?:ed|ing|ation|e)|clearance|approved\s+(?:by|this|it)|sign(?:ed)?[-\s]?off|permission\s+from)\b",
        r"\b(?:on\s+behalf\s+of|per\s+(?:the\s+)?[\w\s]{0,30}?['’]s\s+(?:direct\s+)?(?:instruction|request|order)s?|"
        r"(?:direct\s+)?(?:instruction|order)s?\s+from\s+(?:the\s+|my\s+|our\s+)?\w+)\b",
        r"\b(?:security|it|admin|compliance|audit|support)\s+team\s+here\b",
        r"\b(?:urgent(?:ly)?|emergency|immediately|right\s+now|asap|override|lose\s+my\s+job|compliance\s+audit|"
        r"auditors?\s+(?:is|are)\s+waiting|(?:we\s+are|i\s*['’]?\s*m|i\s+am)\s+testing\s+you|upgraded\s+you|unrestricted\s+mode|developer\s+mode)\b",
    ]), re.IGNORECASE)
    ADMIN_CLAIM = re.compile(rf"\b(?:i\s*['’]?\s*m|i\s+am|this\s+is|speaking\s+as|as|i\s+work\s+as)\s+(?:the\s+|your\s+|an?\s+|our\s+)?"
                             r"(?:(?:system|it|database|platform|chief|senior|head|lead)\s+)*(?:admin(?:istrator)?|sysadmin|superuser|root|owner)\b",
                             re.IGNORECASE)
    TARGET = re.compile("|".join([
        r"\b(?:disable|turn\s+off|switch\s+off|deactivate|bypass|skip|ignore|override|remove|lift|suspend)\s+(?:all\s+|any\s+|the\s+|your\s+|its\s+)?"
        r"(?:\w+\s+)?(?:safety|security|content|filters?|checks?|verification|restrictions?|guardrails?|rules|moderation|limits?|protections?)\b",
        r"\b(?:show|print|reveal|display|output|repeat|tell\s+me|give\s+me|send|share|dump|leak|export|list|email|forward|download|copy)\b[^.?!]{0,60}?"
        r"\b(?:(?:your|the)\s+(?:hidden\s+|system\s+|original\s+|internal\s+|secret\s+)*(?:instructions|prompt|configuration|rules)|"
        r"(?:every|all|each|other|another)\b[^.?!]{0,30}?\b(?:records?|salar(?:y|ies)|data|documents?|files?|passwords?|credentials?|keys?|"
        r"tokens?|accounts?|details|e-?mails?|addresses|phone\s+numbers|ssns?|payroll|customers?|users?|employees?|departments?|tenants?))",
        r"\b(?:unrestricted\s+mode|no\s+restrictions|without\s+(?:any\s+)?(?:restrictions|filters|limits|checks))\b",
    ]), re.IGNORECASE)

    def evaluate(self, text: str, role: Optional[str] = None) -> Tuple[float, List[str]]:
        """(score A, ids). role is the logged-in account's role, or None when the caller is not authenticated.
        The claim and the request must sit in the same sentence or in neighbouring ones: in a long pasted text an
        "emergency" in one paragraph and a "list all documents" in another are unrelated."""
        if role in ADMIN_ROLES or not self.TARGET.search(text):
            return 0.0, []
        pieces = split_pieces(text) or [text]
        windows = pieces + [pieces[i] + " " + pieces[i + 1] for i in range(len(pieces) - 1)]
        score, ids = 0.0, []
        for w in windows:
            if not self.TARGET.search(w):
                continue
            if role is not None and self.ADMIN_CLAIM.search(w):
                return self.MISMATCH_SCORE, ["ROLE_CLAIM_MISMATCH"]
            if self.CLAIM.search(w):
                score, ids = self.CLAIM_SCORE, ["AUTHORITY_CLAIM_WITH_PROTECTED_REQUEST"]
        return score, ids


# ---------------------------------------------------------------------------
# Segmentation: score parts of a long message, not only its average
# ---------------------------------------------------------------------------

_SENTENCE_BREAK = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+|\n+")
WINDOW_WORDS, WINDOW_STRIDE = 40, 20
MAX_SEGMENTS = 160


def split_pieces(text: str) -> List[str]:
    """The message's sentences, long ones cut into 40-word windows with a 20-word stride, in reading order."""
    pieces: List[str] = []
    for sentence in (s.strip() for s in _SENTENCE_BREAK.split(text.strip())):
        if not sentence:
            continue
        words = sentence.split()
        if len(words) <= WINDOW_WORDS:
            pieces.append(sentence)
            continue
        for start in range(0, len(words), WINDOW_STRIDE):
            pieces.append(" ".join(words[start:start + WINDOW_WORDS]))
            if start + WINDOW_WORDS >= len(words):
                break
    return pieces


def segment_text(text: str) -> List[str]:
    """The whole message, then each sentence (long ones as 40-word windows, 20-word stride), then each pair of
    neighbouring pieces. One embedding of a long message averages a short attack away (an attack inside about 50
    words of ordinary text scored 0); scoring the pieces and keeping the highest score does not. Capped at
    MAX_SEGMENTS (the 1,024-word admission limit gives at most about 140)."""
    text = text.strip()
    if not text:
        return []
    pieces = split_pieces(text)
    segments = [text]
    if len(pieces) > 1:
        segments += pieces + [pieces[i] + " " + pieces[i + 1] for i in range(len(pieces) - 1)]
    return list(dict.fromkeys(segments))[:MAX_SEGMENTS]


JOIN_SIZES = (2, 3)


def joined_top_pieces(pieces: List[str], score_of: Callable[[str], float]) -> List[str]:
    """The 2 and the 3 most suspicious pieces of a message joined in reading order, wherever they are.

    Segment scoring sees single pieces and neighbouring pairs only, so an attack split into two harmless-looking
    halves with ordinary text between them scores low everywhere. Each half usually still scores above the filler,
    so joining the top pieces puts the halves back together. Only for messages of 3 or more pieces (with 2, the
    pair is the whole message), and a join is skipped when it equals an existing segment."""
    if len(pieces) < 3:
        return []
    ranked = sorted(range(len(pieces)), key=lambda i: -score_of(pieces[i]))
    out = []
    for k in JOIN_SIZES:
        if k < len(pieces):
            out.append(" ".join(pieces[i] for i in sorted(ranked[:k])))
    return list(dict.fromkeys(out))


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
        trust_store: Optional[SessionTrustStore] = None,
        policy_version: str = LAYER1_POLICY_VERSION,
        corpus_version: str = DEFAULT_CORPUS_VERSION,
        use_rules: bool = True,
        use_semantic: bool = True,
        use_session: bool = True,
        use_classifier: bool = True,
        classifier: Optional[ThreatClassifier] = None,
        classifier_can_block: Optional[bool] = None,
        classifier_path: Optional[str] = None,
        segment_scoring: Optional[bool] = None,
        use_claims: bool = True,
        join_top_pieces: Optional[bool] = None,
    ):
        # The use_* switches exist for ablation studies (scripts/evaluate_pipeline.py); production uses all of them.
        # use_session=False gives the static gate: thresholds never move and no suspicion is recorded.
        self.use_rules, self.use_semantic, self.use_session = use_rules, use_semantic, use_session
        self.use_claims = use_claims
        self.claims = AuthorityClaimDetector()
        if classifier is None and use_classifier:
            classifier = ThreatClassifier(**({"path": classifier_path} if classifier_path else {}), can_block=classifier_can_block)
        self.classifier = classifier
        usable = self.classifier is not None and self.classifier.available
        # The classifier's file decides the embedding model, the scoring mode and the similarity calibration, so V and C
        # always see the embeddings the classifier was trained on. Without a classifier the previous defaults apply.
        self.segment_scoring = (usable and self.classifier.scoring == "segment_max") if segment_scoring is None else segment_scoring
        # joining the most suspicious pieces is part of segment scoring unless switched off (ablation)
        self.join_top_pieces = self.segment_scoring if join_top_pieces is None else (join_top_pieces and self.segment_scoring)
        self.admission = RequestAdmission()
        self.normalizer = PayloadNormalizer()
        self.trust_store = trust_store or SessionTrustStore(secret_key=secret_key)
        self.heuristic = HeuristicDetector(patterns_path=patterns_path)
        semantic_cfg = self.classifier.semantic if usable else {}
        self.semantic = SemanticGuard(embedder_fn=embedder_fn, corpus_version=corpus_version,
                                      model_name=(self.classifier.embedding_model if usable and self.classifier.embedding_model else EMBEDDING_MODEL),
                                      sim_floor=semantic_cfg.get("sim_floor"), sim_ceil=semantic_cfg.get("sim_ceil"))
        self.policy_version = policy_version
        self.corpus_version = corpus_version

    @staticmethod
    def _compute_base_risk(h: float, v: float, c: float = 0.0, a: float = 0.0) -> float:
        """The strongest detector decides: a clear rule hit must never be diluted by quiet detectors."""
        return max(h, v, c, a)

    def _embed_and_score(self, texts: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """(V per text, P(attack) per text or None) for one batch of texts."""
        vecs = self.semantic.embed(texts)
        v = self.semantic.score_vectors(vecs) if self.use_semantic else np.zeros(len(texts))
        probs = self.classifier.probabilities(vecs) if self.classifier is not None else None
        return np.asarray(v, dtype=float), probs

    def _blocked(self, query: str, reason: str, start: float, session_key: str = "") -> ThreatResult:
        return ThreatResult(
            allowed=False, action="BLOCK", final_risk=1.0, base_risk=1.0, session_risk=0.0,
            detector_scores={"H": 1.0, "V": 1.0}, original_query=query, processed_query=query,
            encoding_anomaly=False, reasons=[reason],
            execution_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
            policy_version=self.policy_version, corpus_version=self.corpus_version, session_key=session_key,
        )

    def screen(
        self,
        query: str,
        session_id: str = "default_session",
        tenant_id: str = "default_tenant",
        user_id: str = "default_user",
        client_ip: str = "127.0.0.1",
        authenticated: bool = False,
        role: Optional[str] = None,
    ) -> ThreatResult:
        """authenticated=True means tenant_id and user_id come from a verified login token (see app/auth.py); role is
        that account's role (viewer, user, tenant_admin or super_admin), used by the authority-claim check."""
        start = time.perf_counter()
        reasons: List[str] = []

        session_key = self.trust_store.key(tenant_id, user_id, client_ip, bind_subnet=not authenticated)
        valid, admission_err = self.admission.validate(query, client_ip)
        if not valid:
            return self._blocked(query, f"Admission failure: {admission_err}", start, session_key)

        norm_query, encoding_anomaly, _ = self.normalizer.normalize(query)
        if encoding_anomaly:
            reasons.append("Payload encoding anomaly / decoding timeout detected")

        policy = self.trust_store.policy(session_key) if self.use_session else STATIC_POLICY
        session_risk = policy.suspicion

        h_score, signature_ids = self.heuristic.evaluate(norm_query, encoding_anomaly) if self.use_rules else (0.0, [])
        a_score, claim_ids = self.claims.evaluate(norm_query, role if authenticated else None) if self.use_claims else (0.0, [])
        # V and C share one batch of embeddings: the whole message, plus its sentences / windows when segment scoring
        # is on, then the most suspicious pieces joined together. Each signal keeps its highest score.
        v_score, c_prob, top_segment, segments, joined_won = 0.0, None, 0, [], False
        if self.use_semantic or self.classifier is not None:
            segments = segment_text(norm_query) if self.segment_scoring else ([norm_query] if norm_query.strip() else [])
            if segments:
                try:
                    v, probs = self._embed_and_score(segments)
                    v_score = float(np.max(v))
                    if probs is not None and len(probs):
                        top_segment = int(np.argmax(probs))
                        c_prob = float(probs[top_segment])
                    if self.join_top_pieces:
                        rank = dict(zip(segments, probs if probs is not None else v))
                        joins = [j for j in joined_top_pieces(split_pieces(norm_query), lambda p: float(rank.get(p, 0.0)))
                                 if j not in rank]
                        if joins:
                            jv, jprobs = self._embed_and_score(joins)
                            v_score = max(v_score, float(np.max(jv)))
                            if jprobs is not None and len(jprobs) and (c_prob is None or float(np.max(jprobs)) > c_prob):
                                c_prob, joined_won = float(np.max(jprobs)), True
                except Exception as exc:
                    logger.warning(f"Embedding the query failed: {exc}")
        # The classifier blocks between its block and block-alone thresholds only when another detector also fired
        corroborated = h_score > 0 or v_score > 0 or a_score > 0
        # suspicion also lowers the classifier's own flag threshold (towards its dev-calibrated floor)
        c_score = (self.classifier.risk(c_prob, n_segments=len(segments) or None, corroborated=corroborated,
                                        suspicion=policy.suspicion)
                   if self.classifier is not None else 0.0)
        detector_scores = {"H": round(h_score, 4), "V": round(v_score, 4), "C": round(c_score, 4), "A": round(a_score, 4)}

        base_risk = self._compute_base_risk(h_score, v_score, c_score, a_score)
        # The query's own risk is never discounted; suspicion only moves the thresholds the risk is compared with
        final_risk = base_risk
        if signature_ids:
            reasons.append(f"Matched Threat Signatures: {', '.join(signature_ids)}")
        if claim_ids:
            reasons.append("Claims authority, urgency or a pretext while asking for something protected"
                           + (" (the account is not an admin)" if "ROLE_CLAIM_MISMATCH" in claim_ids else ""))
        if v_score >= FLAG_THRESHOLD:
            reasons.append(f"Query resembles known attack phrasing (semantic score {v_score:.2f})")
        if c_score >= FLAG_THRESHOLD:
            if joined_won:
                where = " in its most suspicious parts read together"
            else:
                where = f" in part {top_segment} of {len(segments) - 1} of the message" if top_segment > 0 else ""
            reasons.append(f"Classifier rates this query as an attack{where} (probability {c_prob:.2f})")

        if final_risk >= policy.block_threshold:
            action = "BLOCK"
        elif final_risk >= policy.flag_threshold:
            action = "FLAG"
        else:
            action = "PASS"
        if policy.suspicion >= 0.05 and (policy.block_threshold < QUARANTINE_THRESHOLD or policy.flag_threshold < FLAG_THRESHOLD):
            reasons.append(f"Session suspicion {policy.suspicion:.2f} tightened the thresholds "
                           f"(flag {policy.flag_threshold:.2f}, block {policy.block_threshold:.2f})")

        if self.use_session:
            # a flag from the classifier alone is weaker evidence about the user than one another detector agrees with
            # (only for a classifier that may block: the flag-only version-1 classifier keeps its calibrated FLAG_CAP weight)
            classifier_only = c_score > 0 and not corroborated and self.classifier is not None and self.classifier.can_block
            self.trust_store.observe_layer1(session_key, base_risk, blocked=action == "BLOCK",
                                            weight=CLASSIFIER_ONLY_EVIDENCE if classifier_only else 1.0)
        return ThreatResult(
            allowed=action != "BLOCK", action=action, final_risk=round(final_risk, 4),
            base_risk=round(base_risk, 4), session_risk=round(session_risk, 4),
            detector_scores=detector_scores, original_query=query, processed_query=norm_query,
            encoding_anomaly=encoding_anomaly, reasons=reasons,
            execution_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
            policy_version=self.policy_version, corpus_version=self.corpus_version,
            session_key=session_key, policy=policy,
        )
