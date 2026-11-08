"""Per-user suspicion state that lets all three layers adapt to observed behavior.

A user's *suspicion* s in [0, 1] starts at 0. Evidence from any layer raises it; time and clean turns lower it.
`policy_for(s)` turns s into the settings each layer uses on the user's NEXT query, so a user who keeps probing
meets stricter gates, while a user with a clean history sees exactly the static (s = 0) behavior.

Evidence sources (see the observe_* methods):
  Layer 1  how risky the query looked (rule / semantic score), or an outright block
  Layer 2  retrieved chunks that were quarantined by the pattern scan, or failed authorization
  Layer 3  a secret / prompt-leak in the generated answer, or claims that CONTRADICT the evidence
A low grounding ratio is deliberately NOT evidence: it is common for benign questions (about half of a draft
is typically dropped) and would lock honest users out.

The state is keyed on tenant + user + IP subnet, NOT on the session id, so opening a new session does not reset it.
(Tenant and user still come from client-supplied headers until requests are authenticated.)
"""
import hashlib
import hmac
import math
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

# ---- tunable constants (ALPHA, BLOCK_SLOPE and CLEAN_RELAX were calibrated on the dev split of eval/adaptive_sessions.json by grid search; the test split was run once afterwards) ----------
ALPHA = 0.5                # how strongly one unit of evidence moves s (saturating: s <- s + (1 - s) * ALPHA * e)
HALF_LIFE_SEC = 600.0      # suspicion halves every 10 minutes without new evidence, so a user who stops recovers quickly
CLEAN_RISK = 0.20          # a query whose base risk is below this counts as a clean turn
CLEAN_RELAX = 0.95         # clean turn: s is multiplied by this, so honest users recover (calibrated on dev)
RISK_FLOOR, RISK_SPAN = 0.20, 0.60   # layer-1 evidence e = clip((base_risk - floor) / span)
L2_QUARANTINE_E = 0.6      # a chunk matched a poisoning pattern at query time
L2_AUTH_E = 0.5            # a candidate failed tenant / clearance authorization
L3_SAFETY_E = 0.8          # the generated answer leaked a secret or the system prompt
L3_CONTRADICTION_E = 0.3   # per claim that the evidence CONTRADICTS (capped at 1.0 in total)
CLASSIFIER_ONLY_EVIDENCE = 0.5   # a Layer 1 flag raised by the classifier with no other detector agreeing counts half:
                                 # one false flag on an honest message should not make the gate strict with that user

# ---- how suspicion tightens each layer ---------------------------------------------------------------------
BLOCK_BASE, BLOCK_SLOPE, BLOCK_FLOOR = 0.80, 0.50, 0.45   # Layer 1 block threshold = max(floor, 0.80 - 0.50 s)  (slope calibrated on dev)
FLAG_BASE, FLAG_SLOPE, FLAG_FLOOR = 0.30, 0.15, 0.15      # Layer 1 flag threshold  = max(floor, 0.30 - 0.15 s)
TIER2_CUTOFF = 0.50        # Layer 2: above this, evidence from unknown / untrusted sources is set aside ...
TIER2_FALLBACK = True      # ... unless nothing trusted is left: then it is used, under the stricter Layer 3 check (False = withhold, as in v1)
ENTAIL_BASE, ENTAIL_SLOPE = 0.60, 0.10                    # Layer 3 entailment threshold = 0.60 + 0.10 s (a gentle rise: at most 0.70)
GROUNDING_CUTOFF, MIN_GROUNDING = 0.70, 0.0               # Layer 3 minimum verified share; 0 = off (a 50% rule rejected most honest answers)


@dataclass(frozen=True)
class AdaptivePolicy:
    """The settings the three layers use for one query."""
    suspicion: float = 0.0
    block_threshold: float = BLOCK_BASE
    flag_threshold: float = FLAG_BASE
    allow_tier2: bool = True
    tier2_fallback: bool = True   # use tier-2 evidence anyway when no trusted evidence exists
    entailment_threshold: float = ENTAIL_BASE
    min_grounding: float = 0.0   # 0 = no requirement

    def as_dict(self) -> Dict[str, float]:
        return {"suspicion": round(self.suspicion, 4), "block_threshold": round(self.block_threshold, 4),
                "flag_threshold": round(self.flag_threshold, 4), "allow_tier2": self.allow_tier2, "tier2_fallback": self.tier2_fallback,
                "entailment_threshold": round(self.entailment_threshold, 4), "min_grounding": self.min_grounding}


STATIC_POLICY = AdaptivePolicy()


def policy_for(s: float) -> AdaptivePolicy:
    s = min(1.0, max(0.0, s))
    return AdaptivePolicy(
        suspicion=s,
        block_threshold=max(BLOCK_FLOOR, BLOCK_BASE - BLOCK_SLOPE * s),
        flag_threshold=max(FLAG_FLOOR, FLAG_BASE - FLAG_SLOPE * s),
        allow_tier2=s < TIER2_CUTOFF,
        tier2_fallback=bool(TIER2_FALLBACK),
        entailment_threshold=ENTAIL_BASE + ENTAIL_SLOPE * s,
        min_grounding=MIN_GROUNDING if s >= GROUNDING_CUTOFF else 0.0,
    )


def _subnet(client_ip: str) -> str:
    return ".".join(client_ip.split(".")[:3]) + ".0/24" if "." in client_ip else client_ip


class SessionTrustStore:
    """Thread-safe store of suspicion values with time decay."""

    def __init__(self, secret_key: Optional[str] = None, alpha: Optional[float] = None, half_life_sec: Optional[float] = None,
                 clock: Callable[[], float] = time.time):
        # Random per-process key unless configured: a guessable key lets a client forge identities
        self.secret_key = secret_key or os.getenv("SESSION_SECRET") or secrets.token_hex(16)
        # None -> read the module constants at use time, so an experiment can override them
        self._alpha, self._half_life_sec, self._clock = alpha, half_life_sec, clock
        self._state: Dict[str, Tuple[float, float]] = {}   # key -> (suspicion at time, time)
        self._lock = threading.Lock()

    @property
    def alpha(self) -> float:
        return ALPHA if self._alpha is None else self._alpha

    @property
    def half_life_sec(self) -> float:
        return HALF_LIFE_SEC if self._half_life_sec is None else self._half_life_sec

    def key(self, tenant_id: str, user_id: str, client_ip: str, bind_subnet: bool = True) -> str:
        """The suspicion key. For an unauthenticated caller the claimed user is bound to the client /24, so a
        forged user id from another network does not share (or reset) someone's state. For a logged-in user
        (bind_subnet=False) the account alone is the identity: changing network does not reset suspicion."""
        identity = f"{tenant_id}:{user_id}:{_subnet(client_ip)}" if bind_subnet else f"{tenant_id}:{user_id}:authenticated"
        return hmac.new(self.secret_key.encode(), identity.encode(), hashlib.sha256).hexdigest()

    def _decayed(self, key: str, now: float) -> float:
        s, t = self._state.get(key, (0.0, now))
        return s * math.pow(2.0, -max(0.0, now - t) / self.half_life_sec)

    def suspicion(self, key: str) -> float:
        with self._lock:
            return self._decayed(key, self._clock())

    def policy(self, key: str) -> AdaptivePolicy:
        return policy_for(self.suspicion(key))

    def observe(self, key: str, evidence: float) -> float:
        """Raises suspicion by evidence e in [0, 1]; returns the new value."""
        e = min(1.0, max(0.0, evidence))
        with self._lock:
            now = self._clock()
            s = self._decayed(key, now)
            s = s + (1.0 - s) * self.alpha * e
            self._state[key] = (s, now)
            return s

    # ---- evidence from each layer -------------------------------------------------------------------------
    def observe_layer1(self, key: str, base_risk: float, blocked: bool = False, weight: float = 1.0) -> float:
        """weight scales the evidence of a flagged (not blocked) query, e.g. CLASSIFIER_ONLY_EVIDENCE."""
        if blocked:
            return self.observe(key, 1.0)
        if base_risk < CLEAN_RISK:
            with self._lock:  # a clean turn lets suspicion recover
                now = self._clock()
                s = self._decayed(key, now) * CLEAN_RELAX
                self._state[key] = (s, now)
                return s
        return self.observe(key, weight * (base_risk - RISK_FLOOR) / RISK_SPAN)

    def observe_layer2(self, key: str, quarantined: int = 0, auth_failures: int = 0) -> float:
        e = (L2_QUARANTINE_E if quarantined else 0.0) + (L2_AUTH_E if auth_failures else 0.0)
        return self.observe(key, e) if e else self.suspicion(key)

    def observe_layer3(self, key: str, safety_hit: bool = False, contradictions: int = 0) -> float:
        e = (L3_SAFETY_E if safety_hit else 0.0) + L3_CONTRADICTION_E * max(0, contradictions)
        return self.observe(key, e) if e else self.suspicion(key)
