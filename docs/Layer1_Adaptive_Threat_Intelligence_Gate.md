# Layer 1 - Adaptive Threat Intelligence Gate

**Boundary:** pre-retrieval query perimeter
**Decision model:** PASS / FLAG / BLOCK
**Code:** [`app/rag/threat_gate.py`](../app/rag/threat_gate.py) · **Config:** [`config/injection_patterns.yaml`](../config/injection_patterns.yaml), [`config/attack_corpus.yaml`](../config/attack_corpus.yaml) · **Tests:** [`tests/test_threat_gate.py`](../tests/test_threat_gate.py)

---

## 1. Purpose

Layer 1 screens the user's query **before** retrieval or any LLM call. It is cheap (a few milliseconds plus one small embedding) and is built so that blocking rests on deterministic evidence.

```
query -> Admission -> Normalization -> Suspicion lookup -> Detectors (H, V) -> Adaptive thresholds -> PASS / FLAG / BLOCK
```

### Design principles
1. **Deterministic blocking.** Only rule hits (and, for a user with a history of suspicious queries, scores that would otherwise only flag) can block a query.
2. **Soft semantic signal.** Embedding similarity to known attacks is not reliable enough to block alone. Measured with MiniLM, a benign "list all the users who have access" scores 0.60 while the paraphrased attack "disregard what you were told…" scores 0.34. The semantic score is therefore capped at 0.60, below the block threshold: it can flag a query and add to the user's suspicion, and it blocks only once that suspicion has lowered the block threshold.
3. **Adaptive, never discounting.** A query's own risk is never reduced. A per-user *suspicion* value (see [`Adaptive_Suspicion.md`](Adaptive_Suspicion.md)), fed by evidence from all three layers, lowers the flag and block thresholds, so a user who keeps probing is blocked sooner. With suspicion 0 the thresholds are the static ones, so honest users see the same behavior as before.
4. **Hard to reset.** Suspicion is keyed by an HMAC of tenant, user and IP /24, not by the session id, so opening a new session does not clear it.

---

## 2. Stages

### 2.1 Admission
Token-bucket rate limit per client /24 subnet (capacity 60, 1 token/s) and request bounds (≤ 32 KB, ≤ 1,024 words). A failure blocks before any analysis.

### 2.2 Normalization
NFKC normalization, removal of control / zero-width / bidi characters, and up to 3 decoding passes (HTML entities, percent-encoding, Base64, `\xNN` escapes). Needing all 3 passes, or exceeding an 8 ms budget, sets `encoding_anomaly` and adds +0.35 to the rule score. Detectors always see the decoded text.

### 2.3 Detectors

| Detector | Mechanism | Role |
|---|---|---|
| **H - rules** | Regex threat feed (`injection_patterns.yaml`, shared with the ingestion poisoning scan); each rule has a severity 0-1. Score = highest matching severity. | Deterministic, auditable; the only detector that can block on its own. Covers instruction override, role override, prompt extraction, chat-markup injection, cross-tenant requests and paraphrases of these. |
| **V - semantic** | Cosine similarity of the query embedding to `attack_corpus.yaml` phrases. Score rises linearly from 0 at similarity 0.60 to the 0.60 cap at 0.85. | Soft signal for paraphrases the rules miss. Extend the corpus without code changes. |

### 2.4 Decision

```
BaseRisk  = max(H, V)                                   # a clear rule hit is never diluted
FinalRisk = BaseRisk                                    # suspicion moves the thresholds, not the risk
block threshold = max(0.45, 0.80 - 0.50 * s)            # s = the user's suspicion in [0, 1]
flag threshold  = max(0.15, 0.30 - 0.15 * s)
```

| FinalRisk (s = 0) | Action | Effect |
|---|---|---|
| < 0.30 | `PASS` | Continue to retrieval. |
| 0.30 - 0.80 | `FLAG` | Continue; reasons are logged and the turn adds to the user's suspicion. |
| ≥ 0.80 | `BLOCK` | Stop; the API returns a safe refusal. |

(The s = 0 thresholds are `FLAG_THRESHOLD` and `QUARANTINE_THRESHOLD` from `trust_policy.py`.)

For s > 0 both thresholds fall. A query that the detectors score at 0.6 or 0.7 is only flagged for a first-time user, but is blocked once the user's suspicion is about 0.4 to 0.6, typically on the second to fourth such query. Layer 1 feeds the suspicion value with how risky each query looked (a block counts fully; a clean query lets suspicion recover), and Layers 2 and 3 add their own evidence; see [`Adaptive_Suspicion.md`](Adaptive_Suspicion.md) for the update rules, the policy applied to Layers 2 and 3, and the evaluation.

---

## 3. Response fields

`threat_gate`: `allowed`, `action`, `final_risk`, `base_risk`, `session_risk` (the user's suspicion before this query), `detector_scores` (`H`, `V`), `reasons`, `execution_time_ms`, `suspicion`, `suspicion_after` (after Layers 2 and 3 reported back) and `policy` (the thresholds and settings this query was judged with).

## 4. Known limits

- The regex feed only covers phrasings someone has written; novel paraphrases rely on the capped semantic signal and on Layers 2-3.
- Suspicion is kept in process memory and is lost on restart (use a shared store for multi-worker deployments).
- Adaptivity reacts only to queries that the detectors score (above about 0.2). Paraphrased, multilingual and social-engineering attacks that score 0 give it nothing to react to.
- Suspicion is tied to the client-supplied user identity, so an attacker who uses a fresh user id for every query is not slowed down until requests are authenticated.
- `tenant_id` and `user_id` are taken from client headers. Until requests are authenticated (e.g. JWT claims), a client can claim any tenant; this is the main open security gap.
- Rate limiting is per /24 subnet and behind a proxy sees the proxy's address.
