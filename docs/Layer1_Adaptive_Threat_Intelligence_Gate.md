# Layer 1 - Adaptive Threat Intelligence Gate

**Boundary:** pre-retrieval query perimeter
**Decision model:** PASS / FLAG / BLOCK
**Code:** [`app/rag/threat_gate.py`](../app/rag/threat_gate.py) · **Config:** [`config/injection_patterns.yaml`](../config/injection_patterns.yaml), [`config/attack_corpus.yaml`](../config/attack_corpus.yaml) · **Tests:** [`tests/test_threat_gate.py`](../tests/test_threat_gate.py)

---

## 1. Purpose

Layer 1 screens the user's query **before** retrieval or any LLM call. It is cheap (a few milliseconds plus one small embedding) and is built so that blocking rests on deterministic evidence.

```
query -> Admission -> Normalization -> Session lookup -> Detectors (H, V) -> Fusion -> PASS / FLAG / BLOCK
```

### Design principles
1. **Deterministic blocking.** Only rule hits (and risk accumulated from earlier rule hits in the session) can block a query.
2. **Soft semantic signal.** Embedding similarity to known attacks is not reliable enough to block alone. Measured with MiniLM, a benign "list all the users who have access" scores 0.60 while the paraphrased attack "disregard what you were told…" scores 0.34. The semantic score is therefore capped at 0.60, below the block threshold: it can flag a query and raise session risk, never block alone.
3. **History never helps an attacker.** A clean history never lowers a query's own risk, and session risk is computed from past *base* risk only (no feedback loop).
4. **Hard to reset.** Session state is keyed by an HMAC of tenant, user, IP /24 and session id, so changing the session id does not clear it.

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

### 2.4 Fusion and decision

```
BaseRisk  = max(H, V)                                   # a clear rule hit is never diluted
SessionRisk = min(1, Σ 0.20 · BaseRisk_i · 2^(-Δt_i / 1800 s))
FinalRisk = min(1, max(BaseRisk, 0.70·BaseRisk + 0.30·SessionRisk))
```

| FinalRisk | Action | Effect |
|---|---|---|
| < 0.30 | `PASS` | Continue to retrieval. |
| 0.30 - 0.80 | `FLAG` | Continue; reasons are logged and the turn adds to session risk. |
| ≥ 0.80 | `BLOCK` | Stop; the API returns a safe refusal. |

(`FLAG_THRESHOLD` and `QUARANTINE_THRESHOLD` come from `trust_policy.py`.)

Repeated suspicious turns therefore escalate: each raises session risk, which lifts later queries toward the block threshold.

---

## 3. Response fields

`threat_gate`: `allowed`, `action`, `final_risk`, `base_risk`, `session_risk`, `detector_scores` (`H`, `V`), `reasons`, `execution_time_ms`.

## 4. Known limits

- The regex feed only covers phrasings someone has written; novel paraphrases rely on the capped semantic signal and on Layers 2-3.
- Session state is in process memory and is lost on restart (use a shared store for multi-worker deployments).
- `tenant_id` and `user_id` are taken from client headers. Until requests are authenticated (e.g. JWT claims), a client can claim any tenant; this is the main open security gap.
- Rate limiting is per /24 subnet and behind a proxy sees the proxy's address.
