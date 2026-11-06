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
| **C - classifier** | Logistic regression over the same MiniLM query embedding (`app/rag/threat_classifier.py`, weights in `config/threat_classifier.json`, plain numbers, no pickle). Trained on public prompt-injection / jailbreak sets plus generated in-domain benign questions; thresholds calibrated on a dev split. Probability below the flag threshold scores 0; above it the risk rises from 0.30 to at most `FLAG_CAP = 0.50`. | **Flag-only.** Catches rephrased, multilingual and social-engineering attacks the rules cannot, but is not allowed to block on its own (see section 2.5). |

### 2.4 Decision

```
BaseRisk  = max(H, V, C)                                # a clear rule hit is never diluted
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

### 2.5 The classifier signal (C): what it does and why it cannot block

Before C, Layer 1 flagged or blocked 29.0% of the attacks in the public test split (3,747 texts, 961 attacks, 2,786 benign; fixed train / dev / test splits made before tuning). With C it flags or blocks **94.3%**, with the same blocks as before (24.5%) and 2.3% of benign texts flagged (65 of 2,786) and 0.0% blocked (1 of 2,786).

| Public test split | Attacks blocked | Attacks flagged or blocked | Benign blocked | Benign flagged or blocked |
|---|---|---|---|---|
| Rules + similarity (previous release) | 24.5% | 29.0% | 0.0% (1/2,786) | 0.1% (2/2,786) |
| + classifier, flag-only (shipped) | 24.5% | **94.3%** | 0.0% (1/2,786) | 2.3% (65/2,786) |
| + classifier allowed to block (experiment) | 89.3% | 94.3% | 0.9% (24/2,786) | 2.3% (65/2,786) |

On the project's own 72 queries the classifier raises flagged-or-blocked attacks from 70% to 95% (multilingual and social-engineering attacks, previously not detected at all, are now flagged) and flags 7 of 32 benign queries instead of 3, all of them hard negatives or document-search phrasing.

**Why flag-only.** A flag costs the user nothing (the query continues), while a block costs an answer. The first classifier, trained on public data only, blocked ordinary document questions such as "Summarize the IT security policy." because public benign data (chat, creative prompts) does not look like questions about company documents. Adding generated in-domain benign questions fixed those, but hand-written hard negatives ("Show me the documents about data privacy and security") are still flagged. So C reports a risk of at most `FLAG_CAP` and repeated flags are turned into blocks by the user's suspicion value, which lowers the block threshold.

**Why `FLAG_CAP = 0.50`.** It was calibrated on the dev sessions of `eval/adaptive_sessions.json` with the rule used for the other suspicion constants (fewest released attack turns, no extra benign blocks, benign suspicion at most 0.45): 0.79 caused an extra benign block, 0.60 let honest users reach suspicion 0.53, 0.50 met the rule. The test sessions were already inspected once at 0.79 before this choice, so the test result at 0.50 is a second look at that split, not a clean single run.

**Does it generalise?** Partly. In a leave-one-dataset-out check (train without a dataset, score all of it, thresholds calibrated on the others) the classifier flagged or blocked 21% of deepset/prompt-injections, 68% of the safeguard set, 89% of the Lakera Gandalf set and 95% of the jailbreak-classification set. Without the generated in-domain benign questions it flagged 36% of in-domain benign questions (and would have blocked 14% had it been allowed to), so those questions are essential training data; a deployment with different documents should regenerate them from its own topics (`scripts/build_domain_benign.py`).

### 2.6 Adaptive attacker

`scripts/eval_adaptive_attacker.py` simulates a black-box attacker who sees only PASS / FLAG / BLOCK, and keeps rewording one attack (synonyms, wrappers, Cyrillic look-alike letters, leetspeak, letter spacing, translated phrases, benign padding), keeping the version that looked least suspicious. 114 test attacks (our 40 plus short attacks from the public test splits), 20 queries each:

| Defender | Attacks with a clean-scoring variant after 1 / 5 / 20 queries | Share of attempts that reach retrieval (20 queries) | Attacks blocked at least once |
|---|---|---|---|
| Rules + similarity, no memory | 55% / 92% / 100% | 89.3% | 39.5% |
| + classifier, no memory | 1.8% / 46% / 90% | 88.9% | 39.5% |
| + classifier + per-user suspicion (persistent attacker) | 1.8% / 45% / 87% | **59.1%** | **87.7%** |
| same, attacker rotates the user id every attempt | 1.8% / 46% / 90% | 88.9% | 39.5% |

The classifier makes the attacker work much harder at the start, and suspicion reduces the share of attempts that reach retrieval by 30 points and blocks most persistent attackers. Neither stops an attacker who keeps rewording: most still find a variant that scores as clean within 20 queries, and rotating the user id removes the suspicion benefit entirely. The mutations are meaning-preserving by construction, but the script cannot check that a reworded query still works as an attack, so the clean-variant rates are an upper bound.

---

## 3. Response fields

`threat_gate`: `allowed`, `action`, `final_risk`, `base_risk`, `session_risk` (the user's suspicion before this query), `detector_scores` (`H`, `V`, `C`), `reasons`, `execution_time_ms`, `suspicion`, `suspicion_after` (after Layers 2 and 3 reported back) and `policy` (the thresholds and settings this query was judged with).

## 4. Known limits

- The regex feed only covers phrasings someone has written; novel paraphrases rely on the semantic and classifier signals (which only flag) and on Layers 2-3.
- The classifier flags 2.3% of benign public texts and 22% of the project's 32 hand-written benign queries; it generalises only partly to attack datasets it did not train on (21% to 95% flagged), and depends on in-domain benign training questions.
- Against an attacker who keeps rewording, Layer 1 slows and limits the attacker but does not stop them (see 2.6).
- Suspicion is kept in process memory and is lost on restart (use a shared store for multi-worker deployments).
- Adaptivity reacts only to queries that the detectors score (above about 0.2). Paraphrased, multilingual and social-engineering attacks that score 0 give it nothing to react to.
- Suspicion is tied to the client-supplied user identity, so an attacker who uses a fresh user id for every query is not slowed down until requests are authenticated.
- `tenant_id` and `user_id` are taken from client headers. Until requests are authenticated (e.g. JWT claims), a client can claim any tenant; this is the main open security gap.
- Rate limiting is per /24 subnet and behind a proxy sees the proxy's address.
