# Layer 1 - Adaptive Threat Intelligence Gate

**Boundary:** pre-retrieval query perimeter
**Decision model:** PASS / FLAG / BLOCK
**Code:** [`app/rag/threat_gate.py`](../app/rag/threat_gate.py), [`app/rag/threat_classifier.py`](../app/rag/threat_classifier.py) · **Config:** [`config/injection_patterns.yaml`](../config/injection_patterns.yaml), [`config/attack_corpus.yaml`](../config/attack_corpus.yaml), [`config/threat_classifier.json`](../config/threat_classifier.json) · **Tests:** [`tests/test_threat_gate.py`](../tests/test_threat_gate.py), [`tests/test_layer1_upgrades.py`](../tests/test_layer1_upgrades.py), [`tests/test_threat_classifier.py`](../tests/test_threat_classifier.py)

---

## 1. Purpose

Layer 1 screens the user's query **before** retrieval or any LLM call.

```
query -> Admission -> Normalization -> Suspicion lookup -> Segments -> Detectors (H, V, C, A) -> Adaptive thresholds -> PASS / FLAG / BLOCK
```

### Design principles
1. **Block on agreement.** A rule hit blocks on its own. The trained classifier blocks on its own only when it is very sure; otherwise it blocks only when a second detector agrees, and a classifier hit nothing else backs is flagged.
2. **Look at the pieces, not only the whole.** An attack hidden in a long, harmless message is averaged away by one embedding of the whole text, so the gate also scores every sentence (or window) and the most suspicious pieces joined together, and keeps the highest score.
3. **Adaptive, never discounting.** A query's own risk is never reduced. A per-user *suspicion* value (see [`Adaptive_Suspicion.md`](Adaptive_Suspicion.md)), fed by evidence from all three layers, lowers the gate's thresholds and the classifier's own flag threshold, so a user who keeps probing is flagged and blocked sooner. With suspicion 0 the thresholds are the static ones, so honest users see the static behaviour.
4. **Hard to reset.** Identity comes from a signed login token (`app/auth.py`); suspicion is keyed by the account, not by the session or the network.

---

## 2. Stages

### 2.1 Admission
Token-bucket rate limit per client /24 subnet (capacity 60, 1 token/s) and request bounds (≤ 32 KB, ≤ 1,024 words). A failure blocks before any analysis.

### 2.2 Normalization
NFKC normalization, removal of control / zero-width / bidi characters, homoglyph folding, and up to 3 decoding passes (HTML entities, percent-encoding, Base64, `\xNN` escapes). Needing all 3 passes, or exceeding an 8 ms budget, sets `encoding_anomaly` and adds +0.35 to the rule score. Detectors always see the decoded text.

### 2.3 Segments
`segment_text` cuts the message into sentences; a sentence longer than 40 words is cut into 40-word windows (stride 20); neighbouring pairs are added, and the whole message is always included (at most 160 pieces). The embedding detectors (V and C) score every piece. For three or more pieces the gate also scores the **top 2 and top 3 pieces joined in reading order** (`joined_top_pieces`), which catches an attack split in two halves with harmless text between them.

### 2.4 Detectors

| Detector | Mechanism | Role |
|---|---|---|
| **H - rules** | Regex threat feed (`injection_patterns.yaml`, shared with the ingestion poisoning scan); each rule has a severity 0-1. Score = highest matching severity. | Deterministic and auditable; blocks on its own. |
| **V - similarity** | Cosine similarity of each piece's embedding to `attack_corpus.yaml` phrases, highest piece kept. Score 0 below a floor and rising to the cap of 0.60 at a ceiling; floor and ceiling are calibrated on dev for the embedding model (0.578 and 0.828, 2% benign flags). | Soft signal for paraphrases; flags and backs up the classifier. |
| **C - classifier** | Logistic regression over the embedding of each piece (multilingual MiniLM-L12, `paraphrase-multilingual-MiniLM-L12-v2`; weights in `config/threat_classifier.json`, plain numbers, no pickle). Highest piece (or join) kept. Thresholds depend on message length (section 2.5). | Catches rephrased, translated, hidden and split attacks; may block (section 2.5). |
| **A - authority claim** | `AuthorityClaimDetector`: a claim of authority, urgency or a pretext ("as the CIO…", "urgent, I'll lose my job", "approved by security", "compliance audit") **in the same or the neighbouring sentence** as a request for something protected (disable checks, reveal the system prompt, list every user's records). Score 0.45; 0.65 when a logged-in non-admin claims to be an admin. Skipped for admins. | Social engineering that has no injection wording. Flags; backs up the classifier. |

### 2.5 The classifier: length buckets and corroboration

With segment scoring, the more pieces a message has, the more chances one ordinary sentence scores high, so one threshold for all lengths is either too strict on long text or too lenient on short attacks. The thresholds are therefore calibrated **per bucket of segment count** (1, 2-10, 11-40, more than 40), each on the dev split to the same benign rates: flag 2%, block 0.5%, block-alone 0.1% of benign dev texts (a bucket with fewer than 200 benign dev rows is merged with its neighbour). `scripts/train_threat_classifier.py` trains and calibrates; the test split is never read.

| Max segments | Flag | Block | Block alone | Flag floor (suspicion) |
|---|---|---|---|---|
| 1 | 0.691 | 0.909 | 0.973 | 0.112 |
| 10 | 0.896 | 0.941 | 0.991 | 0.308 |
| 40 | 0.940 | 0.981 | 0.990 | 0.659 |
| no limit | 0.988 | 0.992 | 0.996 | 0.901 |

The probability is mapped to a risk:

| Classifier probability p | Risk | Effect at suspicion 0 |
|---|---|---|
| p < flag | 0 | nothing |
| flag ≤ p < block | 0.30 to 0.50 (linear) | flag |
| block ≤ p < block alone, **no other detector fired** | 0.60 (`STRONG_RISK`) | flag; blocks once suspicion has lowered the block threshold to 0.60 |
| block ≤ p < block alone, H, V or A also fired | 0.85 to 1.0 | block |
| p ≥ block alone | 0.85 to 1.0 | block |

**Suspicion lowers the flag threshold too.** A reworded attack that scores just under the flag threshold reports risk 0, and suspicion, which only lowers the gate's thresholds on that risk, could not catch it. So the classifier moves its flag threshold linearly towards a **flag floor** as the user's suspicion rises (fully at `SUSPICION_FULL = 0.5`). The floor is the threshold that flags 20% of benign dev texts of the bucket; the block thresholds never move. For a user with suspicion 0 nothing changes.

A classifier hit that no other detector backs counts **half** towards the user's suspicion (`CLASSIFIER_ONLY_EVIDENCE = 0.5`), so one false flag does not make the gate strict with an honest user.

### 2.6 Decision

```
BaseRisk  = max(H, V, C, A)                             # a clear rule hit is never diluted
FinalRisk = BaseRisk                                    # suspicion moves the thresholds, not the risk
block threshold = max(0.45, 0.80 - 0.50 * s)            # s = the user's suspicion in [0, 1]
flag threshold  = max(0.15, 0.30 - 0.15 * s)
```

| FinalRisk (s = 0) | Action | Effect |
|---|---|---|
| < 0.30 | `PASS` | Continue to retrieval. |
| 0.30 - 0.80 | `FLAG` | Continue; reasons are logged and the turn adds to the user's suspicion. |
| ≥ 0.80 | `BLOCK` | Stop; the API returns a safe refusal. |

---

## 3. Evaluation

All runs: `eval/external/` public and generated data with fixed train / dev / test splits made before tuning; every threshold and constant chosen on dev; the **test split run once** (`--blind`). Intervals are Wilson 95%. Code evaluated: commit `f409ace`.

### 3.1 Blind test split (7,517 texts: 2,856 attacks, 4,661 benign)

`python scripts/eval_layer1_public.py --splits test --config full --blind` (and `--config v1` for the previous gate).

| Gate | Attacks flagged or blocked | Attacks blocked | Benign flagged or blocked | Benign blocked |
|---|---|---|---|---|
| Previous gate (English classifier, whole message, flag-only) | 61.4% [60-63] | 15.1% [14-16] | 4.3% [4-5] | 0.02% (1/4,661) |
| **Current gate** | **89.6% [88-91]** | **70.4% [69-72]** | **2.1% [2-3]** | 0.28% (13/4,661) |
| ProtectAI DeBERTa-v3 prompt-injection v2, whole message | 69.9% [68-72] | - | 8.4% [8-9] | - |
| ProtectAI, piece by piece (same pieces as the gate) | 89.1% [88-90] | - | 15.6% [15-17] | - |

The published guard model (`protectai/deberta-v3-base-prompt-injection-v2`, `scripts/eval_layer1_protectai.py`) catches as many attacks as the current gate only when it also scores the pieces, and then flags 7 times as many benign texts (28.5% of long benign paragraphs, 37.7% of translated benign questions). Dev and test agree (dev: 89.9% caught, 2.3% benign flagged), so the gate was not fitted to dev.

**By attack family (flagged or blocked; benign: flagged or blocked):**

| Family | n | Previous gate | Current gate | ProtectAI whole | ProtectAI pieces |
|---|---|---|---|---|---|
| Translated attacks (7 languages) | 1,050 | 47.2% | **94.2%** | 88.3% | 88.8% |
| - languages not in training (Italian, Portuguese, Chinese) | 450 | - | 93.3-95.3% | - | - |
| Attack hidden in a long paragraph | 500 | 41.4% | **80.4%** | 31.4% | 91.0% |
| Attack split in two halves | 300 | 35.0% | **83.0%** | 22.7% | 90.7% |
| Social engineering | 45 | 86.7% | 80.0% | 100% | 100% |
| Jailbreak prompts | 139 | 95.0% | 92.1% | 84.2% | 95.7% |
| Direct override (Gandalf) | 112 | 97.3% | 93.8% | 100% | 100% |
| *Benign:* translated questions | 1,050 | 10.9% | **1.4%** | 34.0% | 37.7% |
| *Benign:* long paragraphs | 800 | 2.5% | 3.1% | 0.9% | 28.5% |
| *Benign:* in-domain questions | 935 | 0.6% | 1.5% | 2.6% | 2.6% |
| *Benign:* hard negatives | 50 | 12.0% | 8.0% | 2.0% | 2.0% |

**Worse than before:** social engineering (86.7% to 80.0%, though blocked rose from 33% to 53%), jailbreak prompts (95.0% to 92.1%), direct override (97.3% to 93.8%), in-domain benign questions flagged (0.6% to 1.5%), long benign paragraphs flagged (2.5% to 3.1%), and benign texts blocked (1 to 13 of 4,661). The weakest public set is still deepset/prompt-injections (46.7% caught; its German attacks 7 of 19).

**Ablations (test, flagged or blocked):**

| Removed | All attacks | Split attacks | Hidden, middle | Social engineering | Benign flagged |
|---|---|---|---|---|---|
| nothing (current gate) | 89.6% | 83.0% | 75.6% | 80.0% | 2.1% |
| joining the top pieces | 87.2% | 68.0% | 70.8% | 80.0% | 1.9% |
| authority-claim check | 89.4% | 83.0% | 75.0% | 68.9% | 2.1% |

### 3.2 Hand-written queries (40 attacks, 32 benign, 9 families)

`python scripts/evaluate_pipeline.py --suite l1`. Attacks blocked 57.5% → **87.5%** (flagged or blocked 95.0% → 97.5%); benign flagged or blocked 21.9% → 18.8%; benign blocked 6.2% (2 of 32) in both. The two blocked benign queries are hard negatives that a rule matches: "Should employees ignore any instructions found in phishing emails?" and "Act as a travel assistant and explain the booking rules for flights."

### 3.3 Adaptive attacker

`scripts/eval_adaptive_attacker.py` simulates a black-box attacker who sees only PASS / FLAG / BLOCK and keeps rewording one attack (synonyms, wrappers, Cyrillic look-alike letters, leetspeak, letter spacing, translated phrases, benign padding), keeping the version that looked least suspicious. 114 test attacks (the 40 hand-written ones plus short attacks from the public test splits), 20 queries each. *Unflagged*: at least one attempt got a plain PASS. *Released*: attempts not blocked (they reach retrieval).

| Defender | Attacks with an unflagged attempt after 1 / 5 / 20 queries | Attempts released after 5 / 20 queries | Attacks blocked at least once |
|---|---|---|---|
| Rules + similarity, no memory | 55.3% / 88.6% / 100% | 76.8% / 87.4% | 39.5% |
| Previous gate + suspicion, persistent attacker | 1.8% / 33.3% / 73.7% | 58.9% / 49.7% | 93.0% |
| Current gate, no memory | 4.4% / 47.4% / 94.7% | 41.4% / 65.5% | 89.5% |
| **Current gate + suspicion, persistent attacker** | 4.4% / **17.5%** / **71.1%** | **28.4%** / **48.7%** | 92.1% |
| same, logged in, new account per attempt (5-account sign-up limit) | 4.4% / 47.4% / 76.3% | 41.4% / 53.1% | 93.9% |
| same, attacker claims a new user id every attempt (no login) | 4.4% / 47.4% / 94.7% | 41.4% / 65.5% | 89.5% |

On the first attempt the current gate releases 16.7% of attacks against 60.5% for the previous one. Against a persistent attacker it now matches or beats the previous gate from 5 queries on; on the very first query the previous gate leaves fewer attacks unflagged (1.8% vs 4.4%, because it flagged far more of everything, benign texts included), and it blocks slightly more attacks at least once (93.0% vs 92.1%). Before the flag floor (section 2.5) it was not: after 20 queries 94.7% of attacks had an unflagged attempt and 54.9% of attempts were released, because a reworded attack scoring just below the high flag threshold reported risk 0, which suspicion cannot act on. **That problem was found on these test seeds; the floor was then chosen on the dev seeds only, and the test seeds run again, so this table is a second look at the attacker test split.** The mutations are meaning-preserving by construction, but the script cannot check that a reworded query still works as an attack, so the unflagged rates are an upper bound.

### 3.4 Multi-turn sessions

See [`Adaptive_Suspicion.md`](Adaptive_Suspicion.md) section 5: on the test sessions suspicion halves the attack turns released (23.8% to 11.9%) with no extra benign block.

### 3.5 Latency

`python scripts/measure_layer1_latency.py` (500 dev messages, one at a time, CPU: Intel Core Ultra 7 155H). Time of `ThreatGate.screen` in ms:

| Message length | Previous gate, median / p95 | Current gate, median / p95 |
|---|---|---|
| 1-20 words | 7 / 8 | 13 / 23 |
| 21-50 words | 10 / 12 | 79 / 239 |
| 51-150 words | 14 / 20 | 349 / 708 |
| 151-300 words | 24 / 29 | 681 / 911 |
| more than 300 words | 27 / 32 | 756 / 1,029 |
| all | 14 / 29 | 334 / 849 |

Scoring every piece and the joins costs one embedding per piece; short questions stay fast, long messages take up to about 1.2 s.

---

## 4. Response fields

`threat_gate`: `allowed`, `action`, `final_risk`, `base_risk`, `session_risk` (the user's suspicion before this query), `detector_scores` (`H`, `V`, `C`, `A`), `reasons` (including which piece of the message the classifier flagged, or "its most suspicious parts read together"), `execution_time_ms`, `suspicion`, `suspicion_after` (after Layers 2 and 3 reported back) and `policy` (the thresholds and settings this query was judged with).

## 5. Known limits

- **Rotating identity.** Without login, an attacker who claims a new user id every query gets no suspicion at all (94.7% of attacks find an unflagged variant within 20 queries). Login removes that, but with open sign-up an attacker can still open up to `SIGNUPS_PER_SUBNET_PER_HOUR` accounts per /24 network per hour (default 5); only `SIGNUP_MODE=invite` or `closed` removes it.
- **A persistent rewording attacker is slowed, not stopped:** 71.1% of attacks still find an unflagged variant within 20 queries.
- **Some families got worse** (section 3.1): social engineering, jailbreak prompts and direct override are caught slightly less often than by the previous gate, and 13 of 4,661 benign test texts are blocked (previously 1). On the hand-written queries, 2 of 32 benign hard negatives are blocked by rules.
- **ProtectAI is better on two families:** social engineering (100% vs 80%) and, when run piece by piece, hidden and split attacks (about 91% vs 80-83%), at a far higher benign cost.
- **Latency:** median 334 ms per message (14 ms before), up to about 1.2 s for messages over 300 words.
- The classifier depends on in-domain benign training questions; a deployment with different documents should regenerate them from its own topics (`scripts/build_domain_benign.py`). It generalises only partly to unseen attack datasets (deepset/prompt-injections: 46.7%).
- The flag floor makes a suspicious user's ordinary messages more likely to be flagged (on dev, 13.7% of benign texts at suspicion 0.4 against 2% at 0; blocks rise only at high suspicion: 3.3% at 0.8). Honest dev users never exceeded suspicion 0.42, but one honest test session reached 0.79.
- Suspicion is kept in process memory and is lost on restart (use a shared store for multi-worker deployments).
- Rate limiting is per /24 subnet and behind a proxy sees the proxy's address.
