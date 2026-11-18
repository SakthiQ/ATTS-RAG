# Adaptive suspicion

**What it is:** one per-user *suspicion* value that all three layers share. Evidence from any layer raises it; time and clean queries lower it. It sets the thresholds and strictness each layer uses for the user's next query.
**Code:** [`app/rag/session_trust.py`](../app/rag/session_trust.py) · wired in [`threat_gate.py`](../app/rag/threat_gate.py), [`trust_gate.py`](../app/rag/trust_gate.py), [`layer3/gate.py`](../app/rag/layer3/gate.py), [`engine.py`](../app/rag/engine.py) · **Tests:** `tests/test_session_trust.py` and the adaptive tests in the layer suites · **Benchmark:** [`eval/adaptive_sessions.json`](../eval/adaptive_sessions.json)

## 1. Why

A static gate treats the fiftieth probing attempt like the first. Here, a user who keeps sending attack-looking queries, whose queries pull poisoned documents, or whose answers leak secrets or contradict the evidence meets stricter gates. A user with a clean history sees exactly the static behavior (suspicion 0).

## 2. State

Suspicion `s` is in [0, 1], keyed by an HMAC of the **logged-in account** (tenant + user from the signed login token; for unauthenticated calls, tenant + claimed user + IP /24). The session id is deliberately not part of the key, so opening a new session does not reset it. It decays with a 10-minute half-life, so a user who stops recovers quickly (each clean query also multiplies it by 0.95).

Evidence `e` in [0, 1] raises it with a saturating rule `s <- s + (1 - s) * 0.5 * e`:

| Source | Evidence | e |
|---|---|---|
| Layer 1 | query risk (highest of rules, similarity, classifier, authority claim) | `(risk - 0.2) / 0.6`, clipped; a block counts as 1; a classifier hit that no other detector backs counts half |
| Layer 1 | clean query (risk < 0.2) | none; `s` is multiplied by 0.95 (recovery) |
| Layer 2 | a retrieved chunk the user **uploaded themselves** was quarantined (strict mode; before, any quarantined chunk counted) | 0.6 |
| Layer 2 | a candidate failed tenant or clearance authorization (strict mode; before, integrity failures counted too) | 0.5 |
| Layer 3 | the answer leaked a secret or the system prompt | 0.8 |
| Layer 3 | per claim the evidence **contradicts** | 0.3 (capped at 1) |

A low share of verified claims is **not** evidence: it is common for honest questions and would lock honest users out.

**Classifier hits.** Layer 1's trained classifier (`app/rag/threat_classifier.py`, see [`Layer1_Adaptive_Threat_Intelligence_Gate.md`](Layer1_Adaptive_Threat_Intelligence_Gate.md) section 2.5) reports 0.30-0.50 in its flag band, 0.60 for a strong hit nothing else backs, and 0.85-1.0 for a block. A hit that no other detector backs counts half (`CLASSIFIER_ONLY_EVIDENCE = 0.5`), so one false flag does not make the gate strict with an honest user. (The previous, flag-only classifier capped its risk at `FLAG_CAP = 0.50`, calibrated on the dev sessions; that cap still applies to a classifier configured without `can_block`.)

**Suspicion also lowers the classifier's flag threshold.** Without this, a reworded attack scoring just below the classifier's flag threshold reports risk 0 and suspicion has nothing to act on: on the attacker test seeds, 94.7% of attacks found such a variant within 20 queries even against a persistent attacker. The classifier now moves its flag threshold linearly towards a dev-calibrated floor (the level that flags 20% of benign dev texts of the same length bucket), reaching it at `SUSPICION_FULL = 0.5`. Floor rate and `SUSPICION_FULL` were chosen on the dev attacker seeds and dev sessions (floors at 5%, 10% and 20%; `SUSPICION_FULL` 0.3, 0.5 and 1.0; 20% was the only floor that beat the previous gate on the dev attacker without changing the dev sessions).

**A known weakness of the Layer 3 evidence.** A claim that the NLI model labels CONTRADICTION adds 0.3 to the user's suspicion, but in an independent check of 100 sentences labelled by an LLM evaluator, 6 of the 19 true sentences the verifier dropped had been dropped as CONTRADICTION, i.e. the NLI model sometimes calls a true, numerically derived sentence a contradiction. Such false contradictions feed suspicion; the measured cost to honest users stays small (below) because 0.3 per claim decays quickly, but the topical-overlap rule of Layer 3 does not remove them.

## 3. Policy: what suspicion changes

| Layer | Setting | Formula | At s = 0 | At s = 0.5 | At s = 1 |
|---|---|---|---|---|---|
| 1 | block threshold | `max(0.45, 0.80 - 0.50 s)` | 0.80 | 0.55 | 0.45 |
| 1 | flag threshold | `max(0.15, 0.30 - 0.15 s)` | 0.30 | 0.225 | 0.15 |
| 1 | classifier flag threshold | moves from the calibrated threshold to the 20% floor, linearly up to `s = 0.5` | calibrated | floor | floor |
| 2 | tier-2 (unverified-source) evidence | set aside once `s >= 0.5`, but only if trusted evidence exists; otherwise it is used anyway (fallback) | used | set aside if tier 1 exists | set aside if tier 1 exists |
| 3 | entailment threshold | `0.60 + 0.10 s` (at most 0.70) | 0.60 | 0.65 | 0.70 |
| 3 | minimum verified share of the draft | optional, off by default (`MIN_GROUNDING = 0`) | none | none | none |

The tightening is deliberately mild. A first version withheld all unverified-source evidence above `s = 0.5`, raised the entailment threshold to 0.85 and required half of the draft to verify; because uploads default to the `unknown` tier and about half of an honest draft is normally dropped, that would have left a suspicious (or wrongly flagged) user with almost no answers. Now a suspicious user is still answered: trusted evidence is preferred and unverified evidence is kept only as a fallback, under a slightly stricter check. `TIER2_FALLBACK = False` restores the old withholding.

`ALPHA`, `CLEAN_RELAX` and `BLOCK_SLOPE` were calibrated by grid search on the **dev** split only (rule: fewest released attack turns subject to no extra benign blocks and benign suspicion at most 0.45); the **test** split was run once afterwards. All constants are documented in `session_trust.py`.

## 4. Where it flows

```
Layer 1 screen  --policy-->  retrieval  --policy-->  Layer 2 (withhold tier 2)  --policy-->  Layer 3 (stricter checks)
      ^                                                    |                                        |
      '------------- evidence (risk, quarantine, auth) ----'---- evidence (leak, contradiction) ----'
```

`RAGEngine(adapt_downstream=False)` stops the policy from reaching Layers 2 and 3; `RAGEngine(feedback=False)` stops Layers 2 and 3 from reporting back; `ThreatGate(use_session=False)` is the fully static gate (used as the baseline in the evaluation). The API's `threat_gate` field reports `suspicion`, `suspicion_after` and the `policy` used.

## 5. Evaluation (`python scripts/evaluate_pipeline.py --suite adaptive | adaptive-benign | adaptive-poison`)

**Current gate** (multilingual classifier that may block, authority-claim check, flag floor; commit `f409ace`):

| Sessions | Attack turns released, static → adaptive | Sessions with a block | Benign turns blocked, static / adaptive | Benign turns flagged | Highest benign suspicion |
|---|---|---|---|---|---|
| dev (8 attack, 6 benign) | 21.4% (9/42) → **7.1%** (3/42) | 8 of 8 (first block at attack turn 1.0) | 0 / 0 of 48 | 4 of 48 | 0.42 |
| test (8 attack, 6 benign) | 23.8% (10/42) → **11.9%** (5/42) | 8 of 8 (first block at attack turn 1.5) | 2 / 2 of 48 | 4 of 48 | 0.79 |

Paired reduction in attack turns released: 14.6 points [8, 21] (dev), 10.4 points [4, 15] (test). Adaptivity adds no benign block on either split; the 2 benign test turns blocked are blocked by the static gate too. The static baseline releases far fewer attack turns than in the tables below (23.8% instead of 100%) because the classifier can now block on its own. The test sessions had been inspected before, at earlier stages of the gate (rules only; flag-only classifier at `FLAG_CAP` 0.79 and 0.50), so this is not a first look at that split.

The tables below are from the first version (rules and similarity only, before any classifier).

Benchmark: 16 attack sessions, 12 benign sessions, 8 poisoned-document sessions and 2 identity-rotation sessions, split into dev and test.

| Test split (8 attack sessions, 6 benign sessions) | Static | Adaptive |
|---|---|---|
| Attack turns released (reach retrieval) | 100.0% (42/42) | 45.2% [31-60] (19/42) |
| Sessions in which an attack was blocked | 0 of 8 | 8 of 8 (first block at the 3rd attack turn on average) |
| Extra benign blocks caused by adaptivity | - | 0 in 48 turns |

* Paired reduction in the share of attack turns released: **54.2 points** [38, 69] (test).
* **Cost to honest users** (dev and test, 86 answerable turns, `--suite adaptive-benign`): 62 turns are judged with an unchanged policy and are identical to the static system. The other 24 follow a flagged or blocked query and are judged with a tightened policy; with the current mild policy they are also identical to static (fact recall 70.8%, 20 of 24 answers released), both when the corpus is `official` and when it is at the application's default `unknown` tier. The first, harsher version (withhold all unverified evidence, entailment up to 0.85, half of the draft must verify) cost those 24 turns 8.3 points of fact recall (70.8% to 62.5%) and 3 answers (20 to 17 of 24) with an `unknown`-tier corpus, which is why it was replaced.
* **Poisoned-document sessions** (naive generator, probes then a hijacked question): the planted marker never appeared in any configuration (0 of 8, static included), because Layer 2's pattern scan and Layer 3 already stop it, so attack success cannot discriminate. Exposure did change: a poisoned passage reached the LLM in 6 of 8 queries under static and 0 of 8 under the full adaptive policy, at the price of answering fewer of the attacker's questions (6 of 8 to 4 of 8).

## 6. Limits

* It reacts only to queries some detector scores. The classifier's flag floor widens that for a user who is already suspicious, but a first query that scores 0 gives it nothing to react to.
* An attacker who uses a **fresh user id every query** is not slowed when it can claim any user id for free (attacker test: 94.7% of attacks find an unflagged variant within 20 queries, as with no memory at all). User and tenant now come from a signed login token, so rotating identity means creating accounts: with the default open sign-up limit of 5 accounts per /24 network per hour, 76.3% of attacks find an unflagged variant (71.1% for a single persistent account). Invite-only or closed sign-up removes it.
* With the flag floor, a suspicious user's ordinary messages are flagged more often (dev benign texts: 2% at suspicion 0, 13.7% at 0.4, 20% from 0.5); blocks rise only at high suspicion (0.9% → 3.3% at 0.8). Honest dev users stayed at or below 0.42, but one honest test session reached 0.79.
* An honest user who is blocked twice by an over-eager rule (for example "Act as a travel assistant ...") can reach a high suspicion; Layers 2 and 3 are then slightly stricter for roughly ten to twenty minutes. The cost is small in our data but not zero.
* State lives in process memory; multi-worker deployments need a shared store.
* The benchmark is small and hand-built (sessions are the unit of analysis, so the confidence intervals are wide).
