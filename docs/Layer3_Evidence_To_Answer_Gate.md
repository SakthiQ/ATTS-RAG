# Layer 3 - Evidence-to-Answer & Output Verification Gate

**System:** ATTS-RAG (Adaptive Threat and Trust Security for Retrieval-Augmented Generation)
**Boundary:** Output / answer release
**Decision model:** PASS / REJECT (a PASS may release only part of the draft)
**Code:** [`app/rag/layer3/`](../app/rag/layer3/) · **Tests:** [`tests/test_layer3_gate.py`](../tests/test_layer3_gate.py)

---

## 1. Purpose

Layer 2 decides which passages may be used. Layer 3 turns them into a **descriptive, question-aware answer** and decides which of its sentences may be released.

### Security invariant
> **No factual sentence is released unless it is supported by a Layer 2 evidence passage, checked by an independent verifier, and cited to the passage that actually supports it.** Numbers, charts and citations are checked deterministically as well as by the NLI model.

Unsupported sentences are not repaired or rephrased. They are dropped, and the drop is reported (`failed_claims_count`, `grounding_ratio`).

---

## 2. Pipeline

```
Layer 2 evidence + question
        │
        ▼
1. Draft  (generation.py)            ONE LLM call, plain text:
   INTENT:  what the user is really asking and what kind of answer helps
   ANSWER:  direct answer first, then reasons/context, each sentence ending in [E#] tags
        │   evidence is shown to the model as short handles E1..En (never chunk hashes)
        ▼
2. Parse  (generation.py)            sentences + cited handles; markdown labels, "according to [E1]"
        │                            phrasing, tables and chart announcements are normalised away
        ▼
3. Safety scan  (safety.py)          secrets (API keys, private keys, JWTs, high-entropy tokens)
        │                            and system-prompt leakage in the generated text -> REJECT
        ▼
4. Verify  (claim_verify.py)         every sentence, in batch (see §3)
        │
        ▼
5. Reconstruct  (reconstruction.py)  keep verified sentences in the original structure; cite each with
        │                            the passage that supported it; drop headings left with no content
        ▼
6. Chart (optional, generation.py)   only if the evidence has numbers and the question is quantitative;
                                     released only if every label/value is found in the evidence (§4)
```

If no sentence of the draft can be verified, the decision is `REJECT` and the user receives the closest sources instead of an answer. There is no retry: a retry costs a full generation (minutes on CPU) and cannot add facts the evidence lacks.

---

## 3. Claim verification

Each sentence is compared with **windows** of evidence (single sentences, pairs of consecutive sentences, or a whole short passage), not with whole passages. The NLI model truncates at 512 tokens and long premises dilute the entailment signal. PDF line wraps are re-joined first, so a sentence broken across two lines is judged as one.

| Check | Rule |
|---|---|
| **Numbers** | Every number in the claim must appear in the evidence, else `INSUFFICIENT` (no model call). This rejects invented totals such as a "135%" that sums unrelated figures. |
| **Extractive match** | Near-verbatim claims and list items pass if ≥ 90% of their content words and all numbers occur in one window **and** the window has the same number of negations ("may" can never become "may not"). |
| **NLI** | `cross-encoder/nli-deberta-v3-small` (override with `NLI_MODEL`). The best window decides: entailment ≥ 0.60 → `ENTAILMENT`. Contradiction ≥ 0.60 counts only from a window that shares ≥ 40% of the claim's content words, otherwise it is `INSUFFICIENT`. Label order is read from the model config. |
| **Citation repair** | The window that supports the claim decides the citation. A claim cited to the wrong passage is re-attributed to the right one. |

Statuses: `ENTAILMENT` (kept), `CONTRADICTION`, `INSUFFICIENT`, `UNKNOWN` (verifier error) - all but the first are dropped.

Uncited structural text (short lead-ins, "the document does not mention X") is kept without verification; it makes no claim about the evidence.

---

## 4. Charts

A chart is requested only when the evidence contains at least 3 numbers and the question asks for comparison, trend, share, counts or an explicit chart. The LLM returns a JSON spec (bar, line or pie). The spec is released only if:

- it has 2-20 points, and
- **each point's label and value occur together in one evidence passage** (a number cannot be borrowed from elsewhere).

Otherwise the chart is silently dropped and the verified text answer is still released.

---

## 5. Contracts (`contracts.py`)

`Layer3GateDecision`: `decision`, `reconstructed_answer`, `intent`, `chart`, `verified_claims`, `failed_claims`, `grounding_ratio`, `evidence_map` (`E1 -> chunk_uid`), `failure_reason`, `telemetry`.

## 6. Adaptive policy

`Layer3Gate.process(..., policy=...)` accepts the user's `AdaptivePolicy` (see [`Adaptive_Suspicion.md`](Adaptive_Suspicion.md)). For a user with raised suspicion the entailment threshold rises from 0.60 towards 0.85, and once suspicion reaches 0.7 at least half of the draft's checkable sentences must verify or the answer is rejected. With no policy (or suspicion 0) the gate behaves exactly as described above. The decision reports `safety_flagged` (a secret or prompt leak) and `contradictions` (claims the evidence contradicts); the engine passes these back as evidence of suspicious behavior. Low grounding is deliberately not reported as evidence, because it is common for honest questions.

## 7. Defense in depth and audit

- **Cross-tenant check:** evidence whose `tenant_id` differs from the query's is rejected, even though Layer 2 already filters.
- **Audit log:** every decision is appended to `logs/layer3_audit.jsonl` with ids, counts, per-claim status/confidence and timing. Raw prompt and answer text are never stored.
- **Timings:** generation dominates (tens of seconds on CPU with a local 8B model). Verification is about 1-10 s and the safety scan under 1 ms (`scripts/benchmark_layer3_latency.py`).

## 8. Offline improvement flywheel

Runtime thresholds are fixed (`ENTAILMENT >= 0.60`, `CONTRADICTION >= 0.60`) so behaviour is deterministic. They are tuned offline from the audit log; see [`Offline_Flywheel_Operations.md`](Offline_Flywheel_Operations.md).

## 9. Known limits

- A small NLI model is conservative: paraphrases of multi-sentence evidence can be dropped, so answers favour accuracy over completeness (`grounding_ratio` shows how much of the draft survived).
- The verifier checks support, not completeness; an answer may omit relevant evidence.
- Quality depends on the local LLM following the INTENT/ANSWER format. Missing labels trigger one follow-up call.
