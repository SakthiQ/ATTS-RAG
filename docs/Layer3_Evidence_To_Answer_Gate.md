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
| **Ranges** | A number the evidence states only as a range counts as stated: 4 years is covered by "3 to 5 years", 7 by "6 or more". The supporting window must itself contain the range, open ranges are capped at five times their lower bound, and dates such as 2026-10-23 are not read as ranges. |
| **Table check** | A superlative claim about a table ("North America had the highest revenue") cannot be judged by an NLI model that sees one row at a time. `table_check.py` parses the Markdown table and confirms the claim only if the named row holds the extreme value of the column the claim is about (time units are compared in minutes), every number it quotes comes from that row, and it adds no words the passage lacks. |
| **Attribution clauses** | "as stated in the handbook" and "The evidence states that ..." talk about the document, not about a fact, and add words no evidence sentence contains. They are removed from the claim before checking. A sentence is never reduced to a fragment. |
| **Document names** | A sentence-initial document name ("The Acme Corp IT Security Policy requires ...") is replaced by "The policy" in the text the NLI model reads; the model otherwise scores the whole sentence neutral because of the extra name. |
| **Second opinion (optional)** | `second_opinion.py` asks a model of a different family than the generator about claims the NLI model left INSUFFICIENT and that share enough words with the best window. The model's answer is never trusted by itself: it must quote the evidence, and the claim is released only if the quote occurs verbatim in a passage, contains (or ranges over) every number of the claim, has the same number of negations, and the quote and its passage cover the claim's words. Text planted in the evidence cannot make a false claim pass because none of these checks is decided by the model. Enabled with `DirectClaimVerifier(second_opinion=QuoteGroundedJudge(invoker))`; off by default. |

Each addition has an off switch (`use_ranges`, `use_table_check`, `use_nli_text`, `AnswerGenerator.strip_meta`, `second_opinion=None`) so the evaluation can ablate it (`l3_no_ranges`, `l3_no_table`, `l3_no_nlitext`, `l3_no_meta`).

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

`Layer3Gate.process(..., policy=...)` accepts the user's `AdaptivePolicy` (see [`Adaptive_Suspicion.md`](Adaptive_Suspicion.md)). For a user with raised suspicion the entailment threshold rises gently from 0.60 to at most 0.70. The policy can also carry a minimum share of the draft that must verify (`min_grounding`); it is off by default because honest drafts often lose about half their sentences. With no policy (or suspicion 0) the gate behaves exactly as described above. The decision reports `safety_flagged` (a secret or prompt leak) and `contradictions` (claims the evidence contradicts); the engine passes these back as evidence of suspicious behavior. Low grounding is deliberately not reported as evidence, because it is common for honest questions.

## 7. Defense in depth and audit

- **Cross-tenant check:** evidence whose `tenant_id` differs from the query's is rejected, even though Layer 2 already filters.
- **Audit log:** every decision is appended to `logs/layer3_audit.jsonl` with ids, counts, per-claim status/confidence and timing. Raw prompt and answer text are never stored.
- **Timings:** generation dominates (tens of seconds on CPU with a local 8B model). Verification is about 1-10 s and the safety scan under 1 ms (`scripts/benchmark_layer3_latency.py`).

## 8. Offline improvement flywheel

Runtime thresholds are fixed (`ENTAILMENT >= 0.60`, `CONTRADICTION >= 0.60`) so behaviour is deterministic. They are tuned offline from the audit log; see [`Offline_Flywheel_Operations.md`](Offline_Flywheel_Operations.md).

## 9. Why a lower threshold does not fix the recall loss

`scripts/verify_diagnose.py` replays the cached drafts and scores every claim. Sentences that the verifier dropped although they are true score close to zero entailment (almost always below 0.05), so lowering the 0.60 cut-off to 0.30 changes nothing; raising it only loses sentences. The losses come from specific, fixable situations (ranges, table superlatives, attribution wording, document names), which is why they are handled by the checks in section 3 and not by a threshold. Sentences that only give reasons or opinions ("this is a common industry standard") are correctly dropped: the evidence does not state them.

## 10. Known limits

- A small NLI model is conservative: paraphrases of multi-sentence evidence can be dropped, so answers favour accuracy over completeness (`grounding_ratio` shows how much of the draft survived).
- The verifier checks support, not completeness; an answer may omit relevant evidence.
- Quality depends on the local LLM following the INTENT/ANSWER format. Missing labels trigger one follow-up call.
- **Recall is still below the unverified draft.** On held-out questions fact recall is 79.2% against 100% for the unverified draft (development questions: 88.5% against 96.2%). In an independent check of 100 sentences by an LLM evaluator (not a human), the verifier kept 68% of the sentences judged supported, and released 4 of 40 judged unsupported (10%).
- **False contradictions.** The NLI model sometimes labels a true, numerically derived sentence CONTRADICTION (6 of the 19 supported sentences that were dropped). Each such verdict adds 0.3 to the user's suspicion (see `Adaptive_Suspicion.md`).
- **Windows lie inside one passage.** A claim that needs two passages combined cannot be verified. Numbers are checked by exact match, ranges and table extremes, not by general arithmetic.
- **The version-2 fixes were designed on the development failures;** the held-out questions test the same failure types on new content, so they do not show that unseen failure types are handled.
- **The local LLM judge used in the earlier tables is unreliable** (agreement 48%, kappa -0.08 against the independent evaluator: it called all 24 unsupported sentences it saw supported). Its "unsupported sentence" rates understate the problem.
