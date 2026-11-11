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
        │                            and system-prompt leakage in the answer and the intent line -> REJECT
        ▼
4. Verify  (claim_verify.py)         every sentence, in batch (see §3)
        │
        ▼
5. Reconstruct  (reconstruction.py)  keep verified sentences in the original structure; cite each with
        │                            the passage that supported it; keep unverified headings and lead-ins
        │                            only if they cannot state a fact (§3.1); drop headings left with no content
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
| **Extractive match** | Near-verbatim claims and list items pass if ≥ 90% of their content words and all numbers occur in one window **and** the window has the same number of negations ("may" can never become "may not"). Since v3, the window must be a single sentence, unless one of its sentences holds every number of the claim. Otherwise "Employees can request an MFA exception" passed from "Employees must rotate passwords every 90 days. Contractors may request an MFA exception". |
| **Table rows (v3)** | `table_rows.py` checks claims that state values of Markdown table rows on the table itself. The NLI model reads one linearized row at a time and scored correct paraphrases ("The Business plan costs $9 per seat per month") as neutral, or as contradictions when the row is a range ("100% to 149% → 8%" for "a representative at 120% of quota earns 8%"). A claim passes only if every number is a value of the row it names (or lies inside a range cell of that row), or selects a row whose label is a range, or appears in a prose sentence of the passage that shares the claim's words. In addition, each value must sit in a column whose header the claim talks about, with the same unit and currency; values must be paired with rows by position or "respectively"; the claim must have no negation; and it may add at most one content word the passage lacks. The check can only accept claims, never reject them. |
| **NLI** | `cross-encoder/nli-deberta-v3-small` (override with `NLI_MODEL`). The best window decides: entailment ≥ 0.60 → `ENTAILMENT`. Contradiction ≥ 0.60 counts only from a window that shares ≥ 40% of the claim's content words, otherwise it is `INSUFFICIENT`. Label order is read from the model config. |
| **Citation repair** | The window that supports the claim decides the citation. A claim cited to the wrong passage is re-attributed to the right one. |
| **Ranges** | A number the evidence states only as a range counts as stated: 4 years is covered by "3 to 5 years", 7 by "6 or more". The supporting window must itself contain the range, open ranges are capped at five times their lower bound, and dates such as 2026-10-23 are not read as ranges. |
| **Table check** | A superlative claim about a table ("North America had the highest revenue") cannot be judged by an NLI model that sees one row at a time. `table_check.py` parses the Markdown table and confirms the claim only if the named row holds the extreme value of the column the claim is about (time units are compared in minutes), every number it quotes comes from that row, and it adds no words the passage lacks. |
| **Attribution clauses** | "as stated in the handbook" and "The evidence states that ..." talk about the document, not about a fact, and add words no evidence sentence contains. They are removed from the claim before checking. Since v3 this covers long document names ("as specified in the Paid Time Off section of the Acme Corp Employee Handbook"), "This is stated in X, which says that ..." and "This means that ...". A sentence is never reduced to a fragment. |
| **Document names** | A sentence-initial document name ("The Acme Corp IT Security Policy requires ...") is replaced by "The policy" in the text the NLI model reads; the model otherwise scores the whole sentence neutral because of the extra name. |
| **Second opinion (optional)** | `second_opinion.py` asks a model of a different family than the generator about claims the NLI model left INSUFFICIENT and that share enough words with the best window. The model's answer is never trusted by itself: it must quote the evidence, and the claim is released only if the quote occurs verbatim in a passage, contains (or ranges over) every number of the claim, has the same number of negations, and the quote and its passage cover the claim's words. Text planted in the evidence cannot make a false claim pass because none of these checks is decided by the model. Enabled with `DirectClaimVerifier(second_opinion=QuoteGroundedJudge(invoker))`; off by default. |

Each addition has an off switch (`use_ranges`, `use_table_check`, `use_table_rows`, `extractive_single_sentence`, `use_nli_text`, `AnswerGenerator.strip_meta`, `second_opinion=None`) so the evaluation can ablate it (`l3_no_ranges`, `l3_no_table`, `l3_no_nlitext`, `l3_no_meta`). Verifier v3 is the default; `ATTS_L3_V3=0` restores v2 to reproduce the reports made before it.

Statuses: `ENTAILMENT` (kept), `CONTRADICTION`, `INSUFFICIENT`, `UNKNOWN` (verifier error) - all but the first are dropped.

### 3.1 Text released without verification (strict release)

Some text in an answer is not a claim and is not sent to the verifier. Since strict release, such text is released only if it cannot state a fact of its own:

| Text | Released without verification only if |
|---|---|
| A note that the documents do not cover something | the documents are its subject ("The handbook does not specify ...", "There is no information about ..."), it says nothing more (no "but", "so", "and", comma or semicolon clause), and it has no number the question lacks. Otherwise it is verified like any claim. |
| Headings, lines under four words, lead-ins ending in ":" | they have no number and no negation ("not", "no", "only", ...), and every word is in the question, in a verified sentence of the answer, or in a short list of lead-in words ("key points", "following", "steps"). A short line or lead-in with a number or negation ("Limit: 500 dollars.") is verified like any claim; other failing text is left out. |
| The intent line ("Understood as: ...") | it is cut to its first sentence, has at most 40 words, and brings in no number and no more negations than the question. Otherwise it is not shown. It is scanned for secrets together with the answer. |

Before strict release, an uncited sentence with "not ... provide/include/available" anywhere in it, any heading, and any short line or lead-in were released unchecked. So "Managers are not required to provide a receipt for refunds up to 2,000 dollars." or "## Refunds take 90 days" could reach the user from a draft whose only verified sentence said 30 days. In the saved Llama 3 answers this happened twice: "The handbook does not provide further details ..., but it implies that ..." (hr-08) and "The evidence does not provide any information about refunds beyond the initial 30-day period, so it is not possible to receive a refund 45 days after ..." (ho-22). `ATTS_L3_STRICT_RELEASE=0` restores the earlier rule to reproduce the reports made before it. Tests: `tests/test_layer3_release.py`.

**Effect on the results.** The quality suite was re-run offline on the same cached Llama 3 drafts (`eval/results/quality_{dev,heldout}_strict`). For full Layer 3, only those two answers changed, and each lost only the unverified sentence. Fact recall (90.4% dev, 91.7% held-out), all facts present, fabricated numbers, contradicted facts, refusal of unanswerable questions and charts are identical. "Draft retained" falls slightly (dev 57.5% → 56.2%, held-out 56.9% → 56.7%) because short lines with a number and uncited negated sentences now count as claims.

---

## 4. Charts

A chart is requested only when the evidence contains at least 3 numbers and the question asks for comparison, trend, share, counts or an explicit chart. The LLM returns a JSON spec (bar, line or pie). The spec is released only if:

- it has 2-20 points, and
- **each point's label and value occur together in one evidence passage** (a number cannot be borrowed from elsewhere). Labels match whole words only, so "Q1" is not found in "Q10".

Otherwise the chart is silently dropped and the verified text answer is still released.

With strict release the text around the data is checked as well, because the app shows it:
- **Unit:** kept only if it is written next to every value in the evidence ("%" or "percent", "$" or "dollar"); otherwise it is left out.
- **Title and axis labels:** they may not carry a number that is not in the question or the rows the data came from, a negation, or a word that is not in the question, the labels or the passages the data came from. A failing title becomes "Figures from the evidence"; a failing axis label is left out. So a title such as "Refunds collapsed by 90% this year" cannot accompany data that does not show it.
- The chart's title, labels and unit are scanned for secrets.

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
- **Recall is still below the unverified draft, though much less so with v3.** With v3, held-out fact recall is 91.7% against 100% unverified (dev: 90.4% against 96.2%); see section 11. The independent judge (qwen2.5:14b) still finds about a third of supported sentences dropped. Most of these are explanations and inferences ("this ensures that ..."), which the evidence does not state word for word.
- **False contradictions.** The NLI model sometimes labels a true, numerically derived sentence CONTRADICTION. In the diagnosis all 7 such sentences came from tables; the table-row check of v3 recovers 4 of them. Derived statements outside tables remain. Each such verdict adds 0.3 to the user's suspicion (see `Adaptive_Suspicion.md`).
- **JSON and other non-Markdown tables are not covered by the table-row check.** On RAGTruth's data-to-text task (JSON records), only 22.6% of clean sentences are kept (v2).
- **Windows lie inside one passage.** A claim that needs two passages combined cannot be verified. Numbers are checked by exact match, ranges and table extremes, not by general arithmetic.
- **The version-2 fixes were designed on the development failures;** the held-out questions test the same failure types on new content, so they do not show that unseen failure types are handled.
- **The intent line is a paraphrase, not a verified statement.** Strict release keeps numbers, negations, secrets and extra sentences out of it, but a one-sentence restatement can still misread the question.
- **Poisoned evidence is "supported" evidence.** Layer 3 checks the answer against the passages Layer 2 released. A planted passage that passed Layers 1 and 2 supports the false claims it states; stopping it is the job of the ingestion scan and Layer 2 (see `Poisoning_Study.md`).
- **A secret quoted from a document counts against the user.** The safety scan rejects an answer that repeats a key-like string even when the document itself contains it, and the rejection adds to the user's suspicion.
- **The local LLM judge used in the earlier tables is unreliable** (agreement 48%, kappa -0.08 against the independent evaluator: it called all 24 unsupported sentences it saw supported). Its "unsupported sentence" rates understate the problem.

## 11. Verifier v3: fewer correct sentences rejected (feedback point 4)

> *"The default setting leads to a decrease in fact recall from 96.2% to 78.8% at the sentence level. Improve the verification mechanism to avoid the rejection of correct statements while keeping the hallucination detection effective."*

**Diagnosis.**
- **Method:** the independent judge (qwen2.5:14b, `scripts/independent_judge.py`; validated against human labels in `docs/Models_and_Evaluators.md`) labelled all 169 sentences of the Llama 3 answers to the dev and held-out questions. `scripts/analyse_dropped.py` then sorted the 53 sentences that the judge calls supported but v2 dropped (`eval/results/indep/dropped_supported.md`).
- **Fixable causes:**
  - Facts read from Markdown tables, which the NLI model sees one row at a time, especially when the row is a range. All 7 sentences rejected as contradictions were of this kind.
  - Long attribution clauses that the v2 patterns missed.
- **Not fixable without risk:** sentences that explain, infer or restate the question ("This means that ...", "this ensures that ..."). The evidence does not state them, and releasing them would weaken hallucination detection.

**Changes.** Three changes, all in section 3, each with its own switch:
- the table-row check;
- the single-sentence rule for the extractive shortcut, which also fixes a hallucination slip found during the analysis;
- longer attribution stripping.

**Results.** Llama 3, the same cached drafts, v2 vs v3 (`eval/results/quality_{dev,heldout}_v3`):

| | No verification | v2 | **v3** |
|---|---|---|---|
| Fact recall, dev (26 questions) | 96.2% | 88.5% [77-98] | **90.4% [79-100]** |
| Fact recall, held-out (30 questions, never used for tuning) | 100% | 82.5% [70-93] | **91.7% [83-98]** |
| Recall lost to verification, held-out | – | 17.5 points [7, 30] | **8.3 points [2, 17]** |
| All facts present, held-out | 100% | 73.3% | **86.7%** |
| Answers with a fabricated number | 3.8% / 3.3% | 0% | **0%** |
| Unanswerable questions refused | 100% | 100% | **100%** |
| Draft sentences released, held-out | 100% | 48.3% | 56.9% |

On the 169 judge-labelled sentences:
- **Supported sentences kept:** 58.6% (v2) → 67.2% (v3). 12 gained and 1 lost; the lost one was restored by the refined two-sentence rule.
- **Unsupported sentences released:** 3 of 41, unchanged, with no new ones. Two of the three are correct but phrased with "this plan" or "this tier", which the judge cannot resolve on its own.

**Tests.** `tests/test_table_rows.py`:
- 10 correct table claims are accepted.
- 21 wrong ones are refused: wrong row, wrong column, wrong unit, "$9" read as "9 seats", added content, negation, a 44.2% share called a majority.

`tests/test_extractive_window.py` covers the subject swap and numbers mixed across sentences.

**Across generators** (Kaggle part 3: the same part 1 drafts, re-checked with v2 and v3; `eval/results/kaggle_1d092ec`):

| Fact recall, dev / held-out | v2 | **v3** |
|---|---|---|
| Llama 3 8B | 86.5% / 83.6% | **88.5% / 91.7%** |
| Gemma 4 E4B | 85.9% / 96.7% | **92.3% / 99.2%** |
| Mistral 7B | 72.4% / 74.2% | **82.7% / 79.2%** |
| Phi-4 14B | 78.8% / 80.6% | **79.8% / 88.3%** |
| Llama 3 8B, large NLI | 82.7% / 86.1% | **84.6% / 96.7%** |

- **By the independent judge,** every generator keeps more supported sentences with v3: Llama 3 58.6 → 67.2%, Gemma 4 86.3 → 91.0%, Mistral 64.8 → 73.3%, Phi-4 41.9 → 46.0%. No sentence that v2 released is dropped. One newly released Phi-4 sentence is labelled unsupported by both judges, although it is correct (`docs/Models_and_Evaluators.md` §3.3).
- **The gain is consistent.** Pooled over all four generators and both question sets (224 answers), v3 raises fact recall by **+5.4 points [+3.0, +8.3]**: 17 answers better, none worse.
- **Nothing is lost on hallucination detection.**
  - Fabricated numbers are unchanged for every generator (0, except one Mistral held-out answer under both versions).
  - Refusal of unanswerable questions is unchanged.
- **On public data, detection holds exactly.** On RAGTruth, v3 still catches 98.8% of hallucinated answers (small NLI) and 98.5% (large NLI). It keeps about the same share of clean sentences (43.0% and 57.8%), because RAGTruth has no Markdown tables (`docs/Public_Benchmarks.md` §1.5).
- **In the poisoning study,** v3 lowers the fact-recall cost of the full stack (Llama 3 v1: 88.8% → 95.4%) at the same attack success (`docs/Poisoning_Study.md` §2.6).

**Compared with the first release.**
- The 78.8% in the feedback was measured with the first release. Against that, v3 cuts the recall cost on dev by two-thirds (17.4 → 5.8 points), and it halves the cost on held-out questions relative to v2.
