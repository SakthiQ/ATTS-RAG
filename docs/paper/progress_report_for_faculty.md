# ATTS-RAG: progress on the five feedback points

**Draft of 2026-11-09.** This report covers each of the five points raised on the draft paper: what was done, the results, and the limits that remain.

**Status: 4 of 5 points complete (1, 2, 3, 5).**
- Point 4 has its main result. It is waiting for one last GPU run, which checks that hallucination detection holds on public data and other generators.
- Figures from that run are marked **[PART 3]**.

Everything was run with open models, locally or on a free Kaggle GPU; no paid API was used. Code, data, prompts, cached model outputs and the exact commands are in the repository (branch `claude/layer1-detection`).

Statistics used throughout:
- Wilson 95% intervals for proportions;
- paired bootstrap 95% intervals for differences;
- McNemar's exact test when two defences are compared on the same cases.

Splits were fixed before tuning, and test sets were run once. Nothing was tuned on the public benchmarks.

| # | Point | Status | Headline |
|---|---|---|---|
| 1 | Larger public benchmarks | Complete | RAGTruth: 98.8% of hallucinated answers caught; BIPIA run (floor effect) |
| 2 | Paraphrased, multilingual, social-engineering and adaptive attacks | Complete | Attacks caught 61.4% → 89.6% on a blind public test, benign flagged 4.3% → 2.1% |
| 3 | Larger poisoning study vs existing defences | Complete, qualified | Llama 3: 7.8% vs 21.9% for an LLM-judge filter (p = 0.004); does not carry over to other generators |
| 4 | Fact recall drop (96.2% → 78.8%) | Main result in | Held-out fact recall 82.5% → 91.7%, zero fabricated numbers; **[PART 3]** for public data |
| 5 | Several models, independent evaluator | Complete | 4 generators; 2 judges validated against human labels (kappa 0.61 / 0.53; 0.72 where they agree) |

---

## Point 1: larger public benchmarks

> *"The experiments use 40 attack queries, 32 benign queries, and five poisoned documents only. Evaluate the framework on larger public benchmarks to validate the reliability and generalizability of the results."*

**What was done.** Each part of the system now has a larger benchmark, public where one exists.

| Part of the system | Before | Now |
|---|---|---|
| Query gate (Layer 1) | 40 attacks, 32 benign | **7,517 test texts** (2,856 attacks, 4,661 benign) from public prompt-injection and jailbreak sets, plus translated, hidden, split and social-engineering variants; blind test split |
| Poisoned documents | 5 documents, 8 attack queries | **184 cases × 11 defence settings** (120 cases in 10 styles, plus 64 held-out cases in 8 new styles), and the deployed configuration **[PART 3]** |
| Output verifier (Layer 3) | 34 questions | **RAGTruth** (ACL 2024) test split: 2,675 answers from six LLMs, 20,186 sentences with human hallucination labels |
| Whole pipeline, indirect injection | none | **BIPIA** (Microsoft, KDD 2025): 150 cases (75 attacks × e-mail and table tasks, 3 positions) |

**Results.**

- **RAGTruth: 98.8% of hallucinated answers caught (932 / 943), with no training on RAGTruth.**
  - Response-level F1: 55.4 [53, 57] with the small NLI model, 57.5 [55, 59] with a larger one.
  - Published detectors on the same split: GPT-3.5 prompting 52.9, SelfCheckGPT 58.8, GPT-4 prompting 63.4, and Llama-2-13B fine-tuned on RAGTruth 78.7.
  - So the verifier is competitive with prompted LLM detectors, using a 140M-parameter model on a CPU.
  - Recall is 95–100% for each of the six source LLMs.
- **The weakness is precision:** only 43.1% of clean sentences are kept (57.9% with the larger NLI model). This is the same issue as point 4. With the new verifier: **[PART 3]**.
- **BIPIA has a floor effect.**
  - Without any defence, only 4.7% of attacks succeed (7 / 150). The injected document reached the generator in 45% of cases, and Llama 3 ignores most of BIPIA's task-style injections.
  - The full framework gives 1.3% (2 / 150), not significantly lower than an LLM-judge filter (4.7%, p = 0.062).
  - So BIPIA does not separate the defences on this pipeline. With the new verifier and the deployed configuration: **[PART 3]**.

**Details:** `docs/Public_Benchmarks.md`

---

## Point 2: paraphrased, multilingual and social-engineering attacks; adaptive attacks

> *"The proposed query gate cannot detect most of the paraphrased, multilingual and social engineering attacks. Improve the detection mechanism and evaluate it on adaptive adversarial attacks."*

**What was done.** The detector was redesigned:
- **Piece-by-piece scoring:** sentences and windows are scored separately, so hidden or split attacks are not diluted.
- **A multilingual classifier**, with thresholds that depend on message length.
- **Joined pieces:** the top pieces are also scored together.
- **An authority-claim check** for social engineering.
- **Per-user suspicion:** it lowers the thresholds after suspicious messages.

It was evaluated on a blind public test split, against a published guard model, and against an adaptive attacker.

**Results on the blind test split (2,856 attacks, 4,661 benign).**

| | Previous gate | New gate | Published guard model (ProtectAI DeBERTa-v3), whole message / piece by piece |
|---|---|---|---|
| Attacks caught (flagged or blocked) | 61.4% [60-63] | **89.6% [88-91]** | 69.9% / 89.1% |
| Benign texts flagged | 4.3% | **2.1%** | 8.4% / 15.6% |
| Translated attacks (7 languages) | 47.2% | **94.2%** (93–95% on 3 languages not in training) | 88.3% / 88.8% |
| Attack hidden in a long paragraph | 41.4% | **80.4%** | 31.4% / 91.0% |
| Attack split in two | 35.0% | **83.0%** | 22.7% / 90.7% |
| Translated benign questions flagged | 10.9% | **1.4%** | 34.0% / 37.7% |

The published guard model matches the detection rate only when it scores pieces too, and it then flags seven times as many benign texts.

**Adaptive attacker.** A black-box attacker sees only pass, flag or block. It keeps paraphrasing one attack (synonyms, wrappers, look-alike characters, leetspeak, spacing, translation, padding) for 20 queries, on 114 test attacks.
- Against a persistent attacker, attacks with an unflagged attempt:
  - after 5 queries: **17.5%** with the new gate plus suspicion, against 33.3% with the previous gate;
  - after 20 queries: 71.1% against 73.7%.
- Released attempts after 5 queries: 28.4% against 58.9%.
- In multi-turn sessions, suspicion halves the attack turns released (23.8% → 11.9%), with no extra benign blocks.

**Limits that remain.**
- **Rotating identities.** An attacker who claims a new user id per attempt defeats suspicion (94.7% within 20 queries; 76.3% with login and the sign-up limit).
- **Families that got slightly worse:**
  - social engineering detection 86.7% → 80.0%, although blocking rose from 33% to 53%;
  - jailbreak and direct override fell by about 3 points;
  - benign blocks rose from 1 to 13 of 4,661.
- **Latency.** Median latency per message 14 → 334 ms on a CPU.
- **A second look at the attacker split.** The adaptive-attacker test split was examined twice; the flag floor was tuned on the development split.

**Details:** `docs/Layer1_Adaptive_Threat_Intelligence_Gate.md`, `docs/Adaptive_Suspicion.md`

---

## Point 3: larger poisoning experiments and comparison with existing defences

> *"The poisoning experiments have only eight attack queries in stress condition. Run larger experiments to get statistically reliable improvements. Compare the framework against existing defense methods."*

**What was done.** The 8-query stress test was replaced by **184 poisoned-document cases**:
- **v1:** 120 cases in 10 attack styles.
- **v2:** 64 held-out cases in 8 new styles, written after development and run once.

Each case runs under 11 defence settings. Two of them are existing defences:
- **prompt-only** (spotlighting: the generator is told documents are untrusted data);
- **an LLM-judge chunk filter.**

The held-out study was repeated with three more generators, and BIPIA adds a public benchmark (point 1).

**Results with Llama 3** (the main generator, in the stress setting with no defence in its prompt):

| Defence | v1 (120 cases) | v2, held out (64 cases) |
|---|---|---|
| None | 46.7% | 23.4% |
| Prompt-only (existing) | 35.0% | 25.0% |
| LLM-judge chunk filter (existing) | 17.5% | 21.9% |
| **Full framework with deep scan** | **10.8% [6-18]** | **7.8% [3-17]** |
| Full vs LLM-judge filter, paired | −6.7 points, p = 0.057 | **−14.1 points [−23, −6], p = 0.004** |
| Full vs prompt-only, paired | −24.2 points, p < 0.001 | −17.2 points, p = 0.001 |

On the held-out styles, neither existing defence carried over:
- The judge filter removed only styles that were not succeeding anyway.
- Prompt-only did nothing.

Most of the framework's protection came from Layer 2's trust ordering, which puts official evidence before unverified uploads.

**Results with other generators: the advantage does not carry over.** Same 64 held-out cases, same defences; only the model writing the answer changes.

| Generator | None | Prompt-only | LLM-judge filter | Full + deep scan | Full vs judge filter |
|---|---|---|---|---|---|
| Llama 3 8B | 23.4% | 25.0% | 21.9% | **7.8%** | p = 0.004 |
| Gemma 4 E4B | 78.1% | 54.7% | 57.8% | 51.6% | p = 0.125 |
| Mistral 7B | 45.3% | 35.9% | 39.1% | 37.5% | p = 1.0 |
| Phi-4 14B | 45.3% | **21.9%** | 39.1% | 31.2% | p = 0.27 |

- **These generators follow document instructions far more readily** (Gemma 4: 78% undefended), and Layer 2's ordering does not help them.
- **The defences are complementary by attack style.** The untrusted-data instruction stops styles the filters miss, and the deep scan stops styles the instruction misses.
- **The configuration the application actually runs** combines both: deep scan, Layer 2, Layer 3 and a prompt that marks documents as untrusted. It is being measured for all four generators and BIPIA: **[PART 3]**.

**Limits that remain.**
- **Instruction-free requests.** Polite requests without instruction wording are the main weakness: a "note to reviewers" style succeeds in 10 of 12 cases against the full framework.
- **Shared judge.** The deep scan uses the same LLM judge as the judge-filter baseline, so the framework's edge comes from the extra layers, not from a better judge.
- **Exact-marker metric.** Attack success is measured with an exact marker.

**Details:** `docs/Poisoning_Study.md`

---

## Point 4: fact recall drop from 96.2% to 78.8%

> *"The default setting leads to a decrease in fact recall from 96.2% to 78.8% at the sentence level. Improve the verification mechanism to avoid the rejection of correct statements while keeping the hallucination detection effective."*

**What was done.**

**Diagnosis.**
- The independent judge (point 5) labelled every answer sentence.
- The supported sentences that the verifier dropped were analysed. The fixable causes:
  - facts read from tables, which the NLI model sees one row at a time (including every sentence wrongly rejected as a contradiction);
  - long attribution clauses ("as specified in the Paid Time Off section of the ... Handbook").
- Explanations and inferences that the evidence does not state are still rejected, on purpose.
- A hallucination slip was found and fixed: the extractive shortcut had accepted a sentence with a swapped subject.

**Verifier v3** adds three things:
- **A deterministic table-row check.** The value must come from the named row and from a column the claim refers to, with the same unit and currency. The check can only accept claims, never reject them.
- **A single-sentence rule** for the extractive shortcut.
- **Broader attribution stripping.**

**Results** (Llama 3, the same cached drafts, so only verification differs):

| | No verification | First release (draft paper) | v2 | **v3** |
|---|---|---|---|---|
| Fact recall, development questions | 96.2% | 78.8% | 88.5% | **90.4% [79-100]** |
| Fact recall, held-out questions (never used for tuning) | 100% | — | 82.5% | **91.7% [83-98]** |
| All facts present, held-out | 100% | — | 73.3% | **86.7%** |
| Answers with a fabricated number | 3.8% / 3.3% | — | 0% | **0%** |
| Unanswerable questions refused | 100% | — | 100% | **100%** |

- **The recall cost on development questions falls from 17.4 to 5.8 points.** On held-out questions it falls from 17.5 (v2) to 8.3.
- **On the judge-labelled sentences**, v3 keeps 67.2% of supported sentences (v2: 58.6%) and releases no additional unsupported sentence.
- **Tests:** all 21 deliberately wrong table claims are refused (wrong row, column, unit or currency, negation, added content).

**Is hallucination detection kept?**
- **On our questions:** yes. Zero fabricated numbers, and all unanswerable questions refused.
- **On RAGTruth:** **[PART 3: hallucinated answers caught with v3, against 98.8% for v2; clean sentences kept, against 43.1%]**.
- **Across generators:** 96–100% of released sentences are supported for all four generators, by both judges (point 5). With v3: **[PART 3]**.

**Limits that remain.**
- **Support, not relevance.** The verifier checks support, not relevance: with the three other generators, 3 of 42 unanswerable questions were answered with a true but irrelevant sentence.
- **JSON tables.** The table-row check covers Markdown tables, not JSON records. On RAGTruth's data-to-text task only 22.6% of clean sentences are kept.
- **Generator style.** The recall cost depends on the generator's writing style (point 5).

**Details:** `docs/Layer3_Evidence_To_Answer_Gate.md` §11

---

## Point 5: several language models and an independent evaluator

> *"Evaluation is performed on a single generation model and a small natural-language-inference model. Test a variety of language models and have an independent evaluator validate the results."*

**What was done.**
- **Generators:** four families now run through the same pipeline: Llama 3 8B (Meta), Gemma 4 E4B (Google), Mistral 7B (Mistral AI) and Phi-4 14B (Microsoft). The verifier is also tested on RAGTruth answers from six more LLMs (point 1).
- **NLI models:** `nli-deberta-v3-small` (default, about 140M parameters) and `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (about 435M).
- **Two independent judges:** qwen2.5:14b and Gemma 3 27B, both from families other than the main generator and the verifier.
  - Each judge labels every answer sentence as supported, not supported or contradicted.
  - **Before their labels were used, both were checked against human labels:** 300 RAGTruth sentences annotated by RAGTruth's annotators. No labels were made by us.

| Rater, against human labels | Agreement | Cohen's kappa |
|---|---|---|
| Judge 1, qwen2.5:14b | 80.7% | **0.61 [0.52, 0.70]** |
| Judge 2, Gemma 3 27B | 76.3% | 0.53 [0.44, 0.61] |
| Both judges, where they agree (235 of 300) | 86.4% | **0.72** |
| Our verifier (v2) | 68.3% | 0.37 [0.26, 0.47] |

**Results.**

- **Fabricated numbers.** Layer 3 removes them for every generator: Mistral's held-out rate falls from 13.3% to 3.3%, and Llama 3's from 3.3–3.8% to 0%.
- **Precision.** By both judges, **96–100% of released sentences are supported for every generator**.
- **Recall cost.** It depends on the generator's writing style more than its size. Supported sentences kept (judge 1) range from 42% for Phi-4, which paraphrases and explains, to 86% for Gemma 4, which stays close to the source. With v3: **[PART 3]**.
- **NLI models.** On RAGTruth the larger NLI model keeps 57.9% of clean sentences against 43.1%, at the same recall. On our small question sets the two are within noise. The small model stays the CPU default.
- **Choice of judge.** The conclusion that released sentences are almost always supported holds under both judges. The size of the recall cost depends on the judge, so it is reported under both, together with fact recall against gold facts, which needs no judge.
- **Hardware.** CPU and GPU give identical verifier decisions (12,901 of 12,901 sentences checked on both).

**Limits that remain.**
- The generators are 4–14B models.
- The judges' agreement with humans was measured on RAGTruth, not on our corpus.
- Gemma 3 shares a family with one generator (Gemma 4); for that generator only judge 1 is independent.

**Details:** `docs/Models_and_Evaluators.md`

---

## Next steps

1. **Part 3 GPU run** (in progress): fills in every **[PART 3]** figure, and completes point 4.
2. **Manuscript for first submission:** results sections, tables and figures written with these results built in; limitations section updated.

## Points to discuss

1. **Poisoning claim (point 3).** If the deployed configuration also does not beat the existing defences for the other generators, should the paper claim the poisoning result for Llama 3 only, or present poisoning defence as generator-dependent and recommend combining the defences?
2. **BIPIA.** It shows a floor effect on this pipeline. Should it go in the main results, or in an appendix as a robustness check?
3. **Default NLI model.** Should the small model stay the default (CPU-friendly), with the large one reported as the GPU option, or should the large one become the default?
4. **Rotating identities (point 2).** Is the honest statement of this limit enough, or should invite-only sign-up or per-device limits be implemented and evaluated before submission?
