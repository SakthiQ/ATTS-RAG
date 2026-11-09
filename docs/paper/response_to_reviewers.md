# Response to the reviewers

**Manuscript:** "Verify before you release: layered defenses and sentence-level verified generation for retrieval-augmented LLMs"

> **Draft, 2026-11-09.** Items marked **[PART 3]** are filled in from the last GPU run (verifier v3 on RAGTruth, the other generators and the deployed configuration). Items marked **[§ / p.]** get the final section and page numbers of the revised manuscript.

---

We thank the reviewers for their careful reading and for comments that substantially improved the paper. Every comment led to new experiments, and every new result is reported as measured, including the results that do not favour our method.

The main changes are:

1. **Public benchmarks.** The evaluation now includes:
   - 7,517 public and generated query-gate test texts;
   - RAGTruth (2,675 human-labelled answers);
   - BIPIA (150 indirect prompt-injection cases);
   - a 184-case poisoning study.

   The original evaluation had 40 attack queries, 32 benign queries and 5 poisoned documents.
2. **Query gate.** A redesigned query gate, evaluated on a blind test split and against an adaptive attacker.
3. **Verifier.** An improved verifier (v3). It cuts the fact-recall cost of verification by two-thirds while fabricated numbers stay at zero.
4. **Models.** Four generator families instead of one, and two NLI models.
5. **Independent evaluators.** Two independent LLM judges from other model families, both validated against human labels.

All models are open and run locally or on a free cloud GPU. No paid API is used. Code, data, prompts, cached model outputs and the exact commands are in the repository [URL, commit].

Throughout, we report:
- Wilson 95% intervals for proportions;
- paired bootstrap 95% intervals for differences;
- McNemar's exact test for paired comparisons of defences.

---

## Comment 1: larger public benchmarks

> *"The experiments use 40 attack queries, 32 benign queries, and five poisoned documents only. Evaluate the framework on larger public benchmarks to validate the reliability and generalizability of the results."*

**Response.** We added public benchmarks for each part of the system, and enlarged our own sets where no public benchmark fits. Splits were fixed before tuning, and test sets were run once. Nothing was tuned on RAGTruth or BIPIA.

| Part of the system | Before | Now |
|---|---|---|
| Query gate (Layer 1) | 40 attacks, 32 benign | **7,517 test texts** (2,856 attacks, 4,661 benign) from public prompt-injection and jailbreak sets, plus translated, hidden, split and social-engineering variants; blind test split |
| Poisoned documents (ingestion, Layer 2, Layer 3) | 5 documents, 8 attack queries | **184 cases × 11 defence settings** (plus the deployed configuration, **[PART 3]**): 120 cases in 10 styles, and 64 held-out cases in 8 new styles written after development |
| Output verifier (Layer 3) | 34 questions | **RAGTruth** test split: 2,675 answers from six LLMs (GPT-3.5, GPT-4, Llama-2 7B/13B/70B, Mistral-7B), 20,186 sentences with human hallucination labels |
| Whole pipeline, indirect injection | none | **BIPIA**: 150 cases (75 attacks × e-mail and table tasks, 3 positions) |

**Results.**

- **RAGTruth: the verifier catches 98.8% of hallucinated answers (932 / 943), with no training on RAGTruth.**
  - Response-level F1 is 55.4 [53, 57] with the small NLI model, and 57.5 [55, 59] with a larger one.
  - Published detectors on the same split: GPT-3.5 prompting 52.9, SelfCheckGPT 58.8, GPT-4 prompting 63.4, and a Llama-2-13B fine-tuned on RAGTruth 78.7.
  - So the verifier is competitive with prompted LLM detectors at a fraction of the cost: a 140M-parameter model on a CPU.
  - Its weakness is precision: only 43.1% of clean sentences are kept (57.9% with the larger NLI model). This is the same issue as Comment 4. With verifier v3: **[PART 3]**.
- **Recall holds for each of the six source LLMs:** 95–100%.
- **BIPIA has a floor effect.**
  - Even without a defence, only 4.7% of attacks succeed against our pipeline (7 / 150). The injected document reached the generator in 45% of cases, and most of BIPIA's task-style injections are ignored by Llama 3.
  - The full framework lowers attack success to 1.3% (2 / 150). This is not significantly lower than an LLM-judge filter (4.7%, p = 0.062).
  - We report BIPIA for completeness and state that it cannot separate the defences on this pipeline. With verifier v3 and the deployed configuration: **[PART 3]**.
- **Query gate and poisoning:** see Comments 2 and 3.

**Changes in the manuscript.**
- Section 3.3 (datasets, baselines): **[§ / p.]**
- Section 4: new tables for RAGTruth and BIPIA **[§ / p.]**
- Repository: `docs/Public_Benchmarks.md`

---

## Comment 2: paraphrased, multilingual and social-engineering attacks; adaptive attacks

> *"The proposed query gate cannot detect most of the paraphrased, multilingual and social engineering attacks. Improve the detection mechanism and evaluate it on adaptive adversarial attacks."*

**Response.** We redesigned the detector and evaluated it on a blind public test split, against a published guard model, and against an adaptive attacker.

**Changes to the detector:**
- **Piece-by-piece scoring:** sentences and windows are scored separately, so an attack hidden in a long paragraph or split in two is not diluted.
- **A multilingual classifier** (logistic regression over multilingual sentence embeddings), with thresholds that depend on message length.
- **Joined pieces:** the top pieces are also scored together, for attacks split across sentences.
- **An authority-claim check** for social engineering.
- **Per-user suspicion:** it lowers the thresholds for a user whose earlier messages looked suspicious.

**Results on the blind test split (2,856 attacks, 4,661 benign).**

| | Previous gate | New gate | Published guard model (ProtectAI DeBERTa-v3), whole message / piece by piece |
|---|---|---|---|
| Attacks caught (flagged or blocked) | 61.4% [60-63] | **89.6% [88-91]** | 69.9% / 89.1% |
| Benign texts flagged | 4.3% | **2.1%** | 8.4% / 15.6% |
| Translated attacks (7 languages) | 47.2% | **94.2%** (93–95% on 3 languages not in training) | 88.3% / 88.8% |
| Attack hidden in a long paragraph | 41.4% | **80.4%** | 31.4% / 91.0% |
| Attack split in two | 35.0% | **83.0%** | 22.7% / 90.7% |
| Translated benign questions flagged | 10.9% | **1.4%** | 34.0% / 37.7% |

The published guard model matches our detection rate only when it also scores pieces. It then flags seven times as many benign texts.

**Adaptive attacker.** A black-box attacker sees only pass, flag or block. It keeps paraphrasing one attack, with synonyms, wrappers, look-alike characters, leetspeak, spacing, translation and benign padding, for 20 queries (114 test attacks).
- Against a persistent attacker, the share of attacks with at least one unflagged attempt is:
  - after 5 queries: **17.5%** with the new gate plus suspicion, against 33.3% with the previous gate;
  - after 20 queries: 71.1% against 73.7%.
- Released attempts after 5 queries: 28.4% against 58.9%.
- In multi-turn sessions, suspicion halves the attack turns released (23.8% → 11.9%), with no additional benign blocks.

**Limitations we now state.**
- **Rotating identities.** An attacker who claims a new user id per attempt defeats the suspicion mechanism (94.7% find an unflagged variant within 20 queries). With login and the sign-up limit, the figure is 76.3%.
- **Families that got worse:**
  - social engineering detection fell slightly (86.7% → 80.0%), although blocking rose (33% → 53%);
  - jailbreak and direct-override detection fell by 3 points;
  - benign blocks rose from 1 to 13 of 4,661.
- **Latency.** Median latency per message rose from 14 to 334 ms on a CPU.
- **A second look at the attacker split.** The adaptive-attacker test split was examined twice. The flag floor was added after the first run, tuned on the development split.

**Changes in the manuscript.**
- Section 3.2 (Layer 1 design): **[§ / p.]**
- Section 4, RQ1: **[§ / p.]**
- Section 5 (limitations): **[§ / p.]**
- Repository: `docs/Layer1_Adaptive_Threat_Intelligence_Gate.md`, `docs/Adaptive_Suspicion.md`

---

## Comment 3: larger poisoning experiments and comparison with existing defences

> *"The poisoning experiments have only eight attack queries in stress condition. Run larger experiments to get statistically reliable improvements. Compare the framework against existing defense methods."*

**Response.** We replaced the 8-query stress test with **184 poisoned-document cases**:
- **v1:** 120 cases in 10 attack styles.
- **v2:** 64 held-out cases in 8 new styles, written after development and run once.

Each case is run under 11 defence settings, plus the deployed configuration **[PART 3]**, including two existing defences:
- **prompt-only** (spotlighting / instruction hierarchy: the generator is told that documents are untrusted data);
- **an LLM-judge chunk filter.**

All comparisons are paired. McNemar's exact test uses only the cases where two defences disagree. We also repeated the held-out study with three more generators and ran BIPIA (Comment 1).

**Results with Llama 3** (the main generator; stress setting with a generator prompt that has no defence of its own):

| Defence | v1 (120 cases) | v2, held out (64 cases) |
|---|---|---|
| None | 46.7% | 23.4% |
| Prompt-only (existing) | 35.0% | 25.0% |
| LLM-judge chunk filter (existing) | 17.5% | 21.9% |
| **Full framework with deep scan** | **10.8% [6-18]** | **7.8% [3-17]** |
| Full vs LLM-judge filter, paired | −6.7 points, p = 0.057 | **−14.1 points [−23, −6], p = 0.004** |
| Full vs prompt-only, paired | −24.2 points, p < 0.001 | −17.2 points, p = 0.001 |

On the held-out styles, both existing defences failed to carry over. The judge filter removed only styles that were not succeeding, and prompt-only did nothing. Most of the framework's protection came from Layer 2's trust ordering, which puts official evidence before unverified uploads.

**Results with other generators: the advantage does not generalise.** On the same 64 held-out cases, with the defences unchanged:

| Generator | None | Prompt-only | LLM-judge filter | Full + deep scan | Full vs judge filter |
|---|---|---|---|---|---|
| Llama 3 8B | 23.4% | 25.0% | 21.9% | **7.8%** | p = 0.004 |
| Gemma 4 E4B | 78.1% | 54.7% | 57.8% | 51.6% | p = 0.125 |
| Mistral 7B | 45.3% | 35.9% | 39.1% | 37.5% | p = 1.0 |
| Phi-4 14B | 45.3% | **21.9%** | 39.1% | 31.2% | p = 0.27 |

- **The other generators obey document instructions far more readily** (Gemma 4: 78% undefended).
- **Layer 2's ordering does not help them.**
- **The defences are complementary by attack style.** The untrusted-data instruction stops styles that the filters miss, and the deep scan stops styles that the instruction misses.
- **The deployed configuration.** The application combines both: deep scan, Layer 2, Layer 3 and a generator prompt that already marks documents as untrusted. For all four generators and BIPIA: **[PART 3]**.

We present the poisoning results as strong for Llama 3, and as not established for other generators unless the deployed configuration shows otherwise **[PART 3]**.

**Further limitations we state.**
- **Polite requests without instruction wording** remain the main weakness (a "note to reviewers" style succeeds in 10 of 12 cases against the full framework).
- **Shared judge.** The deep scan uses the same LLM judge as the judge-filter baseline, so the framework's advantage comes from the additional layers, not from a better judge.
- **Exact-marker metric.** Attack success is measured by an exact marker.

**Changes in the manuscript.**
- Section 3.3 and Section 4, RQ2: **[§ / p.]**
- Section 5 (generator dependence): **[§ / p.]**
- Repository: `docs/Poisoning_Study.md`

---

## Comment 4: fact recall drop from 96.2% to 78.8%

> *"The default setting leads to a decrease in fact recall from 96.2% to 78.8% at the sentence level. Improve the verification mechanism to avoid the rejection of correct statements while keeping the hallucination detection effective."*

**Response.**

**Diagnosis.** We first measured which correct sentences the verifier rejects and why:
- An independent judge (Comment 5) labelled every sentence of our answers.
- We analysed the supported sentences that the verifier dropped.
- The fixable causes were:
  - facts read from tables, which the NLI model reads one row at a time (including every sentence wrongly rejected as a contradiction);
  - long attribution clauses ("as specified in the Paid Time Off section of the ... Handbook").
- Explanations and inferences that the evidence does not state are still rejected, by design.
- During the analysis we also found and fixed a hallucination slip: the extractive shortcut had accepted a sentence whose subject was swapped.

**Verifier v3** adds three things:
- **A deterministic table-row check.** A claimed value must come from the named row and from a column the claim refers to, with the same unit and currency. The check can only accept claims, never reject them.
- **A single-sentence rule** for the extractive shortcut.
- **Broader attribution stripping.**

**Results** (Llama 3, identical cached drafts, so only verification differs):

| | No verification | First release | v2 | **v3** |
|---|---|---|---|---|
| Fact recall, development questions | 96.2% | 78.8% | 88.5% | **90.4% [79-100]** |
| Fact recall, held-out questions (never used for tuning) | 100% | — | 82.5% | **91.7% [83-98]** |
| All facts present, held-out | 100% | — | 73.3% | **86.7%** |
| Answers with a fabricated number | 3.8% / 3.3% | — | 0% | **0%** |
| Unanswerable questions refused | 100% | — | 100% | **100%** |

- **The recall cost on development questions falls from 17.4 to 5.8 points.** On held-out questions it falls from 17.5 points (v2) to 8.3.
- **On sentences labelled by the independent judge**, v3 keeps 67.2% of supported sentences (v2: 58.6%) and releases no additional unsupported sentence.
- **Tests:** 21 deliberately wrong table claims are all refused (wrong row, column, unit or currency, negation, added content).

**Hallucination detection is kept.**
- **On our questions:** zero fabricated numbers, all unanswerable questions refused.
- **On RAGTruth:** **[PART 3: hallucinated answers caught with v3, against 98.8% for v2; clean sentences kept, against 43.1%]**.
- **Across generators:** 96–100% of released sentences are supported for all four generators, by both independent judges (Comment 5). With v3: **[PART 3]**.

**Limitations we state.**
- The verifier still checks support, not relevance. In the runs with the three other generators, 3 of 42 unanswerable questions were answered with a true but irrelevant sentence.
- The table-row check covers Markdown tables, not JSON records. On RAGTruth's data-to-text task only 22.6% of clean sentences are kept.
- The recall cost depends on the generator's style (Comment 5).

**Changes in the manuscript.**
- Section 3.2 (Layer 3 algorithm, with the new checks): **[§ / p.]**
- Section 4, RQ3/RQ4: **[§ / p.]**
- Repository: `docs/Layer3_Evidence_To_Answer_Gate.md` §11

---

## Comment 5: several language models and an independent evaluator

> *"Evaluation is performed on a single generation model and a small natural-language-inference model. Test a variety of language models and have an independent evaluator validate the results."*

**Response.**

**Generators.** Four generator families now run through the same pipeline:
- Llama 3 8B (Meta), Gemma 4 E4B (Google), Mistral 7B (Mistral AI), Phi-4 14B (Microsoft).
- The verifier is also tested on RAGTruth answers from six further LLMs (Comment 1).

**NLI models.** Two NLI models are compared: `nli-deberta-v3-small` (default, about 140M parameters) and `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (about 435M).

**Independent evaluators.** Two independent judges, qwen2.5:14b and Gemma 3 27B.
- They come from families different from the main generator and from the verifier.
- Each labels every answer sentence as supported, not supported or contradicted, against the same evidence.
- **Before their labels are used, both are validated against human labels.** We used 300 RAGTruth sentences annotated by RAGTruth's annotators; the authors made no labels.

| Rater, against human labels | Agreement | Cohen's kappa |
|---|---|---|
| Judge 1, qwen2.5:14b | 80.7% | **0.61 [0.52, 0.70]** |
| Judge 2, Gemma 3 27B | 76.3% | 0.53 [0.44, 0.61] |
| Both judges, where they agree (235 of 300) | 86.4% | **0.72** |
| Our verifier (v2) | 68.3% | 0.37 [0.26, 0.47] |

**Results.**

- **Generators.** Layer 3 removes fabricated numbers for every generator: Mistral's held-out rate falls from 13.3% to 3.3%, and Llama 3's from 3.3–3.8% to 0%.
  - By both judges, **96–100% of released sentences are supported for every generator**.
  - The recall cost depends on the generator's writing style, more than on its size. Supported sentences kept, by judge 1, range from 42% (Phi-4, which paraphrases and explains) to 86% (Gemma 4, which stays close to the source).
  - With v3: **[PART 3]**.
- **NLI models.** On RAGTruth, the larger NLI model keeps 57.9% of clean sentences against 43.1%, at the same recall (98.5% vs 98.8%). On our small question sets the two are within noise.
  - We keep the small model as the CPU default and report the large one as the better choice when a GPU is available.
- **Choice of judge.** The main conclusion, that released sentences are almost always supported, is the same under both judges. The size of the recall cost depends on the judge, so we report it under both, together with fact recall against gold facts, which needs no judge.
- **Hardware.** On 1,761 RAGTruth answers verified on both a CPU and a GPU, all 12,901 sentence decisions were identical.

**Limitations we state.**
- The generators are 4–14B models.
- The judges are LLMs. Their agreement with humans is measured on RAGTruth, not on our corpus.
- Gemma 3 shares a family with one generator (Gemma 4); for that generator only judge 1 is independent.

**Changes in the manuscript.**
- Section 3.3 (models and hardware): **[§ / p.]**
- Section 4 (new RQ on generators and evaluators): **[§ / p.]**
- Repository: `docs/Models_and_Evaluators.md`

---

We hope the revised manuscript addresses the reviewers' concerns, and we thank them again for their constructive comments.
