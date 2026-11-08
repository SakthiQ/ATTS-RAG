# Reviewer checklist (Q1 revision)

Status as of 2026-11-08, 21:45. `[x]` done, `[~]` running or queued, `[ ]` not started.
A point counts as complete only when its experiments are run **and** written up.

**Complete: 2 of 5** (points 2 and 3).

---

## 1. Larger public benchmarks — *in progress*
> The experiments use 40 attack queries, 32 benign queries, and five poisoned documents only. Evaluate the framework on larger public benchmarks to validate the reliability and generalizability of the results.

- [x] Layer 1 on public prompt-injection / jailbreak sets: 7,517 test texts, blind test run once, ProtectAI DeBERTa baseline (`docs/Layer1_Adaptive_Threat_Intelligence_Gate.md` §3)
- [x] Poisoning: 120 + 64 cases instead of 5 documents (point 3, `docs/Poisoning_Study.md`)
- [~] RAGTruth (2,675 answers, human hallucination labels) for the Layer 3 verifier, against the paper's published baselines (`scripts/eval_ragtruth.py`; running, ~23:00)
- [~] BIPIA (Microsoft indirect prompt injection) through the whole pipeline: 150 cases, 5 defences (`scripts/bipia_cases.py`; queued overnight, ~05:00)
- [ ] Write up RAGTruth and BIPIA (docs + manuscript datasets/results)

## 2. Paraphrased, multilingual and social-engineering attacks; adaptive attacks — **complete**
> The proposed query gate cannot detect most of the paraphrased, multilingual and social engineering attacks. Improve the detection mechanism and evaluate it on adaptive adversarial attacks.

- [x] New detection: piece-by-piece scoring, multilingual classifier, authority-claim check, joined pieces, suspicion-driven flag floor
- [x] Blind public test: attacks caught 61.4% → 89.6%, benign flagged 4.3% → 2.1%; translated 47.2% → 94.2%, hidden 41.4% → 80.4%, split 35.0% → 83.0%
- [x] Adaptive attacker (20 queries) and multi-turn sessions evaluated
- [x] Written up (`docs/Layer1_Adaptive_Threat_Intelligence_Gate.md`, `docs/Adaptive_Suspicion.md`, manuscript RQ1 draft)
- Known limits stated in the paper: a new user id per attempt defeats suspicion; social engineering slightly worse (86.7% → 80.0%); median latency 14 → 334 ms; the attacker test split was looked at twice

## 3. Larger poisoning study, compared with existing defences — **complete**
> The poisoning experiments have only eight attack queries in stress condition. Run larger experiments to get statistically reliable improvements. Compare the framework against existing defense methods.

- [x] v1: 120 cases (10 styles x 12 questions), 11 defence settings, paired McNemar tests. No defence 46.7% → full stack with deep scan 10.8% (p < 0.001 vs prompt-only, 35.0%); vs LLM-judge filter 17.5%, p = 0.057
- [x] Existing defences as baselines: prompt-only (spotlighting) and an LLM-judge chunk filter
- [x] v2: 64 held-out cases, 8 new styles, run once. Full framework with deep scan 7.8% vs LLM-judge filter 21.9% (p = 0.004) and prompt-only 25.0% (p = 0.001); both baselines barely work on the new styles
- [~] BIPIA adds a public indirect-injection benchmark (reported under point 1)
- [x] Written up (`docs/Poisoning_Study.md`, manuscript RQ2 draft), with the caveats: the deep scan uses the same judge as the LLM-judge baseline, the 11-13 point fact-recall cost of Layer 3, v1 examined during development
- Known limit stated: polite requests with no instruction wording still succeed (reviewer_request 10/12, email_forward 3/8)

## 4. Fact recall drop (96.2% → 78.8%) without losing hallucination detection — *in progress*
> The default setting leads to a decrease in fact recall from 96.2% to 78.8% at the sentence level. Improve the verification mechanism to avoid the rejection of correct statements while keeping the hallucination detection effective.

- [x] Verifier v2 fixes (ranges, table check, document names, attribution stripping); dev recall 88.5%, no fabricated numbers
- [~] Held-out rerun with the attribution fix (queued, ~23:30; previous held-out 79.2%)
- [~] Which correct sentences are still dropped, and why (`scripts/analyse_dropped.py`; after the judge, ~00:00)
- [ ] Fix the main causes found, rerun dev + held-out
- [ ] Fix a hallucination slip found 2026-11-08: the extractive shortcut accepts a sentence whose words all appear in a two-sentence window even when the subject is swapped ("Employees can request an MFA exception" from "...Contractors may request an MFA exception"); both NLI models flag it as a contradiction
- [ ] Show hallucination detection is kept: RAGTruth recall, fabricated-number rate, unsupported sentences released
- [ ] Write up (Layer 3 doc, manuscript RQ3/RQ4)

## 5. Several language models and an independent evaluator — *in progress*
> Evaluation is performed on a single generation model and a small natural-language-inference model. Test a variety of language models and have an independent evaluator validate the results.

- [x] Independent evaluator built: qwen2.5:14b, a different model family from the generator (Llama 3) and the verifier (DeBERTa NLI), local and free (`scripts/independent_judge.py`)
- [~] Judge labels every dev + held-out sentence (queued tonight, ~23:00)
- [~] Judge validated against **human** labels without new labelling: 300 RAGTruth sentences, Cohen's kappa, verifier on the same sentences (`scripts/judge_ragtruth.py`; queued after BIPIA). Optional: the author labels 100 of our own sentences (`label_sheet.csv`) as an extra in-domain check
- [~] RAGTruth answers come from 6 generators (GPT-4, GPT-3.5, Mistral-7B, Llama-2 7B/13B/70B), so the verifier is tested on several models' output
- [~] Second generator through the pipeline: Gemma 4 E4B on dev + held-out (`reasoning=False` added to the harness; queued after the judge check)
- [ ] Second, larger NLI model (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, downloaded to D:) compared with `nli-deberta-v3-small` on dev, held-out and RAGTruth QA (after the point 4 fix)
- [ ] Write up (manuscript models + RQ on evaluator agreement)
