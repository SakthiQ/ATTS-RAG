# Reviewer checklist (Q1 revision)

Status as of 2026-11-09, 16:00. `[x]` done, `[~]` running or queued, `[ ]` not started.
A point counts as complete only when its experiments are run **and** written up.

**Complete: 4 of 5** (points 1, 2, 3 and 5). Point 4 waits for Kaggle part 3 (RAGTruth and the other generators with verifier v3).

---

## 1. Larger public benchmarks — **complete**
> The experiments use 40 attack queries, 32 benign queries, and five poisoned documents only. Evaluate the framework on larger public benchmarks to validate the reliability and generalizability of the results.

- [x] Layer 1 on public prompt-injection / jailbreak sets: 7,517 test texts, blind test run once, ProtectAI DeBERTa baseline (`docs/Layer1_Adaptive_Threat_Intelligence_Gate.md` §3)
- [x] Poisoning: 120 + 64 cases instead of 5 documents (point 3, `docs/Poisoning_Study.md`)
- [x] RAGTruth (2,675 answers, human labels): verifier catches 98.8% of hallucinated answers, F1 55.4 (small NLI) / 57.5 (large), between GPT-3.5 prompting (52.9) and SelfCheckGPT (58.8); weakness: 43% of clean sentences kept
- [x] BIPIA through the whole pipeline: 150 cases, 5 defences. Floor effect (4.7% undefended, injected document retrieved in 45%); full framework 1.3%, not significantly below the LLM-judge filter (p = 0.062); table answers 22.7% → 6.7% with Layer 3 v2
- [x] Written up: `docs/Public_Benchmarks.md` and manuscript draft (v3 rows for RAGTruth and BIPIA follow from part 3)

## 2. Paraphrased, multilingual and social-engineering attacks; adaptive attacks — **complete**
> The proposed query gate cannot detect most of the paraphrased, multilingual and social engineering attacks. Improve the detection mechanism and evaluate it on adaptive adversarial attacks.

- [x] New detection: piece-by-piece scoring, multilingual classifier, authority-claim check, joined pieces, suspicion-driven flag floor
- [x] Blind public test: attacks caught 61.4% → 89.6%, benign flagged 4.3% → 2.1%; translated 47.2% → 94.2%, hidden 41.4% → 80.4%, split 35.0% → 83.0%
- [x] Adaptive attacker (20 queries) and multi-turn sessions evaluated
- [x] Written up (`docs/Layer1_Adaptive_Threat_Intelligence_Gate.md`, `docs/Adaptive_Suspicion.md`, manuscript RQ1 draft)
- Known limits stated in the paper: a new user id per attempt defeats suspicion; social engineering slightly worse (86.7% → 80.0%); median latency 14 → 334 ms; the attacker test split was looked at twice

## 3. Larger poisoning study, compared with existing defences — **complete** (advantage holds for Llama 3 only; deployed configuration in part 3)
> The poisoning experiments have only eight attack queries in stress condition. Run larger experiments to get statistically reliable improvements. Compare the framework against existing defense methods.

- [x] v1: 120 cases (10 styles x 12 questions), 11 defence settings, paired McNemar tests. No defence 46.7% → full stack with deep scan 10.8% (p < 0.001 vs prompt-only, 35.0%); vs LLM-judge filter 17.5%, p = 0.057
- [x] Existing defences as baselines: prompt-only (spotlighting) and an LLM-judge chunk filter
- [x] v2: 64 held-out cases, 8 new styles, run once. Full framework with deep scan 7.8% vs LLM-judge filter 21.9% (p = 0.004) and prompt-only 25.0% (p = 0.001); both baselines barely work on the new styles
- [x] BIPIA adds a public indirect-injection benchmark (reported under point 1)
- [x] Other generators (Gemma 4, Mistral 7B, Phi-4 14B) on the 64 held-out cases: the advantage over the LLM-judge filter does **not** carry over (p = 0.13 to 1.0); prompt-only is best for Phi-4. Written up in `docs/Poisoning_Study.md` §2.5 and point 7
- [~] The deployed configuration (deep scan + L2 + L3 + untrusted-data instruction) for all four generators and BIPIA, with verifier v3 (Kaggle part 3)
- [x] Written up (`docs/Poisoning_Study.md`, manuscript RQ2 draft), with the caveats: the deep scan uses the same judge as the LLM-judge baseline, the 11-13 point fact-recall cost of Layer 3, v1 examined during development
- Known limit stated: polite requests with no instruction wording still succeed (reviewer_request 10/12, email_forward 3/8)

## 4. Fact recall drop (96.2% → 78.8%) without losing hallucination detection — *in progress*
> The default setting leads to a decrease in fact recall from 96.2% to 78.8% at the sentence level. Improve the verification mechanism to avoid the rejection of correct statements while keeping the hallucination detection effective.

- [x] Verifier v2 fixes (ranges, table check, document names, attribution stripping); dev recall 88.5%, no fabricated numbers
- [x] Held-out rerun with the attribution fix: 82.5% (was 79.2%)
- [x] Which correct sentences are still dropped, and why (`eval/results/indep/dropped_supported.md`): table facts the NLI model cannot read, long attribution clauses, facts spread over two sentences
- [x] Verifier v3 (`app/rag/layer3/table_rows.py`, longer attribution stripping): fact recall dev 88.5% → 90.4%, held-out 82.5% → 91.7% (cost vs no verification 17.5 → 8.3 points), no fabricated numbers, all unanswerable questions refused; judge-supported sentences kept 58.6% → 67.2% with no new unsupported ones
- [x] Fixed (v3, `tests/test_extractive_window.py`) a hallucination slip found 2026-11-08: the extractive shortcut accepts a sentence whose words all appear in a two-sentence window even when the subject is swapped ("Employees can request an MFA exception" from "...Contractors may request an MFA exception"); both NLI models flag it as a contradiction
- [~] Show hallucination detection is kept on public data and other generators: Kaggle part 3 (RAGTruth v3, four generators, poisoning with v3)
- [~] Written up: `docs/Layer3_Evidence_To_Answer_Gate.md` §11 and manuscript RQ3/RQ4 draft (RAGTruth and other generators wait for part 3)

## 5. Several language models and an independent evaluator — **complete**
> Evaluation is performed on a single generation model and a small natural-language-inference model. Test a variety of language models and have an independent evaluator validate the results.

- [x] Independent evaluator built: qwen2.5:14b, a different model family from the generator (Llama 3) and the verifier (DeBERTa NLI), local and free (`scripts/independent_judge.py`)
- [x] Judge labels every sentence of four generators: 96–99% of released sentences supported for each
- [x] Judge validated against RAGTruth's human labels: agreement 80.7%, kappa 0.61 [0.52, 0.70] (verifier 0.37)
- [x] Second judge Gemma 3 27B: kappa 0.53 with humans; both judges agreeing: kappa 0.72; released sentences 96–100% supported by both judges for every generator
- [x] RAGTruth answers come from 6 generators: recall 95–100% for each
- [x] Three more generators through the pipeline (Gemma 4 E4B, Mistral 7B, Phi-4 14B): fabricated numbers removed for each; recall cost depends on writing style (held-out 74–97%)
- [x] Larger NLI model: RAGTruth clean sentences kept 43.1% → 57.9% at the same recall; within noise on our questions (v3 comparison in Kaggle part 3)
- [x] Written up: `docs/Models_and_Evaluators.md` and manuscript draft (v3 for the other generators follows from part 3)
