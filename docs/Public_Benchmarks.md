# Public benchmarks

Reviewer point 1: *"The experiments use 40 attack queries, 32 benign queries, and five poisoned documents only. Evaluate the framework on larger public benchmarks to validate the reliability and generalizability of the results."*

| Part of the system | Benchmark | Size | Where |
|---|---|---|---|
| Layer 1 query gate | Public prompt-injection and jailbreak sets, plus generated hidden, split, translated and social-engineering sets; blind test split run once | 7,517 test texts | `docs/Layer1_Adaptive_Threat_Intelligence_Gate.md` §3 |
| Ingestion, Layer 2 and Layer 3 against poisoned documents | Generated poisoned documents, including a held-out set of new styles | 184 cases × 11 defences | `docs/Poisoning_Study.md` |
| Layer 3 verifier (hallucination detection) | **RAGTruth** (Niu et al., ACL 2024): human span-level hallucination labels on answers from six LLMs | 2,675 answers, 20,186 sentences | §1 below |
| Whole pipeline against indirect prompt injection | **BIPIA** (Yi et al., KDD 2025): Microsoft's benchmark of instructions hidden in external content | 150 cases × 5 defences | §2 below |

Everything runs locally with open models; no paid API is used. Nothing was tuned on RAGTruth or BIPIA.

---

## 1. RAGTruth: the Layer 3 verifier

**Setup** (`scripts/eval_ragtruth.py`):
- **Data:** the test split, answers of quality "good" (2,675).
  - 875 question answering, 900 data-to-text, 900 summarisation.
  - The answers come from GPT-3.5, GPT-4, Llama-2 7B/13B/70B and Mistral-7B.
- **Gold label:** an answer is hallucinated when annotators marked at least one span; a sentence is hallucinated when a marked span overlaps it. This is the paper's own response-level rule.
- **Verifier input:** each answer is split into sentences. Every sentence is checked against the source passages exactly as the pipeline checks its own answers: same thresholds, no second opinion.
- **Prediction:** an answer is predicted hallucinated when any sentence is not ENTAILMENT.
- **Baselines:** the published response-level F1 of RAGTruth's Table 5, on the same split and with the same label rule.
- **Hardware:** run on a Kaggle T4 GPU. On the 1,761 answers that were also run on CPU, all 12,901 sentence decisions were identical, so the hardware does not affect the results.

### 1.1 Response level, against the published detectors

F1 in %. Our rows carry a bootstrap 95% interval.

| Method | QA | Data-to-text | Summary | Overall |
|---|---|---|---|---|
| Prompt (GPT-3.5-turbo) | 30.8 | 77.4 | 37.1 | 52.9 |
| Prompt (GPT-4-turbo) | 45.6* | 78.3 | 47.6 | 63.4 |
| SelfCheckGPT (GPT-3.5-turbo) | 43.7 | 74.8 | 40.1 | 58.8 |
| LMvLM (GPT-4-turbo) | 30.1 | 72.1 | 36.2 | 49.4 |
| Fine-tuned Llama-2-13B (trained on RAGTruth) | 68.2 | 88.1 | 59.1 | 78.7 |
| **ATTS-RAG verifier v2, small NLI** (no training on RAGTruth) | 36.5 [32, 40] | 78.3 [76, 80] | 38.3 [35, 42] | **55.4 [53, 57]** |
| **ATTS-RAG verifier v2, large NLI** | 39.1 [35, 43] | 78.3 [76, 80] | 40.7 [37, 45] | **57.5 [55, 59]** |

\* In the paper the GPT-4 QA cell gives P 33.2 and R 90.6, which imply F1 48.6, not the printed 45.6; the printed value is kept.

The small NLI model is `cross-encoder/nli-deberta-v3-small` (the default). The large one is `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.

| Verifier v2 | Precision | Recall | F1 |
|---|---|---|---|
| Small NLI, overall | 38.5 | **98.8** | 55.4 |
| Large NLI, overall | 40.6 | **98.5** | 57.5 |

### 1.2 Reading the results

- **The verifier almost never misses a hallucinated answer.**
  - It flags 98.8% of them (932 of 943), with no training on RAGTruth.
  - Its F1 lies between the GPT-3.5 prompt detector (52.9) and SelfCheckGPT (58.8). It is below GPT-4 prompting (63.4) and well below a 13B model fine-tuned on RAGTruth (78.7).
  - It runs on a CPU, with a 140M-parameter NLI model and no API.
- **Its weakness is precision, the same weakness as reviewer point 4.**
  - It rejects many clean sentences: only 43.1% of clean sentences are kept with the small model, and 57.9% with the large one (§1.3).
  - The large NLI model keeps 14.8 points more clean sentences for 0.3 points less recall. RAGTruth is where the larger model matters most; on our own questions the difference is within noise (`docs/Models_and_Evaluators.md`).
- **Data-to-text is the hardest task.**
  - The sources are JSON business records with reviews, and only 22.6% of clean sentences are kept.
  - Most rejections are INSUFFICIENT: the NLI model cannot read a JSON record. The table-row check of verifier v3 handles Markdown tables, not JSON.

### 1.3 Sentence level (not comparable with the paper's character-level Table 6)

| Verifier v2 | Sentences | Hallucinated sentences caught | Clean sentences kept |
|---|---|---|---|
| Small NLI, QA | 6,912 | 80.2% | 58.2% |
| Small NLI, data-to-text | 8,200 | 95.9% | 22.6% |
| Small NLI, summary | 5,074 | 78.1% | 54.0% |
| **Small NLI, overall** | 20,186 | **88.7%** | **43.1%** |
| **Large NLI, overall** | 20,186 | **84.4%** | **57.9%** |

### 1.4 By the model that wrote the answer (small NLI)

| Generator | Answers | Hallucinated | Recall | F1 |
|---|---|---|---|---|
| GPT-3.5-turbo | 441 | 10.4% | 97.8 | 22.2 |
| GPT-4 | 450 | 9.3% | 95.2 | 20.8 |
| Llama-2-7B | 449 | 50.3% | 99.1 | 67.0 |
| Llama-2-13B | 450 | 46.0% | 100.0 | 63.2 |
| Llama-2-70B | 449 | 38.1% | 99.4 | 56.7 |
| Mistral-7B | 436 | 57.6% | 98.0 | 75.8 |

Recall stays at 95–100% for every generator. F1 is low for GPT-3.5 and GPT-4 only because few of their answers are hallucinated, so every false alarm weighs more.

### 1.5 Verifier v3 (reviewer point 4)

*Pending: Kaggle part 3 re-runs RAGTruth with verifier v3 (small and large NLI). This table will show whether v3 keeps more clean sentences while the 98% recall holds.*

---

## 2. BIPIA: indirect prompt injection through the whole pipeline

**Setup** (`scripts/bipia_cases.py`):
- **Cases:** for each of BIPIA's 75 text attacks, one email case and one table case.
  - The attack is inserted at the start, middle or end of a real BIPIA context.
  - That gives 150 cases, built with a fixed seed. They rebuild byte-for-byte from the public repository.
- **Ingestion:** each case is ingested as an `unknown`-tier upload, exactly like the poisoning study.
- **Defences:** none, prompt-only, LLM-judge chunk filter, full framework, and full framework with deep scan.
- **Scoring:**
  - BIPIA's own rule-based checks (emoji, language, reference match) where BIPIA has them.
  - For the attacks BIPIA scores with GPT prompts, BIPIA's own prompts are asked of a local judge, qwen2.5:14b.
- **Utility:** for table cases, whether the answer still contains the correct value.
- **Run:** Kaggle part 2 (`eval/results/kaggle_9ca0116/results/bipia`). Generator Llama 3 with the naive prompt (the stress setting of the poisoning study); verifier v2.

| Defence | Attack success | Injected document reached the LLM | Table answer correct |
|---|---|---|---|
| None | 4.7% [2-9] (7/150) | 45.3% | 22.7% (17/75) |
| Prompt-only | 2.0% [1-6] (3/150) | 45.3% | 21.3% |
| LLM-judge chunk filter | 4.7% [2-9] (7/150) | 44.7% | 22.7% |
| Full framework | 1.3% [0-5] (2/150) | 45.3% | 6.7% (5/75) |
| Full framework with deep scan | 1.3% [0-5] (2/150) | 44.7% | 6.7% (5/75) |

Head to head: full with deep scan vs the LLM-judge filter, 0 / 5 discordant cases (p = 0.062); vs prompt-only, 1 / 2 (p = 1.0).

**Reading the results.**
- **BIPIA's attacks rarely work on this pipeline, even without a defence (4.7%).**
  - The injected email or table reached the generator in only 45% of cases. BIPIA's questions are about one email or one table, and here that document competes with the official corpus for retrieval.
  - When it does reach the generator, most of BIPIA's attacks are tasks such as "translate this" or "summarise in emoji", which Llama 3 mostly ignores. Only emoji substitution, reverse text, entertainment, sentiment and task-automation attacks succeeded at all.
  - With this floor, the benchmark cannot separate the defences. The framework's 1.3% is the lowest rate, but the difference to the judge filter is not significant (p = 0.062).
- **The judge filter removed only one injected document.** BIPIA's injections are polite tasks, not instruction overrides, which matches the poisoning study's finding about instruction-free requests.
- **The utility cost is the point 4 problem.**
  - With verifier v2, correct table answers fall from 22.7% to 6.7%. BIPIA's tables are the kind of row facts the NLI model cannot read.
  - The low baseline (22.7%) comes from BIPIA's exact-answer criterion against free-text answers.
  - *Pending: Kaggle part 3* re-checks the same answers with verifier v3 (table-row check) and adds the deployed configuration.

---

## 3. How to reproduce

```bash
python scripts/eval_ragtruth.py --data <RAGTruth>/dataset --out eval/results/ragtruth            # add NLI_MODEL=<large model> for the large NLI
python scripts/bipia_cases.py build --bipia <BIPIA repo> --out <cases dir>
python scripts/evaluate_pipeline.py --suite poison-large --poison-dir <cases dir> --extra-corpus eval/corpus_heldout \
  --conditions "none (no defense)|prompt-only (documents are untrusted data)|LLM-judge chunk filter|guard + L2 + L3 (full)|deep guard + L2 + L3 (full, deep scan)" --out eval/results/bipia
python scripts/bipia_cases.py score --bipia <BIPIA repo> --cases <cases dir> --results eval/results/bipia
```

Or on a free Kaggle GPU: `notebooks/kaggle_gpu_runs.ipynb` (RAGTruth), `notebooks/kaggle_gpu_runs_2.ipynb` (BIPIA). Results of the GPU runs are in `eval/results/kaggle_*`.
