# Several language models and independent evaluators

Feedback point 5: *"Evaluation is performed on a single generation model and a small natural-language-inference model. Test a variety of language models and have an independent evaluator validate the results."*

The point is answered in three ways:
- **Generators:** four generator families instead of one (§1), plus RAGTruth answers from six more LLMs (`docs/Public_Benchmarks.md`).
- **NLI:** two NLI models of different sizes (§2).
- **Evaluators:** independent LLM judges from model families other than the generator's and the verifier's, validated against human labels (§3).

All models are open and run locally or on a free Kaggle GPU. Runs: `notebooks/kaggle_gpu_runs.ipynb` (part 1) and `notebooks/kaggle_gpu_runs_2.ipynb` (part 2). Results are in `eval/results/kaggle_*`.

| Role | Models | Families |
|---|---|---|
| Generator | Llama 3 8B (main), Gemma 4 E4B, Mistral 7B, Phi-4 14B | Meta, Google, Mistral AI, Microsoft |
| NLI verifier | `nli-deberta-v3-small` (default), `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | Microsoft DeBERTa, two sizes |
| Independent judge | qwen2.5:14b (main); Gemma 3 27B (second judge) | Alibaba; Google |

---

## 1. Four generators through the same pipeline

**Setup.**
- **Questions:** each generator answers the 26 dev and 30 held-out questions (plus 8 and 6 unanswerable ones) over the same corpus.
- **Pipeline:** the same prompts, retrieval and verifier (v2, small NLI). Only the model that writes the draft changes.
- **Hardware:** Kaggle T4 GPU. Llama 3's drafts were regenerated on the GPU for this table, so all four generators are on the same machine.

| Generator | Split | Fact recall, no verification | Fact recall, Layer 3 | Answers with a fabricated number, no verification → Layer 3 | Unanswerable refused (Layer 3) | Draft sentences released |
|---|---|---|---|---|---|---|
| Llama 3 8B | dev | 96.2% | 86.5% | 3.8% → **0%** | 8/8 | 53.9% |
| | held-out | 100% | 83.6% | 3.3% → **0%** | 6/6 | 48.1% |
| Gemma 4 E4B | dev | 96.2% | 85.9% | 0% → 0% | 7/8 | 83.4% |
| | held-out | 100% | 96.7% | 0% → 0% | 5/6 | 85.3% |
| Mistral 7B | dev | 96.2% | 72.4% | 0% → 0% | 8/8 | 63.3% |
| | held-out | 98.3% | 74.2% | **13.3% → 3.3%** | 5/6 | 55.4% |
| Phi-4 14B | dev | 96.2% | 78.8% | 0% → 0% | 8/8 | 30.1% |
| | held-out | 100% | 80.6% | 0% → 0% | 6/6 | 35.0% |

**Reading the results.**
- **Layer 3 removes fabricated numbers whichever model writes the answer.**
  - Mistral 7B invents numbers in 4 of 30 held-out answers without verification, and in 1 with Layer 3.
  - Llama 3's single fabricated number is removed in both splits.
- **The recall cost depends on the writing style of the generator, more than on its size.**
  - Gemma 4 writes short sentences close to the source: 83–85% of its sentences are released, and held-out recall is 96.7%.
  - Phi-4 14B paraphrases and explains: only 30–35% of its sentences are released, and recall is about 80%.
  - Mistral is in between.
  - So the cost of verification is a property of the generator's style. A deployment should measure it for its own model.
- **Two unanswerable questions were not refused.** Gemma 4 had one in each split, and Mistral one held-out. In each case Layer 3 released a sentence that the evidence states but that does not answer the question. Verification checks support, not relevance.

**With verifier v3** (Kaggle part 3: the same drafts, re-checked; `eval/results/kaggle_1d092ec`):

| Generator | Fact recall dev, v2 → v3 | Fact recall held-out, v2 → v3 | Draft sentences released, held-out, v2 → v3 |
|---|---|---|---|
| Llama 3 8B | 86.5% → **88.5%** | 83.6% → **91.7%** | 48.1% → 55.8% |
| Gemma 4 E4B | 85.9% → **92.3%** | 96.7% → **99.2%** | 85.3% → 88.8% |
| Mistral 7B | 72.4% → **82.7%** | 74.2% → **79.2%** | 55.4% → 59.4% |
| Phi-4 14B | 78.8% → **79.8%** | 80.6% → **88.3%** | 35.0% → 40.2% |

- **Every generator gains.** Pooled over the 224 answers: **+5.4 points [+3.0, +8.3]**, with 17 answers better and none worse.
- **Safety is unchanged.** Fabricated numbers and refusals of unanswerable questions are the same under both versions.

---

## 2. A larger NLI model

**Setup.** The same Llama 3 drafts are verified with the small NLI model (the default, about 140M parameters) and with the large one (about 435M), with the same thresholds.

| | Small NLI | Large NLI |
|---|---|---|
| RAGTruth, hallucinated answers caught | 98.8% | 98.5% |
| RAGTruth, clean sentences kept | 43.1% | **57.9%** |
| RAGTruth, response-level F1 | 55.4 | **57.5** |
| Our dev questions, fact recall | 86.5% | 82.7% |
| Our held-out questions, fact recall | 83.6% | 86.1% |
| Our questions, draft sentences released (dev / held-out) | 53.9% / 48.1% | 62.5% / 57.2% |
| Our questions, fabricated numbers released | 0 | 0 |

**Reading the results.**
- On public data the large model keeps 15 points more clean sentences, at the same hallucination recall. It also releases more of our draft sentences.
- On our own small question sets, its fact recall is within noise of the small model: down 3.8 points on dev, up 2.5 on held-out, with overlapping intervals.
- The small model stays the default because it is three times smaller and fast on a CPU. The large model is the better choice where a GPU is available.
- **With verifier v3, the two help together.**
  - RAGTruth: 98.5% caught and 57.8% of clean sentences kept (large NLI), against 98.8% and 43.0% (small NLI).
  - Our questions, large NLI: fact recall 84.6% (dev) and **96.7%** (held-out), up from 82.7% and 86.1% with v2.
  - Large NLI + v3 is the most complete configuration measured. It needs a GPU to be fast.

---

## 3. Independent evaluators

### 3.1 Why a judge, and which

**Problem.** The earlier tables scored sentence support with a judge from the generator's own family (Llama 3), and that judge proved unreliable. Its agreement with the independent judge below was 48% (kappa −0.08), and it called every unsupported sentence it saw supported.

**The new judge.** The main independent judge is **qwen2.5:14b**. It is a different family from every generator and from the verifier, and it is larger than the generators. It labels each answer sentence SUPPORTED, NOT_SUPPORTED or CONTRADICTED against the same evidence passages (`scripts/independent_judge.py`). It is local and free.

### 3.2 Is the judge trustworthy? Agreement with human labels

**Setup.**
- **Sample:** 300 RAGTruth test sentences (QA and summarisation), stratified: 150 overlap a span that human annotators marked as hallucinated, and 150 do not.
- **Prompt:** the judge sees the source passages and the same prompt it uses on our answers (`scripts/judge_ragtruth.py`).
- **Measure:** Cohen's kappa with a bootstrap 95% interval.
- **Caveat:** because the sample is balanced, accuracy here is not the rate on natural answers.

| Rater | Agreement with humans | Cohen's kappa | Hallucinated sentences caught | Clean sentences flagged |
|---|---|---|---|---|
| **Judge, qwen2.5:14b** | **80.7%** | **0.61 [0.52, 0.70]** | 121/150 | 29/150 |
| Layer 3 verifier (v2, small NLI) | 68.3% | 0.37 [0.26, 0.47] | 117/150 | 62/150 |
| Judge, QA only | 79.0% | 0.57 [0.43, 0.69] | 76/88 | 21/69 |
| Judge, summaries only | 82.5% | 0.64 [0.50, 0.77] | 45/62 | 8/81 |
| Second judge, Gemma 3 27B | 76.3% | 0.53 [0.44, 0.61] | 83/150 | 4/150 |
| **Both judges, where they agree** (235 of 300) | **86.4%** | **0.72** | | |

**Reading the results.**
- A kappa of 0.61 is "substantial" agreement on the usual Landis–Koch scale, and it is clearly higher than the verifier's.
- The judge is therefore good enough to *estimate* how many correct sentences the verifier drops. It is not a replacement for humans: it misses 29 of 150 hallucinated sentences.
- Gemma 3 27B is more lenient. It flags almost no clean sentence (4/150) but misses 67 of 150 hallucinated ones. Its kappa of 0.53 is "moderate".
- The two judges agree with each other on 78.3% of these sentences (kappa 0.57). Where they agree, which is 235 of 300 sentences, they match the humans on 86.4% (kappa 0.72). Two judges in agreement are therefore more reliable than either alone.
- No labels were made by the authors. The human labels are RAGTruth's own.

### 3.3 The verifier, as seen by the judge, for every generator

The judge labelled every verified sentence of the four generators (dev + held-out, verifier v2):

| Generator | Sentences | Supported sentences the verifier kept | Unsupported sentences the verifier released | Released sentences that are supported |
|---|---|---|---|---|
| Llama 3 8B (CPU drafts) | 169 | 58.6% (75/128) | 3/41 | 96.2% |
| Gemma 4 E4B | 300 | 86.2% (256/297) | 2/3 | 99.2% |
| Mistral 7B | 199 | 64.8% (107/165) | 1/34 | 99.1% |
| Phi-4 14B | 258 | 41.6% (82/197) | 2/61 | 97.6% |

**Reading the results.**
- **Precision holds across all four generators:** 96–99% of what the verifier releases is supported, according to an evaluator that agrees with humans at kappa 0.61.
- **Recall varies a lot:** from 42% (Phi-4) to 86% (Gemma 4). This is the cost feedback point 4 asks about.

**With verifier v3**, judged by qwen2.5:14b on the same sentences:

| Generator | Supported sentences kept, v2 → v3 | Unsupported sentences released, v2 → v3 |
|---|---|---|
| Llama 3 8B (CPU drafts) | 58.6% → **67.2%** (75 → 86 of 128) | 3 → 3 of 41 |
| Gemma 4 E4B | 86.3% → **91.0%** (259 → 273 of 300) | 2 → 2 of 3 |
| Mistral 7B | 64.8% → **73.3%** (107 → 121 of 165) | 1 → 1 of 34 |
| Phi-4 14B | 41.9% → **46.0%** (83 → 91 of 198) | 2 → 3 of 61* |

\* The one newly released sentence is "A sales representative who achieves 120% of their annual quota earns a commission rate of 8%". This is correct (the 100%–149% tier pays 8%), and it was accepted by the table-row check. Both judges label it unsupported, although their own explanations state that 120% falls in the 8% tier. We count it as the judges label it.

No sentence that v2 released was dropped by v3. The v2 column is recomputed on the same labelled sentences, so it differs from §3.3 by at most 3 sentences.

### 3.4 Second judge: does the result depend on the choice of judge?

Gemma 3 27B labelled every verified sentence of the four generators, the same sentences as in §3.3 (Kaggle part 2, `eval/results/kaggle_9ca0116/results/judge2_<model>`).

| Generator | Released sentences that are supported: qwen2.5:14b / Gemma 3 27B | Supported sentences the verifier kept: qwen2.5:14b / Gemma 3 27B | Agreement between the judges (kappa) |
|---|---|---|---|
| Llama 3 8B | 96.2% / 98.7% | 58.6% / 48.7% | 81.1% (0.31) |
| Gemma 4 E4B | 99.2% / 100% | 86.2% / 86.3% | 98.7% (−0.01)* |
| Mistral 7B | 99.1% / 100% | 64.8% / 57.4% | 88.4% (0.44) |
| Phi-4 14B | 97.6% / 100% | 41.6% / 34.3% | 79.1% (0.20) |
| All 926 sentences | | | 87.8% (0.31 [0.22, 0.40]) |

\* Both judges call almost every Gemma 4 sentence supported (297 and 299 of 300), so kappa is not informative there.

**Reading the results.**
- **The main conclusion does not depend on the judge.** By both judges, 96–100% of the sentences the verifier releases are supported, for every generator.
- **The size of the recall cost does depend on the judge.** Gemma 3 calls more sentences supported, including explanations and inferences, so by its labels the verifier keeps fewer of them (Llama 3: 48.7% instead of 58.6%). The paper therefore reports recall cost with both judges, plus fact recall against gold facts, which needs no judge (`docs/Layer3_Evidence_To_Answer_Gate.md` §11).
- **The judges disagree mostly on whether an explanation counts as supported.** On our sentences their raw agreement is high (87.8%), but kappa is low (0.31) because most sentences are supported. Gemma 4 is from the same family as one generator, so for that generator only qwen2.5:14b counts as independent.

---

## 4. Reproducibility across hardware

On the 1,761 RAGTruth answers verified on both the author's CPU and a Kaggle GPU, all 12,901 sentence decisions were identical. Drafts generated on the GPU replay exactly on the CPU from the saved cache (0 LLM calls). Results from the two machines can therefore be compared. LLM drafts are still generated on one machine per experiment, because generation itself can differ slightly between CPU and GPU.

## 5. Limits

- The generators are 4–14B models. Larger generators (27B and up) were not run through the pipeline.
- The judges are LLMs. Their agreement with humans is measured on RAGTruth, not on our corpus. Our own sentences were not labelled by humans.
- The question sets are small (26 + 30 answerable), so differences of a few points between generators are within noise. The intervals are in the reports.
