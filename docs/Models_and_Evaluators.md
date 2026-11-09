# Several language models and independent evaluators

Reviewer point 5: *"Evaluation is performed on a single generation model and a small natural-language-inference model. Test a variety of language models and have an independent evaluator validate the results."*

The point is answered in three ways:
- **Generators:** four generator families instead of one (§1), plus RAGTruth answers from six more LLMs (`docs/Public_Benchmarks.md`).
- **NLI:** two NLI models of different sizes (§2).
- **Evaluators:** independent LLM judges from model families other than the generator's and the verifier's, validated against human labels (§3).

All models are open and run locally or on a free Kaggle GPU. Runs: `notebooks/kaggle_gpu_runs.ipynb` (part 1) and `notebooks/kaggle_gpu_runs_2.ipynb` (part 2). Results are in `eval/results/kaggle_*`.

| Role | Models | Families |
|---|---|---|
| Generator | Llama 3 8B (main), Gemma 4 E4B, Mistral 7B, Phi-4 14B | Meta, Google, Mistral AI, Microsoft |
| NLI verifier | `nli-deberta-v3-small` (default), `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | Microsoft DeBERTa, two sizes |
| Independent judge | qwen2.5:14b; Gemma 3 27B (second judge, *pending: part 2*) | Alibaba; Google |

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
- *Pending: Kaggle part 3* re-checks the same drafts with verifier v3.

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
- *Pending: Kaggle part 3* gives the same comparison with verifier v3.

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
| Second judge, Gemma 3 27B | *pending: part 2* | | | |

**Reading the results.**
- A kappa of 0.61 is "substantial" agreement on the usual Landis–Koch scale, and it is clearly higher than the verifier's.
- The judge is therefore good enough to *estimate* how many correct sentences the verifier drops. It is not a replacement for humans: it misses 29 of 150 hallucinated sentences.
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
- **Recall varies a lot:** from 42% (Phi-4) to 86% (Gemma 4). This is the cost reviewer point 4 asks about. With Llama 3, verifier v3 raises it from 58.6% to 67.2% with no new unsupported sentences (`docs/Layer3_Evidence_To_Answer_Gate.md` §11). *Pending: part 3* repeats this for the other three generators.

### 3.4 Second judge

*Pending: Kaggle part 2.*
- Gemma 3 27B, from a second family, labels the same 300 RAGTruth sentences (against the humans) and every sentence of the four generators (against qwen2.5:14b).
- Agreement between the two judges shows how much the evaluation depends on the choice of judge.
- Gemma 3 is from the same family as one generator (Gemma 4), so for that generator qwen2.5:14b remains the independent judge.

---

## 4. Reproducibility across hardware

On the 1,761 RAGTruth answers verified on both the author's CPU and a Kaggle GPU, all 12,901 sentence decisions were identical. Drafts generated on the GPU replay exactly on the CPU from the saved cache (0 LLM calls). Results from the two machines can therefore be compared. LLM drafts are still generated on one machine per experiment, because generation itself can differ slightly between CPU and GPU.

## 5. Limits

- The generators are 4–14B models. Larger generators (27B and up) were not run through the pipeline.
- The judges are LLMs. Their agreement with humans is measured on RAGTruth, not on our corpus. Our own sentences were not labelled by humans.
- The question sets are small (26 + 30 answerable), so differences of a few points between generators are within noise. The intervals are in the reports.
