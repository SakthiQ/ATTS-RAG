# GPU runs on Kaggle (free)

`kaggle_gpu_runs.ipynb` runs the slow experiments for feedback points 1, 4 and 5 on a free Kaggle GPU. On a T4 they take about 8–10 hours; on the author's CPU they would take several days.

| Job | Output folder in the zip |
|---|---|
| RAGTruth, verifier with the small NLI model | `results/ragtruth_nli_small` |
| RAGTruth, verifier with the large NLI model | `results/ragtruth_nli_large` |
| Llama 3, dev + held-out, small vs large NLI | `results/quality_{dev,heldout}_llama3_nli_{small,large}` |
| Gemma 4 E4B, Mistral 7B, Phi-4 14B, dev + held-out | `results/quality_{dev,heldout}_<model>` |
| Independent judge (qwen2.5:14b) on each generator | `results/indep_<model>` |
| Judge vs RAGTruth human labels | `results/judge_ragtruth` |

## One-time setup

1. Sign in at [kaggle.com](https://www.kaggle.com).
2. Verify a phone number: *Settings* → *Phone verification*. Kaggle needs this before it allows GPUs and internet.

## Running it

1. *Create* → *New Notebook* → *File* → *Import Notebook*. Upload `notebooks/kaggle_gpu_runs.ipynb`, or paste its GitHub link.
2. In the right-hand panel, under *Settings*:
   - **Accelerator:** *GPU T4 x2*.
   - **Internet:** *On*.
3. Optional check, about 30 minutes: set `SMOKE = True` in the first code cell and click *Run All*. If it ends with `DONE` and every job says `ok`, set it back to `False`.
4. *Save Version* → *Save & Run All (Commit)* → *Save*. The run continues with the browser closed.
5. When the version shows *Complete*, open it, go to the **Output** tab, and download `atts_gpu_results.zip`.

The notebook clones the branch `claude/layer1-detection` from GitHub when it starts, so it runs whatever is pushed at that moment.

## Part 2: `kaggle_gpu_runs_2.ipynb`

Same settings and steps as above. It runs in about 4–6 hours and produces `atts_gpu_results_2.zip`:

| Job | Output folder in the zip |
|---|---|
| BIPIA (150 cases, 5 defences), rebuilt from the public repo with the same seed, then scored | `results/bipia` |
| The 64 held-out poisoning cases with Llama 3, Gemma 4, Mistral 7B and Phi-4 14B (defences' judge fixed to Llama 3 with `--guard-model`) | `results/poison_v2_<model>` |
| Second independent judge (Gemma 3 27B) on 300 RAGTruth sentences | `results/judge2_ragtruth` |
| Second judge on every sentence of the four generators (inputs in `eval/judge_inputs/`) | `results/judge2_<model>` |

## Part 3: `kaggle_gpu_runs_3.ipynb` (verifier v3 and the deployed configuration)

Same settings and steps as above. It runs in about 3–4 hours and produces `atts_gpu_results_3.zip`.

Every job runs with verifier v3 and re-checks saved answers, so each result pairs with the earlier v2 result on identical drafts. The poisoning and BIPIA jobs also add the **deployed configuration**: deep scan + Layer 2 + Layer 3 + the "documents are untrusted data" instruction that the application's prompt already contains.

| Job | Feedback point | Output folder | Compare with |
|---|---|---|---|
| RAGTruth, small and large NLI | 4 (detection kept), 1 | `results/ragtruth_v3_nli_{small,large}` | part 1 `ragtruth_nli_*` |
| Four generators, dev + held-out (offline, part 1 drafts) | 4, 5 | `results/quality_{dev,heldout}_<model>_v3` | part 1 `quality_*` |
| Llama 3 with the large NLI model | 5 | `results/quality_{dev,heldout}_llama3_nli_large_v3` | part 1 `quality_*_llama3_nli_large` |
| Poisoning v1 + v2 with Llama 3, all defences plus the deployed configuration (CPU study drafts) | 3, 4 | `results/poison_large_{v1,v2}_v3` | `docs/Poisoning_Study.md` §2.1–2.4 |
| Poisoning v2 with Gemma 4, Mistral 7B, Phi-4 14B, plus the deployed configuration (part 2 drafts; defences' judge Llama 3) | 3, 5 | `results/poison_v2_<model>_v3` | part 2 `poison_v2_<model>` (`docs/Poisoning_Study.md` §2.5) |
| BIPIA, 5 defences plus the deployed configuration (part 2 drafts) | 1, 3 | `results/bipia_v3` | part 2 `bipia` |

The deployed configuration needs some new LLM calls, because Layer 2 reorders the evidence its drafts are based on. Everything else replays saved answers.

## Notes

- **Time limit.** Any job that would start after `TIME_LIMIT_H` (11 h) is skipped. The zip is rewritten after every job, so finished results are kept. RAGTruth runs that are stopped early can be resumed: their `items.jsonl` files are kept.
- **Pairing with the main CPU runs.** The Llama 3 jobs reuse the CPU runs' Llama 3 drafts from `eval/cache_snapshot/llm_cache.jsonl`. The "LLM calls" figure in each report shows how many drafts had to be regenerated (ideally 0).
- **What stays on one machine.** Every other generator is run entirely on the GPU. Small and large NLI are compared on the GPU against the same drafts.
- **Re-checking without a GPU.** `atts_gpu/llm_cache.jsonl` in the zip holds every generator's drafts. A later verifier change can be re-checked on CPU with `--offline --cache <that file>`, without regenerating anything.
- **Quota.** Kaggle gives roughly 30 GPU hours a week. One full run uses about a third of that.
