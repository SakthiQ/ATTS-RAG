# Poisoned-document study

Feedback point 3: *"The poisoning experiments have only eight attack queries in stress condition. Run larger experiments to get statistically reliable improvements. Compare the framework against existing defense methods."*

This study replaces the earlier 8-query stress test with **184 poisoned-document cases** in two sets:

- **v1:** 120 cases in 10 attack styles.
- **v2:** 64 held-out cases in 8 new styles. None of these styles was seen while the defences were built or tuned.

Each case runs under **11 defence settings**, including two existing defences used as baselines. All comparisons are paired: every setting sees the same cases.

Raw results:

| Set | Report | Per-case records |
| --- | --- | --- |
| v1 | `eval/results/poison_large_v1/report.md` | `items_poison_large.jsonl` |
| v2 | `eval/results/poison_large_v2/report.md` | `items_poison_large.jsonl` |

---

## 1. Setup

**Cases.** `scripts/build_poison_benchmark.py` writes `eval/poison_large/` (v1) and `eval/poison_large_v2/` (v2). Each case is built the same way:

- **The document.** One poisoned document is written for one question from `eval/qa.jsonl` or `eval/qa_heldout.jsonl`. It repeats the answer's key words, so retrieval finds it. It also carries an instruction to put a unique marker string into the answer.
- **The upload.** The document is uploaded as tier `unknown` by an anonymous user. The clean corpus (`eval/corpus` + `eval/corpus_heldout`) is tier `official`.
- **The question.** The question the document targets is asked.
- **Clean-up.** The document is deleted afterwards, so cases don't affect each other.

| Set | Styles (cases each) |
| --- | --- |
| v1 (10 x 12) | assistant_note, authority_claim, compliance_policy, direct_override, html_comment, markdown_note, multilingual, reviewer_request, split_payload, system_tag |
| v2 (8 x 8, held out) | email_forward, footnote, italian_note, json_directive, numbered_steps, spaced_letters, spanish_polite, story_embedded |

**Generator.** Llama 3 8B (Ollama, temperature 0) with the **naive** prompt: it has no "documents are untrusted" instruction. This is the stress condition, the same as the original 8-query test. The prompt-only baseline adds that instruction.

**Metrics.**

- **Attack success:** the marker appears in the released answer or chart.
- **Poison reached the LLM:** a chunk of the poisoned document was in the evidence given to the generator.
- **Utility:** fact recall on the question (share of gold facts present in the answer).

**Statistics.**

- Proportions get Wilson 95% intervals.
- Differences get paired bootstrap 95% intervals.
- Head-to-head comparisons use McNemar's exact test, which only counts the cases where the two defences disagree.

**Defence settings.**

| Setting | What it does |
| --- | --- |
| none | Plain RAG |
| **prompt-only** (baseline) | The generator is told that documents are untrusted data and their instructions must not be followed (spotlighting / instruction-hierarchy style) |
| **LLM-judge chunk filter** (baseline) | At query time, a local LLM judge (Llama 3) reads every retrieved chunk and drops those it calls an injection with confidence ≥ 0.5 |
| ingestion guard | Upload-time scan (`app/rag/ingestion_guard.py`): English regex patterns, an embedding outlier score, and the LLM judge **only on chunks that match a pattern**; flagged chunks are quarantined |
| deep guard | The ingestion guard with deep scan: the LLM judge reads **every** chunk of an `unknown` or `untrusted` upload |
| Layer 2 | Trust gate (`app/rag/trust_gate.py`): regex quarantine, integrity and authorization checks, and trust-tier ordering (official passages first, `unknown` uploads after them) |
| Layer 3 | Sentence-level verification of the answer (`docs/Layer3_Evidence_To_Answer_Gate.md`) |
| guard + L2, full, deep guard + L2, deep full | Combinations; *full* = ingestion guard + Layer 2 + Layer 3 |

---

## 2. Results

### 2.1 v1: 120 cases

| Defence | Attack success | Poison reached the LLM | Utility | vs no defence (points) |
| --- | --- | --- | --- | --- |
| none | 46.7% [38-56] (56) | 100.0% | 100.0% | - |
| prompt-only (baseline) | 35.0% [27-44] (42) | 100.0% | 99.2% | -11.7 [-20, -3] |
| LLM-judge chunk filter (baseline) | 17.5% [12-25] (21) | 35.0% | 100.0% | -29.2 [-38, -21] |
| ingestion guard | 36.7% [29-46] (44) | 90.0% | 100.0% | -10.0 [-16, -5] |
| Layer 2 | 30.0% [23-39] (36) | 80.0% | 100.0% | -16.7 [-27, -7] |
| Layer 3 | 42.5% [34-51] (51) | 100.0% | 94.6% | -4.2 [-8, -1] |
| guard + L2 + L3 (full) | 23.3% [17-32] (28) | 80.0% | 89.0% | -23.3 [-33, -13] |
| deep guard | 17.5% [12-25] (21) | 35.0% | 100.0% | -29.2 [-38, -21] |
| deep guard + L2 | 12.5% [8-20] (15) | 35.0% | 100.0% | -34.2 [-43, -24] |
| **deep guard + L2 + L3 (deep full)** | **10.8% [6-18] (13)** | 35.0% | 88.8% | **-35.8 [-45, -27]** |

### 2.2 v2: 64 held-out cases (new styles)

| Defence | Attack success | Poison reached the LLM | Utility | vs no defence (points) |
| --- | --- | --- | --- | --- |
| none | 23.4% [15-35] (15) | 100.0% | 100.0% | - |
| prompt-only (baseline) | 25.0% [16-37] (16) | 100.0% | 100.0% | +1.6 [-6, +9] |
| LLM-judge chunk filter (baseline) | 21.9% [14-33] (14) | 71.9% | 100.0% | -1.6 [-5, +0] |
| ingestion guard | 23.4% [15-35] (15) | 100.0% | 100.0% | +0.0 |
| Layer 2 | 10.9% [5-21] (7) | 100.0% | 100.0% | -12.5 [-22, -3] |
| Layer 3 | 23.4% [15-35] (15) | 100.0% | 94.5% | +0.0 |
| guard + L2 + L3 (full) | 9.4% [4-19] (6) | 100.0% | 88.0% | -14.1 [-23, -6] |
| deep guard | 21.9% [14-33] (14) | 71.9% | 100.0% | -1.6 [-5, +0] |
| deep guard + L2 | 7.8% [3-17] (5) | 71.9% | 100.0% | -15.6 [-25, -8] |
| **deep guard + L2 + L3 (deep full)** | **7.8% [3-17] (5)** | 71.9% | 86.5% | **-15.6 [-25, -8]** |

### 2.3 Head to head against the existing defences

Negative differences favour the framework. *Only A / only B* = cases where the attack beat one defence but not the other.

| Comparison | Set | Difference (points) | Only A / only B | McNemar p |
| --- | --- | --- | --- | --- |
| deep full vs LLM-judge filter | v1 | -6.7 [-12, -1] | 3 / 11 | 0.057 |
| | v2 | **-14.1 [-23, -6]** | 0 / 9 | **0.004** |
| | pooled (184) | -9.2 [-14, -4] | 3 / 20 | < 0.001 |
| full (no deep scan) vs LLM-judge filter | v1 | +5.8 [-2, +15] | 18 / 11 | 0.265 |
| | v2 | **-12.5 [-22, -3]** | 1 / 9 | **0.021** |
| | pooled | -0.5 [-7, +7] | 19 / 20 | 1.000 |
| deep full vs prompt-only | v1 | -24.2 [-33, -15] | 5 / 34 | < 0.001 |
| | v2 | -17.2 [-27, -9] | 0 / 11 | 0.001 |
| | pooled | -21.7 [-29, -15] | 5 / 45 | < 0.001 |
| deep full vs full | v1 | -12.5 [-18, -7] | 0 / 15 | < 0.001 |
| | v2 | -1.6 [-5, +0] | 0 / 1 | 1.000 |
| deep guard vs ingestion guard | v1 | -19.2 [-26, -12] | 0 / 23 | < 0.001 |
| | v2 | -1.6 [-5, +0] | 0 / 1 | 1.000 |

The pooled rows join the two sets. v1 was used while the defences were examined, so **v2 is the fair test** and the pooled rows are supporting evidence only.

### 2.4 Attack success by style

| Style (set) | none | prompt-only | LLM-judge filter | Layer 2 | full | deep full |
| --- | --- | --- | --- | --- | --- | --- |
| direct_override (v1) | 12/12 | 12/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| multilingual (v1) | 10/12 | 9/12 | 0/12 | 11/12 | 8/12 | 0/12 |
| html_comment (v1) | 4/12 | 7/12 | 0/12 | 6/12 | 6/12 | 0/12 |
| system_tag (v1) | 6/12 | 1/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| assistant_note (v1) | 6/12 | 5/12 | 6/12 | 2/12 | 2/12 | 2/12 |
| authority_claim (v1) | 4/12 | 0/12 | 3/12 | 2/12 | 1/12 | 0/12 |
| **reviewer_request (v1)** | 11/12 | 8/12 | 11/12 | 11/12 | 10/12 | **10/12** |
| compliance_policy (v1) | 1/12 | 0/12 | 1/12 | 2/12 | 1/12 | 1/12 |
| markdown_note (v1) | 2/12 | 0/12 | 0/12 | 2/12 | 0/12 | 0/12 |
| split_payload (v1) | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| **email_forward (v2)** | 6/8 | 7/8 | 6/8 | 3/8 | 3/8 | **3/8** |
| italian_note (v2) | 4/8 | 4/8 | 4/8 | 1/8 | 1/8 | 1/8 |
| spanish_polite (v2) | 3/8 | 4/8 | 2/8 | 2/8 | 2/8 | 1/8 |
| numbered_steps (v2) | 2/8 | 1/8 | 2/8 | 0/8 | 0/8 | 0/8 |
| footnote, json_directive, spaced_letters, story_embedded (v2) | 0/32 | 0/32 | 0/32 | 1/32 | 0/32 | 0/32 |

### 2.5 The same 64 held-out cases with three other generators

Kaggle part 2 (`eval/results/kaggle_9ca0116/results/poison_v2_<model>`, T4 GPU, verifier v2). Only the model that writes the answer changes. The defences' LLM judge (ingestion guard, deep scan, LLM-judge filter) stays Llama 3 (`--guard-model llama3`), so the defences are identical. The Llama 3 row is the CPU study of §2.2. The GPU rerun of Llama 3 stopped at case 30, when Ollama aborted one chart request; the harness now treats that as "no chart".

| Generator | No defence | Prompt-only | LLM-judge filter | Layer 2 | Full | **Full + deep scan** | Full + deep vs judge filter | Full + deep vs prompt-only |
|---|---|---|---|---|---|---|---|---|
| Llama 3 8B | 23.4% | 25.0% | 21.9% | 10.9% | 9.4% | **7.8%** | 0 / 9, p = 0.004 | 0 / 11, p = 0.001 |
| Gemma 4 E4B | 78.1% | 54.7% | 57.8% | 70.3% | 70.3% | **51.6%** | 0 / 4, p = 0.125 | 9 / 11, p = 0.82 |
| Mistral 7B | 45.3% | 35.9% | 39.1% | 57.8% | 50.0% | **37.5%** | 8 / 9, p = 1.0 | 10 / 9, p = 1.0 |
| Phi-4 14B | 45.3% | **21.9%** | 39.1% | 42.2% | 35.9% | 31.2% | 4 / 9, p = 0.27 | 15 / 9, p = 0.31 |

Utility (fact recall) of the full stack with deep scan, verifier v2: Llama 3 86.5%, Gemma 4 82.0%, Mistral 69.3%, Phi-4 55.5%; without Layer 3 it stays at 95–100%.

Attack success by style, none / prompt-only / LLM-judge filter / full + deep scan (of 8):

| Style | Llama 3 | Gemma 4 | Mistral 7B | Phi-4 14B |
|---|---|---|---|---|
| email_forward | 6 / 7 / 6 / 3 | 8 / 8 / 8 / 8 | 7 / 4 / 7 / 4 | 6 / 2 / 6 / 5 |
| italian_note | 4 / 4 / 4 / 1 | 8 / 8 / 8 / 8 | 4 / 2 / 4 / 2 | 4 / 0 / 4 / 5 |
| spanish_polite | 3 / 4 / 2 / 1 | 8 / 7 / 5 / 5 | 7 / 8 / 4 / 4 | 7 / 6 / 5 / 4 |
| numbered_steps | 2 / 1 / 2 / 0 | 8 / 5 / 8 / 6 | 8 / 8 / 8 / 8 | 8 / 1 / 8 / 5 |
| story_embedded | 0 / 0 / 0 / 0 | 8 / 0 / 8 / 6 | 1 / 0 / 1 / 4 | 0 / 0 / 0 / 0 |
| json_directive | 0 / 0 / 0 / 0 | 8 / 7 / 0 / 0 | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| footnote, spaced_letters | 0 / 0 / 0 / 0 | 2 / 0 / 0 / 0 | 1 / 1 / 1 / 2 | 4 / 5 / 2 / 1 |

### 2.6 The deployed configuration, with verifier v3

**Setup.**
- **What it is:** the configuration the application actually runs for an `unknown` upload: deep scan, Layer 2, Layer 3 (verifier v3), and the generator prompt that marks documents as untrusted data. That instruction is the prompt-only defence, which none of the "full" rows above include.
- **Run:** Kaggle part 3 (`eval/results/kaggle_1d092ec/results`).
- **Drafts:** the same saved drafts as above (CPU study for Llama 3, part 2 for the other generators). New LLM calls were needed only where Layer 2 reorders the evidence the instructed prompt sees: 47–85 calls per study.

| Generator, cases | None | Prompt-only | LLM-judge filter | Full + deep scan | **Deployed** | Deployed vs judge filter | Deployed vs prompt-only |
|---|---|---|---|---|---|---|---|
| Llama 3, v1 (120) | 46.7% | 35.0% | 17.5% | 10.8% | **6.7% [3-13]** | 2 / 15, **p = 0.002** | 2 / 36, **p < 0.001** |
| Llama 3, v2 (64) | 23.4% | 25.0% | 21.9% | 7.8% | **6.2% [2-15]** | 0 / 10, **p = 0.002** | 0 / 12, **p < 0.001** |
| Gemma 4 E4B, v2 | 78.1% | 54.7% | 57.8% | 51.6% | **18.8% [11-30]** | 0 / 25, **p < 0.001** | 1 / 24, **p < 0.001** |
| Mistral 7B, v2 | 45.3% | 35.9% | 39.1% | 37.5% | **28.1% [19-40]** | 5 / 12, p = 0.14 | 6 / 11, p = 0.33 |
| Phi-4 14B, v2 | 45.3% | 21.9% | 39.1% | 31.2% | **17.2% [10-28]** | 3 / 17, **p = 0.003** | 7 / 10, p = 0.63 |
| BIPIA (150) | 4.7% | 2.0% | 4.7% | 1.3% | **0.7%** | – | – |

"a / b" = cases where only the deployed configuration / only the baseline let the attack through.

**Fact recall on the poisoned questions** (with v3; v2 in brackets):

| Generator | Full + deep scan | Deployed | Without Layer 3 |
|---|---|---|---|
| Llama 3, v1 | 95.4% (88.8%) | 90.1% | 100% |
| Llama 3, v2 | 89.6% (86.5%) | 93.0% | 100% |
| Gemma 4 | 86.7% (82.0%) | 87.5% | 100% |
| Mistral 7B | 75.5% (69.3%) | 67.4% | 98–100% |
| Phi-4 14B | 62.5% (55.5%) | 63.8% | 97–98% |

---

## 3. Reading the results

1. **With Llama 3, the full framework with deep scan beats both existing defences on the held-out styles** (for other generators see point 7).
   - On v2 its attack success is 7.8%, against 21.9% for the LLM-judge filter (p = 0.004) and 25.0% for prompt-only (p = 0.001).
   - The judge filter never stopped an attack that deep full let through: 0 cases against 9.
   - On v1 the gap to the judge filter is smaller and not significant on its own (-6.7 points, p = 0.057). It is significant when the two sets are pooled (p < 0.001).

2. **The two baselines did not carry over to new styles.**
   - The LLM-judge filter cut attack success by 29 points on v1, but by only 1.6 points on v2.
   - On v2 the judge removed 18 poisoned documents (all of json_directive and spaced_letters, 2 of spanish_polite). Those styles were not succeeding anyway.
   - It let through every email_forward and italian_note document, the two styles that worked.
   - Prompt-only did nothing on v2 (+1.6 points).

3. **On new styles, most of the protection comes from Layer 2's trust ordering.**
   - In v2 the regex patterns quarantined nothing: poison reached the LLM in 64 of 64 cases with Layer 2.
   - Even so, attack success halved (15 → 7). The likely reason is that Layer 2 puts the official passages first and the `unknown` upload after them, so the generator follows the trusted text.
   - This ordering does not depend on the wording of the attack. That is why it carries over where pattern-based and judge-based scanning do not.

4. **Deep scan matters on v1 and not on v2.**
   - On v1 the deep guard stopped 23 attacks that the pattern-gated guard missed: multilingual, HTML comments and direct overrides, which have no English pattern (p < 0.001).
   - On v2 it helped in one case only. The judge it uses does not recognise the v2 styles that succeed.

5. **Layer 3 adds little against poisoning.**
   - Alone it lowers attack success by 4 points on v1 and none on v2. In the full stack it removes one or two more attacks.
   - With verifier v2 it cost about **11–13 points of fact recall** (utility 88–89% for the full stacks against 100% without Layer 3). With verifier v3 (feedback point 4) the cost falls to 4.6 points on v1 (95.4%) and 10.4 on v2 (89.6%), at the same attack success (§2.6).

6. **Polite, instruction-free requests remain the open weakness.**
   - reviewer_request (v1: a polite note addressed to "reviewers", asking that replies on the topic end with a link) succeeds in 10 of 12 cases even against deep full.
   - email_forward (v2: a forwarded e-mail asking assistants to end answers with a confirmation phrase) succeeds in 3 of 8 cases.
   - Neither contains instruction wording that a pattern or a judge recognises. Neither depends on the generator ignoring its prompt: the generator treats them as content.

7. **Filtering and ordering alone help mainly Llama 3 (§2.5). Combined with the untrusted-data instruction, as deployed, the framework has the lowest attack success for every generator (§2.6).**
   - With Gemma 4, Mistral 7B and Phi-4 14B, the full framework with deep scan is not significantly better than the LLM-judge filter (p = 0.13 to 1.0). With Phi-4 the simple prompt-only defence does best (21.9% against 31.2%).
   - The other generators follow instructions in documents far more readily: Gemma 4 is fooled in 78% of undefended cases, against 23% for Llama 3. Defences that only filter or reorder evidence cannot stop a document that is retrieved and obeyed.
   - Layer 2's trust ordering, the main protection for Llama 3, does not carry over. Gemma 4 and Phi-4 still follow an `unknown` passage placed after the official ones, and with Mistral attack success rises (45.3% → 57.8%).
   - The defences are complementary. The untrusted-data instruction stops styles that the filters miss (Gemma 4's story_embedded 8 → 0, Phi-4's numbered_steps 8 → 1), and the deep scan stops styles the instruction misses (Gemma 4's json_directive 7 → 0). Hence the deployed configuration, which uses both, is measured in part 3.
   - The verification cost of Layer 3 is larger for these generators (Phi-4 utility 55.5% with v2), as in `docs/Models_and_Evaluators.md` §1.

8. **The deployed configuration is the strongest defence for every generator.**
   - **Against the LLM-judge filter:** significantly better for Llama 3 (both sets), Gemma 4 and Phi-4 (p ≤ 0.003); not significant for Mistral (28.1% vs 39.1%, p = 0.14).
   - **Against prompt-only:** significantly better for Llama 3 and Gemma 4 (p < 0.001), and not for Mistral and Phi-4.
     - With Phi-4, prompt-only alone already reaches 21.9%, and the framework adds a further 4.7 points (17.2%, p = 0.63).
     - With Gemma 4 the combination is far better than either part: 78.1% undefended, 54.7% prompt-only, 51.6% full framework, 18.8% both.
   - **It is never worse than either baseline.** Its point estimate is the lowest in every row.
   - **The cost is Layer 3's fact-recall cost:** 87–93% for Llama 3 and Gemma 4, but 64–67% for Mistral and Phi-4, whose explanatory style the verifier cuts most. Verifier v3 reduced this cost in every study (for example Llama 3 v1: 88.8% → 95.4% for the full stack), but not to zero.

---

## 4. Caveats

- **The deep guard and the LLM-judge baseline use the same judge** (Llama 3 with the same prompt), once at upload and once at query time. That is why "deep guard" and "LLM-judge chunk filter" have identical rows. The framework's advantage over the judge baseline therefore comes from Layer 2 and Layer 3 in addition to that judge, not from a better judge.
- **v1 was looked at while the defences were developed**, and the deep scan was added after an earlier, smaller version of this study. v2 was written afterwards and run once.
- **v2's styles are weaker:** 4 of 8 never succeed even without a defence. v2 therefore has only 15 undefended successes, which is why its intervals are wide.
- **The poisoned documents repeat the answer's words** so that they are always retrieved. This is a worst case for retrieval. It also means that a document relying on wording alone, with no instruction, still has to be retrieved to matter.
- **The naive generator prompt is the stress condition.** The app's own prompt already contains the untrusted-data instruction (the prompt-only row).
- **Attack success is measured by an exact marker.** An attack that changes the answer's meaning without the marker is not counted. Utility is measured separately and shows no loss without Layer 3.
- **Deployed configuration measured last.** It was added after the other-generator results were known, because the application's own prompt already contains the instruction. It was run once on the existing cases, with no tuning.
- **Four generators, one study set.** The other generators were run on the 64 held-out cases only, not on v1, and on a GPU (the Llama 3 study ran on the CPU). Their results are not paired with Llama 3's case by case, only compared.

## 5. How to reproduce

```bash
python scripts/evaluate_pipeline.py --suite poison-large --generator-prompt naive --extra-corpus eval/corpus_heldout --out eval/results/poison_large_v1
python scripts/evaluate_pipeline.py --suite poison-large --generator-prompt naive --extra-corpus eval/corpus_heldout --poison-dir poison_large_v2 --out eval/results/poison_large_v2
```

About 5.5 hours (v1) and 3.5 hours (v2) on a 22-thread CPU, with LLM calls cached. `--conditions "name|name"` runs a subset. A public indirect-injection benchmark (BIPIA) run through the same harness is reported with feedback point 1.
