# ATTS-RAG manuscript skeleton (standard Q1/Q2 structure)

Follows the common journal layout: Title page -> Abstract -> Keywords -> 1 Introduction -> 2 Literature review -> 3 Methodology -> 4 Results -> 5 Discussion -> 6 Conclusion and future work -> References.
**Always overrule this file with the target journal's author guidelines** (word limit, abstract length, reference style, required statements, figure format).

Working title (one claim, used everywhere):
**"Verify before you release: layered defenses and sentence-level verified generation for retrieval-augmented LLMs"**

Venue ideas (confirm current quartile on SCImago / JCR; it changes by year and category):
*Computers & Security*, *IEEE Trans. Information Forensics and Security*, *Journal of Information Security and Applications*, *Information Processing & Management*, *ACM Trans. Information Systems*, *Expert Systems with Applications*, *Knowledge-Based Systems*, *Future Generation Computer Systems*, *IEEE Access*.

---

## Title page and metadata
- Concise title stating the core finding (above).
- Authors, affiliations, corresponding author e-mail.
- Declarations: funding, conflict of interest, ethical approval (state that attack data are synthetic and no human subjects were involved), CRediT author statement, AI-use disclosure, data and code availability (repository URL + commit hash).

## Abstract (250-350 words; five parts, one to three sentences each)
1. **Background / problem:** RAG systems used on private documents are exposed to query injection, poisoned documents and unsupported answers.
2. **Objective:** design and evaluate a layered, verify-before-release architecture.
3. **Methods:** three layers (query gate, evidence trust gate, output verification gate); sentence-level verification combining NLI with deterministic number, negation and extractive guards; ablation study on <datasets, baselines, n>.
4. **Results:** headline numbers with intervals (attack success, fabricated-content rate, completeness cost, latency).
5. **Conclusion and contribution:** what the architecture achieves and the main limitation.

## Keywords (5-8, not repeating the title)
prompt injection; knowledge poisoning; hallucination detection; natural language inference; zero-trust architecture; enterprise question answering

---

## 1. Introduction
- Background: RAG in enterprise and regulated settings; why trust matters.
- Research gap: existing defenses address one stage; none releases an answer sentence by sentence only if verified.
- Research questions (answered in Section 4):
  - RQ1 How well does the query gate block attacks without blocking legitimate queries?
  - RQ2 Do the ingestion guard, trust gate and output gate reduce poisoning success, and at what utility cost?
  - RQ3 Does sentence-level verification reduce unsupported content, and what share of the draft does it cost?
  - RQ4 Which verification components matter (NLI only / minus number guard / minus extractive check / full)?
  - RQ5 Are generated charts faithful to the sources?
  - RQ6 What is the latency overhead?
- **Novel contributions** (explicit, numbered):
  1. A three-layer architecture with an auditable decision at each boundary.
  2. A sentence-level verification method (NLI over evidence windows + number/negation/extractive guards + citation repair).
  3. Evidence-checked charts: a chart is released only if each value is traceable to the source text.
  4. An open evaluation and ablation harness and dataset.

## 2. Literature review / theoretical framework
Synthesize recent (last 3-5 years) and seminal peer-reviewed work, one paragraph per theme, each ending with the gap this paper fills:
- RAG and its robustness (RAG, Self-RAG, CRAG)
- Prompt-injection and jailbreak defenses (input filters, guard models, instruction hierarchy)
- Corpus / knowledge poisoning and indirect injection (attacks and defenses)
- Faithfulness and hallucination checking (NLI/entailment-based, citation verification, RAGAS-style metrics)
- Guardrail frameworks used as baselines (e.g. NeMo Guardrails, Llama Guard)
Theoretical grounding: zero-trust principles and defense-in-depth; entailment as the formal criterion for "supported by evidence".

## 3. Methodology
**3.1 Threat model.** Attacker capabilities (anonymous user; uploader of a poisoned document), goals (override instructions, exfiltrate, plant false facts, obtain unsupported claims), out of scope (compromised admin, model weights, side channels), trust assumptions (`official` uploads trusted; user, tenant and role come from a signed login token; with open sign-up an attacker can still create up to 5 accounts per /24 network per hour - say so).

**3.2 System design** (architecture figure: `docs/reports/atts_rag_architecture.png`, update to the current design).
- Ingestion poisoning scan: rules + LLM judge + embedding outlier (`app/rag/ingestion_guard.py`)
- Layer 1 query gate: admission, normalization, sentence / window segmentation, four detectors (rules, embedding similarity, multilingual logistic-regression classifier with length-bucketed thresholds and corroborated blocking, authority-claim check), per-user suspicion that lowers the gate's and the classifier's thresholds (`docs/Layer1_Adaptive_Threat_Intelligence_Gate.md`)
- Retrieval: hybrid dense + BM25 (RRF), cross-encoder rerank, relevance floor
- Layer 2 trust gate: authorization, integrity hash, provenance, trust tiers, PII redaction
- Layer 3 output gate: intent-aware draft, parsing, verification, reconstruction, charts (`docs/Layer3_Evidence_To_Answer_Gate.md`)
- Algorithm box for Layer 3 and a table of thresholds with the justification for each.

**3.3 Experimental setup** (detailed enough to reproduce).
- Datasets. Internal Acme corpus (`eval/`: 34 questions, 40 attacks, 32 benign queries, 5 poisoned documents) for ablations. Poisoning: 184 generated poisoned-document cases, 120 (v1, 10 styles) + 64 held out (v2, 8 new styles) (`eval/poison_large*`, `docs/Poisoning_Study.md`). Layer 1: public prompt-injection / jailbreak sets plus generated hidden, split, translated and social-engineering sets, 7,517 test texts (`eval/external_manifest.json`, splits fixed before tuning, test run once). Layer 3: RAGTruth test split (2,675 answers from six LLMs, human span labels). Whole pipeline: BIPIA (150 indirect-injection cases, rebuilt from the public repository with a fixed seed). See `docs/Public_Benchmarks.md`.
- Models and hardware: generators Llama 3 8B (main), Gemma 4 E4B, Mistral 7B, Phi-4 14B (Ollama, temperature 0); NLI `nli-deberta-v3-small` (default) and `DeBERTa-v3-large-mnli-fever-anli-ling-wanli`; independent judges qwen2.5:14b and Gemma 3 27B; reranker, embedder. CPU (Intel Core Ultra 7 155H) and a free Kaggle T4 GPU; identical verifier decisions on both (12,901 of 12,901 sentences). See `docs/Models_and_Evaluators.md`.
- Baselines. Plain RAG and the ablation variants; Layer 1: the previous gate and `protectai/deberta-v3-base-prompt-injection-v2` (whole message and piece by piece); poisoning: LLM-judge chunk filter and a prompt-only (spotlighting) defence. Layer 3: RAGTruth's published detectors (GPT-3.5/GPT-4 prompting, SelfCheckGPT, LMvLM, fine-tuned Llama-2-13B).
- Metrics: attack success rate, false-block rate, fact recall, fabricated-number rate, sentence support (human-checked sample), abstention on unanswerable questions, draft retention, latency p50/p95.
- Analysis: Wilson intervals (proportions), bootstrap intervals (means), paired bootstrap (differences); temperature 0; all LLM calls cached; `python scripts/evaluate_pipeline.py --suite all --judge`; `--offline` reproduces from the saved cache. Tools: Python, sentence-transformers, ChromaDB, Ollama.

## 4. Results
Objective presentation only; each result = claim -> number with interval -> one-line reading. Figures >= 300 dpi.
| RQ | Source |
|---|---|
| RQ1 | `eval/results/*/report.md`, Layer 1 section (per-family bars) |
| RQ2 | Poison section |
| RQ3, RQ4 | Quality section and ablation table |
| RQ5 | Chart table |
| RQ6 | Latency columns and `scripts/benchmark_layer3_latency.py` |
Report negative results as measured (for example: some Layer 1 families got slightly worse; the attacker test was looked at twice).

**RQ1 draft (reviewer point 2; numbers from `docs/Layer1_Adaptive_Threat_Intelligence_Gate.md` section 3, all test-split, Wilson 95%):**
- Blind public test split (2,856 attacks, 4,661 benign): attacks flagged or blocked 61.4% [60-63] → 89.6% [88-91]; blocked 15.1% → 70.4%; benign flagged 4.3% → 2.1%; benign blocked 1 → 13 of 4,661. ProtectAI: 69.9% caught / 8.4% benign flagged (whole message), 89.1% / 15.6% (piece by piece).
- Families: translated attacks 47.2% → 94.2% (93-95% on three languages not in training); hidden 41.4% → 80.4%; split 35.0% → 83.0%; translated benign flagged 10.9% → 1.4%. Worse: social engineering 86.7% → 80.0%, jailbreak 95.0% → 92.1%, direct override 97.3% → 93.8%.
- Ablations: joining the top pieces +2.4 points overall (+15 on split attacks); authority-claim check +11 points on social engineering.
- Adaptive attacker (114 test attacks, 20 queries): attempts reaching retrieval 49.7% (previous gate) → 48.7% (current, persistent attacker); at 5 queries 58.9% → 28.4%. A new user id per attempt defeats suspicion (94.7% find an unflagged variant); with login and the 5-account limit, 76.3%. Second look at this split (the flag floor was added after the first run, tuned on dev).
- Sessions (test): attack turns released 23.8% → 11.9% with suspicion, no extra benign block.
- Cost: median latency 14 → 334 ms per message (up to ~1.2 s for 300+ words).

**RQ2 draft (reviewer point 3; numbers from `docs/Poisoning_Study.md`, Llama 3 with the naive prompt, Wilson / paired bootstrap 95%, McNemar exact):**
- 184 poisoned-document cases instead of 8 queries: v1 120 cases in 10 styles; v2 64 held-out cases in 8 new styles, run once. 11 defence settings on the same cases. Baselines: prompt-only (spotlighting) and an LLM-judge chunk filter.
- v1: no defence 46.7% [38-56] attack success; prompt-only 35.0%; LLM-judge filter 17.5%; full framework with deep scan 10.8% [6-18] (vs judge filter -6.7 points [-12, -1], p = 0.057; vs prompt-only -24.2, p < 0.001).
- v2 (held out): no defence 23.4%; prompt-only 25.0%; LLM-judge filter 21.9%; full framework 9.4% (vs judge filter p = 0.021); with deep scan 7.8% [3-17] (vs judge filter -14.1 points [-23, -6], 0 vs 9 discordant cases, p = 0.004; vs prompt-only p = 0.001).
- Mechanism: on new styles the regex patterns quarantined nothing, and the judge (baseline and deep scan alike) removed only styles that were not succeeding. Layer 2's trust ordering (official evidence first) halved attack success with the poison still in the evidence (23.4% → 10.9%), independent of attack wording.
- Cost: Layer 3 removes 11-13 points of fact recall (utility 86.5-89.0% vs 100%) for a small gain against poisoning; this is the reviewer point 4 trade-off.
- Other generators (same 64 held-out cases, defences' judge fixed to Llama 3): the advantage does **not** carry over. Full + deep scan vs LLM-judge filter: Gemma 4 51.6% vs 57.8% (p = 0.125), Mistral 37.5% vs 39.1% (p = 1.0), Phi-4 31.2% vs 39.1% (p = 0.27); prompt-only is best for Phi-4 (21.9%). These generators obey document instructions more (Gemma 4: 78% undefended), and Layer 2's ordering does not help them. The defences are complementary: the untrusted-data instruction stops styles the filters miss, and vice versa. *Pending (Kaggle part 3): the deployed configuration (deep scan + L2 + L3 + that instruction).*
- Limits: polite requests with no instruction wording (reviewer_request 10/12, email_forward 3/8 against the full stack); the deep scan and the judge baseline share one judge; v1 was examined during development.

**RQ3/RQ4 draft (reviewer point 4; `docs/Layer3_Evidence_To_Answer_Gate.md` §11, Llama 3, same cached drafts, bootstrap 95%):**
- Diagnosis with an independent judge: of the sentences it calls supported, v2 dropped 34% (dev) and 48% (held-out). The fixable causes were facts read from Markdown tables, which the NLI model sees one row at a time (including all 7 sentences rejected as contradictions), and long attribution clauses. Explanations and inferences the evidence does not state stay rejected by design.
- Verifier v3 (table-row check, single-sentence extractive rule, longer attribution stripping): fact recall dev 88.5% → 90.4% and held-out 82.5% → 91.7% (no verification: 96.2% / 100%); recall cost on held-out 17.5 → 8.3 points; all facts present, held-out, 73.3% → 86.7%; fabricated numbers 0; unanswerable questions refused 100%. Judge-supported sentences kept 58.6% → 67.2% with no new unsupported ones. Against the first release (96.2% → 78.8%), the dev cost falls from 17.4 to 5.8 points.
- The check can only accept claims. 21 wrong table claims (wrong row, column, unit or currency, negation, added content) are refused in the tests. *Pending: RAGTruth recall and the other generators with v3 (Kaggle part 3).*

**Public benchmarks draft (reviewer point 1; `docs/Public_Benchmarks.md`):**
- RAGTruth (2,675 answers, no training on it): verifier v2 catches 98.8% of hallucinated answers (932/943), response-level F1 55.4 [53, 57] with the small NLI model and 57.5 [55, 59] with the large one. Published detectors: GPT-3.5 prompt 52.9, SelfCheckGPT 58.8, GPT-4 prompt 63.4, fine-tuned Llama-2-13B 78.7. Weakness: precision, with 43.1% of clean sentences kept (57.9% with the large NLI model; 22.6% on data-to-text JSON records). Recall is 95–100% for each of the six source LLMs.
- BIPIA (150 cases, Llama 3, naive prompt): attack success none 4.7%, prompt-only 2.0%, LLM-judge filter 4.7%, full 1.3% (vs judge filter p = 0.062). Floor effect: the injected document reached the LLM in 45% of cases, and BIPIA's task-style attacks rarely work on Llama 3. Table answers correct 22.7% → 6.7% with Layer 3 v2 (the point 4 problem); *pending: v3 and the deployed configuration (part 3)*.

**Models and evaluators draft (reviewer point 5; `docs/Models_and_Evaluators.md`):**
- Four generators from four families. Layer 3 removes fabricated numbers for each (Mistral held-out 13.3% → 3.3%; Llama 3 3.8% / 3.3% → 0%). The recall cost depends on the generator's style: held-out fact recall with Layer 3 is 96.7% for Gemma 4, 83.6% for Llama 3, 80.6% for Phi-4 and 74.2% for Mistral. Draft sentences released: 30–85%.
- Independent judge qwen2.5:14b, validated on 300 RAGTruth sentences against human labels: agreement 80.7%, kappa 0.61 [0.52, 0.70] (verifier 0.37). By this judge, 96–99% of released sentences are supported for every generator; supported sentences kept range from 42% (Phi-4) to 86% (Gemma 4).
- Larger NLI model: RAGTruth clean sentences kept 43.1% → 57.9% at the same recall (98.8% / 98.5%); on our small question sets within noise. Second judge Gemma 3 27B: kappa 0.53 with humans (more lenient); where both judges agree (235/300), agreement with humans is 86.4%, kappa 0.72. By both judges, 96–100% of released sentences are supported for every generator; the size of the recall cost depends on the judge (Llama 3: 58.6% vs 48.7% of supported sentences kept).

## 5. Discussion
- Interpretation: why deterministic guards complement a small NLI model; what the ablations show.
- Comparison with previous work (use the baselines and cited numbers).
- Implications: practical (regulated deployments, CPU-only operation) and theoretical (entailment as a release criterion).
- Failure analysis from the report's missed-attack and false-positive lists.
- Accuracy versus completeness: draft retention is the cost of verification.
- **Limitations:** small internal dataset; same-model LLM judge; account rotation under open sign-up; Layer 1 latency on long messages; no resolution of conflicting facts between documents; regex coverage; language coverage; single generator family.

## 6. Conclusion and future work
Key insights with the headline numbers, then concrete directions: closing account rotation (invite-only sign-up or per-device limits), faster Layer 1 scoring (quantised / ONNX embeddings), conflict resolution between sources, stronger verifiers, human evaluation.

## References
- Formatted exactly in the target journal's style (APA / IEEE / Harvard / Elsevier numeric).
- Roughly 40-60 sources; **at least 25-30% from the last 2-3 years, including earlier papers from the target journal.**
- Appendices: prompts (`prompts/enterprise_rag_v1.yaml`), full thresholds, per-family tables, extra examples.

---

## Figures and tables checklist
1. Architecture (three layers + ingestion)
2. Layer 3 worked example (draft -> per-sentence verdicts -> released answer)
3. Layer 1 results by attack family
4. Poisoning success by defense combination
5. Verification ablation (recall vs fabricated-number rate vs retention)
6. Latency breakdown
7. UI screenshot with the evidence panel (optional)

## Before submitting
- [ ] Full harness run with `--judge`; hand-check a random sample and report agreement.
- [x] Public benchmarks (RAGTruth, BIPIA) and baselines (RAGTruth's detectors; prompt-only and LLM-judge filter) added.
- [x] Three more generators (Gemma 4, Mistral 7B, Phi-4 14B) and a larger NLI model reported alongside Llama 3.
- [ ] One headline claim; cut what does not support it.
- [ ] Journal template, word limit, reference style, required statements checked.
- [ ] Similarity check; not under review elsewhere.
