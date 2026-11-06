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
**3.1 Threat model.** Attacker capabilities (anonymous user; uploader of a poisoned document), goals (override instructions, exfiltrate, plant false facts, obtain unsupported claims), out of scope (compromised admin, model weights, side channels), trust assumptions (`official` uploads trusted; tenant identity currently unauthenticated - say so).

**3.2 System design** (architecture figure: `docs/reports/atts_rag_architecture.png`, update to the current design).
- Ingestion poisoning scan: rules + LLM judge + embedding outlier (`app/rag/ingestion_guard.py`)
- Layer 1 query gate: admission, normalization, rules, capped semantic signal, session risk (`docs/Layer1_Adaptive_Threat_Intelligence_Gate.md`)
- Retrieval: hybrid dense + BM25 (RRF), cross-encoder rerank, relevance floor
- Layer 2 trust gate: authorization, integrity hash, provenance, trust tiers, PII redaction
- Layer 3 output gate: intent-aware draft, parsing, verification, reconstruction, charts (`docs/Layer3_Evidence_To_Answer_Gate.md`)
- Algorithm box for Layer 3 and a table of thresholds with the justification for each.

**3.3 Experimental setup** (detailed enough to reproduce).
- Datasets. Internal Acme corpus (`eval/`: 34 questions, 40 attacks, 32 benign queries, 5 poisoned documents) for ablations. **TODO: add public benchmarks** (a QA set with gold answers, a sentence-labelled hallucination set, a prompt-injection set, a RAG-poisoning set).
- Models and hardware: generator, NLI verifier, reranker, embedder; versions and CPU/GPU. **TODO: add a stronger generator and NLI model.**
- Baselines. Plain RAG and the ablation variants; **TODO: at least one published guard or faithfulness filter.**
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
Report negative results as measured (for example: session risk currently changes no decision; paraphrased and multilingual attacks are mostly missed).

## 5. Discussion
- Interpretation: why deterministic guards complement a small NLI model; what the ablations show.
- Comparison with previous work (use the baselines and cited numbers).
- Implications: practical (regulated deployments, CPU-only operation) and theoretical (entailment as a release criterion).
- Failure analysis from the report's missed-attack and false-positive lists.
- Accuracy versus completeness: draft retention is the cost of verification.
- **Limitations:** small internal dataset; same-model LLM judge; unauthenticated tenant identity; no resolution of conflicting facts between documents; regex coverage; language coverage; single generator family.

## 6. Conclusion and future work
Key insights with the headline numbers, then concrete directions: authenticated tenancy, multilingual and paraphrase detection, conflict resolution between sources, stronger verifiers, human evaluation.

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
- [ ] Public benchmarks and at least two baselines added.
- [ ] Stronger generator reported alongside Llama 3.
- [ ] One headline claim; cut what does not support it.
- [ ] Journal template, word limit, reference style, required statements checked.
- [ ] Similarity check; not under review elsewhere.
