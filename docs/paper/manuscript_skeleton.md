# ATTS-RAG manuscript skeleton (Q1/Q2 journal format)

Working title (pick one claim and keep it everywhere):
**"Verify before you release: layered defenses and sentence-level verified generation for retrieval-augmented LLMs"**

Venue fit (check the current quartile on SCImago / JCR and the author guidelines before choosing; quartiles change by year and category):
- Security-leaning: *Computers & Security*, *IEEE Transactions on Information Forensics and Security*, *Journal of Information Security and Applications*
- IR / NLP-leaning: *Information Processing & Management*, *ACM Transactions on Information Systems*
- Applied AI: *Expert Systems with Applications*, *Knowledge-Based Systems*, *Future Generation Computer Systems*, *IEEE Access* (broad, faster)

Typical limits to confirm per journal: 8,000-12,000 words (Elsevier, free layout) or 12-14 pages (IEEE two-column); abstract 150-250 words; 5-7 keywords; Elsevier adds **Highlights** (3-5 bullets, max 85 characters each) and an optional **graphical abstract**; most require a **Data/Code availability** statement, **Declaration of competing interest**, **CRediT author statement** and an **AI-use disclosure**.

---

## 0. Front matter
- **Title, authors, affiliations, corresponding author**
- **Abstract** (structured in one paragraph): problem (RAG pipelines are exposed to query injection, poisoned documents and unsupported answers) -> method (three layers; verified sentence-level generation) -> evaluation (datasets, baselines, headline numbers with intervals) -> finding and limitation.
- **Keywords:** retrieval-augmented generation; prompt injection; knowledge poisoning; hallucination detection; natural language inference; zero-trust.
- **Highlights** (Elsevier): 3-5 short claims, each backed by a table below.

## 1. Introduction (1-1.5 pages)
1. RAG in enterprise settings and why trust matters (regulated industries).
2. Three failure modes, with one concrete example each: query-time injection, poisoned or untrusted documents, unsupported or fabricated answers.
3. Gap: existing defenses address one stage; none releases answers sentence by sentence only if verified.
4. **Contributions** (numbered, each testable):
   1. A three-layer architecture (query gate, evidence trust gate, output verification gate).
   2. A sentence-level verification method combining NLI over evidence windows with deterministic number, negation and extractive guards, and citation repair.
   3. Evidence-checked charts: a visualization is released only if each value is traceable to the source.
   4. An open evaluation and ablation harness (`scripts/evaluate_pipeline.py`) and dataset (`eval/`).
5. Paper roadmap (one sentence per section).

## 2. Related work (1-1.5 pages)
Group by theme, end each group with the gap this paper fills:
- RAG and its robustness (RAG, Self-RAG, CRAG)
- Prompt injection and jailbreak defenses (input filters, guard models, instruction hierarchy)
- Knowledge/corpus poisoning attacks and defenses (e.g. PoisonedRAG-style attacks, indirect injection benchmarks)
- Hallucination and faithfulness checking (NLI/entailment-based, RAGAS-style metrics, citation verification)
- Guardrail frameworks (e.g. NeMo Guardrails, Llama Guard) as baselines

## 3. Threat model (0.5-1 page)
Attacker capabilities (anonymous user; uploader of a poisoned document), goals (override instructions, exfiltrate, plant false facts, make the system assert unsupported claims), what is out of scope (compromised admin, model weights, side channels). State the **trust assumptions**: documents uploaded as `official` are trusted; `tenant_id` is currently not authenticated (say so).

## 4. System design (2-3 pages, one architecture figure)
Figure: `docs/reports/atts_rag_architecture.png` (update to the current design).
- 4.1 Ingestion poisoning scan (rules + LLM judge + embedding outlier) - `app/rag/ingestion_guard.py`
- 4.2 Layer 1 query gate: admission, normalization, rules (H), capped semantic signal (V), session risk - `docs/Layer1_...md`
- 4.3 Retrieval: hybrid dense + BM25 with RRF, cross-encoder rerank, relevance floor
- 4.4 Layer 2 trust gate: authorization, integrity hash, provenance, tiers, PII redaction
- 4.5 Layer 3 output gate: intent-aware draft, claim parsing, verification, reconstruction, charts - `docs/Layer3_...md`
- 4.6 Algorithm box for Layer 3 (pseudocode) and a table of thresholds with the reason for each.

## 5. Experimental setup (1-1.5 pages)
- **Datasets** - this is the section reviewers will attack hardest:
  - Internal: the Acme corpus in `eval/` (34 questions, 40 attacks, 32 benign, 5 poisoned docs). Useful for ablations; **too small to stand alone**.
  - **Add public benchmarks** (TODO): a QA set with gold answers (e.g. Natural Questions / HotpotQA subsets), a hallucination benchmark with sentence labels (e.g. RAGTruth), a prompt-injection set (e.g. BIPIA / HackAPrompt-style) and a RAG-poisoning set.
- **Models:** generator (currently Llama 3 8B), NLI verifier (DeBERTa-v3-small), reranker, embedder. Report sizes, versions and hardware. **Add at least one stronger generator and one stronger NLI model** (TODO).
- **Baselines** (TODO beyond the ablations): plain RAG; RAG with a guard model; RAG with a faithfulness filter from prior work.
- **Metrics:** attack success rate, false-block rate; fact recall, fabricated-number rate, sentence support (human-checked sample), abstention on unanswerable questions, draft retention, latency (p50/p95).
- **Statistics:** Wilson intervals for proportions, bootstrap intervals for means, paired bootstrap for differences; fixed temperature 0; all LLM calls cached.
- **Reproducibility:** commit hash, `--offline` re-run from the saved cache, code and data availability.

## 6. Results (2-3 pages) - one table or figure per research question
| RQ | Question | Source in this repo |
|---|---|---|
| RQ1 | How well does Layer 1 block attacks without blocking legitimate queries? | `eval/results/*/report.md`, Layer 1 section |
| RQ2 | Do ingestion guard, Layer 2 and Layer 3 reduce poisoning success, and at what utility cost? | Poison section |
| RQ3 | Does verification reduce unsupported content, and how much of the draft does it cost? | Quality section |
| RQ4 | Which verification components matter? (NLI only / - number guard / - extractive / full) | Quality ablation table |
| RQ5 | Are charts faithful? | Chart table |
| RQ6 | What is the latency overhead? | Latency columns, `scripts/benchmark_layer3_latency.py` |

Write each result as: claim -> number with interval -> one-sentence interpretation. Report negative results (e.g. session risk currently changes no decision; paraphrased and multilingual attacks are mostly missed).

## 7. Discussion (1 page)
- What worked and why (deterministic guards complement a small NLI model).
- **Failure analysis** from the report's missed-attack and false-positive lists.
- Accuracy vs completeness: draft retention is the cost of verification.
- Practical deployment notes (CPU-only latency, local models).

## 8. Limitations and threats to validity (0.5 page; do not skip)
Small internal dataset; same-model LLM judge; unauthenticated tenant identity; no defense against conflicting facts between documents; regex coverage; language coverage; single generator family.

## 9. Conclusion and future work (0.5 page)
Restate the three contributions with the headline numbers; future work = authenticated tenancy, multilingual detection, conflict resolution, stronger verifiers.

## Back matter
Data and code availability (repository URL, commit, dataset license) · Ethics statement (attack data are synthetic) · AI-use disclosure · Acknowledgements · Funding · CRediT author statement · Declaration of competing interest · **References** (aim for 40-60, mostly from the last 3 years, in the journal's style) · Appendices (prompts from `prompts/enterprise_rag_v1.yaml`, full thresholds, per-family tables, extra examples).

---

## Figures and tables checklist
1. Architecture diagram (three layers + ingestion)
2. Layer 3 flow with a worked example (draft -> per-sentence verdicts -> released answer)
3. Layer 1 results by attack family (bar chart)
4. Poisoning success by defense combination (bar chart)
5. Verification ablation table (recall vs fabricated-number rate vs retention)
6. Latency breakdown
7. UI screenshot with the evidence panel (optional, supports the explainability claim)

## Before submitting
- [ ] Run the full harness with `--judge`, then re-judge a random sample by hand (report agreement).
- [ ] Add public benchmarks and at least two baselines.
- [ ] Run with a stronger generator and report it next to Llama 3.
- [ ] Decide the one headline claim and cut anything that does not support it.
- [ ] Check the target journal's template, word limit, reference style and declaration sections.
- [ ] Plagiarism / similarity check; confirm the same work is not under review elsewhere.
