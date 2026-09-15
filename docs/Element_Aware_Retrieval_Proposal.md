# Proposal: Element-Aware Retrieval with Budgeted Context

**Status:** Proposed. Not implemented.
**Scope:** Knowledge Ingestion and the retrieval half of Layer 2.
**Builds on:** strategy v1.2 (see `docs/Knowledge_Ingestion.md`).

---

## 1. Problem

PDFs mix prose with **tables, figures and code**. Any chunking strategy has to balance three conflicting requirements:

| # | Requirement | What goes wrong today |
| :--- | :--- | :--- |
| P1 | **Keep relationships intact** | Splitting a table, even with the header repeated on every piece, separates rows from their totals, merged group cells and footnotes. Questions that compare or add up rows get wrong answers. Figures lose their captions, and code loses the explanation around it. |
| P2 | **Search must stay precise** | Embedding a whole element blurs it into one vector. Our embedding model also reads only 256 tokens, so a large element is mostly cut off. |
| P3 | **Latency must stay bounded** | Sending whole elements to the LLM makes it read more. Measured on our machine, Llama 3 reads about **40 tokens per second, so every extra 100 tokens of context adds about 2.5 s**. |

**Measured on the current system:**

| Context sent to Llama 3 | Reading time | Writing ~50 tokens | Total |
| ---: | ---: | ---: | ---: |
| 325 tokens | 7.9 s | 3.5 s | 11.4 s |
| 603 tokens | 11.4 s | 8.9 s | 20.3 s |
| 1,540 tokens | 42.4 s | 8.8 s | 51.2 s |

Other current problems:
- pdfplumber reports **69 "tables" in the sample PDF**, but half have a single row. They are layout boxes, not data tables.
- Table text is currently **added twice**: once in the page text and once as a Markdown table.
- **Code blocks** aren't detected.
- **Figures** are read by OCR only when a whole page has almost no text.

---

## 2. Proposed solution

**What gets searched is separate from what the LLM reads, and a token budget caps what it reads.**

```
INGESTION (once per upload)                        QUERY (every question)
───────────────────────────                        ──────────────────────
PDF ──► ELEMENT PARSER                             question
          typed elements: paragraph, table,           │
          figure, code, caption, list                 ▼
          + section path, page, neighbours     1. search VIEWS (hybrid + RRF), top 20
          │                                           ▼
          ├──► ELEMENT STORE (SQLite)          2. rerank views (cross-encoder)
          │     the whole element, never split        ▼
          │     + big tables also as data      3. group views by parent_id → dedupe
          │                                           ▼
          └──► SEARCH VIEWS (Chroma + BM25)    4. route: comparison question over a big
                small, ≤ 220 tokens, each with       table? ──yes──► STRUCTURED PATH (SQL)
                parent_id + trust metadata            │ no
                                                      ▼
                                               5. ASSEMBLE CONTEXT within a token budget
                                                  (whole / compact form, by size)
                                                      ▼
                                               6. LLM answer + citations
```

**How each problem is solved:**
- **P2 (search):** only small views are embedded, so search precision matches today's chunks.
- **P1 (relationships):** the LLM receives the whole element, or its data is queried directly. It never receives an arbitrary slice.
- **P3 (latency):** a fixed token budget, not element size, decides how much the LLM reads.

---

## 3. Components

### 3.1 Element parser (new: `app/rag/elements.py`)

Turns each page into typed elements. It reuses v1.2's structure detection (running headers, titles, subheadings, section paths).

| Element | How it's detected | Normalisation |
| :--- | :--- | :--- |
| **Table** | Table outlines from `page.find_tables()` | **Checked first:** fewer than 2 rows or 2 columns → treated as text, which removes the fake tables. Merged cells filled down; multi-level headers flattened (`2024 › Q3 › Revenue`); tables that continue across pages joined back together (same column layout, no new header row). **The table's area is removed from the page text**, which ends the duplication. |
| **Figure** | Image areas from `page.images` | Linked to the nearest caption (`Figure N` / `Fig.`) and the nearest paragraph; text in the image read by OCR |
| **Code** | Monospace font (`Courier`, `Consolas`, `*Mono*`) plus consistent indentation | Kept as one block |
| **Caption / footnote** | Pattern (`Table N`, `Figure N`, `*`, `†`, `Note:`) and position just above or below an element | Attached to that element |
| **Paragraph / list** | Everything else | Grouped by section, as in v1.2 |

Every element records: `element_id`, `type`, `section`, `page_start`/`page_end`, `prev_id`/`next_id`, what it references (e.g. "see Table 3" → that table's ID), and its size in tokens.

### 3.2 Element store (new: `app/rag/element_store.py`, SQLite)

- **`elements` table:** one row per element, holding its full content, caption, the paragraph before it, token count, content hash, and the trust fields inherited from its document.
- **Large tables** (bigger than the token budget) are **also stored as real data tables** in a separate, read-only SQLite file, with cleaned column names. The structured path queries these.
- Fetching by `element_id` is a primary-key lookup: microseconds, no model call.

### 3.3 Search views (changed: `chunker.py`, `vectorstore.py`)

Small search units, each **under 220 tokens** and starting with the `Document > Section` prefix. Each one carries `parent_id`, `view_type`, and the parent's trust metadata.

| Element | Views indexed | Why |
| :--- | :--- | :--- |
| Paragraph / section | The v1.2 chunks | Unchanged, so text search quality is kept |
| **Table** | **Summary:** caption, column names, row count, section. Built from a template, so upload stays instant. | Matches broad questions ("the revenue table") |
| | **One view per row,** written as `Row: Region=APAC, Q3=4.2M, Growth=12%` | Matches specific values |
| | **Total / subtotal rows** as their own views | Matches "total" questions |
| **Figure** | Caption + surrounding text + OCR text. Optionally a description from a local vision model (off by default because of CPU cost). | Figures are found by what they show |
| **Code** | Code split at function or class boundaries | Searching for code itself |
| | Plain-English summary (docstring or first comment + the explanation above) | People search in English, not code |

### 3.4 Query-time context assembly (changed: `engine.py`)

1. **Search views:** hybrid search with RRF, top 20.
2. **Rerank views** with the cross-encoder. It scores the small views, so it stays fast.
3. **Group by `parent_id`:** each parent keeps its best view score. Three matching rows of one table become **one** table.
4. **Route:** does the question compare or total rows (rule-based: *total, sum, average, highest, lowest, most, least, rank, compare, how many*) **and** is the top parent a table larger than the budget? If so → structured path (3.5).
5. **Assemble the context** in rank order, within a **budget of B = 1,200 tokens** (≈ 30 s of reading at 40 tokens/s, about the same as today's 5 × 220 tokens):

| Parent size | Sent to the LLM |
| :--- | :--- |
| **Small**: ≤ 450 tokens (the largest table in the sample PDF is 447) | The **whole element** plus its caption |
| **Medium**: 450 tokens up to what's left of the budget | The **whole element** if it ranks first and fits; otherwise its **compact form**: caption, header, the rows that matched, total rows, footnotes, and the line *"(table has N rows; M shown)"*, so the LLM knows it's seeing part of the table |
| **Doesn't fit** | Its best view only, or skipped |

6. **Add neighbours if budget remains:** the element's caption and the paragraph before it.

### 3.5 Structured path for large tables

1. The LLM receives the **schema** (table name, column names, 3 example rows: about 100 tokens) plus the question, and writes a single `SELECT` query (about 40 tokens).
2. **Checks before running it:**
   - the database connection is read-only (`mode=ro`)
   - exactly one statement, which must start with `SELECT`
   - no `ATTACH`, `PRAGMA` or `load_extension`
   - at most 50 result rows, and a time limit
3. The database runs the query over **every row** in milliseconds.
4. The LLM answers from the result rows, and the citation points to the table.
5. **Fallback:** if the query fails or returns nothing, use the table's compact form.

The read-only, SELECT-only checks matter for ATTS-RAG, because the question comes from the user and is itself an attack surface.

### 3.6 Trust (Layer 2)

- **Views inherit** their parent's tier, version and scan status.
- **The poisoning scan runs on the parent and on each view.** If any view is quarantined, the parent is flagged, so a planted row can't pass as clean on its own.
- **The parent's content hash** goes in the registry, so Layer 2's integrity check covers the whole element.
- **A bonus for Layer 3:** results from the structured path are exact values from the source table, so a number in the answer can be checked against its source row.

---

## 4. How each problem is solved

| Problem | Mechanism | Why it works |
| :--- | :--- | :--- |
| **P1: relationships lost when tables are split** | Parents are never split for the LLM. Small elements are sent whole. Comparison questions over big tables are answered by a query over all rows. | Comparisons, totals, merged cells and footnotes all live in the parent the LLM sees, or in the data the query runs over |
| **P1: figures and code lose context** | Captions, preceding explanations and cross-references attached to the element | They travel with the element as a unit |
| **P2: whole elements are hard to search** | Only small views are embedded (under 220 tokens), each with a context prefix | Search precision equals v1.2's; several views per element make hits more likely |
| **P3: latency grows with element size** | Fixed budget of 1,200 tokens; compact forms; duplicates merged; query path for big tables | Reading time is capped at about 30 s, whatever the document contains. The query path turns a table of about 9,000 tokens (≈ 3.7 min to read, and over Llama 3's 8K limit) into about 10 s. |
| Model context limit | The budget is always below Llama 3's 8K limit | Nothing gets cut off |
| Duplicate table text | Table areas removed from the page text | Each fact is indexed once |
| Fake tables | Checked for at least 2 rows and 2 columns | Layout boxes are treated as text |
| Slow upload | Template summaries; OCR only on figures; vision model optional | No LLM call per element at upload by default |
| SQL injection through the question | Read-only, SELECT-only, limited queries | The worst possible outcome is a wrong read-only answer |

**Latency per question, before and after:**

| Step | Today | Proposed |
| :--- | :--- | :--- |
| Search and rerank | < 1 s | < 1 s (views are the same size as today's chunks) |
| Fetch parents | none | microseconds (by ID) |
| LLM reading | ≤ 1,100 tokens ≈ ≤ 27 s | ≤ 1,200 tokens ≈ ≤ 30 s; usually less, because duplicate hits are merged |
| Structured path (large tables only) | none | one extra LLM call: about 150 tokens read and 40 written, ≈ 10 s |

---

## 5. Evaluation plan

The sample PDF has no real data tables and no code, so the test set needs a **second document**. Build a test PDF containing:
- a 300-row table with totals and merged group cells
- a table that continues across two pages
- a chart with a caption
- a code listing with an explanation

**New questions,** added to `tests/data/retrieval_questions.json`:
- single-value lookups ("Q3 revenue for APAC")
- comparisons ("which region grew fastest")
- totals
- footnote questions
- figure questions
- code questions ("how does the retry work")

| Metric | What it shows |
| :--- | :--- |
| Recall@5, MRR@10, measured on parent elements | Whether the right element is found |
| **Exact-match accuracy on numbers** in table questions | Whether relationships survive |
| Context tokens per question (median, 95th percentile) | Whether the budget holds |
| End-to-end latency (median, 95th percentile) | Whether P3 is solved |
| Fake tables rejected / real tables kept | Parser quality |

**Comparisons:**
1. v1.2 chunks (baseline)
2. \+ views and parents
3. \+ token budget
4. \+ structured path

Run with `scripts/eval_retrieval.py`, extended to report numeric accuracy and tokens per question.

---

## 6. Implementation plan

| Phase | Deliverable | Main files |
| :--- | :--- | :--- |
| **1** | Element parser for tables and text: fake-table check, table text removed from pages, merged-cell fill. SQLite element store. Table summary and row views. Grouping by parent. Budgeted assembly (small and medium sizes). | `elements.py`, `element_store.py`, `chunker.py`, `vectorstore.py`, `engine.py` |
| **2** | Figures (caption, OCR, surrounding text) and code blocks (split at functions, plus summaries) | `elements.py`, `chunker.py` |
| **3** | Structured path: large tables stored as data, SQL generation, read-only checks, fallback | `element_store.py`, `engine.py` |
| **4** | Optional: vision-model figure descriptions; tables joined across pages; switching the parser to Docling for harder layouts | `elements.py` |

Each phase ships with its tests, a `strategy_version` bump, `reingest_corpus.py --rebuild`, a PASS from `check_index.py`, and a before-and-after run of the retrieval test.

---

## 7. Risks and limitations

- **Detection by font and table outline is heuristic.** Tables drawn without lines, or code not in a monospace font, can be missed. Docling (phase 4) is the fallback.
- **An 8B model can write wrong SQL.** Mitigated by the checks, the fallback, and returning the query in the reasoning trace so it can be inspected. Accuracy has to be measured.
- **The budget can push out lower-ranked evidence.** Mitigated by merging duplicates and using compact forms. The 95th-percentile token count should be tracked.
- **Reading is still slow on CPU** (about 30 s at the budget). The budget stops it getting worse; a GPU is what makes it faster.
- **A second document type adds maintenance:** two stores (elements and views) must stay consistent. `check_index.py` must be extended to compare view `parent_id`s with the element store.
