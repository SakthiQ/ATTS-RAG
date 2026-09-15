# Knowledge Ingestion

How ATTS-RAG turns an uploaded document into searchable, trust-tagged chunks, what changed in each
version, and how the changes were measured.

**Current strategy version: `v1.2`** (recorded in each document's registry entry as `strategy_version`;
the table mechanism in §2.3a shipped inside `v1.2` — the code was added mid-version, so existing `v1.2`
documents ingested before it will have no table elements even if their source PDF has real tables).

---

## 1. Pipeline

```
POST /upload  (file, source_tier, document_id, X-Admin-Token)
   │  permission checks run before anything is written
   ▼
LOAD           loader.py      text lines + font sizes; a table is extracted as its own document
                              ONLY if real drawn grid lines back it (§2.1a) -- otherwise its text
                              stays as ordinary paragraph text; OCR for scanned pages
   ▼
STRUCTURE      loader.py      drop running footers/page numbers, running header → section label,
                              large-font lines → title / subheading, split pages at subheadings
   ▼
CHUNK          chunker.py     paragraphs: ≤ 220 tokens (embedding model's tokenizer), 30-token
                              overlap, prefixed "Document > Section" (§2.3)
                              tables: kept WHOLE (never split for the LLM); a small summary view
                              plus one small view per row are generated and searched instead,
                              all sharing one parent_id (§2.3a)
   ▼
DEDUPLICATE    ingestion.py   SHA-256 of the document's content; identical content is skipped
   ▼
SCAN           ingestion_guard.py   regex threat feed + LLM judge + embedding outliers
                              → clean / flagged / quarantined; if any view of a table is
                              quarantined, every view of that table is (§2.4a)
   ▼
STORE          vectorstore.py Chroma (vectors) + BM25 (keywords) + registry, written atomically;
                              trust metadata on every chunk, per-chunk hashes in the registry
```

Files: `app/rag/loader.py`, `app/rag/chunker.py`, `app/rag/ingestion.py`, `app/rag/ingestion_guard.py`,
`app/rag/trust_policy.py`, `app/rag/vectorstore.py`, `config/injection_patterns.yaml`.

---

## 2. Stages in detail

### 2.1 Loading
- **PDF:** `pdfplumber` reads each page as lines with their average font size.
- **Scanned pages:** if a page has fewer than 50 characters of text but contains images, it is rendered at 200 dpi and read with RapidOCR, locally.
- **DOCX, TXT, MD:** plain text extraction; no page or section structure.

### 2.1a Table detection: only real ruled tables are trusted

**Finding.** pdfplumber's default table detector also fires on styled callout boxes that have a
coloured background rectangle but **no internal grid lines at all** — it then guesses column
boundaries from text alignment. On this project's sample PDF, that guess was unreliable and
**corrupted the extracted text**, splitting words mid-way ("Quickly" → "Q" / "uickly" as separate
cells). Checked directly: the page in question has 21 background rectangles and **zero drawn line
objects**, and the whole 50-page document has zero. Every "table" `find_tables()` reports there is
a false positive — a two-column label/description list, not a data table.

**Fix.** A table is only extracted as a table when both hold:

| Check | Threshold |
| :--- | :--- |
| Real rows and columns | ≥ 2 rows and ≥ 2 columns after cleaning |
| Real ruling lines inside the bbox | at least 2 drawn `page.lines` objects, not just background fill |

Failing either check means the text is left as ordinary paragraph content — nothing is corrupted,
nothing is lost, it just isn't treated as a table. Verified on two real documents:

| Document | Real ruled tables found |
| :--- | :--- |
| Sample PDF (styled boxes, 0 real lines document-wide) | **0** — correctly rejected all of them |
| A generated fixture with an actual drawn-line grid (`tests/data/table_sample.pdf`) | **1** — extracted with clean cell text |
| A 140-page real report with genuine grid-lined tables (640 real line objects total) | **27**, cleanly extracted |

A caption is looked for on the line immediately above the table's bounding box, matching
`"Table N"` / `"Figure N"`; if none is found, the caption falls back to `"Table on page N"`.

### 2.2 Structure detection (new in v1.2)
All pages are read first, because running headers and footers are only visible across pages.

| Rule | How it's detected | What happens |
| :--- | :--- | :--- |
| **Running footer** | A last line that repeats on at least 30% of pages (and at least 3), comparing with digits masked, so "Page 7" and "Page 8" match; or any bare page number | Removed |
| **Running header** | A first line that repeats the same way, or a short upper-case first line in a smaller font than the body text ("PART 3", "ABOUT") | Removed from the text and **kept as the section label** |
| **Title** | A line at least **2×** the body font size | Starts a new section; wrapped title lines are joined |
| **Subheading** | A line at least **1.5×** the body font size | Starts a new segment under the current title |
| **Continuation page** | A page with no heading of its own | Inherits the previous page's section |

Each page becomes one document per section segment, with a `section` path such as
`PART 2 > Phase 1: Pre-Pilot Assessment and Strategy > Exit Criteria`.

The thresholds are ratios to the document's median font size, not fixed sizes. On the sample PDF, the body text is 10.5 pt, titles 27 pt and subheadings 18 pt.

### 2.3 Chunking (changed in v1.2)
- **Size:** at most **220 tokens**, counted with the embedding model's own tokenizer (`all-MiniLM-L6-v2`), including the prefix and the model's two special tokens. The model's hard limit is 256.
- **Overlap:** 30 tokens.
- **Split points**, in order of preference: paragraph break, line break, sentence end, space.
- **Context prefix:** every chunk begins with `Document > Section` on its own line, for example:
  ```
  Beyond the Pilot - How Regulated Industries Can Successfully Scale AI > PART 2 > Phase 1: Pre-Pilot Assessment and Strategy > Exit Criteria
  ☐ Executive sponsor committed to the budget allocated
  ...
  ```
  Files without detected sections are prefixed with the file name only.

### 2.3a Table chunking: keep the table whole, search small pointers to it

A table cut into pieces — even with the header repeated on each piece — breaks anything that
needs more than one row: a total, a comparison, a superlative ("which region grew fastest"), a
merged group cell, a footnote below the table. Splitting the table for *search* is fine; splitting
it for the *answer* is what loses information.

**So the two are separated.** A real table (per §2.1a) is chunked into small **search views**, but
what the LLM eventually reads is the **whole table**, fetched by a shared ID — never a fragment.

| Piece | Content | Tokens | Purpose |
| :--- | :--- | :--- | :--- |
| 1 summary view | `"Table summary: <caption> Columns: <names>. <N> rows."` | small | matches broad questions about the table |
| 1 row view, per row | `"Row -> Col1=Val1, Col2=Val2, ..."` | small | matches questions about a specific value |

Every view for one table carries the same `parent_id` (a random ID assigned per table, e.g.
`tbl_7c984629564d`) and the same `whole_content` in its metadata — the full table as Markdown,
with its caption, prefixed `"Document > Section"` like a paragraph chunk. A row view also carries
`row_index` (its 0-based position among the table's data rows; `-1` on the summary view), so a
citation or audit trail can point at the exact source row without re-parsing the view's text.

**At query time** (`engine.py`): whichever view is retrieved, its `parent_id` is used to collapse
every other hit on the same table into one entry, and the *whole table* — not the matched view —
is what gets sent to the model. Multiple matching rows from one table therefore never duplicate
the table in the context; they collapse to a single copy of it.

**The size limit.** A table larger than `TABLE_WHOLE_MAX_TOKENS` (1,800 tokens, matching the
engine's per-question context budget) is never given a `whole_content` at ingestion — there
would be nowhere for it to fit anyway. Its views are still searched normally, but a hit on one
only returns that single row or the summary, the same partial view a plain chunker would give.
**This is a known, not-yet-closed gap** — see §7. When it happens, the engine logs it explicitly
(§2.5a) rather than silently answering from a fragment.

Measured on the 140-page real report used to build this: **27 of 27** real tables fit the 1,800-
token limit and got the whole-table treatment; the one table that didn't (a 60-row continuation of
a matrix split across two pages, 1,440 tokens against the *previous* 1,200-token limit) is exactly
why the limit was raised.

### 2.4 Security scan and storage (v1.1)
- **Source tiers:** `official` 1.00, `verified_internal` 0.90, `approved_external` 0.80, `unknown` 0.40, `untrusted` 0.10. Any tier above `unknown` needs the admin token.
- **Poisoning scan:**
  - Regex threat feed; an LLM judge checks only the chunks that match.
  - Embedding-outlier score, capped so it can flag but never quarantine.
  - Anomaly score = the higher of the two. At 0.8 or above → quarantined; at 0.3 or above → flagged.
- **Table quarantine propagates (new).** If any single view of a table — one row, or the summary
  — is quarantined, every view sharing that table's `parent_id` is quarantined with it, and none of
  them are indexed. Without this, a table with one poisoned row could still have its other rows and
  summary indexed and searchable — a partial, misleadingly-clean-looking table. Paragraph chunks
  carry no `parent_id` and are never grouped this way.
- **Version ownership:** only an admin can add a version to an existing `document_id`.
- **Storage:**
  - Vectors, keyword index and registry are written together, with rollback on failure.
  - Per-chunk SHA-256 hashes are kept in the registry.
  - Trust metadata is stored on every chunk: `chunk_uid`, `document_id`, `version`, `source_tier`, `source_score`, `anomaly_score`, `scan_status`, `section`, `page`, plus `element_type`, `parent_id`, `view_type`, and (for tables) `row_index`, `table_rows`, `table_too_large`, `whole_content`.
- **Consistency:** `scripts/check_index.py` checks that the registry, BM25 and Chroma hold exactly the same chunks.

### 2.5a Query-time assembly (engine.py, not strictly "ingestion" but depends entirely on it)
- Candidates are reranked over a wider pool (20, not 5) before deduplication, so several rows of
  one table have room to compete before being collapsed.
- `_dedupe_by_parent`: collapses every hit sharing a `parent_id` into its single highest-ranked
  occurrence. Paragraph chunks (`parent_id == ""`) fall back to their own `chunk_uid` and are never
  merged with each other.
- `_assemble_within_budget`: takes deduped elements in rank order until `CONTEXT_TOKEN_BUDGET`
  (1,800 tokens) or `MAX_CONTEXT_ITEMS` (10) is reached. Always includes at least the top-ranked
  element, even if it alone exceeds the budget.
- **Degraded-table logging.** If a used element is a table with no `whole_content` (too large at
  ingestion, §2.3a), the reasoning trace logs it explicitly: *"Degraded table context: the table on
  page N (M rows) exceeded the context budget, so only its matched '`view_type`' view was used, not
  the whole table."* This makes the one known gap visible in every answer it affects, instead of
  letting it look like the model saw the complete table when it didn't.

---

## 3. Change log

### Table-aware chunking (shipped inside v1.2, added after the section above)

**Problem found.** pdfplumber's default table detector produces false positives on styled boxes
with no real grid lines, and unguarded extraction corrupts the text — see §2.1a. Separately,
splitting a real table into pieces (even with the header repeated) loses anything that needs more
than one row: totals, comparisons, footnotes, merged cells.

**What changed:**

| # | Change | Why |
| :--- | :--- | :--- |
| 1 | Tables only extracted when backed by real drawn ruling lines, not just background fill (§2.1a) | Stops corrupted extraction of non-tables; verified to reject all false positives on the sample PDF and correctly extract 27 genuine tables on a separate 140-page real report |
| 2 | Table cell text excluded from the surrounding paragraph text | No more duplication between the page text and the table |
| 3 | A table's own content is stored whole (`whole_content`), never split for the LLM (§2.3a) | Totals, comparisons and footnotes stay intact in whatever the model reads |
| 4 | Small summary + per-row search views generated instead, all sharing one `parent_id` | Search stays precise (small pieces) while the answer still gets the whole table |
| 5 | Row views carry `row_index` (0-based; `-1` for the summary view) | A citation can point at the exact source row without re-parsing view text |
| 6 | Quarantine propagates across a table's views (§2.4) | One poisoned row can no longer leave the rest of the table indexed and searchable |
| 7 | Engine reranks a wider pool (20, not 5) before deduplicating by `parent_id`, then assembles the answer within a token budget instead of a fixed chunk count (§2.5a) | Several matching rows from one table collapse to one copy of it instead of crowding out other relevant content |
| 8 | `TABLE_WHOLE_MAX_TOKENS` / `CONTEXT_TOKEN_BUDGET` raised from 1,200 to 1,800 | The one real table found too large for the old limit (1,440 tokens, a 60-row continuation table) now fits |
| 9 | Degraded-table use is logged explicitly in the reasoning trace | The one known remaining gap (§7) is visible when it happens, not silent |

### v1.2: Chunking and structure

**Problem found.** The embedding model reads at most **256 tokens** and silently drops the rest. The old chunker targeted 600 tokens, and counted them with `tiktoken` (OpenAI's tokenizer), not the embedding model's.

| Measured on the sample PDF | v1.1 | v1.2 |
| :--- | :---: | :---: |
| Chunks | 71 | 221 |
| Median chunk length (model tokens) | 320 | 118 |
| Longest chunk (model tokens) | 527 | 220 |
| Chunks longer than the model's limit | **50 of 71** | **0** |
| Text never embedded | **27%** | **0%** |

**What changed:**

| # | Change | Why |
| :--- | :--- | :--- |
| 1 | Chunk length measured with the embedding model's tokenizer; target 220 tokens, 30-token overlap | Every chunk is now embedded in full |
| 2 | "Document > Section" prefix on every chunk | Chunks keep their context. Continuation pages, like page 16's exit criteria, previously had no link to their phase. |
| 3 | Running footers and page numbers removed; running headers moved into the section label | Removes noise repeated in 45 of 50 pages' chunks while keeping the "PART N" context |
| 4 | Pages split at titles and subheadings | Needed so each chunk gets the correct section; also keeps one topic per chunk |

The strategy version went from `v1.1` to `v1.2`.

### v1.1: Ingestion security
- Source tiers with admin-token control.
- Poisoning scan with quarantine.
- Version ownership.
- Per-chunk integrity hashes and trust metadata.
- Index consistency check, and `reingest_corpus.py --rebuild`.

### v1.0: Original pipeline
- 600-token `tiktoken` chunks with 100-token overlap.
- One document hash for duplicate detection.
- Transactional write.

---

## 4. Results

**Setup:**
- 35 questions, each paired with the page (or file) that answers it: `tests/data/retrieval_questions.json`.
- Run with `scripts/eval_retrieval.py`. No LLM is involved, so the results are deterministic. The LLM query-expansion step of the full pipeline is not included.
- A result counts as correct when a chunk comes from the expected file and page.
- **Metrics:**
  - **Recall@1:** share of questions where the first chunk is correct.
  - **Recall@5:** share where a correct chunk is in the top 5.
  - **MRR@10:** average of 1 ÷ the rank of the first correct chunk.

| Mode | Recall@1 | Recall@5 | MRR@10 |
| :--- | :---: | :---: | :---: |
| Dense only, v1.1 | 0.600 | **0.943** | 0.720 |
| Dense only, v1.2 | **0.771** | 0.886 | **0.826** |
| Hybrid (dense + BM25, RRF), v1.1 | 0.800 | 0.971 | 0.862 |
| Hybrid, v1.2 | **0.857** | **1.000** | **0.903** |
| Hybrid + cross-encoder, v1.1 | 0.886 | 0.971 | 0.929 |
| **Hybrid + cross-encoder, v1.2 (used by the app)** | **0.943** | **1.000** | **0.971** |

Raw results are in `logs/retrieval_eval_before_v1.1.json` and `logs/retrieval_eval_after_v1.2.json`.

**Reading the results:**
- **The mode the app uses improved on every metric.** The first result is now correct for 94% of questions, up from 89%, and every question's answer is in the top 5.
- **Dense-only search ranks better at the top:** Recall@1 rose from 0.60 to 0.77.
- **Dense-only Recall@5 went down, from 0.943 to 0.886.** Four questions fell outside the top 5, compared with two before. A likely cause: there are now three times as many chunks, and every chunk from the PDF starts with the same long document title, so short chunks look more alike to the embedding model. This hasn't been tested. Hybrid search and the reranker recover these questions, so the app isn't affected. A shorter prefix is the obvious next experiment.

**Caveats:**
- One PDF and 35 questions, so one question changes Recall by about 0.03.
- The questions were written by reading the document. That tends to favour keyword matches, and it's the same question set for both versions.

---

## 5. How to verify

```powershell
# Tests: loader structure, chunk sizes, real-table detection, table views, quarantine
# propagation, engine dedupe/budget/degradation-logging, versioning, upload permissions (50 tests)
python -m pytest tests/ -q

# The index is complete and consistent, and every document has trust metadata
python scripts/check_index.py

# Retrieval quality on the question set
python scripts/eval_retrieval.py --label current

# After changing ingestion code: rebuild (the old index is kept as a timestamped backup)
python scripts/reingest_corpus.py --rebuild --tier official
```

Set `HF_HUB_OFFLINE=1` to skip HuggingFace network checks. The test suite takes about 45 seconds instead of about 4 minutes.

---

## 6. Settings

| Setting | Where | Value |
| :--- | :--- | :--- |
| Target chunk size | `DocumentChunker(target_tokens=...)` | 220 |
| Chunk overlap | `DocumentChunker(chunk_overlap=...)` | 30 |
| Embedding model limit | `chunker.MODEL_MAX_TOKENS` | 256 |
| Title / subheading font ratio | `loader.TITLE_SIZE_RATIO` / `SUBHEADING_SIZE_RATIO` | 2.0 / 1.5 |
| Running header/footer threshold | `loader.RUNNING_LINE_MIN_SHARE` / `RUNNING_LINE_MIN_PAGES` | 30% of pages / 3 |
| OCR trigger | `loader.MIN_TEXT_CHARS` | fewer than 50 characters, with images |
| Quarantine / flag thresholds | `trust_policy.py` | 0.8 / 0.3 |
| Real-table row/column minimum | `loader.MIN_TABLE_ROWS` / `MIN_TABLE_COLS` | 2 / 2 |
| Real-table ruling-line minimum | `loader.MIN_TABLE_RULE_LINES` | 2 drawn lines inside the bbox |
| Table caption lookup distance | `loader.CAPTION_LOOKUP_PT` | 24 pt above the table |
| Table whole-content size cap | `chunker.TABLE_WHOLE_MAX_TOKENS` | 1,800 |
| Per-question context budget | `engine.CONTEXT_TOKEN_BUDGET` | 1,800 (keep equal to the cap above) |
| Reranked pool before dedup | `engine.RERANK_CANDIDATES` | 20 |
| Max distinct elements per answer | `engine.MAX_CONTEXT_ITEMS` | 10 |

If you change the embedding model, set `MODEL_MAX_TOKENS` to that model's limit and rebuild the index.
If you change `TABLE_WHOLE_MAX_TOKENS`, change `CONTEXT_TOKEN_BUDGET` to match — a table stored as
"whole" that then can't fit the per-question budget just consumes the entire budget by itself.

---

## 7. Known limitations and next steps

Tables are now handled by the mechanism in §2.3a, not the proposal document. Figures and code are
still fully covered by the proposed design in
[Element_Aware_Retrieval_Proposal.md](Element_Aware_Retrieval_Proposal.md) — neither is built.

- **A table larger than 1,800 tokens degrades to a single matched row or the summary, not the
  whole table.** This is the one gap in the whole-table mechanism itself — see §2.3a. It's logged
  explicitly when it happens (§2.5a), but not yet fixed. The proposal's text-to-SQL fallback would
  close it, but hasn't been built: on a real 140-page report used to test this, only 1 of 27 real
  tables hit this case, so the cheaper fix (raising the size cap, already done once from 1,200 to
  1,800) may cover most real documents without needing the SQL path at all.
- **Tables split across a page break aren't rejoined.** Each page's portion becomes its own
  separate table with its own `parent_id`. A real case of this exists in the 140-page report used
  to test this change — a 100-row matrix splits at row 40, and a question needing rows from both
  halves currently won't see them as one table.
- **No parent–child retrieval for plain paragraphs yet** — only tables get it. A paragraph match
  returns just that ~220-token chunk, not the section it came from. Worth building only if
  questions commonly need several paragraphs from one section stitched together; not yet confirmed
  as a real pattern for this project's documents.
- **Row-view search format is untested against alternatives.** Rows are searched as
  `"Column=Value, Column=Value"` text. A more natural-language phrasing might retrieve better with
  some embedding models — plausible, not measured, and would need a table-focused question set
  (the current 35-question eval set has none, since the sample PDF has no real tables) to check
  honestly rather than guess.
- **Section detection relies on font sizes.** PDFs without a clear size hierarchy get only a
  running-header label, or none. DOCX and TXT files get no sections at all.
- **The prefix is long for this PDF**, because the file name is the full title. That may explain
  the dense-only Recall@5 drop in §4. Try a shorter title.
- **The evaluation set is small and has no table questions** (35 questions, one PDF, zero real
  tables in it). Grow it — ideally against a document with genuine data tables — before quoting
  either the paragraph or table retrieval numbers widely.
- **Table detection is heuristic** (row/column counts, drawn-line presence). A layout-detection
  model like Docling would likely be more robust on documents messier than what's been tested here,
  but hasn't been evaluated against this project's real cases.
