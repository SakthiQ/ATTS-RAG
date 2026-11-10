import os
import re
import statistics
from collections import Counter
from typing import List, Dict, Any, Tuple, Set
import numpy as np
import pdfplumber
from docx import Document as DocxDocument
from loguru import logger

TITLE_SIZE_RATIO = 2.0  # Page titles use at least twice the body font size
SUBHEADING_SIZE_RATIO = 1.5  # Subheadings use at least 1.5x the body font size
RUNNING_LINE_MIN_SHARE = 0.3  # A first/last line repeated on 30%+ of pages is a running header/footer
RUNNING_LINE_MIN_PAGES = 3
SMALL_LABEL_RATIO = 0.9  # A short upper-case first line smaller than body text is a running label
PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
MIN_TEXT_CHARS = 50  # Below this, a page with images is treated as scanned and sent to OCR
OCR_DPI = 300  # Measured on a slide diagram: 200 dpi misread 4 of its labels, 300 dpi 1, 400 dpi 2

MIN_TABLE_ROWS = 2  # Below this, a "table" pdfplumber finds is almost always a layout artifact
MIN_TABLE_COLS = 2
MIN_TABLE_RULE_LINES = 2  # Real drawn lines required inside the bbox; background fills don't count
CAPTION_PATTERN = re.compile(r"^(table|fig(?:ure)?\.?)\s*\d+", re.IGNORECASE)
CAPTION_LOOKUP_PT = 24  # How far above a table to look for a "Table N" / "Figure N" caption line

# Multi-column pages (slides, brochures). pdfplumber reads a page line by line across its full width,
# which interleaves side-by-side columns into nonsense ("Constrain LLM outputs to Process PHI using").
# A column layout is only assumed when a vertical gap runs through a block of body text and every
# resulting column holds real prose; anything less clear-cut keeps the plain full-width reading.
COLUMN_GUTTER_MIN_PT = 12  # A gap between columns is at least this wide
COLUMN_MIN_WIDTH_SHARE = 0.15  # Each column is at least this share of the text width...
COLUMN_MIN_CHAR_SHARE = 0.08  # ...holds at least this share of the block's characters...
COLUMN_PROSE_LINE_WORDS = 3  # ...and has at least one line of this many words (not a column of page numbers)
COLUMN_MIN_BLOCK_LINES = 3  # A single line with a wide space in it is not a column layout
COLUMN_HEADING_SIZE_RATIO = 1.3  # Words this much larger than the page median are headings, not column text
SPANNING_LINE_SHARE = 0.6  # A gap-free line wider than this share of the text width spans all columns
LINE_TOLERANCE_PT = 3  # Words whose tops differ by at most this much are on the same line
BULLET_GLYPH = re.compile(r"^[\u2022\u00b7\u25aa\u25cf\u25e6\u2023\u2043\u2219\uf0b7\uf0a7*\-\u2013\u2014]$")

Line = Tuple[str, float]  # (text, mean font size)
Segment = Tuple[str, List[str]]  # (section path, body lines)

class DocumentLoader:
    """Handles extraction of text, tables, and OCR from various file formats."""

    def _format_table(self, table_data: List[List[str]]) -> str:
        """Converts raw table grid into a clean Markdown table."""
        if not table_data or not any(table_data):
            return ""

        # Clean data: replace None with empty string and strip
        cleaned = [[str(cell or "").strip() for cell in row] for row in table_data]

        # Filter out empty rows or rows with only one cell if it's empty
        cleaned = [row for row in cleaned if any(row)]
        if not cleaned: return ""

        headers = cleaned[0]
        rows = cleaned[1:]

        md_table = "\n| " + " | ".join(headers) + " |\n"
        md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        for row in rows:
            # Ensure row length matches header length for valid markdown
            padded_row = row + [""] * (len(headers) - len(row))
            md_table += "| " + " | ".join(padded_row) + " |\n"

        return md_table + "\n"

    def _ocr_page(self, page, file_path: str = None, page_num: int = 0) -> str:
        """Fallback: Converts PDF page to image and runs RapidOCR.
        
        Uses PyMuPDF (fitz) when available for fast page rendering, falling back
        to pdfplumber.to_image().
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()

            img_np = None
            if file_path and os.path.exists(file_path):
                try:
                    import fitz  # PyMuPDF fast rendering
                    doc = fitz.open(file_path)
                    fitz_page = doc[page_num]
                    pix = fitz_page.get_pixmap(dpi=OCR_DPI)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                    doc.close()
                except Exception as fe:
                    logger.debug(f"PyMuPDF rendering fallback to pdfplumber: {fe}")

            if img_np is None:
                img = page.to_image(resolution=OCR_DPI).original
                img_np = np.array(img)

            result, _ = engine(img_np)
            if result:
                return "\n".join([line[1] for line in result])
        except Exception as e:
            page_idx = getattr(page, "page_number", page_num + 1)
            logger.error(f"OCR failed for page {page_idx}: {e}")
        return ""

    # ---------- Reading order: columns are read one after another, not across ----------

    @staticmethod
    def _group_word_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        lines: List[List[Dict[str, Any]]] = []
        for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
            if lines and abs(w["top"] - lines[-1][0]["top"]) <= LINE_TOLERANCE_PT:
                lines[-1].append(w)
            else:
                lines.append([w])
        return [sorted(line, key=lambda w: w["x0"]) for line in lines]

    @staticmethod
    def _is_spanning_line(line: List[Dict[str, Any]], width: float) -> bool:
        """A wide line with no column-sized gap in it (bullet glyphs aside) runs across all columns."""
        text = [w for w in line if not BULLET_GLYPH.match(w["text"])] or line
        gaps = [b["x0"] - a["x1"] for a, b in zip(text, text[1:])]
        return (text[-1]["x1"] - text[0]["x0"]) >= SPANNING_LINE_SHARE * width and all(g < COLUMN_GUTTER_MIN_PT for g in gaps)

    @classmethod
    def _column_layout(cls, page, exclude_bboxes) -> Any:
        """Returns (column x-ranges, block top, block bottom) for a multi-column page, else None."""
        def outside_tables(w):
            return not any(x0 - 2 <= w["x0"] and w["x1"] <= x1 + 2 and top - 2 <= w["top"] and w["bottom"] <= bottom + 2
                           for x0, top, x1, bottom in exclude_bboxes)

        words = [w for w in page.extract_words(extra_attrs=["size"]) if outside_tables(w)]
        if len(words) < 12:
            return None
        median = statistics.median(w["size"] for w in words)
        body = [w for w in words if w["size"] < COLUMN_HEADING_SIZE_RATIO * median]
        if len(body) < 12:
            return None
        left, right = min(w["x0"] for w in body), max(w["x1"] for w in body)
        width = right - left

        # The column block runs from the first non-spanning line to the next full-width line after it.
        # Full-width lines above it are a header; everything from that full-width line on is a footer.
        lines = cls._group_word_lines(body)
        spanning = [cls._is_spanning_line(line, width) for line in lines]
        start = next((i for i, f in enumerate(spanning) if not f), None)
        if start is None:
            return None
        end = next((i for i in range(start, len(lines)) if spanning[i]), len(lines))
        if end - start < COLUMN_MIN_BLOCK_LINES:
            return None
        block = [w for line in lines[start:end] for w in line]
        text_words = [w for w in block if not BULLET_GLYPH.match(w["text"])]
        if len(text_words) < 12:
            return None

        covered = [False] * (int(width) + 2)
        for w in text_words:
            for x in range(int(w["x0"] - left), int(w["x1"] - left) + 1):
                covered[x] = True
        gaps, gap_start = [], None
        for x, is_covered in enumerate(covered):
            if not is_covered and gap_start is None:
                gap_start = x
            elif is_covered and gap_start is not None:
                if x - gap_start >= COLUMN_GUTTER_MIN_PT:
                    gaps.append((left + gap_start, left + x))
                gap_start = None
        if not gaps:
            return None

        # A column's bullets sit in the gap just before its text, so the column starts at its first bullet
        edges = [left]
        for a, b in gaps:
            bullets = [w["x0"] for w in block if BULLET_GLYPH.match(w["text"]) and a <= w["x0"] < b]
            edges.append(min(bullets) - 1 if bullets else (a + b) / 2)
        edges.append(right + 1)
        columns = list(zip(edges[:-1], edges[1:]))

        def members(col):
            return [w for w in text_words if col[0] <= (w["x0"] + w["x1"]) / 2 < col[1]]

        total_chars = sum(len(w["text"]) for w in text_words)

        def substantial(col):
            ws = members(col)
            prose_lines = sum(1 for line in cls._group_word_lines(ws) if len(line) >= COLUMN_PROSE_LINE_WORDS)
            return ((col[1] - col[0]) >= COLUMN_MIN_WIDTH_SHARE * width
                    and sum(len(w["text"]) for w in ws) >= COLUMN_MIN_CHAR_SHARE * total_chars
                    and prose_lines >= 1)

        # Fold narrow, sparse or prose-less columns into a neighbour until every column is substantial
        changed = True
        while changed and len(columns) > 1:
            changed = False
            for i, col in enumerate(columns):
                if not substantial(col):
                    j = 1 if i == 0 else i - 1
                    lo, hi = min(i, j), max(i, j)
                    columns[lo:hi + 1] = [(columns[lo][0], columns[hi][1])]
                    changed = True
                    break
        if len(columns) < 2:
            return None
        return columns, min(w["top"] for w in block), max(w["bottom"] for w in block)

    @classmethod
    def _reading_order_lines(cls, page, full_lines: List[Dict[str, Any]], exclude_bboxes) -> List[Dict[str, Any]]:
        """Text lines in reading order: header, then each column top to bottom, then footer."""
        layout = cls._column_layout(page, exclude_bboxes)
        if not layout:
            return full_lines
        columns, top, bottom = layout
        header = [l for l in full_lines if l["bottom"] <= top + 1]
        footer = [l for l in full_lines if l["top"] >= bottom - 1]
        body = []
        for x0, x1 in columns:
            region = page.crop((max(x0, 0), max(top - 1, 0), min(x1, page.width), min(bottom + 1, page.height)), strict=False)
            body.extend(l for l in region.extract_text_lines(return_chars=True, strip=True) if l["text"].strip() and l["chars"])
        logger.debug(f"Page {page.page_number}: read as {len(columns)} columns")
        return header + body + footer

    # ---------- Tables: only trust extraction when a real ruled grid backs it ----------
    #
    # pdfplumber's default table detector also fires on styled callout boxes that have a
    # coloured background rectangle but no internal grid lines at all -- it then guesses column
    # boundaries from text alignment, which is unreliable and can split words mid-way ("Quickly"
    # -> "Q" / "uickly"). Requiring real drawn lines inside the bbox rejects those false
    # positives; their text is left as ordinary paragraph content instead of being corrupted.

    @staticmethod
    def _table_has_real_rules(page, bbox) -> bool:
        x0, top, x1, bottom = bbox
        def inside(obj):
            return obj["top"] >= top - 2 and obj["bottom"] <= bottom + 2 and obj["x0"] >= x0 - 2 and obj["x1"] <= x1 + 2
        return sum(1 for l in page.lines if inside(l)) >= MIN_TABLE_RULE_LINES

    @staticmethod
    def _find_table_caption(lines: List[Dict[str, Any]], bbox) -> str:
        """The closest 'Table N' / 'Figure N' line within CAPTION_LOOKUP_PT above the table."""
        _, top, _, _ = bbox
        best = ""
        for l in lines:
            if l["bottom"] <= top and (top - l["bottom"]) <= CAPTION_LOOKUP_PT and CAPTION_PATTERN.match(l["text"].strip()):
                best = l["text"].strip()  # lines are in reading order, so the last match is closest
        return best

    @classmethod
    def _extract_valid_tables(cls, page, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Real, ruled tables on this page, each with its bbox, column names, data rows and caption."""
        tables = []
        for t in page.find_tables():
            grid = t.extract()
            cleaned = [[str(c or "").strip() for c in row] for row in grid]
            cleaned = [r for r in cleaned if any(r)]
            if len(cleaned) < MIN_TABLE_ROWS or (cleaned and len(cleaned[0]) < MIN_TABLE_COLS):
                continue
            if not cls._table_has_real_rules(page, t.bbox):
                continue
            tables.append({
                "bbox": t.bbox,
                "columns": cleaned[0],
                "rows": cleaned[1:],
                "caption": cls._find_table_caption(lines, t.bbox),
            })
        return tables

    @staticmethod
    def _line_in_bbox(line: Dict[str, Any], bbox) -> bool:
        """Majority-vertical-overlap test: is this text line inside a table's bbox?"""
        _, top, _, bottom = bbox
        l_top, l_bottom = line["top"], line["bottom"]
        overlap = max(0.0, min(l_bottom, bottom) - max(l_top, top))
        return overlap > 0.5 * max(l_bottom - l_top, 1e-6)

    # ---------- Page structure: running headers/footers and section headings ----------

    @staticmethod
    def _normalize(line: str) -> str:
        """Lower-cases and masks digits, so 'Page 7' and 'Page 8' compare equal."""
        return re.sub(r"\d+", "#", line.strip().lower())

    @classmethod
    def _find_running_lines(cls, pages: List[List[Line]]) -> Tuple[Set[str], Set[str]]:
        """Returns the normalized first and last lines that repeat across many pages."""
        threshold = max(RUNNING_LINE_MIN_PAGES, RUNNING_LINE_MIN_SHARE * len(pages))
        firsts = Counter(cls._normalize(p[0][0]) for p in pages if p)
        lasts = Counter(cls._normalize(p[-1][0]) for p in pages if p)
        return ({l for l, n in firsts.items() if n >= threshold},
                {l for l, n in lasts.items() if n >= threshold})

    @classmethod
    def _is_running_label(cls, line: Line, headers: Set[str], body_size: float) -> bool:
        text, size = line
        if cls._normalize(text) in headers:
            return True
        short_caps = len(text.strip()) <= 40 and text.strip().isupper()
        return bool(body_size) and size < SMALL_LABEL_RATIO * body_size and short_caps

    @staticmethod
    def _level(size: float, body_size: float) -> str:
        if not body_size:
            return "body"
        if size >= TITLE_SIZE_RATIO * body_size:
            return "title"
        if size >= SUBHEADING_SIZE_RATIO * body_size:
            return "subheading"
        return "body"

    @staticmethod
    def _section(label: str, title: str, subheading: str) -> str:
        return " > ".join(part for part in (label, title, subheading) if part)

    @classmethod
    def _structure_pages(cls, pages: List[List[Line]], body_size: float) -> List[List[Segment]]:
        """Splits each page into (section, lines) segments.

        Running headers are removed from the text and become the section label; running footers
        and page numbers are dropped. Large-font lines set the title or subheading, and a new
        segment starts at each one. Pages without their own heading inherit the previous section,
        so continuation pages keep their context.
        """
        headers, footers = cls._find_running_lines(pages)
        label = title = subheading = ""
        structured = []

        for page_lines in pages:
            lines = list(page_lines)
            if lines and cls._is_running_label(lines[0], headers, body_size):
                new_label = lines.pop(0)[0].strip()
                if new_label != label:
                    label, title, subheading = new_label, "", ""
            if lines and (cls._normalize(lines[-1][0]) in footers or PAGE_NUMBER.match(lines[-1][0].strip())):
                lines.pop()

            segments: List[Segment] = []
            current: List[str] = []
            prev_level = "body"
            for text, size in lines:
                level = cls._level(size, body_size)
                if level == "body":
                    current.append(text)
                else:
                    if current:
                        segments.append((cls._section(label, title, subheading), current))
                        current = []
                    # Consecutive heading lines of the same level are one wrapped heading
                    if level == "title":
                        title = f"{title} {text}" if prev_level == "title" else text
                        subheading = ""
                    else:
                        subheading = f"{subheading} {text}" if prev_level == "subheading" else text
                prev_level = level
            if current or not segments:
                segments.append((cls._section(label, title, subheading), current))
            structured.append(segments)

        return structured

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extracts text and tables using pdfplumber with OCR fallback, split into titled sections.

        Real, ruled tables become their own documents (type='table'), kept whole rather than
        chunked, with their cell text excluded from the surrounding paragraph text so nothing is
        duplicated. Tables pdfplumber finds without real ruling lines are treated as unreliable
        and left as ordinary paragraph text -- see _table_has_real_rules.
        """
        documents = []
        source = os.path.basename(file_path)
        try:
            with pdfplumber.open(file_path) as pdf:
                # 1. Read every page first: running headers/footers are only visible across pages,
                #    and a valid table's lines must be excluded before paragraph text is built.
                raw_pages, page_tables, sizes = [], [], []
                for page in pdf.pages:
                    full_lines = [l for l in page.extract_text_lines(return_chars=True, strip=True) if l["text"].strip() and l["chars"]]
                    tables = self._extract_valid_tables(page, full_lines)
                    lines = self._reading_order_lines(page, full_lines, [t["bbox"] for t in tables])
                    kept = [l for l in lines if not any(self._line_in_bbox(l, t["bbox"]) for t in tables)]

                    raw_pages.append([(l["text"], statistics.mean(c["size"] for c in l["chars"])) for l in kept])
                    sizes.extend(c["size"] for l in kept for c in l["chars"] if c["text"].strip())
                    page_tables.append(tables)

                body_size = statistics.median(sizes) if sizes else 0.0
                structured = self._structure_pages(raw_pages, body_size)

                for i, (page, segments, tables) in enumerate(zip(pdf.pages, structured, page_tables)):
                    # 2. OCR Fallback: if paragraph text is nearly empty, no tables were found, but
                    #    the page has images, it's likely a scan
                    body_chars = sum(len(" ".join(seg_lines)) for _, seg_lines in segments)
                    if body_chars < MIN_TEXT_CHARS and len(page.images) > 0 and not tables:
                        logger.info(f"Page {i+1} appears image-heavy. Triggering OCR...")
                        segments = [(segments[-1][0] if segments else "", [self._ocr_page(page, file_path=file_path, page_num=i)])]

                    # 3. Paragraph documents, one per section segment
                    for section, seg_lines in segments:
                        content = "\n".join(seg_lines)
                        if content.strip():
                            documents.append({
                                "content": content,
                                "metadata": {
                                    "source": source,
                                    "page": i + 1,
                                    "type": "pdf",
                                    "section": section,
                                }
                            })

                    # 4. Table documents, kept whole and separate from paragraph text
                    current_section = segments[-1][0] if segments else ""
                    for t in tables:
                        caption = t["caption"] or f"Table on page {i + 1}"
                        content = caption + self._format_table([t["columns"]] + t["rows"])
                        documents.append({
                            "content": content,
                            "metadata": {
                                "source": source,
                                "page": i + 1,
                                "type": "table",
                                "section": current_section,
                                "has_tables": True,
                            },
                            "table_data": {"columns": t["columns"], "rows": t["rows"], "caption": caption},
                        })
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise

        return documents

    @staticmethod
    def load_docx(file_path: str) -> List[Dict[str, Any]]:
        """Extracts text from DOCX files."""
        doc = DocxDocument(file_path)
        full_text = [para.text for para in doc.paragraphs if para.text.strip()]
        return [{
            "content": "\n".join(full_text),
            "metadata": {
                "source": os.path.basename(file_path),
                "type": "docx"
            }
        }]

    @staticmethod
    def load_txt(file_path: str) -> List[Dict[str, Any]]:
        """Extracts text from plain text or markdown files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [{
            "content": content,
            "metadata": {
                "source": os.path.basename(file_path),
                "type": "text"
            }
        }]

    def load_any(self, file_path: str) -> List[Dict[str, Any]]:
        """Entry point to load a file based on its extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext == ".docx":
            return self.load_docx(file_path)
        elif ext in [".txt", ".md"]:
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

# Example Usage (for testing)
if __name__ == "__main__":
    loader = DocumentLoader()
    # print(loader.load_any("path/to/your/test.pdf"))
