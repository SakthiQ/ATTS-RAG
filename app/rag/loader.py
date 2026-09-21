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

MIN_TABLE_ROWS = 2  # Below this, a "table" pdfplumber finds is almost always a layout artifact
MIN_TABLE_COLS = 2
MIN_TABLE_RULE_LINES = 2  # Real drawn lines required inside the bbox; background fills don't count
CAPTION_PATTERN = re.compile(r"^(table|fig(?:ure)?\.?)\s*\d+", re.IGNORECASE)
CAPTION_LOOKUP_PT = 24  # How far above a table to look for a "Table N" / "Figure N" caption line

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
                    pix = fitz_page.get_pixmap(dpi=200)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                    doc.close()
                except Exception as fe:
                    logger.debug(f"PyMuPDF rendering fallback to pdfplumber: {fe}")

            if img_np is None:
                img = page.to_image(resolution=200).original
                img_np = np.array(img)

            result, _ = engine(img_np)
            if result:
                return "\n".join([line[1] for line in result])
        except Exception as e:
            page_idx = getattr(page, "page_number", page_num + 1)
            logger.error(f"OCR failed for page {page_idx}: {e}")
        return ""

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
                    lines = [l for l in page.extract_text_lines(return_chars=True, strip=True) if l["text"].strip() and l["chars"]]
                    tables = self._extract_valid_tables(page, lines)
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
