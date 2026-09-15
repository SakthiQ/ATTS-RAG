"""Generates tests/data/table_sample.pdf: a real, ruled (grid-lined) table plus surrounding prose.

Used to test and demonstrate table-aware chunking on a genuine gridded table, since the project's
sample PDF (Beyond the Pilot...) contains only styled text boxes with no real ruling lines -- see
docs/Knowledge_Ingestion.md, "Known limitations", for that finding.

Run once to (re)build the fixture:
    python scripts/generate_table_fixture.py
"""
import os
import fitz  # PyMuPDF

OUT_PATH = "tests/data/table_sample.pdf"

# (region, Q1, Q2, Q3, Growth%) -- Growth is Q3 vs Q1, Total's Growth is revenue-weighted (not the
# average of the other rows), so it can only be read from the table, never recomputed from the rest.
ROWS = [
    ("Region", "Q1", "Q2", "Q3", "Growth"),
    ("North America", "4.10M", "4.35M", "4.60M", "12%"),
    ("EMEA", "2.10M", "2.30M", "2.55M", "21%"),
    ("APAC", "3.10M", "3.40M", "4.20M", "35%"),
    ("LATAM", "1.20M", "1.25M", "1.30M", "8%"),
    ("Total", "10.50M", "11.30M", "12.65M", "20%"),
]

HEADING = "Regional Performance Review"
INTRO = (
    "The table below summarises quarterly revenue by region for the current fiscal year. "
    "APAC delivered the strongest quarter-over-quarter growth, driven by new enterprise "
    "contracts signed in the second quarter. LATAM growth remained modest, reflecting a "
    "smaller existing customer base in the region."
)
CAPTION = "Table 1. Quarterly revenue by region (Growth = Q3 vs Q1)."
FOOTNOTE = "* Total Growth is revenue-weighted across all regions, not a simple average of the rows above."


def draw_table(page, x0, y0, col_widths, rows, row_height=18, header=True):
    x1 = x0 + sum(col_widths)
    y1 = y0 + row_height * len(rows)

    # Outer border + row/column rules -- REAL drawn lines, not just a background fill
    for i in range(len(rows) + 1):
        y = y0 + i * row_height
        page.draw_line((x0, y), (x1, y), width=0.75)
    x = x0
    for w in [0] + col_widths:
        x += w
        page.draw_line((x, y0), (x, y1), width=0.75)

    for r, row in enumerate(rows):
        x = x0
        font = "hebo" if (header and r == 0) else "helv"
        for text, w in zip(row, col_widths):
            page.insert_text((x + 4, y0 + r * row_height + row_height - 6), text, fontsize=9, fontname=font)
            x += w
    return y1


def build():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    # A real subheading (>= 1.5x the ~10pt body size) so the loader's section detection has
    # something to find -- a single-page fixture can never get a running-header label (that
    # needs 3+ repeating pages), so a heading must come from font size instead.
    page.insert_text((50, 70), HEADING, fontsize=16, fontname="hebo")
    page.insert_textbox(fitz.Rect(50, 90, 545, 190), INTRO, fontsize=10, fontname="helv")
    page.insert_text((50, 210), CAPTION, fontsize=9, fontname="hebo")

    col_widths = [140, 85, 85, 85, 85]
    bottom = draw_table(page, 50, 220, col_widths, ROWS, row_height=18)

    page.insert_text((50, bottom + 14), FOOTNOTE, fontsize=8, fontname="helv")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc.save(OUT_PATH)
    doc.close()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
