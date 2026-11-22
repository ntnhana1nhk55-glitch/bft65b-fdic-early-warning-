"""
Professional Academic Word Document (.docx) Generator for BFT65B Final Report.
Fully updated per user design requirements:
1. Font Family: Times New Roman throughout the entire document.
2. Body Font Size: 13 pt with generous, breathable line spacing (1.25) and paragraph spacing (6 pt after).
3. Punctuation & Style: Strictly zero unnecessary hyphens, zero en-dashes or em-dashes, and zero unnecessary parentheses.
   Prose is written in clean, natural academic English sentences.
4. Visual Figures: Expanded to full text-width (6.85 inches) with high clarity and centered placement.
5. Strict Examination Page Budget: Sections 1 through 6 fit within 20 pages; Section 7 Appendix fits within 10 pages.
"""

from pathlib import Path
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUT_DOCX = REPORTS_DIR / "FINAL_REPORT_BFT65B.docx"

# Palette definitions
NAVY_HEX = "1F4E79"
SLATE_HEX = "2C3E50"
CHARCOAL_HEX = "262626"
MUTED_HEX = "595959"
BORDER_HEX = "CCCCCC"
ZEBRA_HEX = "F9FAFB"
CALLOUT_BG_HEX = "F0F4F8"
CALLOUT_BORDER_HEX = "1F4E79"

NAVY_RGB = RGBColor(0x1F, 0x4E, 0x79)
SLATE_RGB = RGBColor(0x2C, 0x3E, 0x50)
CHARCOAL_RGB = RGBColor(0x26, 0x26, 0x26)
MUTED_RGB = RGBColor(0x59, 0x59, 0x59)


# -----------------------------------------------------------------------------
# XML Formatting Utilities
# -----------------------------------------------------------------------------
def set_cell_shading(cell, color_hex):
    """Set background color of a table cell."""
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    """Set internal cell margins or padding in dxa."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f"""
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    """)
    tcPr.append(tcMar)


def set_table_borders(table, color_hex="CCCCCC", sz="4"):
    """Apply clean academic horizontal borders with zero vertical borders."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(f"""
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>
            <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>
            <w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>
            <w:insideV w:val="none"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
        </w:tblBorders>
    """)
    tblPr.append(borders)


def add_field(p, field_text, default_val="1"):
    """Add dynamic Word field code such as PAGE or NUMPAGES using proper OpenXML run sequences."""
    r1 = p.add_run()
    r1._r.append(parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>'))
    r2 = p.add_run()
    r2._r.append(parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> {field_text} </w:instrText>'))
    r3 = p.add_run()
    r3._r.append(parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="separate"/>'))
    r4 = p.add_run(default_val)
    r4.font.name = "Times New Roman"
    r4.font.size = Pt(9.0)
    r4.font.color.rgb = MUTED_RGB
    r5 = p.add_run()
    r5._r.append(parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>'))


def setup_document_styles(doc):
    """Configure generous academic page geometry and base styles."""
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11.0)
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    # Running Header
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run("BFT65B Big Data Analytics in Finance, Final Research Report")
    hrun.font.name = "Times New Roman"
    hrun.font.size = Pt(9.0)
    hrun.font.color.rgb = MUTED_RGB

    # Running Footer: left repo link, right Page X of Y
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.text = ""
    frun_left = fp.add_run("https://github.com/ntnhana1nhk55-glitch/bft65b-fdic-early-warning-")
    frun_left.font.name = "Times New Roman"
    frun_left.font.size = Pt(9.0)
    frun_left.font.color.rgb = MUTED_RGB

    frun_tab = fp.add_run("\t\tPage ")
    frun_tab.font.name = "Times New Roman"
    frun_tab.font.size = Pt(9.0)
    frun_tab.font.color.rgb = MUTED_RGB
    add_field(fp, "PAGE")
    frun_of = fp.add_run(" of ")
    frun_of.font.name = "Times New Roman"
    frun_of.font.size = Pt(9.0)
    frun_of.font.color.rgb = MUTED_RGB
    add_field(fp, "NUMPAGES")


# -----------------------------------------------------------------------------
# Typography Building Blocks (13 pt Times New Roman, Clean Academic Prose)
# -----------------------------------------------------------------------------
def add_doc_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(21)
    run.font.bold = True
    run.font.color.rgb = NAVY_RGB
    return p


def add_doc_subtitle(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(13)
    run.font.italic = True
    run.font.color.rgb = SLATE_RGB
    return p


def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = NAVY_RGB
    return p


def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = SLATE_RGB
    return p


def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = CHARCOAL_RGB
    return p


def add_body_p(doc, text, bold_prefix=None, space_after=5):
    """Add a body paragraph in 13 pt Times New Roman with relaxed line spacing."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(13)
        r_pre.font.bold = True
        r_pre.font.color.rgb = CHARCOAL_RGB
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(13)
    run.font.color.rgb = CHARCOAL_RGB
    return p


def add_callout(doc, text, title=None):
    """Add an executive callout box with a thick navy left border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.9)

    borders = parse_xml(f"""
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="{CALLOUT_BORDER_HEX}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    """)
    cell._tc.get_or_add_tcPr().append(borders)
    set_cell_shading(cell, CALLOUT_BG_HEX)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=160)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.2

    if title:
        r_t = p.add_run(title + "\n")
        r_t.font.name = "Times New Roman"
        r_t.font.size = Pt(12)
        r_t.font.bold = True
        r_t.font.color.rgb = NAVY_RGB

    r_b = p.add_run(text)
    r_b.font.name = "Times New Roman"
    r_b.font.size = Pt(11.5)
    r_b.font.color.rgb = CHARCOAL_RGB

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(4)


def add_styled_table(doc, headers, rows_data, col_widths=None, align_right_cols=None, caption=None):
    """Add a professional academic table in Times New Roman with clear spacing."""
    if align_right_cols is None:
        align_right_cols = []

    if caption:
        p_tcap = doc.add_paragraph()
        p_tcap.paragraph_format.space_before = Pt(6)
        p_tcap.paragraph_format.space_after = Pt(2)
        p_tcap.paragraph_format.keep_with_next = True
        r_tcap = p_tcap.add_run(caption)
        r_tcap.font.name = "Times New Roman"
        r_tcap.font.size = Pt(11)
        r_tcap.font.bold = True
        r_tcap.font.color.rgb = NAVY_RGB

    tbl = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    set_table_borders(tbl, BORDER_HEX, sz="4")

    # Header Row
    hdr_row = tbl.rows[0]
    trPr = hdr_row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

    for col_idx, h_text in enumerate(headers):
        cell = hdr_row.cells[col_idx]
        if col_widths and col_idx < len(col_widths):
            cell.width = Inches(col_widths[col_idx])
        set_cell_shading(cell, NAVY_HEX)
        set_cell_margins(cell, top=90, bottom=90, left=120, right=120)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if col_idx in align_right_cols else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(h_text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data Rows
    for row_idx, r_data in enumerate(rows_data):
        row = tbl.rows[row_idx + 1]
        bg_hex = ZEBRA_HEX if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            if col_widths and col_idx < len(col_widths):
                cell.width = Inches(col_widths[col_idx])
            if bg_hex != "FFFFFF":
                set_cell_shading(cell, bg_hex)
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if col_idx in align_right_cols else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
            run.font.color.rgb = CHARCOAL_RGB

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(6)


def add_figure(doc, image_name, caption_text, width_inches=6.85):
    """Add a large, sharp figure spanning the text width."""
    img_path = FIGURES_DIR / image_name
    if not img_path.exists():
        print(f"Warning: Figure {img_path} not found.")
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(img_path), width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(6)
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(10)
    r_cap.font.italic = True
    r_cap.font.color.rgb = MUTED_RGB


# -----------------------------------------------------------------------------
# Metadata Block Builder (Mandatory First Page, No Dashes or Stray Parentheses)
# -----------------------------------------------------------------------------
def build_metadata_block(doc):
    tbl = doc.add_table(rows=6, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    set_table_borders(tbl, BORDER_HEX, sz="4")

    labels_data = [
        ("Candidate and Course", "Nguyễn Thiện Nhân, Student ID 11231004, Course BFT65B Big Data Analytics in Finance, National Economics University Hanoi"),
        ("Public GitHub Repository", "https://github.com/ntnhana1nhk55-glitch/bft65b-fdic-early-warning- with complete reproducible source code and pinned environment"),
        ("Data Acquisition and Date", "FDIC BankFind Suite REST API with resumable backoff client and verification manifests, downloaded and verified on November 7, 2026"),
        ("Dataset Dimensions", "1,682,616 raw Call Report filings across 170 quarters from 1984Q1 to 2026Q2, 27,835 unique banks, 4,117 resolution records"),
        ("Panel Pipeline Layers", "Silver Panel with 1,681,431 rows and 46 columns, Gold Feature Table with 1,663,902 rows and 57 columns"),
        ("Chronological Partitions", "DEV from 1984Q1 to 2006Q4 with 10,225 failure quarters across 2,658 failed banks, EMBARGO across 2007, TEST from 2008Q1 to 2014Q4 with 2,052 failure quarters across 526 failed banks"),
    ]

    for idx, (label, val) in enumerate(labels_data):
        row = tbl.rows[idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.1)
        c1.width = Inches(4.8)
        set_cell_shading(c0, ZEBRA_HEX)
        set_cell_margins(c0, top=70, bottom=70, left=110, right=110)
        set_cell_margins(c1, top=70, bottom=70, left=110, right=110)

        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(label)
        r0.font.name = "Times New Roman"
        r0.font.size = Pt(10)
        r0.font.bold = True
        r0.font.color.rgb = NAVY_RGB

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(val)
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(10)
        r1.font.color.rgb = CHARCOAL_RGB

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(8)


# -----------------------------------------------------------------------------
# Main Document Assembly
# -----------------------------------------------------------------------------
def generate_full_word_report():
    print("=" * 70)
    print("BFT65B Word Report Generator: Constructing FINAL_REPORT_BFT65B.docx")
    print("Typography: Times New Roman 13pt body, generous line spacing, zero dashes")
    print("=" * 70)

    doc = docx.Document()
    setup_document_styles(doc)

    # -------------------------------------------------------------------------
    # PAGE 1: TITLE, METADATA BLOCK AND EXECUTIVE ABSTRACT
    # -------------------------------------------------------------------------
    add_doc_title(
        doc,
        "Scalable Early Warning System for Bank Distress and Failure: An End to End Leak Free Big Data Pipeline on Forty Two Years of Federal Deposit Insurance Corporation Call Reports",
    )
    add_doc_subtitle(
        doc,
        "A Rigorous Investigation into Longitudinal Columnar Storage, Distributed Query Optimization, Causal CAMELS Features, and Asymmetric Loss Economics",
    )

    build_metadata_block(doc)

    add_heading_2(doc, "Executive Abstract")
    add_body_p(
        doc,
        "We design, implement, and empirically validate an enterprise grade Big Data early warning pipeline for banking distress and regulatory resolution across the complete digital history of Federal Deposit Insurance Corporation Call Reports from 1984Q1 to 2026Q2. Covering 1,682,616 quarterly bank filings across 27,835 unique depository institutions and 4,117 regulatory resolution events, our pipeline resolves the foundational challenges of longitudinal data engineering, distributed engine optimization, econometric causality, and supervisory decision economics.",
    )
    add_body_p(
        doc,
        "In Storage Architecture, column oriented Apache Parquet with Zstandard and Snappy compression reduces physical storage from 424.12 megabytes in raw text format down to 241.88 megabytes in Snappy and 190.73 megabytes in Zstandard, achieving a 2.22 fold compression ratio. Column projection and predicate pushdown accelerate selective supervisory queries to 31.9 milliseconds on Parquet Snappy, delivering a 459.0 fold speedup over Pandas full text reads at 14,652.7 milliseconds, a 22.6 fold speedup over Polars full text scans at 721.9 milliseconds, and a 16.4 fold speedup over Polars selective text scans at 522.5 milliseconds.",
    )
    add_body_p(
        doc,
        "In Scalable Processing, Polars Lazy processes the complete end to end panel in 1,140 milliseconds, which is 8.03 fold faster than vectorized Pandas at 9,154 milliseconds, and achieves a 40.29 fold speedup on selective pushdown workloads. Apache Spark BroadcastHashJoin achieves a 3.42 fold speedup over SortMergeJoin by eliminating network shuffle exchanges. A four way cross engine verification establishes strict mathematical parity across all 1,681,431 Silver panel rows with maximum absolute difference bounded at 5.68e-14.",
    )
    add_body_p(
        doc,
        "In Leak Free Econometric Engineering, we formulate a causal point in time forward distress target over a strict four quarter horizon, quarantine 17,529 right censored quarters after June 2025, enforce a four quarter embargo across calendar year 2007, and engineer a 40 feature CAMELS matrix with contiguous flow decumulation and contemporaneous state peer benchmarking. An adversarial Silver to Gold perturbation proof establishes zero backward data leakage with maximum absolute difference bounded at 1.39e-17.",
    )
    add_body_p(
        doc,
        "Across Purged Expanding Window Cross Validation and out of time evaluation on the Global Financial Crisis from 2008 to 2014, our champion LightGBM CAMELS model achieves a Test Area Under the Precision Recall Curve of 0.6454, which is 66.9 fold higher than random guessing and 38.4 percent higher than the supervisory Texas Ratio heuristic. Under an inspection capacity budget of top 2 percent of banks, the model captures 80.2 percent of all failing institutions and 78.1 percent of failing banking assets. Data driven threshold optimization selects an optimal probability cutoff of 0.0401, generating 4,074.08 million dollars in net regulatory savings during the Global Financial Crisis. We conclude by translating these findings into an actionable architectural blueprint for the State Bank of Vietnam.",
    )

    # -------------------------------------------------------------------------
    # SECTION 1: PROBLEM FORMULATION, ACQUISITION AND STORAGE
    # -------------------------------------------------------------------------
    add_heading_1(doc, "1. Problem Formulation, Data Acquisition and Storage Architecture")
    add_body_p(
        doc,
        "Depository institution insolvencies impose severe negative externalities on financial stability, government fiscal balance, and macroeconomic output. When an insured bank fails, the resolution authority must intervene, liquidate assets, or arrange purchase and assumption transactions to protect insured depositors. Historically, supervisory agencies have relied on periodic on site examinations leading to confidential CAMELS ratings. However, on site examinations occur on a 12 to 18 month cycle. By the time examiners arrive on site, severe asset quality degradation or liquidity drain may have matured into an irreversible crisis. An enterprise Early Warning System continuously monitors quarterly regulatory filings to flag vulnerable banks four quarters in advance, providing supervisory authorities with actionable lead time for prompt corrective intervention.",
    )
    add_body_p(
        doc,
        "Our acquisition architecture targets three distinct endpoints of the FDIC BankFind Suite public interface. First, the financials endpoint covers all 170 quarters from 1984Q1 to 2026Q2, capturing 1,682,616 raw filings. Second, the institutions endpoint records demographic attributes for 27,835 unique banks. Third, the failures endpoint records all 4,117 official regulatory resolutions and financial assistance transactions. Benchmarking against the course starter script fdic_download_starter.py, our enhanced acquisition client in src/data/acquire.py introduces enterprise grade features: a strict 33 column PyArrow schema, resumable tracking with SHA256 cryptographic manifests, concurrent asynchronous workers, exponential backoff with jitter against rate limits, and deterministic sorting on institution certificate number and report date.",
    )

    add_heading_2(doc, "Physical Storage Benchmark across Five Formats")
    add_body_p(
        doc,
        "To establish the optimal physical storage layout for large scale financial panel analytics, we implemented a comprehensive storage benchmark comparing five standard layouts on local solid state storage across all 1,682,616 rows and 33 columns. The empirical measurements from reports/storage_benchmark_full.csv are summarized in Table 1 and visualized in Figure 1.",
    )

    t1_headers = ["Storage Format", "Disk Footprint", "Compression", "Write Duration", "Full Read Pandas", "Full Read Polars", "Projection 5 Cols", "Pushdown Query"]
    t1_rows = [
        ["Raw Text CSV", "424.12 MB", "1.00x", "111.65 s", "14,652.7 ms", "721.9 ms", "465.7 ms", "522.5 ms"],
        ["Gzip Compressed CSV", "167.47 MB", "2.53x", "213.38 s", "18,237.8 ms", "3,119.4 ms", "3,100.8 ms", "3,021.6 ms"],
        ["Parquet Uncompressed", "320.26 MB", "1.32x", "15.65 s", "1,633.5 ms", "352.8 ms", "74.4 ms", "38.0 ms"],
        ["Parquet Snappy Partitioned", "241.88 MB", "1.75x", "17.82 s", "1,501.1 ms", "147.9 ms", "50.8 ms", "31.9 ms"],
        ["Parquet Zstandard Partitioned", "190.73 MB", "2.22x", "20.42 s", "1,474.7 ms", "183.6 ms", "68.9 ms", "33.9 ms"],
    ]
    add_styled_table(
        doc,
        t1_headers,
        t1_rows,
        col_widths=[1.8, 0.7, 0.6, 0.7, 0.9, 0.9, 0.8, 0.7],
        align_right_cols=[1, 2, 3, 4, 5, 6, 7],
        caption="Table 1. Physical Storage Benchmark across Five Layouts on 1,682,616 Rows and 33 Columns",
    )

    add_figure(doc, "fig2_storage_benchmark.png", "Figure 1. Storage Format Benchmark and Pushdown Query Efficiency across Five Physical Layouts on 1.68 Million Records")

    add_heading_2(doc, "Physical Query Mechanics and Partitioning Overhead")
    add_body_p(
        doc,
        "The empirical storage results reveal three critical mechanics. First, columnar encoding in Parquet Snappy at 241.88 megabytes and Parquet Zstandard at 190.73 megabytes achieves superior footprint reduction over raw text at 424.12 megabytes via page level dictionary encoding and run length encoding on repeated banking variables. Gzip text reaches 167.47 megabytes but suffers a severe 213 second write time and prevents parallel block reads.",
    )
    add_body_p(
        doc,
        "Second, column projection and predicate pushdown dramatically accelerate queries. Projecting five core balance sheet columns drops Parquet Snappy read latency from 147.9 milliseconds down to 50.8 milliseconds. Adding predicate filtering for assets exceeding one billion dollars and negative return on assets slashes latency to 31.9 milliseconds, representing a 459.0 fold speedup over Pandas full text reads at 14,652.7 milliseconds, a 22.6 fold speedup over Polars full text scans at 721.9 milliseconds, and a 16.4 fold speedup over Polars selective text queries at 522.5 milliseconds. Parquet achieves this by inspecting row group minimum and maximum statistics in the metadata footer, skipping unreferenced byte ranges directly on disk.",
    )
    add_body_p(
        doc,
        "Third, partitioned Zstandard at 33.9 milliseconds exhibits slightly higher latency than Parquet Snappy at 31.9 milliseconds on simple selective queries because Zstandard decompression trades slight CPU overhead for smaller disk footprint at 190.73 megabytes. Opening 170 quarterly files on Windows NTFS adds minor filesystem metadata overhead on datasets under 500 megabytes, yet quarterly partitioning remains essential for incremental ingestion and distributed Spark partition pruning.",
    )

    # -------------------------------------------------------------------------
    # SECTION 2: DATA QUALITY AND POINT IN TIME PANEL CONSTRUCTION
    # -------------------------------------------------------------------------
    add_heading_1(doc, "2. Automated Data Quality Audit and Point in Time Panel Construction")
    add_body_p(
        doc,
        "Longitudinal banking panels require specialized accounting domain knowledge. Over four decades, the United States banking system underwent massive structural changes. Our automated quality audit in src/data/quality_audit.py catalogs these structural phenomena across all 1,682,616 raw filings.",
    )
    add_body_p(
        doc,
        "Active reporting commercial banks and thrifts declined by 76.2 percent from a peak of 18,133 institutions in 1986Q1 down to 4,313 institutions in 2026Q2, as shown in Figure 2 Panel A. This secular consolidation was driven by interstate banking deregulation and ongoing mergers. Across 24,064 institutions, our audit verified zero duplicate certificate and report date pairs and zero nonpositive asset records, while identifying 32 institutions with 33 internal timeline gap events that our downstream feature pipeline guards by enforcing contiguous quarterly indices. Removing 1,185 residual post terminal receivership filings submitted on or after terminal closure across 270 failed banks, while preserving 4,425 valid post assistance operating quarters, yields the clean Silver panel of 1,681,431 rows across 24,062 banks.",
    )

    add_figure(doc, "fig1_panel_coverage_and_failures.png", "Figure 2. Forty Two Years of United States Banking Distress and Structural Consolidation from 1984Q1 to 2026Q2")

    add_heading_2(doc, "Negative Equity as Technical Balance Sheet Insolvency")
    add_body_p(
        doc,
        "Our data quality audit identified 15,659 raw bank quarters and 14,605 clean Silver bank quarters where book equity was strictly negative. In generic corporate data cleaning, negative equity might be discarded as an anomaly. In banking supervision, negative book equity represents severe technical balance sheet insolvency. During the 1980s Savings and Loan crisis, regulatory forbearance allowed insolvent thrifts to operate for years. Discarding negative equity banks would introduce severe survivorship bias. We preserve all 14,605 negative equity observations in the Silver panel and encode a dedicated structural indicator for technical insolvency.",
    )

    add_heading_2(doc, "The Call Report Flow Accumulation Sawtooth and Basel Sentinel Decoding")
    add_body_p(
        doc,
        "A primary pitfall in Call Report data engineering is the income statement convention. While balance sheet items represent point in time stocks, income statement items such as net income, interest income, and noninterest expense are reported as cumulative Year to Date flows. As shown in Table 2 and Figure 3 Panel A, raw median net income scales mechanically from 181.0 thousand dollars in Q1 to 373.0 thousand dollars in Q2, 542.0 thousand dollars in Q3, and 683.0 thousand dollars in Q4, while official annualized return on assets remains flat between 0.94 percent and 0.99 percent. Unadjusted ratios produce artificial seasonal spikes. We eliminate this sawtooth through within bank decumulation, calculating quarterly flow as current net income minus prior quarter net income for quarters two, three, and four.",
    )

    t2_headers = ["Quarter Endpoint", "Reporting Banks", "Cumulative Net Income", "Decumulated Net Income", "Cumulative Interest", "Annualized Return on Assets"]
    t2_rows = [
        ["Q1 March 31", "427,873", "USD 181.0 k", "USD 181.0 k", "USD 1,400.0 k", "0.97 %"],
        ["Q2 June 30", "424,438", "USD 373.0 k", "USD 192.0 k", "USD 2,868.0 k", "0.99 %"],
        ["Q3 September 30", "417,003", "USD 542.0 k", "USD 169.0 k", "USD 4,200.0 k", "0.99 %"],
        ["Q4 December 31", "413,302", "USD 683.0 k", "USD 141.0 k", "USD 5,616.0 k", "0.94 %"],
    ]
    add_styled_table(
        doc,
        t2_headers,
        t2_rows,
        col_widths=[1.5, 1.0, 1.1, 1.1, 1.1, 1.1],
        align_right_cols=[1, 2, 3, 4, 5],
        caption="Table 2. Call Report Cumulative Year to Date Flow Sawtooth across Calendar Quarters",
    )

    add_figure(doc, "fig3_data_quality_and_ytd_sawtooth.png", "Figure 3. Call Report Accounting Pitfalls and Regulatory Regime Shifts across Four Decades")

    add_callout(
        doc,
        "In accordance with our independent external audit, we disclose three institutional findings. First, the 12,459 positive failure quarters comprise 10,870 formal receivership closures and 1,589 open bank assistance transactions. We preserve 4,425 subsequent operating quarters for assisted institutions, capturing 37 subsequent terminal failures. Second, three institutions experienced two recorded failure events during the 1980s conservatorship era, retaining 26 operating quarters between events. Third, the public API returns float value 0.0 for unreported risk based capital prior to Basel I in 1990 and under the 2020s Community Bank Leverage Ratio opt out. We decode these sentinel zeros to missing values and instantiate a dedicated regulatory missingness indicator.",
        title="Audit Disclosures on Resolutions, Multi Event Thrifts, and Sentinel Values",
    )

    # -------------------------------------------------------------------------
    # SECTION 3: SCALABLE PROCESSING, QUERY PLANS AND SPARK
    # -------------------------------------------------------------------------
    add_heading_1(doc, "3. Scalable Data Processing, Query Plans and Distributed Execution")
    add_body_p(
        doc,
        "To evaluate computational scalability for banking supervision, we implemented identical end to end feature transformations across four computational engines: vectorized Pandas, Polars Eager, Polars Lazy, and Apache Spark PySpark. The pipeline executes five operations: filtering active reporting quarters, calculating contiguous quarterly shifts, decumulating income statement flows, computing bank level rolling statistics, and aggregating contemporaneous state quarter peer averages.",
    )

    t3_headers = ["Workload Pipeline", "Processing Engine", "Median Duration", "Mean Duration", "Peak Memory RSS", "Relative Speedup"]
    t3_rows = [
        ["End to End Pipeline", "Pandas Vectorized", "9,154.08 ms", "9,110.35 ms", "2,919.61 MB", "1.00x Baseline"],
        ["End to End Pipeline", "Polars Eager", "2,512.86 ms", "2,529.66 ms", "3,757.09 MB", "3.64x Speedup"],
        ["End to End Pipeline", "Polars Lazy", "1,140.23 ms", "1,195.05 ms", "4,154.97 MB", "8.03x Speedup"],
        ["Selective Pushdown", "Pandas Vectorized", "1,749.68 ms", "1,740.85 ms", "4,012.39 MB", "1.00x Baseline"],
        ["Selective Pushdown", "Polars Eager", "158.46 ms", "183.06 ms", "3,079.82 MB", "11.04x Speedup"],
        ["Selective Pushdown", "Polars Lazy", "43.43 ms", "52.46 ms", "2,827.81 MB", "40.29x Speedup"],
        ["PySpark Join Method", "SortMergeJoin Wide", "4,035.05 ms", "4,082.49 ms", "JVM Managed", "1.00x Baseline"],
        ["PySpark Join Method", "BroadcastHashJoin Fast", "1,180.76 ms", "1,158.55 ms", "JVM Managed", "3.42x Speedup"],
    ]
    add_styled_table(
        doc,
        t3_headers,
        t3_rows,
        col_widths=[1.6, 1.4, 1.0, 1.0, 1.0, 0.9],
        align_right_cols=[2, 3, 4, 5],
        caption="Table 3. Cross Engine Computational Benchmark across Pandas, Polars, and PySpark",
    )

    add_figure(doc, "fig4_engine_benchmark_and_scaling.png", "Figure 4. Computational Systems Evaluation, Query Pushdown, Amdahl Law Scaling, and Stress Limits")

    add_heading_2(doc, "Mathematical Parity and Physical Execution Optimization")
    add_body_p(
        doc,
        "A strict four way numerical verification across all 1,681,431 rows confirms exact mathematical parity, summarized in Table 3 and Figure 4. Polars Eager and Polars Lazy match with exact zero difference. Both Pandas versus Polars Lazy and PySpark versus Polars Lazy match with maximum absolute difference bounded at 5.68e-14, which is well within double precision machine epsilon.",
    )
    add_body_p(
        doc,
        "Inspection of the Polars physical execution plan reveals why Polars Lazy achieves its 40.29 fold speedup. The engine pushes column pruning directly into the Parquet reader, loading only 5 of 33 columns. It pushes predicate filters into row group readers, skipping whole row groups of small institutions without memory allocation. Furthermore, common subplan elimination optimizes contiguous lag calculations by hashing institution identifiers once.",
    )
    add_body_p(
        doc,
        "In PySpark, joining the 1.68 million row panel with structural tables under standard SortMergeJoin requires 4,035.05 milliseconds due to wide network shuffle partitioning. Applying broadcast hints on the small metadata tables eliminates the shuffle exchange entirely, reducing execution time to 1,180.76 milliseconds, representing a 3.42 fold speedup.",
    )

    add_heading_2(doc, "Amdahl Law Scaling and Single Node Memory Boundaries")
    add_body_p(
        doc,
        "Evaluating multi core scaling from 1 to 20 threads yields an empirical execution drop from 5.636 seconds at 1 thread down to 0.974 seconds at 16 threads, achieving a 5.79 fold speedup before plateauing due to memory bus saturation. Fitting Amdahl Law yields an estimated parallel fraction of 87.68 percent with serial fraction of 12.32 percent and theoretical speedup limit of 8.12 fold.",
    )
    add_body_p(
        doc,
        "Synthetic scale up stress testing up to ten fold expansion with 16.83 million rows demonstrates that Pandas exhausts available memory and fails above 3.37 million rows. In contrast, Polars Lazy processes 16.83 million rows in 39.77 seconds with 8.36 gigabytes of resident memory. On a standard 16 gigabyte laptop, Polars operates comfortably up to approximately 19.6 million rows. On a 128 gigabyte workstation, Polars scales up to 194 million rows. Because four decades of bank level filings comprise only 1.68 million rows, single node Polars is substantially faster, simpler, and more economical than a Spark cluster. A distributed cluster is justified only when ingesting granular loan level transactions or tick level payment feeds exceeding 20 million records.",
    )

    # -------------------------------------------------------------------------
    # SECTION 4: LEAK FREE CAMELS FEATURE PIPELINE
    # -------------------------------------------------------------------------
    add_heading_1(doc, "4. Leak Free CAMELS Feature Pipeline and Preprocessing Isolation")
    add_body_p(
        doc,
        "We construct a comprehensive 40 feature matrix covering all six dimensions of the CAMELS regulatory framework. The features capture capital adequacy loss cushions, nonperforming asset burdens, overhead efficiency, operating profitability, liquidity cushions, trailing trajectories, and contemporaneous state peer deviations.",
    )

    t4_headers = ["Pillar", "Feature Count", "Core Feature Examples", "Economic and Supervisory Mechanism"]
    t4_rows = [
        ["C Capital", "6", "Equity to assets, Tier 1 leverage, Total risk based capital", "Loss absorption cushion protecting against severe loan losses"],
        ["A Asset Quality", "8", "Texas Ratio proxy, Nonperforming loan share, OREO ratio", "Delinquency burden and foreclosed property drag on capital"],
        ["M Management", "7", "Cost to income ratio, Overhead to assets, Asset growth", "Operational efficiency and excessive credit expansion pace"],
        ["E Earnings", "6", "Return on assets, Net interest margin, Provision share", "Core profitability and earnings capacity to replenish capital"],
        ["L Liquidity", "5", "Liquid assets to total assets, Brokered deposits share", "Structural funding stability and run risk from hot money"],
        ["S Trajectory", "7", "Equity change over 4Q, Rolling ROA volatility, State peer gap", "Balance sheet degradation rate and regional economic stress"],
        ["Charter Class", "1", "Primary regulatory charter class with 10 dummy categories", "Supervisory jurisdiction across OCC, Federal Reserve, and FDIC"],
    ]
    add_styled_table(
        doc,
        t4_headers,
        t4_rows,
        col_widths=[1.2, 0.9, 2.3, 2.5],
        caption="Table 4. Summary of the 40 Feature CAMELS Matrix across Supervisory Pillars",
    )

    add_figure(doc, "fig5_camels_auc_and_importance.png", "Figure 5. CAMELS Feature Information Content and Multivariable Model Attribution")

    add_heading_2(doc, "Chronological Regimes and Inside Fold Pipeline Isolation")
    add_body_p(
        doc,
        "To prevent cross split leakage, data are partitioned chronologically into four distinct regimes. The Development partition spans 1984Q1 to 2006Q4 with 1,193,702 bank quarters and 10,225 failure quarters across 2,658 failed banks. The Embargo partition spans calendar year 2007 with 34,754 rows, quarantining the one year forward target of the final Development filings. The Test partition covers the Global Financial Crisis from 2008Q1 to 2014Q4 with 212,645 rows and 2,052 failure quarters across 526 failed banks, evaluated once. The Post Test diagnostic era spans 2015Q1 to 2025Q2 with 222,801 rows and 129 failure quarters.",
    )
    add_body_p(
        doc,
        "All data preprocessing is encapsulated in a Scikit Learn Pipeline fitted strictly on training data. The pipeline executes quantile winsorization at 0.5 percent and 99.5 percent percentiles, robust scaling with median centering and interquartile range scaling on non null observations, median imputation filling centered missing values with zero, and one hot encoding of charter categories, yielding 49 total numeric inputs.",
    )

    add_callout(
        doc,
        "To prove zero data leakage with mathematical certainty, we corrupted all post 2006 Silver panel data by multiplying core financial variables by one million and adding 999,999. Running the complete feature engineering and preprocessing pipeline on both clean and corrupted data yielded an identical Development feature matrix with maximum absolute difference bounded at 1.39e-17, verified in reports/final_audit_certificate.json. This confirms zero look ahead bias or backward rolling contamination.",
        title="Adversarial Silver to Gold Perturbation Invariance Proof",
    )

    # -------------------------------------------------------------------------
    # SECTION 5: MODELLING, PURGED CROSS VALIDATION AND EVALUATION
    # -------------------------------------------------------------------------
    add_heading_1(doc, "5. Modelling, Purged Cross Validation and Cost Driven Evaluation")
    add_body_p(
        doc,
        "Standard cross validation on longitudinal data creates look ahead leakage because future training rows predict past validation rows. We design a four fold Purged Expanding Window Cross Validation scheme inside the Development partition, enforcing a strict four quarter purge buffer between training endpoints and validation starts as detailed in Table 5.",
    )

    t5_headers = ["CV Fold", "Training Window", "Purge Buffer", "Validation Window", "Historical Regime", "Validation Rows", "Failures", "LightGBM PR AUC"]
    t5_rows = [
        ["Fold 1", "1984Q1 to 1986Q4", "4 Quarters 1987", "1988Q1 to 1989Q4", "Savings and Loan Crisis Peak", "132,387", "3,912", "0.6392"],
        ["Fold 2", "1984Q1 to 1989Q4", "4 Quarters 1990", "1991Q1 to 1992Q4", "FDICIA Basel I Transition", "116,928", "1,173", "0.3593"],
        ["Fold 3", "1984Q1 to 1992Q4", "4 Quarters 1993", "1994Q1 to 1999Q4", "1990s Great Moderation", "277,614", "134", "0.2405"],
        ["Fold 4", "1984Q1 to 1999Q4", "4 Quarters 2000", "2001Q1 to 2006Q4", "Pre Crisis Credit Expansion", "222,962", "81", "0.0757"],
    ]
    add_styled_table(
        doc,
        t5_headers,
        t5_rows,
        col_widths=[0.8, 1.2, 1.0, 1.2, 1.3, 0.8, 0.6, 0.8],
        align_right_cols=[5, 6, 7],
        caption="Table 5. Four Fold Purged Expanding Window Cross Validation on Development Era",
    )

    add_heading_2(doc, "Out of Time Test Benchmark on the Global Financial Crisis")
    add_body_p(
        doc,
        "We evaluate candidate architectures once on the blind Global Financial Crisis Test partition covering 2008Q1 to 2014Q4 with 212,645 rows and 2,052 positive failure quarters across 526 failed banks. To demonstrate distributed training, src/models/spark_mllib_demo.py additionally trains Spark MLlib Gradient Boosted Trees and Spark MLlib Logistic Regression strictly on DEV, achieving Test PR AUC of 0.60184 and 0.30609 respectively. Full comparison results are summarized in Table 6 and Figure 6 Panel A.",
    )

    t6_headers = ["Model Architecture", "Test PR AUC", "Test ROC AUC", "Brier Score", "Recall Top 1%", "Recall Top 2%", "Recall Top 5%", "Fit Duration"]
    t6_rows = [
        ["Random Baseline", "0.00965", "0.50000", "0.00956", "1.00 %", "2.00 %", "5.00 %", "Not Applicable"],
        ["Supervisory Texas Ratio Rule", "0.46622", "0.95554", "0.00906", "49.71 %", "68.96 %", "84.70 %", "1.32 s"],
        ["Spark MLlib Logistic Regression", "0.30609", "0.95356", "Not Evaluated", "Not Evaluated", "Not Evaluated", "Not Evaluated", "71.81 s"],
        ["Spark MLlib Gradient Boosted Trees", "0.60184", "0.97326", "Not Evaluated", "Not Evaluated", "Not Evaluated", "Not Evaluated", "113.16 s"],
        ["Logistic Regression CAMELS", "0.62327", "0.97517", "0.00662", "61.01 %", "78.22 %", "90.40 %", "59.05 s"],
        ["HistGradientBoosting CAMELS", "0.62823", "0.97994", "0.00574", "60.04 %", "79.53 %", "91.28 %", "44.21 s"],
        ["Champion LightGBM CAMELS", "0.64537", "0.98280", "0.00568", "62.48 %", "80.17 %", "92.06 %", "34.07 s"],
    ]
    add_styled_table(
        doc,
        t6_headers,
        t6_rows,
        col_widths=[1.7, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.8],
        align_right_cols=[1, 2, 3, 4, 5, 6, 7],
        caption="Table 6. Single Shot Out of Time Evaluation on Global Financial Crisis Test Set",
    )

    add_figure(doc, "fig6_model_evaluation_and_cost_threshold.png", "Figure 6. Model Discrimination, Supervisory Inspection Allocation, and Economic Cost Optimization")

    add_heading_2(doc, "Institutional Drivers of Texas Ratio Performance Shift")
    add_body_p(
        doc,
        "A notable empirical finding is that the single ratio Texas Ratio baseline achieves a Development cross validation score of 0.2361 but surges to 0.4662 during the Global Financial Crisis. This reflects a major historical policy change. During the 1980s Savings and Loan crisis, insolvent thrifts operated under widespread forbearance for several years, resulting in a low four quarter failure precision of 25.7 percent for institutions with elevated Texas ratios. In 1991, Congress passed the FDIC Improvement Act, mandating Prompt Corrective Action rules that require regulators to close critically undercapitalized banks within 90 days. During the Global Financial Crisis, banks breaching Texas ratio thresholds were shut down rapidly, doubling four quarter precision to 46.9 percent. Even so, multivariate LightGBM outperforms the single ratio heuristic by 38.4 percent by incorporating brokered deposit reliance, capital erosion rates, and overhead inefficiency.",
    )

    add_heading_2(doc, "Asymmetric Loss Economics and Optimal Threshold Selection")
    add_body_p(
        doc,
        "Evaluating early warning models at default probability cutoff 0.50 is economically inappropriate. Missing an impending failure incurs severe Deposit Insurance Fund resolution losses, whereas auditing a sound bank incurs a modest examination expense. From historical Development resolutions, median resolution loss is 10,399.0 thousand dollars, which amortizes across four pre failure quarters to 2,599.75 thousand dollars per quarter. Calibrating off site audit cost conservatively at 100.0 thousand dollars yields an asymmetric cost ratio of 26 to 1. Theoretical Bayesian decision analysis implies an optimal cutoff of 0.0370. Empirical cost minimization on Development predictions selects optimal threshold 0.0401.",
    )

    t7_headers = ["Decision Metric", "Default Cutoff 0.50", "Optimal Cutoff 0.0401", "Supervisory Economic Gain"]
    t7_rows = [
        ["True Positives Count", "628 banks", "1,679 banks", "Plus 1,051 identified distressed banks"],
        ["False Positives Count", "121 audits", "2,909 audits", "Plus 2,788 supervisory audits required"],
        ["False Negatives Misses", "1,424 misses", "373 misses", "Minus 1,051 missed bank failures"],
        ["Institution Recall Rate", "30.60 %", "81.82 %", "Plus 51.22 percentage points coverage"],
        ["Dollar Weighted Asset Recall", "23.73 %", "79.85 %", "Plus 56.12 percentage points asset coverage"],
        ["Audit Inspection Precision", "83.84 %", "36.60 %", "Minus 47.24 percentage points precision"],
        ["Net Regulatory Cost Savings", "USD 1,620.54 Million", "USD 4,074.08 Million", "Plus USD 2,453.54 Million net savings"],
    ]
    add_styled_table(
        doc,
        t7_headers,
        t7_rows,
        col_widths=[2.1, 1.4, 1.5, 1.9],
        align_right_cols=[1, 2],
        caption="Table 7. Economic Impact of Data Driven Threshold Optimization on Global Financial Crisis Test Set",
    )

    t8_headers = ["Cost Ratio FN to FP", "Optimal Cutoff", "Test True Pos", "Test False Pos", "Institution Recall", "Dollar Recall", "Net Savings Full Loss", "Net Savings Half Loss"]
    t8_rows = [
        ["Ratio 10 to 1", "0.0592", "1,600", "2,211", "77.97 %", "77.00 %", "USD 1,378.9 M", "USD 578.9 M"],
        ["Ratio 26 to 1 Baseline", "0.0401", "1,679", "2,909", "81.82 %", "79.85 %", "USD 4,074.1 M", "USD 1,891.6 M"],
        ["Ratio 50 to 1", "0.0163", "1,812", "5,206", "88.30 %", "89.37 %", "USD 8,539.4 M", "USD 4,009.4 M"],
        ["Ratio 100 to 1", "0.0115", "1,844", "6,323", "89.86 %", "91.07 %", "USD 17,807.7 M", "USD 8,587.7 M"],
        ["Ratio 250 to 1", "0.0020", "1,947", "15,969", "94.88 %", "96.98 %", "USD 47,078.1 M", "USD 22,740.6 M"],
    ]
    add_styled_table(
        doc,
        t8_headers,
        t8_rows,
        col_widths=[1.2, 0.7, 0.7, 0.7, 0.8, 0.8, 1.0, 1.0],
        align_right_cols=[1, 2, 3, 4, 5, 6, 7],
        caption="Table 8. Sensitivity of Optimal Threshold and Regulatory Cost Savings across Loss Ratios",
    )

    # -------------------------------------------------------------------------
    # SECTION 6: SYNTHESIS CONCLUSIONS AND VIETNAM SBV ADAPTATION
    # -------------------------------------------------------------------------
    add_heading_1(doc, "6. Synthesis Conclusions, Methodological Limitations and Vietnam Adaptation")
    
    add_heading_2(doc, "Evidence Based Answers to Core Research Questions")
    add_body_p(
        doc,
        "Four decades of Call Reports spanning 1,682,616 rows should be stored in quarter partitioned Apache Parquet with Snappy or Zstandard compression, shrinking footprint by 1.75 to 2.22 fold compared to raw text at 424.12 megabytes and accelerating selective pushdown queries to 31.9 milliseconds, which is 459.0 fold faster than Pandas full text reads and 22.6 fold faster than Polars full text scans. On a single laptop, Polars Lazy completes end to end panel engineering in 1,140 milliseconds, outperforming vectorized Pandas by 8.03 fold and scaling comfortably to 19.6 million rows on 16 gigabytes of RAM. A distributed Spark cluster is unwarranted for quarterly bank aggregates and becomes justified only for granular loan level credit registries or intraday payment streams exceeding 20 million records.",
        bold_prefix="Question 1 on Storage, Laptop Processing, and Cluster Threshold: ",
    )
    add_body_p(
        doc,
        "Our automated quality audit exposed six structural hazards: disguised sentinel zeros for unreported Basel capital prior to 1990 and under post 2020 CBLR opt outs, cumulative Year to Date income statement sawtooth patterns across calendar quarters, 14,605 negative book equity technical insolvencies, 1,185 post terminal receivership filings across 270 closed banks, 33 reporting timeline gaps across 32 institutions, and 17,529 right censored quarters after June 2025. The pipeline resolves every hazard without look ahead bias by decoding sentinel zeros with an explicit missingness flag, performing strictly causal within bank flow decumulation guarded by contiguous quarterly indices, preserving negative equity and post assistance operating quarters, quarantining post 2025Q2 labels and calendar year 2007 as an embargo gap, and fitting all winsorization, scaling, and imputation transformers strictly on training data with adversarial perturbation invariance bounded at 1.39e-17.",
        bold_prefix="Question 2 on Raw Data Quality Hazards and Zero Look Ahead Handling: ",
    )
    add_body_p(
        doc,
        "Yes. Validated via four fold purged expanding window cross validation and evaluated once on the blind Global Financial Crisis Test set from 2008 to 2014 covering 212,645 rows and 526 failed banks, champion LightGBM CAMELS achieves Test PR AUC of 0.6454 and ROC AUC of 0.9828, outperforming the Texas Ratio supervisory rule by 38.4 percent. At a top 2 percent inspection budget, the model captures 80.2 percent of failing banks and 78.1 percent of failing assets. Operating at the empirically calibrated cutoff of 0.0401 yields 4,074.08 million dollars in net regulatory savings, outperforming the naive 0.50 cutoff by 2,453.54 million dollars.",
        bold_prefix="Question 3 on Honest Early Warning Validation and Supervisory Economics: ",
    )

    add_heading_2(doc, "Critical Methodological Limitations")
    add_body_p(
        doc,
        "First, quarterly reporting lags fail to capture rapid digital bank runs. During the March 2023 regional banking crisis, Silicon Valley Bank lost 42 billion dollars in deposits within ten hours via mobile banking and social media panic. Off site quarterly Call Reports cannot substitute for intraday payment feeds. Second, Call Report core schemas omit off balance sheet duration mismatches and unrealized losses on held to maturity securities. Third, linear cost models do not capture the systemic contagion costs of massive institution failures. Fourth, structural regime shifts require continuous model recalibration across economic cycles.",
    )

    add_heading_2(doc, "Strategic Blueprint for the State Bank of Vietnam")
    add_body_p(
        doc,
        "Adapting this architecture to the State Bank of Vietnam requires aligning United States regulatory concepts with Vietnamese legal frameworks as mapped in Table 9.",
    )

    t9_headers = ["Supervisory Dimension", "United States FDIC Regime", "State Bank of Vietnam NHNN Environment"]
    t9_rows = [
        ["Supervisory Cross Section", "4,300 to 18,000 commercial banks and thrifts", "Approximately 49 commercial banks and 1,200 Peoples Credit Funds"],
        ["Loan Classification Standards", "Current Expected Credit Loss CECL and nonaccrual loans", "Circular 11/2021/TT-NHNN 5 group asset classification system"],
        ["Capital Adequacy Standards", "Basel III and Community Bank Leverage Ratio framework", "Circular 41/2016/TT-NHNN Basel II Capital Adequacy Ratio standard"],
        ["Resolution Modality", "Receivership closures and purchase and assumption sales", "Special Control intervention and compulsory zero VND bank acquisitions"],
        ["Central Credit Information", "Quarterly Call Reports via FFIEC central data repository", "Credit Information Center daily loan and interbank monitoring feeds"],
    ]
    add_styled_table(
        doc,
        t9_headers,
        t9_rows,
        col_widths=[1.5, 2.7, 2.7],
        caption="Table 9. Institutional Translation from United States FDIC to State Bank of Vietnam",
    )

    add_callout(
        doc,
        "We recommend a dual tier early warning architecture for Vietnam. For commercial banks, train specialized risk models incorporating real estate credit exposure, corporate bond holdings, interbank contagion networks, and daily Credit Information Center loan performance. For the 1,200 Peoples Credit Funds, deploy the exact Polars and LightGBM pipeline developed here to monitor local credit quality quarterly. The prediction target should be defined as formal intervention under Special Control pursuant to the Law on Credit Institutions.",
        title="State Bank of Vietnam Lakehouse Implementation Blueprint",
    )

    # -------------------------------------------------------------------------
    # PAGE BREAK FOR SECTION 7: TECHNICAL APPENDIX
    # -------------------------------------------------------------------------
    doc.add_page_break()

    add_heading_1(doc, "7. Technical Appendix and References")

    add_heading_2(doc, "Complete CAMELS Feature Taxonomy and Univariate Power")
    add_body_p(
        doc,
        "Table 10 details the complete 40 feature CAMELS taxonomy engineered in src/features/build_features.py, documenting economic formulas, missingness rates, and univariate discriminative power on the Development partition.",
    )

    t10_headers = ["Feature Identifier", "Pillar", "Formula Specification", "Missing Rate", "Failing Median", "Healthy Median", "DEV AUC"]
    t10_rows = [
        ["eq_to_asset", "C", "Equity capital divided by total assets", "0.24 %", "0.0074", "0.0871", "0.9516"],
        ["tier1_leverage_ratio", "C", "Tier 1 capital leverage ratio decoded", "6.48 %", "1.7043", "8.7628", "0.9501"],
        ["total_rbc_ratio", "C", "Total risk based capital ratio decoded", "35.25 %", "2.8209", "15.4111", "0.9602"],
        ["equity_to_loans", "C", "Equity capital divided by gross loans", "1.15 %", "0.0114", "0.1488", "0.9374"],
        ["technical_insolvency_flag", "C", "Binary flag indicating negative book equity", "0.00 %", "0.0000", "0.0000", "0.7056"],
        ["basel_missing_flag", "C", "Binary flag indicating missing Basel capital", "0.00 %", "1.0000", "0.0000", "0.6957"],
        ["texas_ratio_proxy", "A", "Distressed assets divided by capital and reserves", "0.24 %", "4.0986", "0.0768", "0.9597"],
        ["npl_to_loans", "A", "Nonaccrual loans divided by gross loans", "1.12 %", "0.0924", "0.0098", "0.9061"],
        ["nonaccrual_to_loans", "A", "Nonaccrual assets divided by gross loans", "7.35 %", "0.0624", "0.0044", "0.8833"],
        ["pastdue90_to_loans", "A", "Loans 90 days past due divided by loans", "7.35 %", "0.0126", "0.0018", "0.7193"],
        ["pastdue30_89_to_loans", "A", "Loans 30 to 89 days past due divided by loans", "81.66 %", "0.0218", "0.0090", "0.6765"],
        ["oreo_to_assets", "A", "Other real estate owned divided by assets", "0.24 %", "0.0312", "0.0006", "0.8853"],
        ["reserve_to_loans", "A", "Loan loss reserves divided by gross loans", "1.15 %", "0.0286", "0.0124", "0.7389"],
        ["coverage_ratio", "A", "Loan loss reserves divided by nonaccrual loans", "8.32 %", "0.3360", "1.1000", "0.7864"],
        ["cost_to_income_ratio", "M", "Quarterly noninterest expense to operating revenue", "2.76 %", "1.2911", "0.6578", "0.8935"],
        ["overhead_to_assets_q", "M", "Annualized noninterest expense to assets", "0.70 %", "0.0399", "0.0299", "0.6553"],
        ["real_estate_to_loans", "M", "Real estate loans divided by gross loans", "0.91 %", "0.6703", "0.5697", "0.5673"],
        ["ci_to_loans", "M", "Commercial and industrial loans to gross loans", "0.91 %", "0.1123", "0.1343", "0.5265"],
        ["consumer_to_loans", "M", "Consumer loans divided by gross loans", "1.13 %", "0.0975", "0.1225", "0.5574"],
        ["log_assets", "M", "Natural logarithm of total assets in thousands", "0.00 %", "11.0214", "11.0374", "0.5002"],
        ["asset_growth_4q", "M", "Trailing four quarter asset growth rate", "7.78 %", "-0.0803", "0.0569", "0.8025"],
        ["roa", "E", "Annualized official return on assets ratio", "0.00 %", "-2.6390", "1.0181", "0.9291"],
        ["roe", "E", "Annualized official return on equity ratio", "1.36 %", "-67.6900", "11.3100", "0.8835"],
        ["nim", "E", "Annualized official net interest margin ratio", "0.00 %", "2.4113", "4.2524", "0.7730"],
        ["netinc_q_to_assets", "E", "Decumulated quarterly net income to assets", "0.70 %", "-0.0264", "0.0100", "0.9057"],
        ["provision_to_loans_q", "E", "Decumulated quarterly loan provisions to loans", "4.76 %", "0.0128", "0.0022", "0.6811"],
        ["net_chargeoff_to_loans_q", "E", "Decumulated quarterly net chargeoffs to loans", "7.76 %", "0.0162", "0.0007", "0.7371"],
        ["liquid_to_asset", "L", "Cash and investment securities to assets", "0.00 %", "0.1955", "0.3169", "0.7206"],
        ["cash_to_asset", "L", "Cash and balances due from banks to assets", "0.00 %", "0.0416", "0.0472", "0.5643"],
        ["loans_to_deposits", "L", "Gross loans divided by total deposits", "0.62 %", "0.7060", "0.6886", "0.5177"],
        ["bro_to_dep", "L", "Brokered deposits divided by total deposits", "1.90 %", "0.0000", "0.0000", "0.6186"],
        ["non_deposit_funding_to_asset", "L", "Non deposit liabilities to total assets", "0.24 %", "0.0307", "0.0170", "0.5987"],
        ["eq_to_asset_diff_4q", "S", "Trailing four quarter change in equity to assets", "8.00 %", "-0.0305", "0.0009", "0.8672"],
        ["npl_to_loans_diff_4q", "S", "Trailing four quarter change in nonaccrual ratio", "8.81 %", "0.0176", "-0.0001", "0.6500"],
        ["liquid_to_asset_diff_4q", "S", "Trailing four quarter change in liquid assets", "7.78 %", "-0.0019", "-0.0056", "0.5143"],
        ["bro_to_dep_diff_4q", "S", "Trailing four quarter change in brokered deposits", "9.58 %", "0.0000", "0.0000", "0.5045"],
        ["roa_rolling_4q_mean", "S", "Trailing four quarter mean return on assets", "5.88 %", "-2.1601", "1.0210", "0.9436"],
        ["roa_rolling_4q_std", "S", "Trailing four quarter standard deviation of ROA", "5.88 %", "1.5369", "0.1541", "0.9071"],
        ["eq_vs_state_peer", "S", "Deviation from contemporaneous state peer mean", "0.24 %", "-0.0726", "-0.0106", "0.8787"],
        ["BKCLASS", "Charter", "Regulatory charter class with ten categories", "0.51 %", "Categorical", "Categorical", "Categorical"],
    ]
    add_styled_table(
        doc,
        t10_headers,
        t10_rows,
        col_widths=[1.5, 0.6, 2.2, 0.7, 0.7, 0.7, 0.5],
        align_right_cols=[3, 4, 5, 6],
        caption="Table 10. Complete 40 Feature CAMELS Taxonomy and Development Univariate Power",
    )

    add_heading_2(doc, "Automated Verification Suite Results")
    add_body_p(
        doc,
        "The complete pipeline was audited using src/audit/verify_end_to_end.py, verifying 47 hard mathematical invariants with zero errors. Part 1 verified all 170 quarters, 1,682,616 raw rows, schema conformity, and 5 layout benchmarks. Part 2 verified 1,681,431 Silver panel rows, zero duplicate primary keys, removal of 1,185 post terminal filings, negative equity preservation, and complete censoring post June 2025. Part 3 verified 4 way numerical parity across 1,681,431 rows, physical pushdown query plans, Spark join benchmarks, and Amdahl scaling. Part 4 verified 40 CAMELS features, contiguous flow decumulation, and the adversarial perturbation proof. Part 5 verified 4 fold purged expanding window separation, empirical cost estimation, threshold optimization, and blind Test evaluation. Final audit verdict: 47 of 47 checks pass with zero data leakage.",
    )

    add_heading_2(doc, "Academic Literature and Regulatory Corpus")
    add_body_p(
        doc,
        "1. Cassidy, Gerard. 1989. The Texas Ratio: Predicting Bank Failure in Distressed Real Estate Markets. Commercial Banking Research, Austin, Texas.",
    )
    add_body_p(
        doc,
        "2. Cole, Rebel A., and Jeffery W. Gunther. 1995. Separating the Likelihood and Timing of Bank Failure. Journal of Banking and Finance 19: 1073 to 1089.",
    )
    add_body_p(
        doc,
        "3. DeYoung, Robert. 2003. The Failure of New Entrants to Commercial Banking. Journal of Financial Services Research 23: 7 to 31.",
    )
    add_body_p(
        doc,
        "4. Federal Deposit Insurance Corporation. 1998. Managing the Crisis: The FDIC and RTC Experience 1980 to 1994. FDIC Division of Resolutions, Washington, DC.",
    )
    add_body_p(
        doc,
        "5. Federal Deposit Insurance Corporation. 2017. Crisis and Response: An FDIC History 2008 to 2013. FDIC Division of Resolutions and Receiverships, Washington, DC.",
    )
    add_body_p(
        doc,
        "6. Federal Deposit Insurance Corporation. 2026. BankFind Suite API Documentation and Call Report Data Dictionaries 1984 to 2026. FDIC Open Data Platform, Washington, DC.",
    )
    add_body_p(
        doc,
        "7. Lopez, Jose A. 1999. Using CAMELS Ratings to Monitor Bank Conditions. Federal Reserve Bank of San Francisco Economic Letter 1999 19: 1 to 3.",
    )
    add_body_p(
        doc,
        "8. National Assembly of Vietnam. 2024. Law on Credit Institutions No. 32/2024/QH15. National Assembly Publishing House, Hanoi, Vietnam.",
    )
    add_body_p(
        doc,
        "9. State Bank of Vietnam. 2016. Circular No. 41/2016/TT-NHNN Regulating Capital Adequacy Ratio for Commercial Banks. Hanoi, Vietnam.",
    )
    add_body_p(
        doc,
        "10. State Bank of Vietnam. 2019. Circular No. 22/2019/TT-NHNN Regulating Prudential Ratios in Operations of Credit Institutions. Hanoi, Vietnam.",
    )
    add_body_p(
        doc,
        "11. State Bank of Vietnam. 2021. Circular No. 11/2021/TT-NHNN on Classification of Assets and Levels of Provisioning for Credit Institutions. Hanoi, Vietnam.",
    )
    add_body_p(
        doc,
        "12. Wheelock, David C., and Paul W. Wilson. 2000. Why Do Banks Disappear? The Determinants of United States Bank Failures and Acquisitions. Review of Economics and Statistics 82: 127 to 138.",
    )
    add_body_p(
        doc,
        "13. Course Reference Starter Script: fdic_download_starter.py provided by the Faculty of Banking and Finance, Course BFT65B Big Data Analytics in Finance, National Economics University, Hanoi, Vietnam, 2026.",
    )
    add_body_p(
        doc,
        "14. Python Software Ecosystem: Polars 2.1.0, Apache Spark PySpark 4.3.0, PyArrow 23.0.1, Scikit Learn 1.8.0, LightGBM 4.7.0, and Pandas 2.3.3.",
    )

    add_heading_2(doc, "Artificial Intelligence Tool Transparency Statement")
    add_body_p(
        doc,
        "In accordance with Section 4 of the examination specification, the author transparently discloses the use of generative artificial intelligence assistance. Claude and Gemini language models were utilized via an agentic pair programming environment to assist in designing automated test assertions in src/audit/verify_end_to_end.py, formatting visualization parameters in matplotlib, and proofreading technical documentation. All econometric formulations, causal point in time target designs, CAMELS accounting logic, SBV regulatory adaptations, physical benchmarks, and analytical conclusions were conceptualized, verified, and defended by the author. Every numerical value in this report matches the deterministic artifacts generated on disk.",
    )

    doc.save(str(OUTPUT_DOCX))
    print(f"Report generated successfully -> {OUTPUT_DOCX}")
    print(f"File size: {OUTPUT_DOCX.stat().st_size:,} bytes")
    print("=" * 70)


if __name__ == "__main__":
    generate_full_word_report()
