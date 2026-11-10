"""Create a compact Markdown report of PDF page layout predictions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pdfplumber

from src.preprocessing.layout_detector import detect_page_layout


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "documents" / "machine_learning_pdf.pdf"
DEFAULT_REPORT = PROJECT_ROOT / "extracted_text" / "pdf_layout_report.md"


def generate_layout_report(
    pdf_path: str | Path = DEFAULT_PDF,
    report_path: str | Path = DEFAULT_REPORT,
) -> Path:
    """Inspect a PDF and write one readable row per page."""
    pdf_path = Path(pdf_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        layouts = [
            detect_page_layout(page, page_number)
            for page_number, page in enumerate(pdf.pages, start=1)
        ]

    counts = Counter(layout["layout_type"] for layout in layouts)
    review_pages = [
        str(layout["page_number"])
        for layout in layouts
        if layout["diagnostics"].get("uncertain")
    ]
    report = [
        "# PDF layout report",
        "",
        f"PDF: `{pdf_path.name}`",
        f"Pages analyzed: {len(layouts)}",
        "",
        "## Layout counts",
        "",
    ]
    report.extend(f"- {name}: {count}" for name, count in sorted(counts.items()))
    report.extend([
        "",
        "Pages flagged for review: " + (", ".join(review_pages) or "None"),
        "",
        "| Page | Layout | Confidence | Split x | Words | Gutter rows | Review |",
        "|---:|---|---:|---:|---:|---:|---|",
    ])

    for layout in layouts:
        boundaries = layout["column_boundaries"] or {}
        diagnostics = layout["diagnostics"]
        split = f"{boundaries['split_x']:.1f}" if boundaries else "-"
        review = "Yes" if diagnostics.get("uncertain") else "No"
        report.append(
            f"| {layout['page_number']} | {layout['layout_type']} | "
            f"{layout['confidence']:.3f} | {split} | {diagnostics['word_count']} | "
            f"{diagnostics.get('paired_gutter_rows', 0)} | {review} |"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF layout report.")
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(generate_layout_report(args.pdf, args.report))


if __name__ == "__main__":
    main()
