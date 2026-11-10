"""Layout aware PDF text extraction using pdfplumber only."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pdfplumber

from src.preprocessing.layout_detector import detect_page_layout, get_valid_words, group_words_into_lines


def _line_text(words: list[dict]) -> str:
    """Join one visual line, correcting pdfplumber's reversed vertical English."""
    parts = []
    previous = None
    for word in words:
        text = str(word.get("text", "")).strip()
        if word.get("upright") is False and word.get("direction") in {"ttb", "btt"}:
            text = text[::-1]
        if text:
            gap = word["x0"] - previous["x1"] if previous else None
            if (
                parts
                and previous
                and previous.get("upright") is not False
                and word.get("upright") is not False
                and gap is not None
                and gap <= 0.5
            ):
                parts[-1] += text
            else:
                parts.append(text)
            previous = word
    return " ".join(parts)


def _repair_pdfplumber_text(text: str, words: list[dict]) -> str:
    """Fix only word fragments and reversed vertical labels supported by geometry."""
    for line in group_words_into_lines([word for word in words if word.get("upright") is not False]):
        for left, right in zip(line, line[1:]):
            gap = right["x0"] - left["x1"]
            left_text = str(left.get("text", ""))
            if gap <= 0.5 and len(left_text) == 1 and left_text.isupper():
                pattern = (
                    r"(?m)([◼▪■•]\s+)"
                    + re.escape(left_text)
                    + r"\s+"
                    + re.escape(str(right.get("text", "")))
                )
                replacement = rf"\1{left_text}{right.get('text', '')}"
                text = re.sub(pattern, replacement, text, count=1)

    rotated = [word for word in words if word.get("upright") is False]
    vertical_columns = {}
    for word in rotated:
        key = round(word["x0"])
        vertical_columns.setdefault(key, []).append(word)

    for column in vertical_columns.values():
        column.sort(key=lambda word: word["top"])
        groups = []
        for word in column:
            if groups and word["top"] - groups[-1][-1]["bottom"] <= 20:
                groups[-1].append(word)
            else:
                groups.append([word])

        for group in groups:
            if len(group) > 3:
                continue
            source_words = [str(word.get("text", "")) for word in group]
            source_words = [word for word in source_words if word]
            if not source_words:
                continue
            source_pattern = r"\s+".join(re.escape(word) for word in source_words)
            corrected = _line_text(sorted(group, key=lambda word: word["top"], reverse=True))
            text, _ = re.subn(source_pattern, corrected, text, count=1)
    return text


def _column_text(page: Any, words: list[dict], layout: dict) -> str:
    """Read a confident column page left to right, with full width lines between."""
    boundaries = layout.get("column_boundaries") or {}
    split = boundaries.get("split_x")
    if split is None:
        return (page.extract_text() or "").strip()

    flow_words = [word for word in words if word.get("upright") is not False]
    rotated_words = [word for word in words if word.get("upright") is False]
    lines = group_words_into_lines(flow_words)
    full_width_tops = {line["top"] for line in layout.get("full_width_lines", [])}

    left_lines = []
    right_lines = []
    output = []

    def flush_columns() -> None:
        if left_lines:
            output.extend(left_lines)
            left_lines.clear()
        if right_lines:
            output.extend(right_lines)
            right_lines.clear()

    for line in lines:
        top = round(min(word["top"] for word in line), 1)
        if top in full_width_tops:
            flush_columns()
            output.append(_line_text(line))
            continue

        left = [word for word in line if (word["x0"] + word["x1"]) / 2 < split]
        right = [word for word in line if (word["x0"] + word["x1"]) / 2 >= split]
        if left:
            left_lines.append(_line_text(left))
        if right:
            right_lines.append(_line_text(right))

    flush_columns()
    text = "\n\n".join(line for line in output if line).strip()
    rotated = [_line_text([word]) for word in rotated_words]
    rotated = [line for line in rotated if line]
    if rotated:
        text = f"{text}\n\n[Rotated diagram text]\n" + "\n".join(rotated)
    return text


def load_pdf_with_layout(pdf_path: str | Path) -> list[dict]:
    """Extract pages and retain the metadata shape expected by the chunker.

    Single column, diagram heavy, and uncertain pages use pdfplumber's normal
    page extraction. Only confident column or mixed pages use coordinate order.
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    documents = []
    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages, start=1):
            words, invalid_count = get_valid_words(page)
            layout = detect_page_layout(
                page,
                page_number,
                words=words,
                invalid_word_count=invalid_count,
            )
            use_columns = (
                layout["layout_type"] in {"two_column", "mixed"}
                and layout["column_boundaries"] is not None
                and not layout["diagnostics"].get("uncertain", False)
                and invalid_count == 0
            )
            if use_columns:
                page_text = _column_text(page, words, layout)
                extraction_method = "pdfplumber_columns"
            else:
                page_text = (page.extract_text() or "").strip()
                extraction_method = "pdfplumber_page"
            page_text = _repair_pdfplumber_text(page_text, words)

            metadata = {
                "source": path.name,
                "page": page_number,
                "total_pages": total_pages,
                "layout_type": layout["layout_type"],
                "layout_confidence": layout["confidence"],
                "extraction_method": extraction_method,
                # Keep Chroma-compatible scalar metadata values.
                "column_boundaries": json.dumps(layout["column_boundaries"])
                if layout["column_boundaries"] else "",
            }
            documents.append({
                # Retain the original pdfplumber loader result keys.
                "page_number": page_number,
                "total_pages": total_pages,
                "text": page_text,
                # The RAG chunker consumes these two fields.
                "page_content": page_text,
                "metadata": metadata,
                "layout": layout,
            })

    return documents


# Backwards compatible public name used by earlier scripts and tests.
extract_pdf_by_columns = load_pdf_with_layout
