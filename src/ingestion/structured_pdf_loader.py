"""Extract searchable text and reviewable visual elements from a PDF."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from statistics import median
from typing import Any

import pdfplumber

from src.ingestion.pdfplumber_loader import load_pdf_with_layout, _line_text
from src.ingestion.text_cleaner import clean_text
from src.preprocessing.layout_detector import group_words_into_lines


CAPTION_PATTERN = re.compile(
    r"^(?:figure|fig\.?|table|chart|diagram)\s*\d*\s*[:.\-]?\s*",
    re.IGNORECASE,
)
CHART_HINT_PATTERN = re.compile(
    r"\b(chart|graph|performance curve|accuracy|uncertainty|"
    r"number of samples|training loss|epochs)\b",
    re.IGNORECASE,
)
VISUAL_MODEL = "gemini-3.8-flash"


def _bbox(obj: dict) -> tuple[float, float, float, float] | None:
    """Get a valid pdfplumber bbox in x0, top, x1, bottom order."""
    try:
        x0 = float(obj["x0"])
        x1 = float(obj["x1"])
        top = float(obj["top"])
        bottom = float(obj["bottom"])
        if not all(math.isfinite(value) for value in (x0, x1, top, bottom)):
            return None
        if x1 <= x0 or bottom <= top:
            return None
        return (x0, top, x1, bottom)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _inside_bbox(box: tuple[float, float, float, float], region: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = box
    rx0, rtop, rx1, rbottom = region
    center_x = (x0 + x1) / 2
    center_y = (top + bottom) / 2
    return rx0 <= center_x <= rx1 and rtop <= center_y <= rbottom


def _overlap_ratio(box: tuple[float, float, float, float], region: tuple[float, float, float, float]) -> float:
    x0, top, x1, bottom = box
    rx0, rtop, rx1, rbottom = region
    area = max(1.0, (x1 - x0) * (bottom - top))
    overlap_width = max(0.0, min(x1, rx1) - max(x0, rx0))
    overlap_height = max(0.0, min(bottom, rbottom) - max(top, rtop))
    return overlap_width * overlap_height / area


def _table_markdown(rows: list[list[Any]]) -> str:
    """Format extracted table cells as a readable Markdown table."""
    cleaned = [
        [" ".join(str(cell or "").split()).replace("|", "\\|") for cell in row]
        for row in rows
    ]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header = cleaned[0]
    separator = ["---"] * width
    body = cleaned[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _extract_tables(page: Any) -> list[dict]:
    """Extract likely real tables, filtering out page-sized text candidates."""
    found = []
    try:
        candidates = page.find_tables()
    except Exception:
        return found

    page_area = max(1.0, float(page.width) * float(page.height))
    seen_boxes = set()
    for table in candidates:
        try:
            box = tuple(float(value) for value in table.bbox)
            rows = table.extract() or []
        except Exception:
            continue

        if len(box) != 4:
            continue
        x0, top, x1, bottom = box
        if x1 <= x0 or bottom <= top:
            continue
        box = (x0, top, x1, bottom)
        box_key = tuple(round(value, 1) for value in box)
        if box_key in seen_boxes:
            continue
        seen_boxes.add(box_key)

        normalized_rows = [
            [" ".join(str(cell or "").split()) for cell in row]
            for row in rows
        ]
        width = max((len(row) for row in normalized_rows), default=0)
        multi_cell_rows = sum(sum(bool(cell) for cell in row) >= 2 for row in normalized_rows)
        area_ratio = ((x1 - x0) * (bottom - top)) / page_area

        # Requiring several rows and columns avoids labeling ordinary paragraphs
        # and small Yes/No flowchart labels as tables.
        if len(normalized_rows) < 3 or width < 2 or multi_cell_rows < 3 or area_ratio > 0.65:
            continue

        row_widths = [len(row) for row in normalized_rows if any(row)]
        consistent = bool(row_widths) and max(row_widths) - min(row_widths) <= 1
        text = _table_markdown(normalized_rows)
        if not text:
            continue

        found.append({
            "bounding_box": list(box),
            "text": text,
            "rows": len(normalized_rows),
            "columns": width,
            "needs_review": not consistent,
            "extraction_method": "pdfplumber_table",
        })
    return found


def _vector_regions(page: Any, table_boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Group nearby PDF drawing objects into likely diagram or chart regions."""
    objects = []
    for name in ("rects", "curves", "lines"):
        try:
            candidates = getattr(page, name, []) or []
        except Exception:
            candidates = []
        for item in candidates:
            box = _bbox(item)
            if box and not any(_inside_bbox(box, table_box) for table_box in table_boxes):
                objects.append(box)

    if not objects:
        return []

    # Connected components: objects close to one another usually form a figure.
    parents = list(range(len(objects)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    padding = 16.0
    for left in range(len(objects)):
        ax0, atop, ax1, abottom = objects[left]
        for right in range(left + 1, len(objects)):
            bx0, btop, bx1, bbottom = objects[right]
            if (
                ax0 <= bx1 + padding and bx0 <= ax1 + padding
                and atop <= bbottom + padding and btop <= abottom + padding
            ):
                union(left, right)

    groups: dict[int, list[tuple[float, float, float, float]]] = {}
    for index, box in enumerate(objects):
        groups.setdefault(find(index), []).append(box)

    page_area = max(1.0, float(page.width) * float(page.height))
    regions = []
    for group in groups.values():
        if len(group) < 8:
            continue
        region = (
            min(box[0] for box in group),
            min(box[1] for box in group),
            max(box[2] for box in group),
            max(box[3] for box in group),
        )
        x0, top, x1, bottom = region
        area_ratio = (x1 - x0) * (bottom - top) / page_area
        if 0.015 <= area_ratio <= 0.80:
            regions.append(region)
    return regions


def _find_visuals(page: Any, table_boxes: list[tuple[float, float, float, float]]) -> list[dict]:
    """Return raster image and grouped vector visual regions."""
    visuals = []
    page_area = max(1.0, float(page.width) * float(page.height))

    try:
        images = page.images or []
    except Exception:
        images = []
    for image_number, image in enumerate(images, start=1):
        box = _bbox(image)
        if not box:
            continue
        x0, top, x1, bottom = box
        if (x1 - x0) * (bottom - top) / page_area < 0.002:
            continue
        if any(_overlap_ratio(box, table_box) > 0.7 for table_box in table_boxes):
            continue
        visuals.append({"box": box, "element_type": "image", "object_number": image_number})

    for box in _vector_regions(page, table_boxes):
        if any(_overlap_ratio(box, image["box"]) > 0.35 for image in visuals):
            continue
        visuals.append({"box": box, "element_type": "diagram", "object_number": len(visuals) + 1})

    return visuals


def _extract_word_lines(page: Any) -> list[dict]:
    """Build line records that retain coordinates and basic font information."""
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"]) or []
    except Exception:
        words = page.extract_words() or []

    valid_words = []
    for word in words:
        box = _bbox(word)
        if box and str(word.get("text", "")).strip():
            valid_words.append({**word, "x0": box[0], "top": box[1], "x1": box[2], "bottom": box[3]})

    records = []
    for line in group_words_into_lines(valid_words):
        text = _line_text(line).strip()
        if not text:
            continue
        sizes = [float(word["size"]) for word in line if isinstance(word.get("size"), (int, float))]
        fonts = [str(word.get("fontname", "")) for word in line]
        records.append({
            "text": text,
            "words": line,
            "bounding_box": [
                min(word["x0"] for word in line),
                min(word["top"] for word in line),
                max(word["x1"] for word in line),
                max(word["bottom"] for word in line),
            ],
            "font_size": sum(sizes) / len(sizes) if sizes else 0.0,
            "bold": any("bold" in font.casefold() for font in fonts),
        })
    return records


def _order_lines(lines: list[dict], layout: dict) -> list[dict]:
    """Apply the existing left-column then right-column reading order."""
    boundaries = layout.get("column_boundaries") or {}
    split = boundaries.get("split_x")
    if split is None:
        return sorted(lines, key=lambda line: (line["bounding_box"][1], line["bounding_box"][0]))

    full_width_tops = {line["top"] for line in layout.get("full_width_lines", [])}
    left_lines: list[dict] = []
    right_lines: list[dict] = []
    output: list[dict] = []

    def flush() -> None:
        output.extend(sorted(left_lines, key=lambda item: item["bounding_box"][1]))
        output.extend(sorted(right_lines, key=lambda item: item["bounding_box"][1]))
        left_lines.clear()
        right_lines.clear()

    for line in sorted(lines, key=lambda item: (item["bounding_box"][1], item["bounding_box"][0])):
        top = round(line["bounding_box"][1], 1)
        if top in full_width_tops:
            flush()
            output.append(line)
            continue

        left_words = [word for word in line["words"] if (word["x0"] + word["x1"]) / 2 < split]
        right_words = [word for word in line["words"] if (word["x0"] + word["x1"]) / 2 >= split]
        for words, bucket in ((left_words, left_lines), (right_words, right_lines)):
            if not words:
                continue
            bucket.append({
                **line,
                "text": _line_text(words).strip(),
                "words": words,
                "bounding_box": [
                    min(word["x0"] for word in words),
                    min(word["top"] for word in words),
                    max(word["x1"] for word in words),
                    max(word["bottom"] for word in words),
                ],
            })

    flush()
    return output


def _make_text_elements(
    page_number: int,
    lines: list[dict],
    excluded_boxes: list[tuple[float, float, float, float]],
    visual_boxes: list[tuple[float, float, float, float]] | None = None,
) -> list[dict]:
    """Create paragraph, heading, and caption elements from positioned lines."""
    kept = [
        line for line in lines
        if not any(_inside_bbox(tuple(line["bounding_box"]), box) for box in excluded_boxes)
    ]
    body_sizes = [line["font_size"] for line in kept if line["font_size"] > 0]
    typical_size = median(body_sizes) if body_sizes else 0.0
    visual_boxes = visual_boxes or []

    line_types = []
    for line in kept:
        text = line["text"].strip()
        word_count = len(text.split())
        explicit_caption = bool(CAPTION_PATTERN.match(text))
        box = line["bounding_box"]
        center_x = (box[0] + box[2]) / 2
        adjacent_to_visual = any(
            (
                0 <= visual_box[1] - box[3] <= 22
                or 0 <= box[1] - visual_box[3] <= 22
            )
            and abs(center_x - (visual_box[0] + visual_box[2]) / 2)
            <= max(18.0, (visual_box[2] - visual_box[0]) * 0.20)
            for visual_box in visual_boxes
        )
        short_visual_caption = (
            adjacent_to_visual
            and word_count <= 14
            and len(text) <= 110
            and (not typical_size or line["font_size"] <= typical_size * 1.15)
        )
        is_caption = explicit_caption or short_visual_caption
        is_heading = (
            not is_caption
            and typical_size > 0
            and line["font_size"] >= typical_size * 1.22
            and word_count <= 14
            and len(text) <= 110
            and not text.endswith((".", ";", ","))
        )
        line_types.append((line, "caption" if is_caption else "heading" if is_heading else "paragraph"))

    elements = []
    current_lines: list[dict] = []

    def flush_paragraph() -> None:
        if not current_lines:
            return
        text = clean_text("\n".join(line["text"].strip() for line in current_lines))
        if text:
            boxes = [line["bounding_box"] for line in current_lines]
            elements.append({
                "page_number": page_number,
                "element_type": "paragraph",
                "text": text,
                "bounding_box": [
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ],
                "extraction_method": "pdfplumber_words_and_coordinates",
                "needs_review": False,
            })
        current_lines.clear()

    previous_line = None
    for line, element_type in line_types:
        box = line["bounding_box"]
        if element_type != "paragraph":
            flush_paragraph()
            elements.append({
                "page_number": page_number,
                "element_type": element_type,
                "text": line["text"].strip(),
                "bounding_box": box,
                "extraction_method": "pdfplumber_font_and_text",
                "needs_review": short_visual_caption and not explicit_caption,
            })
            previous_line = None
            continue

        if current_lines and previous_line:
            previous_box = previous_line["bounding_box"]
            vertical_gap = box[1] - previous_box[3]
            typical_line_height = max(1.0, previous_box[3] - previous_box[1])
            if vertical_gap > typical_line_height * 0.65:
                flush_paragraph()

        current_lines.append(line)
        previous_line = line

    flush_paragraph()
    return elements


def _text_in_region(lines: list[dict], box: tuple[float, float, float, float]) -> str:
    return " ".join(
        line["text"] for line in lines
        if _inside_bbox(tuple(line["bounding_box"]), box)
    ).strip()


def _render_region(page: Any, box: tuple[float, float, float, float], output_path: Path) -> bytes:
    """Render and save one visual crop, returning the PNG bytes."""
    clipped_box = (
        max(0.0, box[0]),
        max(0.0, box[1]),
        min(float(page.width), box[2]),
        min(float(page.height), box[3]),
    )
    if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
        raise ValueError("Visual region falls outside the PDF page.")
    image = page.crop(clipped_box).to_image(resolution=130)
    buffer = BytesIO()
    image.original.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    return png_bytes


def _ocr_image(png_bytes: bytes) -> str:
    """OCR a visual only when its PDF text layer has little readable text."""
    executable = shutil.which("tesseract")
    if not executable:
        return ""
    result = subprocess.run(
        [executable, "stdin", "stdout", "--psm", "6"],
        input=png_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", errors="replace").strip()


def _describe_visual(png_bytes: bytes, page_number: int, element_type: str) -> str:
    """Ask the existing Gemini client for a cautious, short visual description."""
    from PIL import Image

    from src.embeddings.gemini_embeddings import client

    image = Image.open(BytesIO(png_bytes))
    prompt = f"""Describe this PDF {element_type} in one or two factual sentences.
Only report text, shapes, labels, and relationships that are clearly visible.
Do not invent values, axis labels, arrows, or relationships. For charts, do not
estimate exact numbers. If it cannot be interpreted reliably, say so explicitly.
This crop comes from PDF page {page_number}."""
    response = client.models.generate_content(
        model=VISUAL_MODEL,
        contents=[prompt, image],
    )
    return (response.text or "").strip()


def _element_id(page_number: int, number: int) -> str:
    return f"page_{page_number}_element_{number}"


def extract_structured_pdf(
    pdf_path: str | Path,
    visuals_dir: str | Path,
    describe_visuals: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Return page documents for chunking and a flat list of structured elements.

    Visual descriptions are opt-in because they make Gemini API calls. Crops and
    OCR are still produced locally; uninterpretable visuals remain reviewable.
    """
    path = Path(pdf_path).expanduser().resolve()
    visual_root = Path(visuals_dir).expanduser().resolve()
    base_documents = load_pdf_with_layout(path)
    page_documents = []
    all_elements = []

    with pdfplumber.open(path) as pdf:
        for page_number, (page, base_document) in enumerate(zip(pdf.pages, base_documents), start=1):
            raw_lines = _extract_word_lines(page)
            ordered_lines = _order_lines(raw_lines, base_document.get("layout", {}))
            tables = _extract_tables(page)
            table_boxes = [tuple(table["bounding_box"]) for table in tables]
            visuals = _find_visuals(page, table_boxes)
            visual_boxes = [visual["box"] for visual in visuals]

            if base_document.get("layout", {}).get("diagnostics", {}).get("uncertain", False):
                text_elements = [{
                    "page_number": page_number,
                    "element_type": "paragraph",
                    "text": base_document.get("page_content", "").strip(),
                    "bounding_box": [0.0, 0.0, float(page.width), float(page.height)],
                    "extraction_method": base_document["metadata"].get(
                        "extraction_method", "pdfplumber_page"
                    ),
                    "needs_review": True,
                }]
            else:
                text_elements = _make_text_elements(
                    page_number,
                    ordered_lines,
                    table_boxes + visual_boxes,
                    visual_boxes=visual_boxes,
                )

            # If coordinate-based filtering loses substantial text, retain the
            # loader's complete page text as a reviewable fallback element.
            # Count lines before region filtering: text inside a table or chart
            # is still preserved by its corresponding structured element.
            extracted_chars = sum(len(line["text"]) for line in ordered_lines)
            original_text = base_document.get("page_content", "").strip()
            if original_text and extracted_chars < len(original_text) * 0.55:
                text_elements = [{
                    "page_number": page_number,
                    "element_type": "paragraph",
                    "text": original_text,
                    "bounding_box": [0.0, 0.0, float(page.width), float(page.height)],
                    "extraction_method": base_document["metadata"].get("extraction_method", "pdfplumber_page"),
                    "needs_review": True,
                }]

            page_elements = []
            for element in text_elements:
                page_elements.append(element)

            for table_number, table in enumerate(tables, start=1):
                page_elements.append({
                    "page_number": page_number,
                    "element_type": "table",
                    "text": table["text"],
                    "bounding_box": table["bounding_box"],
                    "extraction_method": table["extraction_method"],
                    "needs_review": table["needs_review"],
                    "table_number": table_number,
                    "row_count": table["rows"],
                    "column_count": table["columns"],
                })

            visual_elements = []
            for visual_number, visual in enumerate(visuals, start=1):
                box = visual["box"]
                visual_type = visual["element_type"]
                embedded_text = _text_in_region(raw_lines, box)
                nearby_text = " ".join(
                    line["text"] for line in raw_lines
                    if abs(line["bounding_box"][1] - box[3]) <= 22
                    or abs(line["bounding_box"][3] - box[1]) <= 22
                )
                if visual_type == "diagram" and CHART_HINT_PATTERN.search(nearby_text):
                    visual_type = "chart"

                visual_id = _element_id(page_number, len(page_elements) + 1)
                safe_box = "_".join(str(round(value)) for value in box)
                image_path = visual_root / f"page_{page_number:02d}_{visual_type}_{visual_number:02d}_{safe_box}.png"
                png_bytes = b""
                render_error = False
                try:
                    png_bytes = _render_region(page, box, image_path)
                except Exception:
                    render_error = True

                ocr_text = embedded_text
                if png_bytes and len(embedded_text) < 8:
                    try:
                        ocr_text = _ocr_image(png_bytes)
                    except Exception:
                        ocr_text = ""

                description = ""
                page_area = max(1.0, float(page.width) * float(page.height))
                region_area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / page_area
                describe_this_visual = (
                    visual_type != "image" or region_area_ratio >= 0.008
                )
                if describe_visuals and png_bytes and describe_this_visual:
                    print(f"Describing {visual_type} on page {page_number}...")
                    try:
                        description = _describe_visual(png_bytes, page_number, visual_type)
                    except Exception:
                        description = ""

                text_parts = []
                if description:
                    text_parts.append(description)
                if ocr_text:
                    text_parts.append(f"Visible text: {ocr_text}")
                if not text_parts:
                    text_parts.append(
                        f"{visual_type.capitalize()} on PDF page {page_number}; "
                        "see the saved visual crop."
                    )

                review = render_error or not description
                if visual_type == "chart":
                    # Numeric interpretation of graphs needs a human check.
                    review = True
                if description and re.search(r"cannot be interpreted|unclear|not clear", description, re.I):
                    review = True

                visual_element = {
                    "element_id": visual_id,
                    "page_number": page_number,
                    "element_type": visual_type,
                    "text": "\n".join(text_parts),
                    "bounding_box": list(box),
                    "extraction_method": (
                        "pdfplumber_render+gemini_vision"
                        if description else "pdfplumber_render+embedded_text_or_ocr"
                        if png_bytes else "pdfplumber_visual_objects"
                    ),
                    "needs_review": review,
                    "visual_path": (
                        image_path.relative_to(visual_root.parent).as_posix()
                        if not render_error else ""
                    ),
                    "ocr_text": ocr_text,
                    "description": description,
                }
                page_elements.append(visual_element)
                visual_elements.append(visual_element)

            # Captions are preserved as their own element and linked to the
            # nearest visual on the same page when one is available.
            for caption in [item for item in page_elements if item["element_type"] == "caption"]:
                caption_box = tuple(caption["bounding_box"])
                if visual_elements:
                    closest = min(
                        visual_elements,
                        key=lambda item: abs(item["bounding_box"][1] - caption_box[3]),
                    )
                    caption["related_element_id"] = closest["element_id"]

            # Assign stable IDs after all page elements are created.
            for element_number, element in enumerate(page_elements, start=1):
                element["element_id"] = element.get("element_id") or _element_id(page_number, element_number)

            # In a single column, combine all blocks by their vertical position.
            # For column pages, keep the loader's left/right text order intact.
            if not (base_document.get("layout", {}).get("column_boundaries") or {}).get("split_x"):
                page_elements.sort(
                    key=lambda item: (
                        tuple(item.get("bounding_box", [0, 0, 0, 0]))[1],
                        tuple(item.get("bounding_box", [0, 0, 0, 0]))[0],
                    )
                )

            # Preserve reading order for text. Structured non-text elements are
            # labeled separately and never passed through the paragraph cleaner.
            prose = []
            pending_heading = ""
            special = []
            for element in page_elements:
                if element["element_type"] == "heading":
                    pending_heading = element["text"]
                elif element["element_type"] == "paragraph":
                    cleaned = " ".join(element["text"].split())
                    if cleaned:
                        prose.append(f"{pending_heading}\n{cleaned}" if pending_heading else cleaned)
                    pending_heading = ""
                elif element["element_type"] == "caption":
                    prose.append(element["text"])
                    pending_heading = ""
                else:
                    special.append(element)
            if pending_heading:
                prose.append(pending_heading)

            for element in sorted(special, key=lambda item: tuple(item["bounding_box"])[1]):
                label = element["element_type"].capitalize()
                prose.append(f"{label} (page {page_number}):\n{element['text']}")

            page_text = "\n\n".join(part for part in prose if part.strip()).strip()
            page_metadata = dict(base_document["metadata"])
            page_metadata["extraction_mode"] = "structured_pdfplumber"
            original_words = set(re.findall(r"[\w'-]+", original_text.casefold()))
            structured_words = set(re.findall(r"[\w'-]+", page_text.casefold()))
            coverage = (
                len(original_words & structured_words) / len(original_words)
                if original_words else 1.0
            )
            coordinate_coverage = coverage
            if original_text and coordinate_coverage < 0.70:
                # Fall back to the proven page extraction when coordinate
                # filtering drops too much text, while retaining typed tables
                # and visuals as separate reviewable elements.
                special_elements = [
                    element for element in page_elements
                    if element["element_type"] not in {"paragraph", "heading", "caption"}
                ]
                fallback_element = {
                    "page_number": page_number,
                    "element_type": "paragraph",
                    "text": original_text,
                    "bounding_box": [0.0, 0.0, float(page.width), float(page.height)],
                    "extraction_method": base_document["metadata"].get(
                        "extraction_method", "pdfplumber_page"
                    ),
                    "needs_review": True,
                }
                page_elements = [fallback_element] + special_elements
                for element_number, element in enumerate(page_elements, start=1):
                    element["element_id"] = element.get("element_id") or _element_id(page_number, element_number)

                page_text = original_text
                for element in sorted(special_elements, key=lambda item: tuple(item["bounding_box"])[1]):
                    page_text += (
                        f"\n\n{element['element_type'].capitalize()} (page {page_number}):\n"
                        f"{element['text']}"
                    )

                final_words = set(re.findall(r"[\w'-]+", page_text.casefold()))
                coverage = (
                    len(original_words & final_words) / len(original_words)
                    if original_words else 1.0
                )

            page_metadata["coordinate_text_word_coverage"] = round(coordinate_coverage, 3)
            page_metadata["text_word_coverage"] = round(coverage, 3)
            page_metadata["needs_review"] = coordinate_coverage < 0.70 or any(
                element["needs_review"] for element in page_elements
            )
            page_documents.append({
                "page_number": page_number,
                "total_pages": len(pdf.pages),
                "page_content": page_text,
                "metadata": page_metadata,
            })
            all_elements.extend(page_elements)

    return page_documents, all_elements


def write_structured_outputs(
    page_documents: list[dict],
    elements: list[dict],
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Write structured elements and readable page text as separate files."""
    json_path = Path(json_path)
    text_path = Path(text_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source": page_documents[0]["metadata"]["source"] if page_documents else "",
        "page_count": page_documents[0]["total_pages"] if page_documents else 0,
        "elements": elements,
    }
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    readable_text = "\n\n".join(
        f"--- PAGE {document['page_number']} ---\n{document['page_content']}"
        for document in page_documents
    )
    text_path.write_text(readable_text, encoding="utf-8")


def count_element_types(elements: list[dict]) -> dict[str, int]:
    """Return a compact count by element type for the pipeline summary."""
    return dict(sorted(Counter(element["element_type"] for element in elements).items()))
