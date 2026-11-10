"""Small, conservative layout detector for pdfplumber pages."""

from __future__ import annotations

from math import isfinite
from statistics import median
from typing import Any


def get_valid_words(page: Any) -> tuple[list[dict], int]:
    """Return positioned words and the number whose coordinates were unusable."""
    try:
        raw_words = page.extract_words() or []
    except Exception:
        return [], 1

    words = []
    invalid_count = 0
    width = float(getattr(page, "width", 0) or 0)
    height = float(getattr(page, "height", 0) or 0)

    for word in raw_words:
        try:
            x0, x1 = float(word["x0"]), float(word["x1"])
            top = float(word["top"])
            bottom = float(word.get("bottom", top))
            if not all(isfinite(value) for value in (x0, x1, top, bottom)):
                raise ValueError("non-finite coordinate")
            if x1 < x0:
                x0, x1 = x1, x0
            if width and (x0 < -width or x1 > width * 2):
                raise ValueError("word is outside the page")
            if height and (top < -height or bottom > height * 2):
                raise ValueError("word is outside the page")
            if not str(word.get("text", "")).strip():
                raise ValueError("word has no text")
            words.append({**word, "x0": x0, "x1": x1, "top": top, "bottom": bottom})
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_count += 1

    return words, invalid_count


def group_words_into_lines(words: list[dict], tolerance: float = 2.8) -> list[list[dict]]:
    """Group words on nearby baselines and sort each line left to right."""
    lines: list[list[dict]] = []
    baselines: list[float] = []

    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if lines and abs(word["top"] - baselines[-1]) <= tolerance:
            lines[-1].append(word)
            baselines[-1] = sum(item["top"] for item in lines[-1]) / len(lines[-1])
        else:
            lines.append([word])
            baselines.append(word["top"])

    for line in lines:
        line.sort(key=lambda item: item["x0"])
    return lines


def _visual_counts(page: Any) -> tuple[int, int]:
    """Return image count and count of vector drawing objects on the page."""
    try:
        image_count = len(page.images or [])
    except Exception:
        image_count = 0

    vector_count = 0
    for name in ("rects", "curves", "lines"):
        try:
            vector_count += len(getattr(page, name, []) or [])
        except Exception:
            continue
    return image_count, vector_count


def _find_column_split(page_width: float, lines: list[list[dict]]) -> dict:
    """Find one repeated central gutter, or report that evidence is weak."""
    gap_centers = []
    gap_widths = []
    for line in lines:
        central_gaps = []
        for left, right in zip(line, line[1:]):
            gap = right["x0"] - left["x1"]
            center = (left["x1"] + right["x0"]) / 2
            if page_width * 0.30 <= center <= page_width * 0.70:
                if gap >= max(12, page_width * 0.045):
                    central_gaps.append((gap, center))
        if central_gaps:
            gap, center = max(central_gaps)
            gap_widths.append(gap)
            gap_centers.append(center)

    if not gap_centers:
        return {"split_x": None, "support": 0.0, "paired_rows": 0, "median_gap": None}

    split = median(gap_centers)
    consistent = [
        (center, gap)
        for center, gap in zip(gap_centers, gap_widths)
        if abs(center - split) <= page_width * 0.06
    ]
    paired_rows = len(consistent)
    support = paired_rows / max(1, len(lines))

    return {
        "split_x": split,
        "support": support,
        "paired_rows": paired_rows,
        "median_gap": median(gap for _, gap in consistent) if consistent else None,
    }


def detect_page_layout(
    page: Any,
    page_number: int = 1,
    words: list[dict] | None = None,
    invalid_word_count: int | None = None,
) -> dict:
    """Classify a page using repeated text gutters and visible page objects.

    ``words`` may be supplied by the loader so pdfplumber only has to extract
    and validate word coordinates once per page.
    """
    if words is None or invalid_word_count is None:
        words, invalid_word_count = get_valid_words(page)

    words = words or []
    lines = group_words_into_lines(words)
    image_count, vector_count = _visual_counts(page)
    has_visuals = image_count > 0 or vector_count >= 15
    has_any_visual = image_count > 0 or vector_count > 0
    diagnostics = {
        "word_count": len(words),
        "line_count": len(lines),
        "image_count": image_count,
        "vector_object_count": vector_count,
        "words_without_usable_coordinates": invalid_word_count,
    }

    if not words:
        try:
            has_text = bool((page.extract_text() or "").strip())
        except Exception:
            has_text = False
        if has_any_visual:
            layout_type, confidence = "table_or_diagram", 0.80
        elif has_text:
            layout_type, confidence = "single_column", 0.30
        else:
            layout_type, confidence = "empty", 0.98
        diagnostics["uncertain"] = layout_type == "single_column" or invalid_word_count > 0
        return {
            "page_number": page_number,
            "layout_type": layout_type,
            "confidence": confidence,
            "column_boundaries": None,
            "diagnostics": diagnostics,
            "full_width_lines": [],
        }

    page_width = float(getattr(page, "width", 0) or 0)
    geometry = _find_column_split(page_width, lines) if page_width > 0 else {
        "split_x": None, "support": 0.0, "paired_rows": 0, "median_gap": None
    }
    split = geometry["split_x"]
    left_chars = right_chars = 0
    if split is not None:
        for word in words:
            if (word["x0"] + word["x1"]) / 2 < split:
                left_chars += len(str(word["text"]))
            else:
                right_chars += len(str(word["text"]))

    balance = min(left_chars, right_chars) / max(1, left_chars + right_chars)
    enough_repeated_gaps = geometry["paired_rows"] >= max(4, round(len(lines) * 0.10))
    columns_detected = bool(
        split is not None
        and enough_repeated_gaps
        and min(left_chars, right_chars) >= 90
        and balance >= 0.20
    )

    full_width_lines = []
    if columns_detected:
        for line in lines:
            left_edge = min(word["x0"] for word in line)
            right_edge = max(word["x1"] for word in line)
            crosses_split = left_edge < split - page_width * 0.035 and right_edge > split + page_width * 0.035
            has_wide_gutter = any(
                left["x1"] <= split <= right["x0"]
                and right["x0"] - left["x1"] >= page_width * 0.045
                for left, right in zip(line, line[1:])
            )
            if crosses_split and not has_wide_gutter:
                full_width_lines.append({
                    "top": round(min(word["top"] for word in line), 1),
                    "text": " ".join(str(word["text"]) for word in line),
                })

    diagram_heavy = vector_count >= 15 and len(lines) < 25
    if diagram_heavy or (has_visuals and len(words) < 100):
        layout_type, confidence = "table_or_diagram", 0.80
    elif columns_detected:
        layout_type = "mixed" if has_visuals or full_width_lines else "two_column"
        confidence = round(0.60 + min(geometry["support"], 0.50) * 0.45, 3)
    elif has_visuals:
        layout_type, confidence = "mixed", 0.60
    else:
        layout_type = "single_column"
        confidence = 0.78 if len(lines) >= 8 else 0.50

    boundaries = None
    if columns_detected:
        half_gap = (geometry["median_gap"] or 0) / 2
        boundaries = {
            "left_column_start": round(min(w["x0"] for w in words if w["x0"] < split), 2),
            "left_column_end": round(split - half_gap, 2),
            "right_column_start": round(split + half_gap, 2),
            "right_column_end": round(max(w["x1"] for w in words if w["x1"] > split), 2),
            "split_x": round(split, 2),
        }

    mixed_with_visuals = layout_type == "mixed" and (image_count > 0 or vector_count >= 15)
    diagnostics.update({
        "paired_gutter_rows": geometry["paired_rows"],
        "gutter_support": round(geometry["support"], 3),
        "left_text_characters": left_chars,
        "right_text_characters": right_chars,
        "uncertain": invalid_word_count > 0 or confidence < 0.65 or mixed_with_visuals,
    })
    if invalid_word_count:
        diagnostics["reason"] = "Some words had invalid coordinates."
    elif mixed_with_visuals:
        diagnostics["reason"] = "Mixed text and visual content uses normal page extraction."
    elif diagnostics["uncertain"]:
        diagnostics["reason"] = "Layout evidence is weak; use normal page extraction."

    return {
        "page_number": page_number,
        "layout_type": layout_type,
        "confidence": confidence,
        "column_boundaries": boundaries,
        "diagnostics": diagnostics,
        "full_width_lines": full_width_lines,
    }
