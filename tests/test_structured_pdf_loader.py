import json
from pathlib import Path

import pdfplumber
import pytest

from src.chunking.text_chunker import create_recursive_chunks
from src.ingestion import structured_pdf_loader as structured


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = PROJECT_ROOT / "documents" / "machine_learning_pdf.pdf"


class FakePage:
    width = 600
    height = 800
    rects = []
    curves = []
    lines = []

    def __init__(self, words=None, images=None):
        self.words = words or []
        self.images = images or []

    def extract_words(self, **_kwargs):
        return self.words

    def find_tables(self):
        return []


class FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def make_word(text, x0, x1, top, size=10, fontname="Arial"):
    return {
        "text": text,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": top + size,
        "size": size,
        "fontname": fontname,
    }


def fake_document(page_number=1, text="A body paragraph."):
    return {
        "page_number": page_number,
        "total_pages": 1,
        "page_content": text,
        "metadata": {
            "source": "fixture.pdf",
            "page": page_number,
            "total_pages": 1,
            "layout_type": "single_column",
            "extraction_method": "pdfplumber_page",
        },
        "layout": {"column_boundaries": None, "full_width_lines": []},
    }


def patch_fake_pdf(monkeypatch, page, document):
    monkeypatch.setattr(structured, "load_pdf_with_layout", lambda _path: [document])
    monkeypatch.setattr(structured.pdfplumber, "open", lambda _path: FakePdf([page]))


def test_page_34_technology_readiness_table_has_rows_and_columns():
    with pdfplumber.open(SOURCE_PDF) as pdf:
        tables = structured._extract_tables(pdf.pages[33])

    assert len(tables) == 1
    table = tables[0]
    assert table["rows"] == 10
    assert table["columns"] == 2
    assert "TRL" in table["text"]
    assert "Description" in table["text"]
    assert "Basic principles observed and reported." in table["text"]
    assert table["needs_review"] is False


def test_narrative_page_flowchart_is_not_reported_as_a_table():
    with pdfplumber.open(SOURCE_PDF) as pdf:
        tables = structured._extract_tables(pdf.pages[8])

    assert tables == []


def test_text_lines_are_classified_as_heading_and_paragraph(monkeypatch, tmp_path):
    words = [
        make_word("Section", 40, 95, 40, size=16, fontname="Arial-Bold"),
        make_word("Heading", 100, 160, 40, size=16, fontname="Arial-Bold"),
        make_word("This", 40, 62, 70),
        make_word("is", 67, 77, 70),
        make_word("body", 82, 110, 70),
        make_word("text.", 115, 145, 70),
    ]
    page = FakePage(words=words)
    patch_fake_pdf(monkeypatch, page, fake_document(text="Section Heading This is body text."))

    documents, elements = structured.extract_structured_pdf(
        "fixture.pdf", tmp_path / "visuals"
    )

    assert [element["element_type"] for element in elements] == ["heading", "paragraph"]
    assert documents[0]["page_content"].startswith("Section Heading\n")
    assert documents[0]["metadata"]["page"] == 1


@pytest.mark.parametrize("failure", ["render", "ocr"])
def test_visual_processing_failures_are_flagged_for_review(monkeypatch, tmp_path, failure):
    page = FakePage(
        words=[make_word("Body", 40, 70, 40)],
        images=[{"x0": 200, "x1": 350, "top": 200, "bottom": 350}],
    )
    patch_fake_pdf(monkeypatch, page, fake_document(text="Body"))

    if failure == "render":
        def fail_render(*_args, **_kwargs):
            raise OSError("could not render")
        monkeypatch.setattr(structured, "_render_region", fail_render)
    else:
        monkeypatch.setattr(structured, "_render_region", lambda *_args: b"png bytes")

        def fail_ocr(_png_bytes):
            raise RuntimeError("OCR failed")
        monkeypatch.setattr(structured, "_ocr_image", fail_ocr)

    _documents, elements = structured.extract_structured_pdf(
        "fixture.pdf", tmp_path / "visuals"
    )
    visual = next(element for element in elements if element["element_type"] == "image")

    assert visual["needs_review"] is True
    assert visual["page_number"] == 1
    assert visual["bounding_box"] == [200, 200, 350, 350]


def test_empty_page_returns_empty_text_without_failing(monkeypatch, tmp_path):
    patch_fake_pdf(monkeypatch, FakePage(), fake_document(text=""))

    documents, elements = structured.extract_structured_pdf(
        "fixture.pdf", tmp_path / "visuals"
    )

    assert documents[0]["page_content"] == ""
    assert elements == []


def test_structured_json_output_and_recursive_chunk_metadata(tmp_path):
    documents = [{
        "page_number": 1,
        "total_pages": 1,
        "page_content": "Heading\nA body paragraph that continues. " * 50,
        "metadata": {"source": "fixture.pdf", "page": 1, "total_pages": 1},
    }]
    elements = [{
        "element_id": "page_1_element_1",
        "page_number": 1,
        "element_type": "heading",
        "text": "Heading",
        "bounding_box": [0, 0, 100, 20],
        "extraction_method": "test",
        "needs_review": False,
    }]
    json_path = tmp_path / "structured.json"
    text_path = tmp_path / "structured.txt"

    structured.write_structured_outputs(documents, elements, json_path, text_path)
    output = json.loads(json_path.read_text(encoding="utf-8"))
    chunks = create_recursive_chunks(
        documents,
        chunk_size=200,
        overlap=40,
        chunk_id_prefix="structured_page",
    )

    assert output["page_count"] == 1
    assert output["elements"][0]["bounding_box"] == [0, 0, 100, 20]
    assert "--- PAGE 1 ---" in text_path.read_text(encoding="utf-8")
    assert chunks[0]["chunk_id"].startswith("structured_page_1_chunk_")
    assert all(chunk["metadata"]["overlap"] <= 40 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        recorded = current["metadata"]["overlap"]
        assert previous["page_content"][-recorded:] == current["page_content"][:recorded] if recorded else True
