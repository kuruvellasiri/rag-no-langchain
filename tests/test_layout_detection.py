from src.chunking.text_chunker import create_recursive_chunks
from src.ingestion import pdfplumber_loader
from src.preprocessing.layout_detector import detect_page_layout


class FakePage:
    width = 600
    height = 800
    images = []
    rects = []
    curves = []
    lines = []

    def __init__(self, words=None, text=""):
        self.words = words or []
        self.text = text

    def extract_words(self):
        return self.words

    def extract_text(self):
        return self.text


def word(text, x0, x1, top, bottom=None):
    return {
        "text": text,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": bottom if bottom is not None else top + 8,
    }


def two_column_words(with_heading=False, rows=12):
    words = []
    if with_heading:
        words.extend([
            word("Full", 50, 75, 30), word("width", 78, 115, 30),
            word("heading", 118, 175, 30), word("spans", 178, 220, 30),
            word("both", 223, 255, 30), word("columns", 258, 315, 30),
        ])
    for row in range(rows):
        y = 100 + row * 18
        words.extend([
            word(f"left{row}", 50, 92, y), word("column", 96, 145, y),
            word("body", 149, 180, y), word(f"right{row}", 340, 390, y),
            word("column", 394, 443, y), word("body", 447, 478, y),
        ])
    return words


class FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def open_fake_pdf(monkeypatch, pages):
    monkeypatch.setattr(
        pdfplumber_loader.pdfplumber,
        "open",
        lambda _path: FakePdf(pages),
    )


def test_empty_page_is_empty():
    result = detect_page_layout(FakePage(), page_number=4)
    assert result["page_number"] == 4
    assert result["layout_type"] == "empty"


def test_image_only_page_is_not_empty():
    page = FakePage()
    page.images = [{"width": 250, "height": 300}]
    result = detect_page_layout(page, page_number=5)
    assert result["layout_type"] == "table_or_diagram"


def test_vector_only_page_is_not_empty():
    page = FakePage()
    page.lines = [{}]
    assert detect_page_layout(page)["layout_type"] == "table_or_diagram"


def test_single_column_page_is_not_split_at_midpoint():
    words = []
    for row in range(12):
        y = 100 + row * 18
        words.extend([
            word(f"line{row}", 50, 90, y), word("continuous", 95, 175, y),
            word("sentence", 180, 260, y), word("across", 265, 320, y),
            word("center", 325, 390, y),
        ])
    result = detect_page_layout(FakePage(words), page_number=1)
    assert result["layout_type"] == "single_column"
    assert result["column_boundaries"] is None


def test_two_column_page_uses_repeated_gutter():
    result = detect_page_layout(FakePage(two_column_words()), page_number=2)
    assert result["layout_type"] == "two_column"
    assert result["column_boundaries"]["left_column_end"] < result["column_boundaries"]["right_column_start"]
    assert result["diagnostics"]["paired_gutter_rows"] >= 10


def test_full_width_heading_makes_page_mixed():
    result = detect_page_layout(FakePage(two_column_words(with_heading=True)), page_number=3)
    assert result["layout_type"] == "mixed"
    assert any("heading" in line["text"] for line in result["full_width_lines"])


def test_diagram_heavy_page_uses_conservative_layout():
    page = FakePage(two_column_words())
    page.rects = [{}] * 20
    result = detect_page_layout(page, page_number=6)
    assert result["layout_type"] == "table_or_diagram"


def test_missing_coordinates_are_reported():
    result = detect_page_layout(
        FakePage([{"text": "unlocated"}], text="Available fallback text"),
        page_number=7,
    )
    assert result["layout_type"] == "single_column"
    assert result["diagnostics"]["uncertain"] is True
    assert result["diagnostics"]["words_without_usable_coordinates"] == 1


def test_unusual_coordinates_are_reported():
    page = FakePage([word("outside", 10000, 10050, 10000)], text="Safe text")
    result = detect_page_layout(page, page_number=8)
    assert result["diagnostics"]["words_without_usable_coordinates"] == 1
    assert result["diagnostics"]["uncertain"] is True


def test_two_column_reading_order_and_page_metadata(monkeypatch, tmp_path):
    page = FakePage(two_column_words(), text="wrong default order")
    open_fake_pdf(monkeypatch, [page])
    source = tmp_path / "fixture.pdf"
    source.touch()

    document = pdfplumber_loader.load_pdf_with_layout(source)[0]
    text = document["page_content"]
    assert document["text"] == text
    assert document["page_number"] == 1
    assert document["total_pages"] == 1
    assert document["metadata"]["source"] == "fixture.pdf"
    assert document["metadata"]["page"] == 1
    assert document["metadata"]["total_pages"] == 1
    assert text.index("left0") < text.index("left11") < text.index("right0")


def test_mixed_reading_order_keeps_heading_before_columns(monkeypatch, tmp_path):
    page = FakePage(two_column_words(with_heading=True))
    open_fake_pdf(monkeypatch, [page])
    source = tmp_path / "mixed.pdf"
    source.touch()

    text = pdfplumber_loader.load_pdf_with_layout(source)[0]["text"]
    assert text.index("Full width heading spans both columns") < text.index("left0")
    assert text.index("left11") < text.index("right0")


def test_uncertain_layout_uses_normal_pdfplumber_extraction(monkeypatch, tmp_path):
    page = FakePage(two_column_words(rows=3), text="normal page extraction")
    open_fake_pdf(monkeypatch, [page])
    source = tmp_path / "uncertain.pdf"
    source.touch()

    document = pdfplumber_loader.load_pdf_with_layout(source)[0]
    assert document["text"] == "normal page extraction"
    assert document["metadata"]["extraction_method"] == "pdfplumber_page"


def test_bad_coordinates_fall_back_without_losing_text(monkeypatch, tmp_path):
    page = FakePage([{"text": "broken coordinate"}], text="Complete fallback text")
    open_fake_pdf(monkeypatch, [page])
    source = tmp_path / "bad-coordinates.pdf"
    source.touch()

    document = pdfplumber_loader.extract_pdf_by_columns(source)[0]
    assert document["text"] == "Complete fallback text"


def test_metadata_survives_chunking(monkeypatch, tmp_path):
    page = FakePage(two_column_words())
    open_fake_pdf(monkeypatch, [page])
    source = tmp_path / "chunk-metadata.pdf"
    source.touch()

    document = pdfplumber_loader.load_pdf_with_layout(source)[0]
    chunks = create_recursive_chunks([document], chunk_size=1000, overlap=200)
    assert chunks
    assert chunks[0]["metadata"]["source"] == "chunk-metadata.pdf"
    assert chunks[0]["metadata"]["page"] == 1
    assert chunks[0]["metadata"]["chunk_number"] == 1
    assert all(isinstance(value, (str, int, float, bool)) for value in chunks[0]["metadata"].values())


def test_extraction_does_not_reconcile_or_duplicate_labels(monkeypatch, tmp_path):
    page = FakePage(
        [word("Matching", 50, 105, 100), word("Networks", 110, 165, 100)],
        text="Matching Networks",
    )
    open_fake_pdf(monkeypatch, [page])
    source = tmp_path / "labels.pdf"
    source.touch()

    text = pdfplumber_loader.load_pdf_with_layout(source)[0]["text"]
    assert text.count("Matching Networks") == 1
    assert "Additional PDF text labels" not in text
    assert not hasattr(pdfplumber_loader, "PdfReader")


def test_pdfplumber_coordinates_join_split_bullet_word(monkeypatch, tmp_path):
    words = [
        word("M", 100, 106, 100),
        word("atching", 106, 134, 100),
        word("Networks", 140, 185, 100),
    ]
    words.insert(0, word("◼", 80, 90, 100))
    page = FakePage(words, text="◼ M atching Networks")
    open_fake_pdf(monkeypatch, [page])
    source = tmp_path / "split-word.pdf"
    source.touch()

    text = pdfplumber_loader.load_pdf_with_layout(source)[0]["text"]
    assert text == "◼ Matching Networks"


def test_column_line_joining_preserves_tightly_spaced_pdf_characters():
    fragments = [word("W", 100, 105, 100), word("h", 105, 110, 100),
                 word("ilst", 110, 125, 100)]
    assert pdfplumber_loader._line_text(fragments) == "Whilst"


def test_pdfplumber_coordinates_correct_reversed_vertical_label(monkeypatch, tmp_path):
    words = [
        {**word("moc.8noci", 300, 305, 10, 32), "upright": False, "direction": "ttb"},
        {**word("morf", 300, 305, 34, 45), "upright": False, "direction": "ttb"},
        {**word("segamI", 300, 305, 47, 63), "upright": False, "direction": "ttb"},
    ]
    page = FakePage(words, text="moc.8noci\nmorf\nsegamI")
    open_fake_pdf(monkeypatch, [page])
    source = tmp_path / "vertical-label.pdf"
    source.touch()

    text = pdfplumber_loader.load_pdf_with_layout(source)[0]["text"]
    assert "Images from icon8.com" in text
    assert "moc.8noci" not in text


def test_legacy_extraction_name_remains_compatible(monkeypatch, tmp_path):
    page = FakePage(two_column_words())
    open_fake_pdf(monkeypatch, [page])
    source = tmp_path / "compat.pdf"
    source.touch()

    result = pdfplumber_loader.extract_pdf_by_columns(source)[0]
    assert {"page_number", "total_pages", "text", "page_content", "metadata"} <= result.keys()
