
from src.ingestion.text_cleaner import clean_text


def test_paragraph_lines_are_joined():

    text = """Machine learning requires
    sufficient training data.
    Deep learning needs more examples."""

    result = clean_text(text)

    assert result == (
        "Machine learning requires sufficient "
        "training data. Deep learning needs more examples."
    )


def test_bullet_points_are_preserved():

    text = """Self-supervised tasks

    ▪ Relative position in images
    ▪ Image patch prediction"""

    result = clean_text(text)

    assert "• Relative position in images" in result
    assert "• Image patch prediction" in result


def test_numbered_lists_are_preserved():

    text = """Steps

    1. Load PDF
    2. Extract text"""

    result = clean_text(text)

    assert "1. Load PDF" in result
    assert "2. Extract text" in result


def test_hyphenated_words_are_joined():

    text = """Trans-
    fer learning is useful."""

    result = clean_text(text)

    assert "Transfer learning" in result
