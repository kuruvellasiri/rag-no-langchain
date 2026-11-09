
from src.chunking.text_chunker import split_text_recursive


def test_chunk_size_limit():
    text = "This is a sample sentence. " * 200

    chunks = split_text_recursive(
        text,
        chunk_size=1000,
        overlap=200
    )

    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_text_is_preserved_without_overlap():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

    chunks = split_text_recursive(
        text,
        chunk_size=20,
        overlap=0
    )

    combined = "".join(chunks)

    assert combined == text


def test_overlap_is_present():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10

    chunks = split_text_recursive(
        text,
        chunk_size=100,
        overlap=20
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-20:] == current[:20]


def test_empty_text():
    assert split_text_recursive("", 100, 20) == []


def test_invalid_overlap():
    try:
        split_text_recursive("Hello world", 100, 100)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
