
from src.chunking.text_chunker import create_recursive_chunks, split_text_recursive


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


def test_chunk_metadata_records_actual_overlap_and_supports_custom_ids():
    text = "This is a sample sentence. " * 200
    chunks = create_recursive_chunks(
        [{"page_content": text, "metadata": {"page": 3}}],
        chunk_size=100,
        overlap=20,
        chunk_id_prefix="structured_page",
    )

    assert chunks[0]["chunk_id"] == "structured_page_3_chunk_1"
    for previous, current in zip(chunks, chunks[1:]):
        actual = 0
        maximum = min(20, len(previous["page_content"]), len(current["page_content"]))
        for size in range(maximum, 0, -1):
            if previous["page_content"][-size:] == current["page_content"][:size]:
                actual = size
                break
        assert current["metadata"]["overlap"] == actual

    assert any(chunk["metadata"]["overlap"] < 20 for chunk in chunks[1:])
