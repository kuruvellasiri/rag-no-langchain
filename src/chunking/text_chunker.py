
def split_text_recursive(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: list[str] | None = None
) -> list[str]:

    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap.")

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Find the first separator available
    separator = next(
        (s for s in separators if s == "" or s in text),
        ""
    )
    # If no meaningful separator exists, use sliding windows
    if separator == "":
        step = chunk_size - overlap

        return [
        text[i:i + chunk_size]
        for i in range(0, len(text), step)
    ]
    # Split while preserving separators
    if separator == "":
        pieces = list(text)
    else:
        raw_pieces = text.split(separator)
        pieces = []

        for index, piece in enumerate(raw_pieces):
            if piece:
                if index < len(raw_pieces) - 1:
                    pieces.append(piece + separator)
                else:
                    pieces.append(piece)

    remaining_separators = separators[
        separators.index(separator) + 1:
    ]

    # Recursively split pieces that are too large
    smaller_pieces = []

    for piece in pieces:
        if len(piece) > chunk_size:
            smaller_pieces.extend(
                split_text_recursive(
                    piece,
                    chunk_size,
                    overlap,
                    remaining_separators
                )
            )
        elif piece:
            smaller_pieces.append(piece)

    # Merge pieces without losing their original content
    chunks = []
    current = ""

    for piece in smaller_pieces:

        candidate = current + piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            current = piece

    if current:
        chunks.append(current)

    # Apply overlap while preserving the complete new content
    final_chunks = []

    for index, chunk in enumerate(chunks):

        if index == 0 or overlap == 0:
            final_chunks.append(chunk)
            continue

        previous = chunks[index - 1]
        overlap_text = previous[-overlap:]

        # If overlap plus new chunk is too large,
        # reduce overlap to make room for all new content.
        available_overlap = max(
            0,
            chunk_size - len(chunk)
        )

        overlap_text = overlap_text[-available_overlap:] if available_overlap else ""

        final_chunks.append(overlap_text + chunk)

    return final_chunks


def create_recursive_chunks(
    documents: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[dict]:

    all_chunks = []

    for document in documents:

        page_text = document["page_content"].strip()
        metadata = document["metadata"]

        page_chunks = split_text_recursive(
            page_text,
            chunk_size,
            overlap
        )

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1
        ):

            all_chunks.append({
                "chunk_id": (
                    f"page_{metadata['page']}_chunk_{chunk_number}"
                ),
                "page_content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_number": chunk_number,
                    "chunk_size": len(chunk_text),
                    "overlap": overlap if chunk_number > 1 else 0
                }
            })

    return all_chunks
