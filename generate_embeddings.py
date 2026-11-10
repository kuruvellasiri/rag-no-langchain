
import json
from pathlib import Path

from src.embeddings.gemini_embeddings import (
    generate_document_embeddings
)


PROJECT_ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = PROJECT_ROOT / "extracted_text" / "chunks.json"

OUTPUT_PATH = PROJECT_ROOT / "extracted_text" / "embeddings.json"


def main():

    # Step 1: Load existing chunks
    print("Loading chunks...")

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks found.")

    # Step 2: Extract text from each chunk
    texts = [
        chunk["page_content"]
        for chunk in chunks
    ]

    # Step 3: Generate embeddings
    embeddings = generate_document_embeddings(
        texts,
        batch_size=20
    )

    # Step 4: Attach vectors to their chunks
    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):

        embedded_chunks.append({
            **chunk,
            "embedding": embedding
        })

    # Step 5: Save embeddings
    OUTPUT_PATH.write_text(
        json.dumps(
            embedded_chunks,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # Step 6: Display results
    print("\nEmbedding generation completed!")

    print(f"Total chunks: {len(embedded_chunks)}")

    print(f"Vector dimensions: {len(embeddings[0])}")

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
