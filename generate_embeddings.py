
import argparse
import json
from pathlib import Path

from src.embeddings.gemini_embeddings import (
    generate_document_embeddings
)


PROJECT_ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = PROJECT_ROOT / "extracted_text" / "chunks.json"

OUTPUT_PATH = PROJECT_ROOT / "extracted_text" / "embeddings.json"
STRUCTURED_CHUNKS_PATH = PROJECT_ROOT / "extracted_text" / "structured_chunks.json"
STRUCTURED_OUTPUT_PATH = PROJECT_ROOT / "extracted_text" / "structured_embeddings.json"


def main(use_structured: bool = False):

    chunks_path = STRUCTURED_CHUNKS_PATH if use_structured else CHUNKS_PATH
    output_path = STRUCTURED_OUTPUT_PATH if use_structured else OUTPUT_PATH

    # Step 1: Load existing chunks
    print("Loading chunks...")

    chunks = json.loads(
        chunks_path.read_text(encoding="utf-8")
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
    output_path.write_text(
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

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for PDF chunks.")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Use structured_chunks.json and save structured_embeddings.json.",
    )
    main(use_structured=parser.parse_args().structured)
