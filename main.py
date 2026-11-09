
import json
from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_cleaner import clean_text
from src.chunking.text_chunker import create_recursive_chunks


PDF_PATH = Path("documents/machine_learning_pdf.pdf")

EXTRACTED_PATH = Path("extracted_text/document_text.txt")

CHUNKS_PATH = Path("extracted_text/chunks.json")


def main():

    # Step 1: Load PDF
    print("Loading PDF...")

    documents = load_pdf(PDF_PATH)

    print(f"Total pages loaded: {len(documents)}")

    # Step 2: Clean each page
    for document in documents:
        document["page_content"] = clean_text(
            document["page_content"]
        )

    # Step 3: Save extracted text
    complete_text = "\n\n".join(
        f"--- PAGE {doc['metadata']['page']} ---\n"
        f"{doc['page_content']}"
        for doc in documents
    )

    EXTRACTED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    EXTRACTED_PATH.write_text(
        complete_text,
        encoding="utf-8"
    )

    # Step 4: Create recursive chunks
    print("\nCreating chunks...")

    chunks = create_recursive_chunks(
        documents,
        chunk_size=1000,
        overlap=200
    )

    # Step 5: Save chunks
    CHUNKS_PATH.write_text(
        json.dumps(
            chunks,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"Total chunks created: {len(chunks)}")

    print(f"Saved extracted text: {EXTRACTED_PATH}")
    print(f"Saved chunks: {CHUNKS_PATH}")

    # Step 6: Display first chunk
    if chunks:
        print("\n----- FIRST CHUNK -----")
        print(json.dumps(chunks[0], indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
