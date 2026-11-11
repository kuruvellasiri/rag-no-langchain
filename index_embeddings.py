
import argparse
import json
from pathlib import Path

from src.vectorstore.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    STRUCTURED_COLLECTION_NAME,
    get_collection,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDINGS_PATH = PROJECT_ROOT / "extracted_text" / "embeddings.json"
STRUCTURED_EMBEDDINGS_PATH = PROJECT_ROOT / "extracted_text" / "structured_embeddings.json"


def main(use_structured: bool = False):

    embeddings_path = STRUCTURED_EMBEDDINGS_PATH if use_structured else EMBEDDINGS_PATH
    collection_name = STRUCTURED_COLLECTION_NAME if use_structured else DEFAULT_COLLECTION_NAME

    # Step 1: Load existing embeddings
    print("Loading embeddings...")

    records = json.loads(
        embeddings_path.read_text(encoding="utf-8")
    )

    print(f"Total records loaded: {len(records)}")

    if not records:
        raise ValueError("No embeddings found.")

    # Step 2: Prepare data for ChromaDB
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for record in records:

        ids.append(record["chunk_id"])

        documents.append(record["page_content"])

        embeddings.append(record["embedding"])

        metadatas.append(record["metadata"])

    # Step 3: Validate dimensions
    for vector in embeddings:
        if len(vector) != 768:
            raise ValueError("Unexpected embedding dimension.")

    # Step 4: Get ChromaDB collection
    collection = get_collection(collection_name)

    print("\nConnected to collection:", collection.name)

    # Step 5: Insert or update records
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    # Step 6: Verify insertion
    print("\nInsertion completed!")

    print("Total records in ChromaDB:", collection.count())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index PDF embeddings in ChromaDB.")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Index structured_embeddings.json into its separate collection.",
    )
    main(use_structured=parser.parse_args().structured)
