
import json
from pathlib import Path

from src.vectorstore.chroma_store import get_collection


EMBEDDINGS_PATH = Path("extracted_text/embeddings.json")


def main():

    # Step 1: Load existing embeddings
    print("Loading embeddings...")

    records = json.loads(
        EMBEDDINGS_PATH.read_text(encoding="utf-8")
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
    collection = get_collection()

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
    main()
