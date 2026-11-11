
import numpy as np
from src.embeddings.gemini_embeddings import generate_query_embedding

import json
from pathlib import Path

from src.embeddings.gemini_embeddings import (
    generate_query_embedding
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = PROJECT_ROOT / "extracted_text" / "embeddings.json"


def semantic_search(
    query: str,
    top_k: int = 3
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    # Load stored document embeddings
    records = json.loads(
        EMBEDDINGS_PATH.read_text(encoding="utf-8")
    )

    # Generate embedding for the user question
    query_vector = generate_query_embedding(query)

    results = []

    # Compare query against every document vector
    for record in records:

        document_vector = record["embedding"]

        score = cosine_similarity(
            query_vector,
            document_vector
        )

        results.append({
            "chunk_id": record["chunk_id"],
            "page_content": record["page_content"],
            "metadata": record["metadata"],
            "similarity_score": score
        })

    # Sort from highest similarity to lowest
    results.sort(
        key=lambda item: item["similarity_score"],
        reverse=True
    )

    return results[:top_k]

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:

    # Convert lists into NumPy arrays
    a = np.array(vector_a, dtype=np.float64)
    b = np.array(vector_b, dtype=np.float64)

    # Validate dimensions
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Vectors must be one-dimensional.")

    if a.shape != b.shape:
        raise ValueError("Vectors must have equal dimensions.")

    # Calculate vector magnitudes
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Zero vectors cannot be compared.")

    # Calculate dot product
    dot_product = np.dot(a, b)

    # Calculate cosine similarity
    score = dot_product / (magnitude_a * magnitude_b)

    return float(score)

from src.vectorstore.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    get_collection,
)


def chroma_semantic_search(
    query: str,
    top_k: int = 3,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    # Step 1: Generate query embedding
    query_vector = generate_query_embedding(query)

    # Step 2: Get ChromaDB collection
    collection = get_collection(collection_name)

    # Step 3: Search the collection
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Step 4: Prepare readable results
    search_results = []

    for i in range(len(results["ids"][0])):

        distance = results["distances"][0][i]

        search_results.append({
            "chunk_id": results["ids"][0][i],
            "page_content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": distance,
            "similarity_score": 1 - distance
        })

    return search_results
