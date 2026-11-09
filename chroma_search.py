
from src.retrieval.similarity_search import chroma_semantic_search


def main():

    query = input("\nAsk a question about the PDF: ")

    results = chroma_semantic_search(
        query=query,
        top_k=3
    )

    print("\n===== CHROMADB SEARCH RESULTS =====")

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print("Chunk ID:", result["chunk_id"])

        print("Page:", result["metadata"]["page"])

        print(
            "Distance:",
            round(result["distance"], 4)
        )

        print(
            "Similarity:",
            round(result["similarity_score"], 4)
        )

        print("\nContent:")
        print(result["page_content"])

        print("-" * 60)


if __name__ == "__main__":
    main()
