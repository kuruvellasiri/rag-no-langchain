
from src.retrieval.similarity_search import semantic_search


def main():

    query = input("\nAsk a question about the PDF: ")

    results = semantic_search(query, top_k=3)

    print("\n===== SEARCH RESULTS =====")

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print("Chunk ID:", result["chunk_id"])

        print("Page:", result["metadata"]["page"])

        print(
            "Similarity:",
            round(result["similarity_score"], 4)
        )

        print("\nContent:")
        print(result["page_content"])

        print("-" * 60)


if __name__ == "__main__":
    main()
