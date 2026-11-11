
import argparse

from src.generation.gemini_answer import (
    INVALID_QUERY_MESSAGE,
    filter_retrieved_chunks,
    generate_answer,
)


def main(use_structured: bool = False):

    query = input("\nAsk a question about the PDF: ")

    if not query or not query.strip():
        print(INVALID_QUERY_MESSAGE)
        return

    try:
        # Import here so a missing Gemini key can be reported cleanly.
        from src.retrieval.similarity_search import chroma_semantic_search
        from src.vectorstore.chroma_store import (
            DEFAULT_COLLECTION_NAME,
            STRUCTURED_COLLECTION_NAME,
        )

        results = chroma_semantic_search(
            query=query,
            top_k=3,
            collection_name=(
                STRUCTURED_COLLECTION_NAME
                if use_structured else DEFAULT_COLLECTION_NAME
            ),
        )
        relevant_results = filter_retrieved_chunks(results)
    except ValueError as error:
        print(f"Search configuration error: {error}")
        return
    except Exception:
        print("Unable to retrieve PDF context right now. Please try again.")
        return

    answer = generate_answer(query, relevant_results)
    source_pages = list(dict.fromkeys(
        result.get("metadata", {}).get("page", "Unknown")
        for result in relevant_results
    ))

    print("\n===== ANSWER =====")
    print(answer)
    print(
        "\nSource pages:",
        ", ".join(str(page) for page in source_pages) or "None",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the PDF with ChromaDB.")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Search the structured ChromaDB collection.",
    )
    main(use_structured=parser.parse_args().structured)
