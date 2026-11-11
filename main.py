
import argparse
import json
from pathlib import Path
import re

from src.ingestion.pdfplumber_loader import load_pdf_with_layout
from src.ingestion.structured_pdf_loader import (
    count_element_types,
    extract_structured_pdf,
    write_structured_outputs,
)
from src.ingestion.text_cleaner import clean_text
from src.chunking.text_chunker import create_recursive_chunks


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "documents" / "machine_learning_pdf.pdf"

EXTRACTED_PATH = PROJECT_ROOT / "extracted_text" / "document_text.txt"

CHUNKS_PATH = PROJECT_ROOT / "extracted_text" / "chunks.json"
STRUCTURED_JSON_PATH = PROJECT_ROOT / "extracted_text" / "structured_content.json"
STRUCTURED_TEXT_PATH = PROJECT_ROOT / "extracted_text" / "structured_document_text.txt"
STRUCTURED_CHUNKS_PATH = PROJECT_ROOT / "extracted_text" / "structured_chunks.json"
STRUCTURED_COMPARISON_PATH = PROJECT_ROOT / "extracted_text" / "structured_comparison.json"
VISUALS_PATH = PROJECT_ROOT / "extracted_text" / "visuals"


def _run_structured_pipeline(describe_visuals: bool = False) -> None:
    print("Extracting structured PDF content...")
    documents, elements = extract_structured_pdf(
        PDF_PATH,
        VISUALS_PATH,
        describe_visuals=describe_visuals,
    )

    write_structured_outputs(
        documents,
        elements,
        STRUCTURED_JSON_PATH,
        STRUCTURED_TEXT_PATH,
    )

    chunks = create_recursive_chunks(
        documents,
        chunk_size=1000,
        overlap=200,
        chunk_id_prefix="structured_page",
    )
    STRUCTURED_CHUNKS_PATH.write_text(
        json.dumps(chunks, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    element_counts = count_element_types(elements)
    visual_description_count = sum(
        bool(element.get("description"))
        for element in elements
        if element["element_type"] in {"image", "diagram", "chart"}
    )
    low_coverage_pages = [
        document["page_number"]
        for document in documents
        if document["metadata"].get("coordinate_text_word_coverage", 1.0) < 0.70
    ]

    comparison = {
        "source": PDF_PATH.name,
        "structured_page_count": len(documents),
        "structured_element_counts": element_counts,
        "visual_descriptions": visual_description_count,
        "structured_chunk_count": len(chunks),
        "pages_below_70_percent_text_word_coverage": low_coverage_pages,
        "legacy_page_count": 0,
        "legacy_text_characters": 0,
        "structured_text_characters": sum(len(doc["page_content"]) for doc in documents),
        "pages_missing_from_structured_output": [],
    }
    if EXTRACTED_PATH.is_file():
        legacy_text = EXTRACTED_PATH.read_text(encoding="utf-8")
        legacy_pages = {
            int(page_number): content.strip()
            for page_number, content in re.findall(
                r"--- PAGE (\d+) ---\s*(.*?)(?=\n\n--- PAGE \d+ ---|\Z)",
                legacy_text,
                flags=re.DOTALL,
            )
        }
        structured_pages = {document["page_number"] for document in documents}
        structured_text_by_page = {
            document["page_number"]: document["page_content"]
            for document in documents
        }
        legacy_word_coverage = {}
        for page_number, legacy_page_text in legacy_pages.items():
            old_words = set(re.findall(r"[\w'-]+", legacy_page_text.casefold()))
            new_words = set(re.findall(
                r"[\w'-]+",
                structured_text_by_page.get(page_number, "").casefold(),
            ))
            coverage = len(old_words & new_words) / len(old_words) if old_words else 1.0
            legacy_word_coverage[str(page_number)] = round(coverage, 3)

        comparison.update({
            "legacy_page_count": len(legacy_pages),
            "legacy_text_characters": sum(len(text) for text in legacy_pages.values()),
            "pages_missing_from_structured_output": sorted(set(legacy_pages) - structured_pages),
            "pages_added_in_structured_output": sorted(structured_pages - set(legacy_pages)),
            "legacy_page_character_counts": {
                str(page): len(text) for page, text in legacy_pages.items()
            },
            "structured_page_character_counts": {
                str(document["page_number"]): len(document["page_content"])
                for document in documents
            },
            "legacy_word_coverage_by_page": legacy_word_coverage,
            "pages_below_70_percent_legacy_word_coverage": [
                page for page, coverage in legacy_word_coverage.items()
                if coverage < 0.70
            ],
        })

    STRUCTURED_COMPARISON_PATH.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Pages processed: {len(documents)}")
    print(f"Structured elements: {element_counts}")
    print(f"Visual descriptions: {visual_description_count}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Pages below text coverage target: {low_coverage_pages or 'None'}")
    print(f"Structured JSON: {STRUCTURED_JSON_PATH}")
    print(f"Readable text: {STRUCTURED_TEXT_PATH}")
    print(f"Structured chunks: {STRUCTURED_CHUNKS_PATH}")
    print(f"Visual crops: {VISUALS_PATH}")
    print(f"Comparison report: {STRUCTURED_COMPARISON_PATH}")
    if describe_visuals:
        print("Gemini visual descriptions were enabled.")
    else:
        print("Gemini visual descriptions were not requested; uncertain visuals are flagged for review.")


def main():
    parser = argparse.ArgumentParser(description="Extract and chunk the project PDF.")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Write structured extraction outputs without replacing the legacy outputs.",
    )
    parser.add_argument(
        "--describe-visuals",
        action="store_true",
        help="Ask Gemini to describe detected visual crops (uses the Gemini API).",
    )
    args = parser.parse_args()

    if args.describe_visuals and not args.structured:
        parser.error("--describe-visuals requires --structured")

    if args.structured:
        _run_structured_pipeline(describe_visuals=args.describe_visuals)
        return

    # Step 1: Load PDF
    print("Loading PDF...")

    documents = load_pdf_with_layout(PDF_PATH)

    print(f"Total pages loaded: {len(documents)}")

    # Step 2: Clean each page
    for document in documents:
        document["page_content"] = clean_text(document["page_content"])

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
