from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "documents" / "machine_learning_pdf.pdf"
PAGES_TO_TEST = [27, 29]


def main():
    reader = PdfReader(str(PDF_PATH))

    for page_number in PAGES_TO_TEST:
        page = reader.pages[page_number - 1]

        print("\n" + "=" * 60)
        print(f"PAGE {page_number}")
        print("=" * 60)

        print("\n--- DEFAULT EXTRACTION ---")
        print(page.extract_text() or "")

        print("\n--- LAYOUT EXTRACTION ---")
        print(
            page.extract_text(extraction_mode="layout") or ""
        )


if __name__ == "__main__":
    main()
