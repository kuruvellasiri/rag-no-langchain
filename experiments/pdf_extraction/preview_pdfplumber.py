import pdfplumber
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "documents" / "machine_learning_pdf.pdf"
PAGES_TO_TEST = [27, 29]


def main():
    with pdfplumber.open(PDF_PATH) as pdf:

        for page_number in PAGES_TO_TEST:
            page = pdf.pages[page_number - 1]

            midpoint = page.width / 2

            left_page = page.crop(
                (0, 0, midpoint, page.height)
            )

            right_page = page.crop(
                (midpoint, 0, page.width, page.height)
            )

            print("\n" + "=" * 60)
            print(f"PAGE {page_number}")
            print("=" * 60)

            print("\n--- LEFT COLUMN ---")
            print(left_page.extract_text() or "")

            print("\n--- RIGHT COLUMN ---")
            print(right_page.extract_text() or "")


if __name__ == "__main__":
    main()
