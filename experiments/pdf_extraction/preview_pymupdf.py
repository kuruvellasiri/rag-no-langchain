import fitz
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "documents" / "machine_learning_pdf.pdf"
PAGES_TO_TEST = [27, 29]


def main():
    document = fitz.open(PDF_PATH)

    for page_number in PAGES_TO_TEST:
        page = document[page_number - 1]

        print("\n" + "=" * 60)
        print(f"PAGE {page_number}")
        print("=" * 60)

        blocks = page.get_text("blocks", sort=True)

        for block_number, block in enumerate(blocks, start=1):
            x0, y0, x1, y1, text, *_ = block

            print(f"\n--- BLOCK {block_number} ---")
            print(f"Position: x={x0:.1f}, y={y0:.1f}")
            print(text.strip())

    document.close()


if __name__ == "__main__":
    main()
