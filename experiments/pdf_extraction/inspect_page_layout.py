
import pdfplumber
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "documents" / "machine_learning_pdf.pdf"


def inspect_page(page, page_number):

    words = page.extract_words()

    if not words:
        return {
            "page": page_number,
            "words": 0,
            "left": 0,
            "right": 0,
            "layout": "No extractable text"
        }

    midpoint = page.width / 2

    left_words = [
        word for word in words
        if (word["x0"] + word["x1"]) / 2 < midpoint
    ]

    right_words = [
        word for word in words
        if (word["x0"] + word["x1"]) / 2 >= midpoint
    ]

    left_ratio = len(left_words) / len(words)
    right_ratio = len(right_words) / len(words)

    if left_ratio > 0.25 and right_ratio > 0.25:
        layout = "Possible two-column or mixed"
    else:
        layout = "Possible single-column"

    return {
        "page": page_number,
        "words": len(words),
        "left": len(left_words),
        "right": len(right_words),
        "layout": layout
    }


def main():

    with pdfplumber.open(PDF_PATH) as pdf:

        print("Total pages:", len(pdf.pages))
        print("\nPAGE LAYOUT REPORT")
        print("-" * 75)

        for page_number, page in enumerate(pdf.pages, start=1):

            result = inspect_page(page, page_number)

            print(
                f"Page {result['page']:02d} | "
                f"Words: {result['words']:4d} | "
                f"Left: {result['left']:4d} | "
                f"Right: {result['right']:4d} | "
                f"{result['layout']}"
            )


if __name__ == "__main__":
    main()
