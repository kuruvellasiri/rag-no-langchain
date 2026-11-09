from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str | Path) -> list[dict]:

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""

        document = {
            "page_content": page_text,
            "metadata": {
                "source": pdf_path.name,
                "page": page_number,
                "total_pages": len(reader.pages)
            }
        }

        documents.append(document)

    return documents