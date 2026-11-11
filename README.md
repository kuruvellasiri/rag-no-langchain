# Traditional PDF RAG (without LangChain)

This project extracts PDF content, splits it into chunks, creates Gemini embeddings, stores them in ChromaDB, retrieves relevant chunks, and asks Gemini to write a cited answer.

## PDF extraction

The standard workflow remains available with:

```powershell
python main.py
```

The optional structured workflow keeps the existing extraction files unchanged and writes a parallel set of outputs:

```powershell
python main.py --structured
```

It uses **pdfplumber** for positioned text, layout clues, table cells, image placements, and vector drawing regions. Text lines are ordered using the existing layout detector. Headings are identified using relative font size plus short-line checks; caption-like labels are stored separately. Tables are extracted as rows and columns and serialized as Markdown. Ordinary paragraph-sized table candidates are filtered out, and uncertain detections are marked for review.

Image and diagram crops are saved in `extracted_text/visuals/`. OCR runs only when the PDF text layer contains little text inside a crop, and only if the Tesseract command-line program is installed. OCR failure does not stop extraction; the element is retained and marked for review. Vector-heavy regions are classified as diagrams or charts using simple visual and nearby-text clues. Those classifications are estimates and should be reviewed.

Gemini visual descriptions are optional because they send selected PDF crops to the Gemini API:

```powershell
python main.py --structured --describe-visuals
```

Descriptions are prompted to report only clearly visible content. Charts remain marked for review because exact values and relationships need validation. Without this option, readable embedded text and OCR are included where available, and visuals without a description are marked for review.

### Structured outputs

- `extracted_text/structured_content.json` — page count and typed elements with IDs, page numbers, bounding boxes, extraction methods, and review flags.
- `extracted_text/structured_document_text.txt` — readable page-by-page text, tables, and visual descriptions or review markers.
- `extracted_text/structured_chunks.json` — recursive chunks built from the structured page text. Headings stay attached to the following paragraph; tables and visual descriptions remain labeled blocks. Chunk metadata records the overlap found in the actual text.
- `extracted_text/structured_comparison.json` — compares page counts and per-page character counts against the existing `document_text.txt`, and lists low text-coverage pages.
- `extracted_text/visuals/` — rendered visual crops referenced by the JSON elements.

The structured command does not replace `document_text.txt` or `chunks.json`. It checks text-token coverage against the existing pdfplumber page extraction and flags pages below 70% for review.

## Dependencies

Install Python packages with:

```powershell
python -m pip install -r requirements.txt
```

The structured path requires `pdfplumber` (which installs Pillow and its PDF renderer). OCR is optional; install the Tesseract OCR program separately and make sure `tesseract` is on `PATH`.

Set `GEMINI_API_KEY` in `.env` for embeddings, answer generation, or the optional visual-description mode. Do not commit `.env`.

## Full structured RAG workflow

Run each stage explicitly:

```powershell
python main.py --structured
python generate_embeddings.py --structured
python index_embeddings.py --structured
python chroma_search.py --structured
```

Use `--describe-visuals` on the extraction command if you want Gemini to describe image, diagram, and chart crops. Embedding generation makes Gemini API calls. Indexing uses `upsert` in a separate `machine_learning_documents_structured` collection with the same cosine metric; the original `machine_learning_documents` collection remains the default and is not cleared.

The existing default flow is still:

```powershell
python main.py
python generate_embeddings.py
python index_embeddings.py
python chroma_search.py
```

The two output sets use different chunk IDs and Chroma collections. Do not run the structured embedding or indexing commands unless you intend to create/update that separate index.
