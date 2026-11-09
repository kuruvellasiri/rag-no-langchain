
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load environment variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create Gemini client
client = genai.Client(api_key=api_key)


# Embedding model configuration
MODEL_NAME = "gemini-embedding-001"
OUTPUT_DIMENSIONS = 768



def generate_embedding(
    text: str,
    task_type: str
) -> list[float]:

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=OUTPUT_DIMENSIONS
        )
    )

    return response.embeddings[0].values


def generate_document_embedding(text: str) -> list[float]:

    return generate_embedding(
        text,
        task_type="RETRIEVAL_DOCUMENT"
    )


def generate_query_embedding(text: str) -> list[float]:

    return generate_embedding(
        text,
        task_type="RETRIEVAL_QUERY"
    )

def generate_document_embeddings(
    texts: list[str],
    batch_size: int = 20
) -> list[list[float]]:

    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    if not texts:
        return []

    if any(not text or not text.strip() for text in texts):
        raise ValueError("All texts must be non-empty.")

    all_embeddings = []

    # Process texts in batches
    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        print(
            f"Generating embeddings: "
            f"{start + 1} to {start + len(batch)}"
        )

        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=OUTPUT_DIMENSIONS
            )
        )

        batch_embeddings = [
            item.values
            for item in response.embeddings
        ]

        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "Embedding count does not match input count."
            )

        all_embeddings.extend(batch_embeddings)

    return all_embeddings
