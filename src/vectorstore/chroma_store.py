
import chromadb
from pathlib import Path


# Location where ChromaDB will save its data
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "machine_learning_documents"
STRUCTURED_COLLECTION_NAME = "machine_learning_documents_structured"


# Create a persistent ChromaDB client
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


# Create or access our document collection
collection = client.get_or_create_collection(
    name=DEFAULT_COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)


def get_collection(name: str = DEFAULT_COLLECTION_NAME):
    """Return the original collection or create the optional structured one."""
    if name == DEFAULT_COLLECTION_NAME:
        return collection
    return client.get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine"
        },
    )
