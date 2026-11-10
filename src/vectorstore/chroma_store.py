
import chromadb
from pathlib import Path


# Location where ChromaDB will save its data
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"


# Create a persistent ChromaDB client
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


# Create or access our document collection
collection = client.get_or_create_collection(
    name="machine_learning_documents",
    metadata={
        "hnsw:space": "cosine"
    }
)


def get_collection():
    return collection
