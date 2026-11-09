
from src.vectorstore.chroma_store import get_collection


collection = get_collection()

print("ChromaDB connected successfully!")

print("Collection name:", collection.name)

print("Existing records:", collection.count())
