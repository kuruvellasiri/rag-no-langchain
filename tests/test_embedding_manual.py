from src.embeddings.gemini_embeddings import (
    generate_document_embedding,
    generate_query_embedding
)


document_text = "Machine learning requires training data."

query_text = "Why does machine learning need training data?"


document_vector = generate_document_embedding(document_text)

query_vector = generate_query_embedding(query_text)


print("Document embedding dimensions:", len(document_vector))
print("Query embedding dimensions:", len(query_vector))

print("\nDocument first 5 values:")
print(document_vector[:5])

print("\nQuery first 5 values:")
print(query_vector[:5])