from src.embeddings.gemini_embeddings import generate_embedding


text = "Machine learning requires training data."

embedding = generate_embedding(text)

print("\nOriginal text:")
print(text)

print("\nFirst 10 embedding values:")
print(embedding[:10])

print("\nVector dimensions:")
print(len(embedding))