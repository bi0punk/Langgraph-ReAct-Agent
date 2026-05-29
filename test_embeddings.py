#!/usr/bin/env python3
from app.embeddings import get_embeddings

print("Testing embeddings...")
embeddings = get_embeddings()
print(f"Embeddings object: {embeddings}")
print(f"Model name: {embeddings.model_name}")

test_texts = ["This is a test sentence.", "Another test sentence."]
print(f"\nTesting with {len(test_texts)} sample texts...")

try:
    result = embeddings.embed_documents(test_texts)
    print(f"✓ Embedding successful!")
    print(f"  - Number of embeddings: {len(result)}")
    print(f"  - Embedding dimension: {len(result[0]) if result else 'N/A'}")
    print(f"  - First embedding preview: {result[0][:5]}..." if result else "Empty")
except Exception as e:
    print(f"❌ ERROR generating embeddings: {e}")
    import traceback
    traceback.print_exc()
