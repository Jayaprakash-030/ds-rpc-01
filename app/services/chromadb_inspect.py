import chromadb
from pathlib import Path
# from app.services.vector_store import VectorStoreManager

# 1. Initialize the client (ensure path matches your persist_directory)
PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_db"
client = chromadb.PersistentClient(path=str(PERSIST_DIR))

# 2. Get your collection
# Note: In LangChain, the default collection name is often "langchain" 
# unless you specified one in VectorStoreManager
collection = client.get_or_create_collection(name="langchain")

# 3. Peek at the first 5 records
print("--- Peeking at first 5 records ---")
print(collection.peek(limit=5))  # Show available keys

# 4. Check the total count of chunks
print(f"\nTotal chunks in DB: {collection.count()}")

# 5. Get all distinct roles stored (Innovation: RBAC Verification)
all_data = collection.get(include=["metadatas"])
roles = set(m.get("role") for m in all_data["metadatas"] if m)
print(f"Roles found in metadata: {roles}")

results = collection.query(
    query_texts=["What are the marketing expenses?"],
    n_results=3,
    where={"role": "finance"}
)

print("\n--- Finance Team Search Results ---")
for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}: {doc[:100]}...")
    print(f"Source: {results['metadatas'][0][i]['source']}\n")
