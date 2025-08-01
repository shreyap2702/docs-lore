from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment or use the one from your main.py
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

pc = Pinecone(api_key=api_key)

print("=== Pinecone Index Information ===")
print(f"API Key: {api_key[:20]}...")
print(f"Environment: {environment}")

# List all indexes
print("\n=== Available Indexes ===")
try:
    indexes = pc.list_indexes()
    print(f"Found {len(indexes)} indexes:")
    for idx in indexes:
        print(f"  - {idx.name}")
        
    if len(indexes) > 0:
        # Get details of the first index
        first_index = indexes[0]
        print(f"\n=== Details for '{first_index.name}' ===")
        
        # Connect to the index
        index = pc.Index(first_index.name)
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"Total vector count: {stats.total_vector_count}")
        print(f"Dimension: {stats.dimension}")
        print(f"Index fullness: {stats.index_fullness}")
        
        # Get namespaces
        if hasattr(stats, 'namespaces'):
            print(f"Namespaces: {list(stats.namespaces.keys())}")
            
    else:
        print("No indexes found. You may need to create one.")
        
except Exception as e:
    print(f"Error accessing Pinecone: {e}") 