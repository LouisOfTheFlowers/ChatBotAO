from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

collection_name = "documents"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config={
        "size": 384,
        "distance": "Cosine"
    }
)