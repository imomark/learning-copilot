from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from app.embeddings.gemini_embeddings import get_embedding_model
from qdrant_client.http import models

class QdrantStore:
    def __init__(self):
        # In-memory Qdrant for now
        self.client = QdrantClient(":memory:")
        self.collection_name = "learning_copilot"
        self.embeddings = get_embedding_model()

        self.store: QdrantVectorStore | None = None



    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None):
        if self.store is None:
            # 1. Ensure the collection exists in the local client
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    # Adjust vectors_config based on your Gemini embedding dimensions
                    # Typically 768 for models like text-embedding-004
                    vectors_config=models.VectorParams(
                        size=3072,
                        distance=models.Distance.COSINE
                    )
                )

            # 2. Initialize the store linked to that collection
            self.store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )

        # 3. Add the texts
        self.store.add_texts(texts=texts, metadatas=metadatas)

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    def search(self, query: str, k: int = 3):
        if self.store is None:
            return []

        results = self.store.similarity_search(query, k=k)
        return results

