import os

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from app.embeddings.gemini_embeddings import get_embedding_model
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue


class QdrantStore:
    def __init__(self):
        db_path = os.path.join(os.getcwd(), "data", "qdrant")
        self.client = QdrantClient(path=db_path)
        self.collection_name = "learning_copilot"
        self.embeddings = get_embedding_model()

        self.store: QdrantVectorStore | None = None
        self._attach_if_exists()

    def _attach_if_exists(self):
        try:
            self.client.get_collection(self.collection_name)
            # If this does not throw, collection exists
            self.store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            print(f"[QdrantStore] Attached to existing collection: {self.collection_name}")
        except Exception:
            # Collection does not exist yet
            print(f"[QdrantStore] No existing collection found. Will create on first ingest.")
            self.store = None



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

    def list_sources(self) -> list[str]:
        sources = set()
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=100,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                meta = payload.get("metadata") or {}
                src = meta.get("source")
                if src:
                    sources.add(src)
            if offset is None:
                break

        return sorted(list(sources))

    def delete_by_source(self, source: str) -> int:
        flt = Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source),
                )
            ]
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=flt,  # ✅ typed filter, not dict
        )

        return 1




