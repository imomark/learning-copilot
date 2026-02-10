import os

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from app.embeddings.gemini_embeddings import get_embedding_model
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue
import math
from typing import List


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

    def search(self, query: str, k: int = 5):
        if self.store is None:
            return []

        # 1) Expand candidates (fetch more than needed)
        candidate_k = max(20, k * 4)

        # Use vector store to get initial candidates with scores
        results_with_scores = self.store.similarity_search_with_score(query, k=candidate_k)

        if not results_with_scores:
            return []

        docs = [doc for doc, _ in results_with_scores]

        # 2) Get embeddings for MMR
        query_vec = self.embeddings.embed_query(query)
        doc_vecs = [self.embeddings.embed_query(d.page_content) for d in docs]

        # 3) MMR selection
        selected_indices = self._mmr_select(query_vec, doc_vecs, k=k, lambda_param=0.6)
        mmr_docs = [docs[i] for i in selected_indices]

        # 4) Light rerank for lexical relevance
        reranked = self._light_rerank(mmr_docs, query)

        return reranked

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

    def _cosine(self, a: List[float], b: List[float]) -> float:
        # safety: if vectors are empty
        if not a or not b:
            return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _mmr_select(self, query_vec, doc_vecs, k: int, lambda_param: float = 0.5):
        """
        MMR: pick documents that are relevant to query and diverse among themselves.
        """
        if not doc_vecs:
            return []

        selected = []
        candidates = list(range(len(doc_vecs)))

        # First pick: most similar to query
        sims = [self._cosine(query_vec, v) for v in doc_vecs]
        first = max(range(len(sims)), key=lambda i: sims[i])
        selected.append(first)
        candidates.remove(first)

        while len(selected) < min(k, len(doc_vecs)) and candidates:
            mmr_scores = []
            for i in candidates:
                sim_to_query = self._cosine(query_vec, doc_vecs[i])
                sim_to_selected = max(
                    self._cosine(doc_vecs[i], doc_vecs[j]) for j in selected
                )
                mmr = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
                mmr_scores.append((mmr, i))

            _, best = max(mmr_scores, key=lambda x: x[0])
            selected.append(best)
            candidates.remove(best)

        return selected

    def _light_rerank(self, docs, query: str):
        """
        Simple lexical boost: prefer chunks that mention query terms more.
        """
        q_terms = [t.lower() for t in query.split() if t.strip()]

        def score(doc):
            text = doc.page_content.lower()
            hits = sum(1 for t in q_terms if t in text)
            return hits

        return sorted(docs, key=score, reverse=True)





