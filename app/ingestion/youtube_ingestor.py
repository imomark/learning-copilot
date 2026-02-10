import uuid
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore.qdrant_store import QdrantStore

class YouTubeIngestor:
    def __init__(self, vector_store: QdrantStore):
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

    def ingest(self, url: str) -> int:
        # 1. Load ONLY transcript (avoid pytube video info)
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False  # ✅ IMPORTANT: avoids pytube metadata fetch
        )

        documents = loader.load()

        if not documents:
            raise ValueError("No transcript found for this video. It may not have captions.")

        # 2. Split into chunks
        chunks = self.splitter.split_documents(documents)

        # 3. Add metadata
        for doc in chunks:
            doc.metadata["source"] = url
            doc.metadata["chunk_id"] = str(uuid.uuid4())
            doc.metadata["type"] = "youtube"

        # 4. Store in vector DB
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]

        self.vector_store.add_texts(texts=texts, metadatas=metadatas)

        return len(chunks)
