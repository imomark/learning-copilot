import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore.qdrant_store import QdrantStore

class PDFIngestor:
    def __init__(self, vector_store: QdrantStore):
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

    def ingest(self, file_bytes: bytes, filename: str) -> int:
        # 1. Save to temp file (PyPDFLoader needs a path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # 2. Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # 3. Split into chunks
            chunks = self.splitter.split_documents(documents)

            import uuid

            # 4. Add metadata (source, chunk_id, page)
            for doc in chunks:
                doc.metadata["source"] = filename
                doc.metadata["chunk_id"] = str(uuid.uuid4())

                # LangChain PyPDFLoader usually provides page in metadata
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata.get("page")
                else:
                    doc.metadata["page"] = None

            # 5. Store in vector DB
            texts = [doc.page_content for doc in chunks]
            metadatas = [doc.metadata for doc in chunks]

            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

            return len(chunks)
        finally:
            # 6. Cleanup temp file
            os.remove(tmp_path)
