from fastapi import FastAPI
from app.core.config import settings
from app.llm.gemini import get_gemini_llm
from app.vectorstore.qdrant_store import QdrantStore
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str


app = FastAPI(title="AI Learning Copilot")
# create one global store instance for now
vector_store = QdrantStore()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "has_google_api_key": bool(settings.GOOGLE_API_KEY)
    }

@app.get("/llm/ping")
def llm_ping():
    llm = get_gemini_llm()
    response = llm.invoke("Reply with exactly: 'Gemini is alive'")
    return {
        "response": response.content
    }

@app.post("/vector/test-ingest")
def test_ingest():
    texts = [
        "Kafka is a distributed event streaming platform used for high-performance data pipelines.",
        "Spring Boot is a Java framework used to build microservices quickly.",
        "FastAPI is a modern Python web framework for building APIs with high performance."
    ]

    metadatas = [
        {"source": "sample", "topic": "kafka"},
        {"source": "sample", "topic": "spring"},
        {"source": "sample", "topic": "fastapi"},
    ]

    vector_store.add_texts(texts=texts, metadatas=metadatas)

    count = vector_store.count()

    return {
        "status": "stored",
        "total_vectors": count
    }

@app.post("/vector/test-search")
def test_search(req: SearchRequest):
    results = vector_store.search(query=req.query, k=3)

    return {
        "query": req.query,
        "results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]
    }

