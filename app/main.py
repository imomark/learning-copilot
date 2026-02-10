from fastapi import FastAPI
from app.core.config import settings
from app.llm.gemini import get_gemini_llm
from app.rag.prompt import build_rag_prompt
from app.rag.quiz_prompt import build_quiz_prompt
from app.rag.summarize_prompt import build_summarize_prompt
from app.vectorstore.qdrant_store import QdrantStore
from pydantic import BaseModel
from fastapi import UploadFile, File
from app.ingestion.pdf_ingestor import PDFIngestor


class SearchRequest(BaseModel):
    query: str

class AskRequest(BaseModel):
    question: str

class SummarizeRequest(BaseModel):
    focus: str | None = None
    k: int = 5

class QuizRequest(BaseModel):
    focus: str | None = None   # e.g., "Kafka architecture"
    k: int = 5                 # how many chunks to retrieve
    num_questions: int = 5     # how many questions to generate


app = FastAPI(title="AI Learning Copilot")
# create one global store instance for now
vector_store = QdrantStore()

pdf_ingestor = PDFIngestor(vector_store)


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

@app.post("/rag/ask")
def rag_ask(req: AskRequest):
    # 1. Retrieve relevant docs
    results = vector_store.search(query=req.question, k=3)

    if not results:
        return {
            "question": req.question,
            "answer": "I don't have any knowledge yet. Please ingest some documents first.",
            "citations": []
        }

    # 2. Extract text for context
    context_chunks = [doc.page_content for doc in results]

    # 3. Build prompt
    prompt = build_rag_prompt(context_chunks, req.question)

    # 4. Call Gemini
    llm = get_gemini_llm()
    response = llm.invoke(prompt)

    # 5. Build structured citations
    citations = []
    for doc in results:
        citations.append({
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
        })

    return {
        "question": req.question,
        "answer": response.content,
        "citations": citations
    }

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    file_bytes = await file.read()

    chunks_added = pdf_ingestor.ingest(file_bytes, file.filename)

    total_vectors = vector_store.count()

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": chunks_added,
        "total_vectors": total_vectors,
    }

@app.post("/rag/summarize")
def rag_summarize(req: SummarizeRequest):
    # 1. Retrieve relevant docs
    # If focus is provided, use it as the retrieval query; otherwise use a generic query
    query = req.focus if req.focus else "Summarize the main topics of the documents"

    results = vector_store.search(query=query, k=req.k)

    if not results:
        return {
            "summary": "I don't have any knowledge yet. Please ingest some documents first.",
            "citations": []
        }

    # 2. Extract text for context
    context_chunks = [doc.page_content for doc in results]

    # 3. Build prompt
    prompt = build_summarize_prompt(context_chunks, req.focus)

    # 4. Call Gemini
    llm = get_gemini_llm()
    response = llm.invoke(prompt)

    # 5. Build structured citations
    citations = []
    for doc in results:
        citations.append({
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
        })

    return {
        "summary": response.content,
        "citations": citations
    }

@app.post("/rag/quiz")
def rag_quiz(req: QuizRequest):
    # 1. Choose retrieval query
    query = req.focus if req.focus else "Generate a quiz from the main topics of the documents"

    # 2. Retrieve relevant chunks
    results = vector_store.search(query=query, k=req.k)

    if not results:
        return {
            "quiz": "I don't have any knowledge yet. Please ingest some documents first.",
            "citations": []
        }

    # 3. Build context
    context_chunks = [doc.page_content for doc in results]

    # 4. Build prompt
    prompt = build_quiz_prompt(context_chunks, req.focus, req.num_questions)

    # 5. Call Gemini
    llm = get_gemini_llm()
    response = llm.invoke(prompt)

    # 6. Build structured citations
    citations = []
    for doc in results:
        citations.append({
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
        })

    return {
        "quiz": response.content,
        "citations": citations
    }



