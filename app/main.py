from fastapi import FastAPI
from app.core.config import settings
from app.llm.gemini import get_gemini_llm
from app.rag.prompt import build_rag_prompt
from app.rag.quiz_prompt import build_quiz_prompt
from app.rag.summarize_prompt import build_summarize_prompt
from app.rag.test_prompt import build_test_question_prompt, build_test_grader_prompt
from app.sessions.db_store import DBSessionStore
from app.sessions.store import SessionStore
from app.vectorstore.qdrant_store import QdrantStore
from pydantic import BaseModel
from fastapi import UploadFile, File
from app.ingestion.pdf_ingestor import PDFIngestor
from app.db import engine, Base
from app import models


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

class TestQuestionRequest(BaseModel):
    focus: str | None = None
    k: int = 5

class TestAnswerRequest(BaseModel):
    question: str
    user_answer: str
    k: int = 5

class StartSessionRequest(BaseModel):
    focus: str | None = None

class SessionQuestionRequest(BaseModel):
    session_id: str
    k: int = 5

class SessionAnswerRequest(BaseModel):
    session_id: str
    question: str
    user_answer: str
    k: int = 5




app = FastAPI(title="AI Learning Copilot")
# create one global store instance for now
vector_store = QdrantStore()


session_store = DBSessionStore()


pdf_ingestor = PDFIngestor(vector_store)

Base.metadata.create_all(bind=engine)


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


@app.post("/rag/test-me/question")
def test_me_question(req: TestQuestionRequest):
    query = req.focus if req.focus else "Generate a challenging question from the documents"

    results = vector_store.search(query=query, k=req.k)

    if not results:
        return {
            "question": "I don't have any knowledge yet. Please ingest some documents first.",
            "citations": []
        }

    context_chunks = [doc.page_content for doc in results]

    prompt = build_test_question_prompt(context_chunks, req.focus)

    llm = get_gemini_llm()
    response = llm.invoke(prompt)

    citations = [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
        }
        for doc in results
    ]

    return {
        "question": response.content.strip(),
        "citations": citations
    }

@app.post("/rag/test-me/answer")
def test_me_answer(req: TestAnswerRequest):
    # Retrieve context again (simple stateless approach)
    results = vector_store.search(query=req.question, k=req.k)

    if not results:
        return {
            "grade": "Unknown",
            "feedback": "No relevant context found to grade this answer.",
            "citations": []
        }

    context_chunks = [doc.page_content for doc in results]

    prompt = build_test_grader_prompt(context_chunks, req.question, req.user_answer)

    llm = get_gemini_llm()
    response = llm.invoke(prompt)

    citations = [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
        }
        for doc in results
    ]

    return {
        "grade_and_feedback": response.content.strip(),
        "citations": citations
    }

@app.post("/rag/test-me/session/start")
def start_test_session(req: StartSessionRequest):
    s = session_store.create(req.focus)
    return {"session_id": s.id, "focus": s.focus}


@app.post("/rag/test-me/session/question")
def session_question(req: SessionQuestionRequest):
    s = session_store.get(req.session_id)
    if not s:
        return {"error": "Invalid session_id"}

    query = s.focus if s.focus else "Generate a challenging question from the documents"
    results = vector_store.search(query=query, k=req.k)

    if not results:
        return {"question": "No knowledge yet.", "citations": []}

    context_chunks = [doc.page_content for doc in results]
    prompt = build_test_question_prompt(context_chunks, s.focus)

    llm = get_gemini_llm()
    response = llm.invoke(prompt)

    citations = [{"chunk_id": d.metadata.get("chunk_id"), "source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in results]

    return {
        "question": response.content.strip(),
        "citations": citations,
        "session_summary": session_store.summary(req.session_id),
    }


@app.post("/rag/test-me/session/answer")
def session_answer(req: SessionAnswerRequest):
    s = session_store.get(req.session_id)
    if not s:
        return {"error": "Invalid session_id"}

    results = vector_store.search(query=req.question, k=req.k)
    if not results:
        return {"grade_and_feedback": "No context.", "citations": [], "session_summary": session_store.summary(req.session_id)}

    context_chunks = [doc.page_content for doc in results]
    prompt = build_test_grader_prompt(context_chunks, req.question, req.user_answer)

    llm = get_gemini_llm()
    response = llm.invoke(prompt)
    grade_text = response.content.strip()

    topic = s.focus or results[0].metadata.get("topic") or "general"

    session_store.record_attempt(req.session_id, req.question, req.user_answer, grade_text, topic)

    citations = [{"chunk_id": d.metadata.get("chunk_id"), "source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in results]

    return {
        "grade_and_feedback": grade_text,
        "citations": citations,
        "session_summary": session_store.summary(req.session_id),
    }


@app.get("/rag/test-me/session/{session_id}/weak-areas")
def session_weak_areas(session_id: str):
    ranked = session_store.weak_areas(session_id)
    if ranked is None:
        return {"error": "Invalid session_id"}

    recommendations = [
        {
            "topic": item["topic"],
            "suggested_actions": [
                f"Run /rag/summarize with focus='{item['topic']}'",
                f"Run /rag/quiz with focus='{item['topic']}'",
                f"Run /rag/test-me/session/question with focus='{item['topic']}'",
            ],
        }
        for item in ranked[:5]
    ]

    return {
        "session_summary": session_store.summary(session_id),
        "ranked_weak_areas": ranked,
        "recommendations": recommendations,
    }



