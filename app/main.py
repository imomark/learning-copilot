from fastapi import FastAPI
from app.core.config import settings
from app.llm.gemini import get_gemini_llm

app = FastAPI(title="AI Learning Copilot")

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
