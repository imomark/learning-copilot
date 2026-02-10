from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

def get_gemini_llm():
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-lite",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,
    )
    return llm
