import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if not settings.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set")

        _embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            api_key=settings.GOOGLE_API_KEY
        )
    return _embedding_model
