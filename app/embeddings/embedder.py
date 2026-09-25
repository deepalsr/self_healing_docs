# app/embeddings/embedder.py

from google import genai
from app.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 768


def embed_text(text: str) -> list[float]:
    """
    Converts a piece of text into a vector using Gemini's embedding model.
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values