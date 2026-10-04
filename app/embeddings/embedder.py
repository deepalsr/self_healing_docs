# app/embeddings/embedder.py

from google import genai
from google.genai import types
from app.config import settings

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def embed_text(text: str) -> list[float]:
    """
    Converts a piece of text into a vector using Gemini's embedding model.
    Explicitly requests 768 dimensions to match our Pinecone index config.
    """
    result = _get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
    )
    return result.embeddings[0].values