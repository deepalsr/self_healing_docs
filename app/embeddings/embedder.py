# app/embeddings/embedder.py

# app/embeddings/embedder.py

from google import genai
from google.genai import types
from app.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768


def embed_text(text: str) -> list[float]:
    """
    Converts a piece of text into a vector using Gemini's embedding model.
    Explicitly requests 768 dimensions to match our Pinecone index config --
    the model's default is 3072, so this must be set explicitly or every
    upsert will fail with a dimension mismatch.
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
    )
    return result.embeddings[0].values