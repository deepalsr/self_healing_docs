# app/generation/llm_client.py

from google import genai
from google.genai import types
from google.genai import errors
from app.config import settings
import time

_MAX_TRANSIENT_RETRIES = 3
_BASE_DELAY_SECONDS = 2

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def generate_json(prompt: str, temperature: float = 0.2) -> str:
    last_error = None
    for attempt in range(1, _MAX_TRANSIENT_RETRIES + 1):
        try:
            response = _get_client().models.generate_content(
                model=settings.GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            return response.text
        except errors.ServerError as e:
            last_error = e
            if attempt < _MAX_TRANSIENT_RETRIES:
                time.sleep(_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))

    raise last_error