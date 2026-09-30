# app/generation/llm_client.py

import time
from google import genai
from google.genai import types
from google.genai import errors
from app.config import settings

_client = genai.Client(api_key=settings.GEMINI_API_KEY)

_MAX_TRANSIENT_RETRIES = 3
_BASE_DELAY_SECONDS = 2


def generate_json(prompt: str, temperature: float = 0.2) -> str:
    """
    The only function that knows which LLM provider we use.
    Retries transient server errors (503, 5xx) with exponential backoff,
    unchanged -- this has nothing to do with prompt quality, so the prompt
    is never touched. Non-transient errors (401, 400, etc.) are raised
    immediately, since retrying them would just fail the same way again.
    """
    last_error = None
    for attempt in range(1, _MAX_TRANSIENT_RETRIES + 1):
        try:
            response = _client.models.generate_content(
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
                delay = _BASE_DELAY_SECONDS * (2 ** (attempt - 1))  # 2s, 4s, 8s
                time.sleep(delay)

    raise last_error