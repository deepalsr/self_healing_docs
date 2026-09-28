from google import genai
from google.genai import types
from app.config import settings

_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def generate_json(prompt: str, temperature: float = 0.2) -> str:
    """
    The ONLY function that knows which LLM provider we use.
    JSON mode constrains the output format at the API level, which is
    sturdier than begging the model in the prompt and stripping markdown fences.
    """
    response = _client.models.generate_content(
        model=settings.GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return response.text