# app/generation/llm_client.py

import json
from google import genai
from app.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)
GENERATION_MODEL = "gemini-2.0-flash"


def generate_doc_patch(diff_facts: dict, retrieval_result: dict, feedback: str | None = None) -> dict:
    """
    Calls the LLM to produce a structured doc patch.
    diff_facts: the metadata for one modified/added function (old/new signature, new_body, etc.)
    retrieval_result: output of retrieve_context_for_chunk() -- exact match or similarity fallback
    feedback: if this is a retry after a guardrail rejection, the reason it failed
    """
    prompt = _build_prompt(diff_facts, retrieval_result, feedback)

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={"temperature": 0.2},  # low temperature: we want precise, not creative
    )

    raw_text = response.text.strip()
    # Guard against the model wrapping output in markdown fences despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").removeprefix("json").strip()

    return json.loads(raw_text)  # will raise if the model didn't return valid JSON -- caught by caller


def _build_prompt(diff_facts: dict, retrieval_result: dict, feedback: str | None) -> str:
    strategy = retrieval_result["strategy"]
    results = retrieval_result["results"]

    if strategy == "exact_match":
        existing_doc_section = results[0]["metadata"].get("doc_snippet", "[no doc_snippet found in metadata]")
        doc_context = f"EXISTING DOCUMENTATION FOR THIS FUNCTION (must be edited precisely):\n{existing_doc_section}"
    else:
        similar_snippets = "\n---\n".join(
            r["metadata"].get("doc_snippet", "") for r in results
        )
        doc_context = (
            "NO EXISTING DOCUMENTATION FOR THIS FUNCTION. "
            "Here are similar functions' docs for STYLE REFERENCE ONLY "
            "(do not copy their content, only match tone/format):\n"
            f"{similar_snippets}"
        )

    feedback_block = ""
    if feedback:
        feedback_block = f"\nYOUR PREVIOUS ATTEMPT WAS REJECTED. REASON: {feedback}\nFix this and try again.\n"

    return f"""You are a documentation-patching assistant. You must output ONLY valid JSON, nothing else.

FUNCTION: {diff_facts['function_name']}
CHANGE TYPE: {diff_facts.get('modification_type', diff_facts.get('change_type'))}
OLD SIGNATURE: {diff_facts.get('old_signature', 'N/A -- new function')}
NEW SIGNATURE: {diff_facts.get('new_signature', diff_facts.get('signature'))}
NEW IMPLEMENTATION:
{diff_facts.get('new_body', diff_facts.get('body', ''))}

{doc_context}
{feedback_block}
Output a JSON object in this EXACT schema, nothing else, no markdown fences:
{{
  "changes": [
    {{
      "action": "replace",
      "target_anchor": "{diff_facts['function_name']}",
      "old_text": "<exact existing text you are replacing, copied verbatim from the existing documentation above>",
      "new_text": "<your updated documentation text>"
    }}
  ]
}}

Rules:
- old_text must be copied EXACTLY from the existing documentation shown above, character for character.
- Only describe what actually changed. Do not rewrite unrelated sentences.
- If there is no existing documentation (new function), set old_text to an empty string and action to "insert".
"""