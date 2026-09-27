# app/github/repo_reader.py

import base64
import httpx
from app.config import settings
from app.github.doc_extractor import extract_doc_section


GITHUB_API_BASE = "https://api.github.com"


def fetch_file_content(owner: str, repo: str, path: str, ref: str) -> str | None:
    """
    Fetches a file's raw text content from GitHub at a specific ref (commit/branch).
    Returns None if the file doesn't exist at that ref (e.g., new repo with no README yet).
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    params = {"ref": ref}

    response = httpx.get(url, headers=headers, params=params)

    if response.status_code == 404:
        return None

    response.raise_for_status()  # raises for any other error (401, 403, 500, etc.) -- fail loud, not silent

    data = response.json()
    encoded_content = data["content"]
    decoded_bytes = base64.b64decode(encoded_content)
    return decoded_bytes.decode("utf-8")

def fetch_doc_snippets_for_functions(
    owner: str, repo: str, doc_path: str, ref: str, function_names: list[str]
) -> dict[str, str | None]:
    """
    Fetches a doc file once, then extracts each function's existing
    doc section from it. Returns {function_name: snippet_or_None}.
    This is the impure "edge" -- chunking itself stays pure and untouched.
    """
    doc_text = fetch_file_content(owner, repo, doc_path, ref)

    if doc_text is None:
        # doc file doesn't exist at all -- every function gets None
        return {name: None for name in function_names}

    return {
        name: extract_doc_section(doc_text, name)
        for name in function_names
    }