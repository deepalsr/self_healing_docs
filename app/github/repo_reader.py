# app/github/repo_reader.py

import base64
import httpx
from app.config import settings

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