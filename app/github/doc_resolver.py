# app/github/doc_resolver.py

import os
from app.github.repo_reader import fetch_file_content

DEFAULT_FALLBACK = "README.md"


def candidate_doc_path(source_file_path: str) -> str:
    """
    Pure: maps a source file to its conventional per-module doc path.
    src/ingest.py -> docs/ingest.md
    app/auth/login.py -> docs/login.md
    No network, no guessing beyond this one convention.
    """
    base = os.path.splitext(os.path.basename(source_file_path))[0]
    return f"docs/{base}.md"


def resolve_doc_path(owner: str, repo: str, source_file_path: str, ref: str) -> str:
    """
    Impure: checks whether the conventional per-module doc file actually
    exists in the repo at this ref. Falls back to README.md if not --
    most repos won't have adopted a docs/ folder, and that's a normal,
    expected case, not an error (same reasoning as doc_snippet being
    absent for an undocumented function).
    """
    candidate = candidate_doc_path(source_file_path)
    if fetch_file_content(owner, repo, candidate, ref) is not None:
        return candidate
    return DEFAULT_FALLBACK