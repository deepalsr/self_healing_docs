# app/main.py

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Header
import hmac
import hashlib
from app.config import settings

from app.diffing.ast_parser import diff_definitions
from app.chunking.chunker import chunk_diff
from app.vectorstore.pinecone_client import sync_chunks_to_pinecone, ensure_index_exists
from app.retrieval.retriever import retrieve_context_for_chunk
from app.generation.pipeline import generate_verified_patch
from app.github.repo_reader import fetch_file_content, fetch_doc_snippets_for_functions
from app.github.pr_manager import open_doc_pr

app = FastAPI()

DOC_FILE_PATH = "README.md"  # v1: single known doc file; multi-file docs is a v2 scope


def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    if not signature_header:
        return False
    if not signature_header.startswith("sha256="):
        return False

    expected_signature = hmac.new(
        key=settings.WEBHOOK_SECRET.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256
    ).hexdigest()

    received_signature = signature_header.split("sha256=")[-1]
    return hmac.compare_digest(expected_signature, received_signature)


def _extract_owner_repo(payload: dict) -> tuple[str, str]:
    full_name = payload["repository"]["full_name"]
    owner, repo = full_name.split("/", 1)
    return owner, repo


def _changed_python_files(payload: dict) -> set[str]:
    changed = set()
    for commit in payload.get("commits", []):
        for file_list in (commit.get("added", []), commit.get("modified", [])):
            changed.update(f for f in file_list if f.endswith(".py"))
    return changed


def process_webhook_event(payload: dict):
    owner, repo = _extract_owner_repo(payload)
    before_sha = payload["before"]
    after_sha = payload["after"]
    base_branch = payload["ref"].removeprefix("refs/heads/")

    ensure_index_exists()

    for file_path in _changed_python_files(payload):
        old_content = fetch_file_content(owner, repo, file_path, before_sha) or ""
        new_content = fetch_file_content(owner, repo, file_path, after_sha) or ""

        diff_result = diff_definitions(old_content, new_content)
        changed_names = [e["name"] for e in diff_result["added"] + diff_result["modified"]]
        if not changed_names:
            continue

        doc_snippets = fetch_doc_snippets_for_functions(
            owner, repo, DOC_FILE_PATH, after_sha, changed_names
        )

        chunks = chunk_diff(diff_result, file_path, doc_snippets)
        sync_chunks_to_pinecone(chunks, namespace=repo)

        for chunk in chunks:
            if chunk["metadata"]["change_type"] == "deleted":
                continue

            retrieval_result = retrieve_context_for_chunk(chunk, namespace=repo)
            result = generate_verified_patch(chunk, retrieval_result)
            open_doc_pr(
                owner=owner, repo=repo, base_branch=base_branch,
                file_path=DOC_FILE_PATH,
                function_name=chunk["metadata"]["function_name"],
                result=result,
            )


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None),
):
    payload_body = await request.body()

    if not verify_signature(payload_body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event != "push":
        return {"status": "ignored", "reason": f"event type '{x_github_event}' not handled"}

    payload = await request.json()

    if payload.get("ref") != "refs/heads/main":
        return {"status": "ignored", "reason": "not the main branch"}

    background_tasks.add_task(process_webhook_event, payload)

    return {"status": "accepted"}