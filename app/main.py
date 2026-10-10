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
from app.github.doc_resolver import resolve_doc_path

import logging
from app.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("self_healing_docs")

app = FastAPI()


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

    logger.info(f"Processing push: repo={owner}/{repo} after={after_sha[:7]}")
    ensure_index_exists()

    changed_files = _changed_python_files(payload)
    logger.info(f"Changed Python files in this push: {sorted(changed_files)}")

    for file_path in changed_files:
        try:
            old_content = fetch_file_content(owner, repo, file_path, before_sha) or ""
            new_content = fetch_file_content(owner, repo, file_path, after_sha) or ""

            diff_result = diff_definitions(old_content, new_content)
            changed_names = [e["name"] for e in diff_result["added"] + diff_result["modified"]]
            if not changed_names:
                logger.info(f"{file_path}: only deletions, skipping doc generation")
                continue

            doc_path = resolve_doc_path(owner, repo, file_path, after_sha)
            logger.info(f"{file_path}: changed functions = {changed_names}, doc target = {doc_path}")

            doc_snippets = fetch_doc_snippets_for_functions(
                owner, repo, doc_path, after_sha, changed_names
            )
            chunks = chunk_diff(diff_result, file_path, doc_snippets)
            sync_chunks_to_pinecone(chunks, namespace=repo)

            for chunk in chunks:
                if chunk["metadata"]["change_type"] == "deleted":
                    continue

                func_name = chunk["metadata"]["function_name"]
                retrieval_result = retrieve_context_for_chunk(chunk, namespace=repo)
                result = generate_verified_patch(chunk, retrieval_result)

                # For a genuinely new function (insert, not replace), Tier 2's
                # similarity match -- if one exists -- tells us where in the
                # file this new content belongs, instead of always appending
                # at the end. Tier 1 (exact match) means an existing section
                # is being replaced, so there's no "near" to compute.
                insert_reference = None
                if retrieval_result["strategy"] == "similarity_fallback" and retrieval_result["results"]:
                    insert_reference = retrieval_result["results"][0]["metadata"].get("doc_snippet")

                logger.info(
                    f"{func_name}: pipeline status={result.status} attempts={result.attempts}"
                    + (f" reason={result.reason}" if result.reason else "")
                )

                open_doc_pr(
                    owner=owner, repo=repo, base_branch=base_branch,
                    file_path=doc_path,
                    function_name=func_name, result=result,
                    insert_reference=insert_reference,
                )
                logger.info(f"{func_name}: PR opened (status={result.status})")

        except Exception:
            # One file's failure must not silently swallow the rest of the push,
            # and must not vanish without a trace either -- same "fail loud"
            # policy as the guardrail escalation, applied to the orchestrator itself.
            logger.exception(f"Unhandled error processing {file_path} in {owner}/{repo}")


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

    full_name = payload.get("repository", {}).get("full_name", "")
    allowed = settings.allowed_repos_set()
    if allowed and full_name not in allowed:
        logger.warning(f"Rejected push from non-allowlisted repo: {full_name}")
        raise HTTPException(status_code=403, detail="Repository not authorized for this deployment")

    background_tasks.add_task(process_webhook_event, payload)

    return {"status": "accepted"}

