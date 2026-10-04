# app/github/pr_manager.py

import base64
import httpx
from app.config import settings
import time
from app.generation.pipeline import PipelineResult

GITHUB_API_BASE = "https://api.github.com"


def _headers():
    return {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def _get(owner: str, repo: str, path: str, params: dict | None = None) -> httpx.Response:
    resp = httpx.get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/{path}", headers=_headers(), params=params)
    resp.raise_for_status()
    return resp


def _post(owner: str, repo: str, path: str, json_body: dict) -> httpx.Response:
    resp = httpx.post(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/{path}", headers=_headers(), json=json_body)
    resp.raise_for_status()
    return resp


def _put(owner: str, repo: str, path: str, json_body: dict) -> httpx.Response:
    resp = httpx.put(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/{path}", headers=_headers(), json=json_body)
    resp.raise_for_status()
    return resp


def create_branch(owner: str, repo: str, base_branch: str, new_branch: str) -> None:
    base_ref = _get(owner, repo, f"git/ref/heads/{base_branch}").json()
    base_sha = base_ref["object"]["sha"]
    _post(owner, repo, "git/refs", {"ref": f"refs/heads/{new_branch}", "sha": base_sha})


def commit_file(owner: str, repo: str, branch: str, file_path: str, new_content: str, message: str) -> None:
    existing = _get(owner, repo, f"contents/{file_path}", params={"ref": branch}).json()
    encoded = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")
    _put(owner, repo, f"contents/{file_path}", {
        "message": message,
        "content": encoded,
        "sha": existing["sha"],  # required by GitHub to prove we're updating, not clobbering blind
        "branch": branch,
    })


def open_pull_request(
    owner: str, repo: str, base_branch: str, head_branch: str,
    title: str, body: str, draft: bool = False,
) -> dict:
    resp = _post(owner, repo, "pulls", {
        "title": title,
        "head": head_branch,
        "base": base_branch,
        "body": body,
        "draft": draft,
    })
    return resp.json()

def open_doc_pr(
    owner: str, repo: str, base_branch: str,
    file_path: str, function_name: str, result: PipelineResult,
    insert_reference: str | None = None,
) -> dict | None:
    branch = f"docs-update/{function_name}-{int(time.time())}"
    create_branch(owner, repo, base_branch, branch)

    if result.status == "approved":
        from app.github.repo_reader import fetch_file_content
        from app.github.patch_applier import apply_patch

        current_content = fetch_file_content(owner, repo, file_path, branch)
        apply_result = apply_patch(current_content, result.patch, insert_reference=insert_reference)

        if not apply_result.success:
            # The file changed between generation and apply -- escalate,
            # don't silently give up. Same "fail loud" policy as everywhere else.
            body = (
                f"⚠️ Needs human review — automated apply failed after generation succeeded.\n\n"
                f"Reason: {apply_result.reason}"
            )
            commit_file(owner, repo, branch, file_path, current_content,
                        f"docs: attempted update for {function_name} (apply failed)")
            return open_pull_request(
                owner, repo, base_branch, branch,
                title=f"[Needs review] Docs update for {function_name}",
                body=body, draft=True,
            )

        commit_file(owner, repo, branch, file_path, apply_result.new_content,
                    f"docs: update {function_name} documentation")
        return open_pull_request(
            owner, repo, base_branch, branch,
            title=f"Update docs for {function_name}",
            body=f"Automated documentation update for `{function_name}`.\n\nVerified attempts: {result.attempts}.",
            draft=False,
        )

    # status == "needs_human_review"
    from app.github.repo_reader import fetch_file_content

    current_content = fetch_file_content(owner, repo, file_path, branch)
    commit_file(owner, repo, branch, file_path, current_content,
                f"docs: no automated update generated for {function_name} (needs review)")
    return open_pull_request(
        owner, repo, base_branch, branch,
        title=f"[Needs review] Docs update for {function_name}",
        body=(
            f"⚠️ Needs human review — automated generation could not produce a "
            f"verified patch after {result.attempts} attempts.\n\n"
            f"Last rejection reason: {result.reason}"
        ),
        draft=True,
    )