# app/chunking/chunker.py

def build_chunk(entry: dict, change_type: str, file_path: str, doc_snippet: str | None) -> dict:
    if change_type == "renamed":
        name = entry["new_name"]
    else:
        name = entry["name"]
    chunk_id = f"{file_path}::{name}"

    if change_type == "deleted":
        embed_text = f"Function: {name}\nFile: {file_path}\n{entry['signature']}"
    elif change_type == "modified":
        embed_text = f"Function: {name}\nFile: {file_path}\n{entry['new_signature']}\n\n{entry['new_body']}"
    elif change_type == "renamed":
        embed_text = f"Function: {name}\nFile: {file_path}\n{entry['new_signature']}\n\n{entry['new_body']}"
    else:  # added
        embed_text = f"Function: {name}\nFile: {file_path}\n{entry['signature']}\n\n{entry['body']}"

    metadata = {
        "chunk_id": chunk_id,
        "function_name": name,
        "file_path": file_path,
        "change_type": change_type,
    }

    if doc_snippet is not None:
        metadata["doc_snippet"] = doc_snippet

    if change_type == "modified":
        metadata["old_signature"] = entry["old_signature"]
        metadata["new_signature"] = entry["new_signature"]
        metadata["modification_type"] = entry["change_type"]

    if change_type == "renamed":
        metadata["old_name"] = entry["old_name"]
        metadata["old_signature"] = entry["old_signature"]
        metadata["new_signature"] = entry["new_signature"]
        metadata["modification_type"] = entry["change_type"]  # "none" / "signature" / "body_only" / "both"
        metadata["old_chunk_id"] = f"{file_path}::{entry['old_name']}"

    return {"chunk_id": chunk_id, "embed_text": embed_text, "metadata": metadata}


def chunk_diff(diff_result: dict, file_path: str, doc_snippets: dict[str, str | None]) -> list[dict]:
    chunks = []

    for entry in diff_result.get("added", []):
        chunks.append(build_chunk(entry, "added", file_path, doc_snippets.get(entry["name"])))

    for entry in diff_result.get("deleted", []):
        chunks.append(build_chunk(entry, "deleted", file_path, doc_snippets.get(entry["name"])))

    for entry in diff_result.get("modified", []):
        chunks.append(build_chunk(entry, "modified", file_path, doc_snippets.get(entry["name"])))

    for entry in diff_result.get("renamed", []):
        chunks.append(build_chunk(entry, "renamed", file_path, doc_snippets.get(entry["old_name"])))

    return chunks