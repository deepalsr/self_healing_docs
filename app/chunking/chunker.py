# app/chunking/chunker.py

def build_chunk(entry: dict, change_type: str, file_path: str) -> dict:
    """
    Converts one diff entry (added/deleted/modified) into a chunk
    ready for embedding.

    Returns:
    {
      "chunk_id": str,        # e.g. "src/users.py::get_user"
      "embed_text": str,      # clean, current-state text -> goes to embedding model
      "metadata": dict        # diff facts -> stored in Pinecone, used at generation time
    }
    """
    name = entry["name"]
    chunk_id = f"{file_path}::{name}"

    if change_type == "deleted":
        embed_text = f"Function: {name}\nFile: {file_path}\n{entry['signature']}"
    else:
        # added or modified -> use current (new) state
        if change_type == "deleted":
            embed_text = f"Function: {name}\nFile: {file_path}\n{entry['signature']}"
        elif change_type == "modified":
            embed_text = f"Function: {name}\nFile: {file_path}\n{entry['new_signature']}\n\n{entry['new_body']}"
        else:  # added
            embed_text = f"Function: {name}\nFile: {file_path}\n{entry['signature']}\n\n{entry['body']}"
    metadata = {
        "chunk_id": chunk_id,
        "function_name": name,
        "file_path": file_path,
        "change_type": change_type,  # "added" | "deleted" | "modified"
    }

    if change_type == "modified":
        metadata["old_signature"] = entry["old_signature"]
        metadata["new_signature"] = entry["new_signature"]
        metadata["modification_type"] = entry["change_type"]  # "signature" | "body_only" | "both"

    return {
        "chunk_id": chunk_id,
        "embed_text": embed_text,
        "metadata": metadata,
    }


def chunk_diff(diff_result: dict, file_path: str) -> list[dict]:
    """
    Takes the full output of diff_definitions() and produces
    a list of chunks ready for embedding.
    """
    chunks = []

    for entry in diff_result.get("added", []):
        chunks.append(build_chunk(entry, "added", file_path))

    for entry in diff_result.get("deleted", []):
        chunks.append(build_chunk(entry, "deleted", file_path))

    for entry in diff_result.get("modified", []):
        chunks.append(build_chunk(entry, "modified", file_path))

    return chunks