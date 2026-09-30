# scratch_test_live_generation.py

from app.generation.pipeline import generate_verified_patch

chunk = {
    "chunk_id": "src/users.py::get_user",
    "embed_text": (
        "Function: get_user\nFile: src/users.py\n"
        "def get_user(id, include_deleted=False)\n\n"
        "def get_user(id, include_deleted=False):\n"
        "    if include_deleted:\n"
        "        return db.query(id, deleted=True)\n"
        "    return db.query(id)"
    ),
    "metadata": {
        "function_name": "get_user",
        "change_type": "modified",
        "modification_type": "both",
        "old_signature": "def get_user(id)",
        "new_signature": "def get_user(id, include_deleted=False)",
        "doc_snippet": "Retrieves a user by their ID.",
    },
}

retrieval_result = {"strategy": "exact_match", "results": []}

result = generate_verified_patch(chunk, retrieval_result)

print("STATUS:", result.status)
print("ATTEMPTS:", result.attempts)
print("REASON:", result.reason)
if result.patch:
    for c in result.patch.changes:
        print("---")
        print("action:", c.action)
        print("old_text:", c.old_text)
        print("new_text:", c.new_text)