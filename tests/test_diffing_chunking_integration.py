# tests/test_diffing_chunking_integration.py

from app.diffing.ast_parser import diff_definitions
from app.chunking.chunker import chunk_diff

OLD_CODE = """
def get_user(id):
    return db.query(id)

def delete_user(id):
    db.remove(id)
"""

NEW_CODE = """
def get_user(id, include_deleted=False):
    if include_deleted:
        return db.query(id, deleted=True)
    return db.query(id)

def create_user(name):
    return db.insert(name)
"""


def test_diff_detects_added_function():
    result = diff_definitions(OLD_CODE, NEW_CODE)
    added_names = [e["name"] for e in result["added"]]
    assert "create_user" in added_names


def test_diff_detects_deleted_function():
    result = diff_definitions(OLD_CODE, NEW_CODE)
    deleted_names = [e["name"] for e in result["deleted"]]
    assert "delete_user" in deleted_names


def test_diff_detects_modified_function_with_both_changes():
    result = diff_definitions(OLD_CODE, NEW_CODE)
    modified = {e["name"]: e for e in result["modified"]}
    assert "get_user" in modified
    assert modified["get_user"]["change_type"] == "both"


def test_chunk_embed_text_includes_new_body_for_modified():
    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    chunks = chunk_diff(diff_result, "fake_file.py")
    get_user_chunk = next(c for c in chunks if c["chunk_id"] == "fake_file.py::get_user")

    # regression guard for the bug we just fixed:
    # embed_text must contain the actual new body, not just the signature
    assert "if include_deleted:" in get_user_chunk["embed_text"]
    assert "db.query(id, deleted=True)" in get_user_chunk["embed_text"]


def test_chunk_metadata_captures_modification_type():
    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    chunks = chunk_diff(diff_result, "fake_file.py")
    get_user_chunk = next(c for c in chunks if c["chunk_id"] == "fake_file.py::get_user")

    # regression guard for the second bug we fixed
    assert get_user_chunk["metadata"]["modification_type"] == "both"


def test_deleted_chunk_id_is_unique_by_file_and_function():
    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    chunks = chunk_diff(diff_result, "fake_file.py")
    delete_user_chunk = next(c for c in chunks if c["metadata"]["function_name"] == "delete_user")
    assert delete_user_chunk["chunk_id"] == "fake_file.py::delete_user"