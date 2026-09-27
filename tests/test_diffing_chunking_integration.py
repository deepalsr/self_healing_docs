# tests/test_diffing_chunking_integration.py

from app.diffing.ast_parser import diff_definitions
from app.chunking.chunker import chunk_diff
from app.vectorstore.pinecone_client import split_chunks_by_action
from app.embeddings.embedder import embed_text, EMBEDDING_DIMENSION
from app.retrieval.retriever import retrieve_context_for_chunk
from app.vectorstore.pinecone_client import upsert_chunks




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

def test_deleted_chunks_are_never_upserted():
    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    chunks = chunk_diff(diff_result, "fake_file.py")

    to_upsert, to_delete = split_chunks_by_action(chunks)

    upsert_change_types = [c["metadata"]["change_type"] for c in to_upsert]
    assert "deleted" not in upsert_change_types

    delete_names = [c["metadata"]["function_name"] for c in to_delete]
    assert "delete_user" in delete_names

def test_embed_text_returns_correct_dimension():
    vector = embed_text("def get_user(id): return db.query(id)")
    assert isinstance(vector, list)
    assert len(vector) == EMBEDDING_DIMENSION
    assert all(isinstance(x, float) for x in vector)

from app.vectorstore.pinecone_client import (
    ensure_index_exists,
    sync_chunks_to_pinecone,
    get_index,
)

TEST_NAMESPACE = "pytest-integration-test"


def test_sync_chunks_to_pinecone_upserts_and_deletes():
    ensure_index_exists()
    index = get_index()

    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    chunks = chunk_diff(diff_result, "fake_file.py")

    try:
        # Act: sync all chunks (added, modified go to upsert; deleted goes to delete)
        sync_chunks_to_pinecone(chunks, namespace=TEST_NAMESPACE)

        # Give Pinecone a moment to make upserts queryable
        # (Pinecone upserts are eventually consistent, not instant)
        import time
        time.sleep(2)

        # Assert: added/modified chunks exist
        fetched = index.fetch(
            ids=["fake_file.py::create_user", "fake_file.py::get_user"],
            namespace=TEST_NAMESPACE,
        )
        assert "fake_file.py::create_user" in fetched.vectors
        assert "fake_file.py::get_user" in fetched.vectors

        # Assert: deleted chunk was never inserted / does not exist
        fetched_deleted = index.fetch(
            ids=["fake_file.py::delete_user"],
            namespace=TEST_NAMESPACE,
        )
        assert "fake_file.py::delete_user" not in fetched_deleted.vectors

    finally:
        # Cleanup: remove all test data regardless of pass/fail,
        # so this test never leaves orphaned vectors behind
        index.delete(
            ids=["fake_file.py::create_user", "fake_file.py::get_user", "fake_file.py::delete_user"],
            namespace=TEST_NAMESPACE,
        )

def test_retrieve_context_finds_exact_match_when_chunk_exists():
    index = get_index()

    fake_chunk = {
        "chunk_id": "fake_file.py::existing_function",
        "embed_text": "Function: existing_function\nFile: fake_file.py\ndef existing_function(x): return x * 2",
        "metadata": {
            "chunk_id": "fake_file.py::existing_function",
            "function_name": "existing_function",
            "file_path": "fake_file.py",
            "change_type": "added",
        },
    }

    try:
        # Setup: put this chunk into Pinecone first, simulating "already documented before"
        upsert_chunks([fake_chunk], namespace=TEST_NAMESPACE)

        import time
        time.sleep(2)  # eventual consistency, same reasoning as Step 5's test

        # Act
        result = retrieve_context_for_chunk(fake_chunk, namespace=TEST_NAMESPACE)

        # Assert: Tier 1 should find it directly, no similarity fallback needed
        assert result["strategy"] == "exact_match"
        assert result["results"][0]["chunk_id"] == "fake_file.py::existing_function"

    finally:
        # Cleanup, regardless of pass/fail
        index.delete(ids=["fake_file.py::existing_function"], namespace=TEST_NAMESPACE)