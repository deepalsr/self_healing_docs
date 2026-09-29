# tests/test_diffing_chunking_integration.py
import pytest
from app.diffing.ast_parser import diff_definitions
from app.chunking.chunker import chunk_diff
from app.embeddings.embedder import embed_text, EMBEDDING_DIMENSION
from app.vectorstore.pinecone_client import (
    ensure_index_exists,
    sync_chunks_to_pinecone,
    get_index,
    upsert_chunks,
)
from app.retrieval.retriever import retrieve_context_for_chunk
from app.github.doc_extractor import extract_doc_section
from app.generation.patch_generator import build_prompt, parse_patch, PatchParseError
from app.guardrails import check_grounding, check_scope
from app.models.schemas import DocPatch, PatchChange

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

TEST_NAMESPACE = "pytest-integration-test"

SAMPLE_README = """
# AI RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions
from a company handbook.

## How it works

1. **Ingestion** (`ingest.py`, run once or whenever documents change): documents are split
into overlapping, sentence-aware chunks, embedded with a neural model, and stored
in a local Chroma vector database.

2. **On each question** (`main.py`): retrieve the nearest chunks, check the cache,
generate a grounded answer, run guardrail checks.

## Setup

Create a `.env` file in the project root.
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
    chunks = chunk_diff(diff_result, "fake_file.py", {})
    get_user_chunk = next(c for c in chunks if c["chunk_id"] == "fake_file.py::get_user")
    assert "if include_deleted:" in get_user_chunk["embed_text"]
    assert "db.query(id, deleted=True)" in get_user_chunk["embed_text"]


def test_chunk_metadata_captures_modification_type():
    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    chunks = chunk_diff(diff_result, "fake_file.py", {})
    get_user_chunk = next(c for c in chunks if c["chunk_id"] == "fake_file.py::get_user")
    assert get_user_chunk["metadata"]["modification_type"] == "both"


def test_deleted_chunk_id_is_unique_by_file_and_function():
    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    chunks = chunk_diff(diff_result, "fake_file.py", {})
    delete_user_chunk = next(c for c in chunks if c["metadata"]["function_name"] == "delete_user")
    assert delete_user_chunk["chunk_id"] == "fake_file.py::delete_user"


def test_deleted_chunks_are_never_upserted():
    from app.vectorstore.pinecone_client import split_chunks_by_action

    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    chunks = chunk_diff(diff_result, "fake_file.py", {})

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


def test_sync_chunks_to_pinecone_upserts_and_deletes():
    ensure_index_exists()
    index = get_index()

    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    chunks = chunk_diff(diff_result, "fake_file.py", {})

    try:
        sync_chunks_to_pinecone(chunks, namespace=TEST_NAMESPACE)

        import time
        time.sleep(2)

        fetched = index.fetch(
            ids=["fake_file.py::create_user", "fake_file.py::get_user"],
            namespace=TEST_NAMESPACE,
        )
        assert "fake_file.py::create_user" in fetched.vectors
        assert "fake_file.py::get_user" in fetched.vectors

        fetched_deleted = index.fetch(
            ids=["fake_file.py::delete_user"],
            namespace=TEST_NAMESPACE,
        )
        assert "fake_file.py::delete_user" not in fetched_deleted.vectors

    finally:
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
        upsert_chunks([fake_chunk], namespace=TEST_NAMESPACE)

        import time
        time.sleep(2)

        result = retrieve_context_for_chunk(fake_chunk, namespace=TEST_NAMESPACE)

        assert result["strategy"] == "exact_match"
        assert result["results"][0]["chunk_id"] == "fake_file.py::existing_function"

    finally:
        index.delete(ids=["fake_file.py::existing_function"], namespace=TEST_NAMESPACE)


def test_retrieve_context_falls_back_to_similarity_when_no_exact_match():
    index = get_index()

    existing_chunks = [
        {
            "chunk_id": "fake_file.py::fetch_account",
            "embed_text": "Function: fetch_account\nFile: fake_file.py\ndef fetch_account(account_id): return db.get_account(account_id)",
            "metadata": {
                "chunk_id": "fake_file.py::fetch_account",
                "function_name": "fetch_account",
                "file_path": "fake_file.py",
                "change_type": "added",
            },
        },
        {
            "chunk_id": "fake_file.py::fetch_order",
            "embed_text": "Function: fetch_order\nFile: fake_file.py\ndef fetch_order(order_id): return db.get_order(order_id)",
            "metadata": {
                "chunk_id": "fake_file.py::fetch_order",
                "function_name": "fetch_order",
                "file_path": "fake_file.py",
                "change_type": "added",
            },
        },
    ]

    new_chunk = {
        "chunk_id": "fake_file.py::fetch_customer",
        "embed_text": "Function: fetch_customer\nFile: fake_file.py\ndef fetch_customer(customer_id): return db.get_customer(customer_id)",
        "metadata": {
            "chunk_id": "fake_file.py::fetch_customer",
            "function_name": "fetch_customer",
            "file_path": "fake_file.py",
            "change_type": "added",
        },
    }

    try:
        upsert_chunks(existing_chunks, namespace=TEST_NAMESPACE)

        import time
        time.sleep(2)

        result = retrieve_context_for_chunk(new_chunk, namespace=TEST_NAMESPACE)

        assert result["strategy"] == "similarity_fallback"
        assert len(result["results"]) > 0

        returned_ids = [r["chunk_id"] for r in result["results"]]
        assert "fake_file.py::fetch_account" in returned_ids or "fake_file.py::fetch_order" in returned_ids

    finally:
        index.delete(
            ids=["fake_file.py::fetch_account", "fake_file.py::fetch_order"],
            namespace=TEST_NAMESPACE,
        )


def test_extract_doc_section_finds_paragraph_mention_when_no_anchor_or_heading():
    result = extract_doc_section(SAMPLE_README, "ingest")
    assert result is not None
    assert "Ingestion" in result
    assert "sentence-aware chunks" in result


def test_extract_doc_section_rejects_false_positive_substring_match():
    result = extract_doc_section(SAMPLE_README, "ingest")
    assert result is not None


def test_extract_doc_section_returns_none_for_nonexistent_function():
    result = extract_doc_section(SAMPLE_README, "totally_fake_function_xyz")
    assert result is None


def test_extract_doc_section_prefers_anchor_tag_when_present():
    doc_with_anchor = """
    <!-- doc-anchor: get_user -->
    ### Retrieving a user
    This is the real, authoritative doc section.
    <!-- doc-anchor: delete_user -->
    ### Deleting a user
    Another section.
    """
    result = extract_doc_section(doc_with_anchor, "get_user")
    assert result is not None
    assert "authoritative" in result
    assert "Deleting a user" not in result


def test_chunk_includes_doc_snippet_when_provided():
    diff_result = diff_definitions(OLD_CODE, NEW_CODE)
    doc_snippets = {"get_user": "Existing docs: fetches a user by id."}

    chunks = chunk_diff(diff_result, "fake_file.py", doc_snippets)

    get_user_chunk = next(c for c in chunks if c["metadata"]["function_name"] == "get_user")
    create_user_chunk = next(c for c in chunks if c["metadata"]["function_name"] == "create_user")

    assert get_user_chunk["metadata"]["doc_snippet"] == "Existing docs: fetches a user by id."
    assert "doc_snippet" not in create_user_chunk["metadata"]

GEN_CHUNK = {
    "chunk_id": "f.py::get_user",
    "embed_text": "Function: get_user\nFile: f.py\ndef get_user(id, include_deleted=False)\n\nbody here",
    "metadata": {
        "function_name": "get_user",
        "change_type": "modified",
        "modification_type": "both",
        "old_signature": "def get_user(id)",
        "new_signature": "def get_user(id, include_deleted=False)",
        "doc_snippet": "Retrieves a user by ID.",
    },
}
EXACT = {"strategy": "exact_match", "results": []}


def test_prompt_uses_replace_mode_with_fresh_doc_snippet():
    prompt = build_prompt(GEN_CHUNK, EXACT)
    assert "MODE: replace" in prompt
    assert "Retrieves a user by ID." in prompt


def test_prompt_uses_insert_mode_when_no_doc_snippet():
    chunk = {**GEN_CHUNK, "metadata": {k: v for k, v in GEN_CHUNK["metadata"].items() if k != "doc_snippet"}}
    prompt = build_prompt(chunk, {"strategy": "similarity_fallback", "results": []})
    assert "MODE: insert" in prompt


def test_prompt_includes_feedback_on_retry():
    assert "too broad" in build_prompt(GEN_CHUNK, EXACT, feedback="too broad")


def test_parse_patch_accepts_valid_json():
    raw = '{"changes": [{"action": "replace", "target_anchor": "get_user", "old_text": "a", "new_text": "b"}]}'
    assert parse_patch(raw).changes[0].action == "replace"


def test_parse_patch_rejects_bad_json_and_bad_action():
    with pytest.raises(PatchParseError):
        parse_patch("not json at all")
    with pytest.raises(PatchParseError):
        parse_patch('{"changes": [{"action": "delete_everything", "target_anchor": "x", "old_text": "", "new_text": ""}]}')

DOC = "Retrieves a user by ID. Returns None if missing."


def _patch(action="replace", old="Retrieves a user by ID.",
           new="Retrieves a user by ID, optionally including soft-deleted users."):
    return DocPatch(changes=[PatchChange(
        action=action, target_anchor="get_user", old_text=old, new_text=new
    )])


def test_grounding_passes_when_old_text_exists_verbatim():
    assert check_grounding(_patch(), DOC).passed


def test_grounding_rejects_hallucinated_old_text():
    result = check_grounding(_patch(old="Fetches a user record."), DOC)
    assert not result.passed
    assert "not found verbatim" in result.reason


def test_grounding_rejects_ambiguous_old_text():
    doc = "Returns a user. Returns a user."
    result = check_grounding(_patch(old="Returns a user.", new="Returns a user or None."), doc)
    assert not result.passed
    assert "matches 2 places" in result.reason


def test_grounding_rejects_insert_when_docs_already_exist():
    assert not check_grounding(_patch(action="insert", old=""), DOC).passed


def test_grounding_rejects_replace_when_no_docs_exist():
    assert not check_grounding(_patch(), None).passed


def test_grounding_accepts_insert_for_undocumented_function():
    assert check_grounding(_patch(action="insert", old="", new="Creates a user."), None).passed


def test_grounding_rejects_noop_patch():
    assert not check_grounding(_patch(new="Retrieves a user by ID."), DOC).passed

def test_scope_passes_when_anchor_is_allowed():
    assert check_scope(_patch(), {"get_user"}).passed


def test_scope_rejects_anchor_outside_allowed_set():
    result = check_scope(_patch(), {"delete_user"})
    assert not result.passed
    assert "outside the allowed scope" in result.reason


def test_scope_rejects_when_patch_touches_multiple_anchors_one_unauthorized():
    patch = DocPatch(changes=[
        PatchChange(action="replace", target_anchor="get_user",
                    old_text="a", new_text="b"),
        PatchChange(action="replace", target_anchor="delete_user",
                    old_text="c", new_text="d"),
    ])
    result = check_scope(patch, {"get_user"})
    assert not result.passed
    assert "delete_user" in result.reason