# app/vectorstore/pinecone_client.py

from pinecone import Pinecone, ServerlessSpec
from app.config import settings
from app.embeddings.embedder import embed_text, EMBEDDING_DIMENSION

INDEX_NAME = "self-healing-docs"

_pc = None


def _get_pc():
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc


def ensure_index_exists():
    existing_indexes = [idx["name"] for idx in _get_pc().list_indexes()]
    if INDEX_NAME not in existing_indexes:
        _get_pc().create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def get_index():
    return _get_pc().Index(INDEX_NAME)


def upsert_chunks(chunks: list[dict], namespace: str):
    """
    Embeds and upserts a list of chunks (added/modified only --
    deleted chunks must never reach this function, see
    split_chunks_by_action) into Pinecone.
    """
    index = get_index()
    vectors = []

    for chunk in chunks:
        vector = embed_text(chunk["embed_text"])
        vectors.append({
            "id": chunk["chunk_id"],
            "values": vector,
            "metadata": chunk["metadata"],
        })

    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)


def delete_chunk(chunk_id: str, namespace: str):
    """
    Explicitly removes a vector -- required for deleted functions,
    since upsert alone cannot express deletion.
    """
    index = get_index()
    index.delete(ids=[chunk_id], namespace=namespace)


def split_chunks_by_action(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Pure function: splits chunks into (to_upsert, to_delete)
    based on change_type. No I/O, no API calls -- easy to test
    in isolation from Pinecone/network.
    """
    to_upsert = [c for c in chunks if c["metadata"]["change_type"] != "deleted"]
    to_delete = [c for c in chunks if c["metadata"]["change_type"] == "deleted"]
    return to_upsert, to_delete


def sync_chunks_to_pinecone(chunks: list[dict], namespace: str):
    """
    Single entry point for syncing a diff's chunks to Pinecone.
    Routes added/modified/renamed chunks to upsert, deleted chunks to
    delete. A renamed chunk is upserted under its NEW chunk_id, then its
    OLD chunk_id (carried in metadata) is explicitly deleted too, so a
    rename never leaves a stale vector behind under the old name.
    Always call this instead of upsert_chunks/delete_chunk directly.
    """
    to_upsert, to_delete = split_chunks_by_action(chunks)

    if to_upsert:
        upsert_chunks(to_upsert, namespace)

    for chunk in to_delete:
        delete_chunk(chunk["chunk_id"], namespace)

    for chunk in to_upsert:
        old_chunk_id = chunk["metadata"].get("old_chunk_id")
        if old_chunk_id:
            delete_chunk(old_chunk_id, namespace)