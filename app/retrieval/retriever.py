# app/retrieval/retriever.py

from app.vectorstore.pinecone_client import get_index
from app.embeddings.embedder import embed_text

TOP_K_FALLBACK = 3


def find_existing_doc_for_chunk(chunk_id: str, namespace: str) -> dict | None:
    """
    Tier 1: deterministic lookup. Checks if this exact chunk_id
    already has a vector (meaning it was documented/embedded before).
    """
    index = get_index()
    result = index.fetch(ids=[chunk_id], namespace=namespace)

    if chunk_id in result.vectors:
        return {
            "match_type": "exact",
            "chunk_id": chunk_id,
            "metadata": result.vectors[chunk_id].metadata,
        }
    return None


def find_similar_docs(embed_text_query: str, namespace: str, exclude_chunk_id: str) -> list[dict]:
    """
    Tier 2: fallback similarity search. Used only when Tier 1 finds nothing --
    i.e., this is a genuinely new function with no prior documentation.
    """
    index = get_index()
    query_vector = embed_text(embed_text_query)

    results = index.query(
        vector=query_vector,
        top_k=TOP_K_FALLBACK,
        namespace=namespace,
        include_metadata=True,
    )

    return [
        {
            "match_type": "similar",
            "chunk_id": match.id,
            "score": match.score,
            "metadata": match.metadata,
        }
        for match in results.matches
        if match.id != exclude_chunk_id
    ]


def retrieve_context_for_chunk(chunk: dict, namespace: str) -> dict:
    """
    Main entry point: given one chunk (from chunk_diff output),
    find its existing doc via Tier 1, falling back to Tier 2.
    """
    chunk_id = chunk["chunk_id"]

    exact_match = find_existing_doc_for_chunk(chunk_id, namespace)
    if exact_match:
        return {"strategy": "exact_match", "results": [exact_match]}

    similar_matches = find_similar_docs(chunk["embed_text"], namespace, exclude_chunk_id=chunk_id)
    return {"strategy": "similarity_fallback", "results": similar_matches}