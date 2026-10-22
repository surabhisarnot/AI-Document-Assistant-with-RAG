from ai_document_assistant_with_rag.embed import embed_query
from ai_document_assistant_with_rag.store import search


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Embed the question and return the k most similar chunks (lower distance = better)."""
    query = query.strip()
    if not query:
        return []
    return search(embed_query(query), k=k)