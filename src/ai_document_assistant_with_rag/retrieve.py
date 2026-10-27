from .embed import embed_query
from .store import COLLECTION_NAME, search


def retrieve(query: str, k: int = 4, collection_name: str = COLLECTION_NAME) -> list[dict]:
    """Embed the question and return the k most similar chunks (lower distance = better)."""
    query = query.strip()
    if not query:
        return []
    return search(embed_query(query), k=k, collection_name=collection_name)