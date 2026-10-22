import chromadb

from .embed import embed_texts

DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"

_client = None


def get_collection():
    """Open (or create) the persistent collection. Cosine distance."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict]) -> int:
    """Embed chunks and save them. Uses upsert, so re-running won't create duplicates."""
    if not chunks:
        return 0
    collection = get_collection()
    texts = [c["text"] for c in chunks]
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embed_texts(texts),
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return len(chunks)


def search(query_embedding: list[float], k: int = 4) -> list[dict]:
    """Return the k nearest chunks with their distance (lower = more similar)."""
    collection = get_collection()
    res = collection.query(query_embeddings=[query_embedding], n_results=k)
    return [
        {
            "chunk_id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "page": res["metadatas"][0][i]["page"],
            "distance": res["distances"][0][i],
        }
        for i in range(len(res["ids"][0]))
    ]