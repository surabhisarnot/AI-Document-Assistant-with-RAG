import os
from pathlib import Path

import chromadb

from .embed import embed_texts

DB_PATH = os.getenv("CHROMA_PATH", str(Path(__file__).resolve().parents[2] / "chroma_db"))
COLLECTION_NAME = "documents"  # shared sample documents
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
    return _client


def get_collection(name: str = COLLECTION_NAME):
    return _get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def delete_collection(name: str) -> None:
    try:
        _get_client().delete_collection(name)
    except Exception:
        pass


def add_chunks(chunks: list[dict], collection_name: str = COLLECTION_NAME) -> int:
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    get_collection(collection_name).upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embed_texts(texts),
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return len(chunks)


def search(query_embedding: list[float], k: int = 4, collection_name: str = COLLECTION_NAME) -> list[dict]:
    res = get_collection(collection_name).query(query_embeddings=[query_embedding], n_results=k)
    return [
        {"chunk_id": res["ids"][0][i], "text": res["documents"][0][i],
         "source": res["metadatas"][0][i]["source"], "page": res["metadatas"][0][i]["page"],
         "distance": res["distances"][0][i]}
        for i in range(len(res["ids"][0]))
    ]