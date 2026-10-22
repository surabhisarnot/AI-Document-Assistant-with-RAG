from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Load the model once, on first use, and reuse it afterwards."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts (e.g. document chunks). Returns one vector per text."""
    if not texts:
        return []
    model = _get_model()
    return [vec.tolist() for vec in model.embed(texts)]


def embed_query(query: str) -> list[float]:
    """Embed a single search query. Uses fastembed's query-specific method."""
    model = _get_model()
    return next(iter(model.query_embed(query))).tolist()