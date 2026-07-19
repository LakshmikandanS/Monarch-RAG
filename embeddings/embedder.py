"""Legacy RAG embedding adapters backed by ``monarch.embeddings``."""

from __future__ import annotations

try:
    from ._monarch_compat import ensure_src_path
except ImportError:  # pragma: no cover - supports direct top-level imports.
    from embeddings._monarch_compat import ensure_src_path

ensure_src_path()

from monarch.embeddings import embed_query as _embed_query  # noqa: E402
from monarch.embeddings import embed_texts, init_embedder  # noqa: E402

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 128


class SharedEmbeddingModel:
    """Small adapter matching the old FastEmbed ``embed`` method shape."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or MODEL_NAME
        init_embedder(self.model_name)

    def embed(self, documents, batch_size: int = BATCH_SIZE):
        return embed_texts(list(documents))


def load_embedding_model(model_name: str | None = None) -> SharedEmbeddingModel:
    """Initialize and return an adapter for old RAG call sites."""
    return SharedEmbeddingModel(model_name)


def _document_payload(doc: dict) -> str:
    meta = doc.get("metadata", {})
    raw_name = (
        meta.get("file_name", "Document")
        .replace(".html", "")
        .replace("_", " ")
    )
    full_path = meta.get("section", "")
    specific_section = full_path.split(" > ")[-1] if full_path else ""

    if specific_section:
        prefix = f"[{raw_name} - {specific_section}]\n"
    else:
        prefix = f"[{raw_name}]\n"
    return f"{prefix}{doc['content']}".strip()


def embed_documents(documents, model=None):
    """Embed RAG chunk dictionaries using the shared embedding facade."""
    payloads = [_document_payload(doc) for doc in documents]
    if not payloads:
        return []

    adapter = model or SharedEmbeddingModel()
    return list(adapter.embed(payloads, batch_size=BATCH_SIZE))


def embed_query(query, model=None):
    """Embed a query using the shared BGE asymmetric query path."""
    return _embed_query(query)
