"""Monarch-RAG embedding compatibility exports."""

from embeddings.embedder_service import (  # noqa: F401
    embed_query,
    embed_texts,
    init_embedder,
)

__all__ = ["init_embedder", "embed_texts", "embed_query"]
