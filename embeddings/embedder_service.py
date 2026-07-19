"""Compatibility wrapper for the shared Monarch embedding service."""

try:
    from ._monarch_compat import ensure_src_path
except ImportError:  # pragma: no cover - supports direct top-level imports.
    from embeddings._monarch_compat import ensure_src_path

ensure_src_path()

from monarch.embeddings import (  # noqa: E402,F401
    embed_query,
    embed_texts,
    init_embedder,
    is_initialized,
    load_model,
    providers,
    shutdown_embedder,
)

__all__ = [
    "embed_query",
    "embed_texts",
    "init_embedder",
    "is_initialized",
    "load_model",
    "providers",
    "shutdown_embedder",
]
