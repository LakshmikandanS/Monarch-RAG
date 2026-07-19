"""Diagnostic: try binding the cached embedding ONNX file directly to CUDA."""

from __future__ import annotations

import sys
from pathlib import Path

import onnxruntime as ort

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monarch.embeddings import get_model, load_model  # noqa: E402
from monarch.embeddings.cuda import enable_cuda_runtime  # noqa: E402


def _model_dir(embedder) -> Path | None:
    inner = getattr(embedder, "model", None)
    model_dir = getattr(inner, "_model_dir", None)
    if model_dir is not None:
        return Path(model_dir)
    nested = getattr(inner, "model", None) or getattr(inner, "_model", None)
    model_dir = getattr(nested, "_model_dir", None)
    return Path(model_dir) if model_dir is not None else None


def main() -> None:
    print("1. Enabling CUDA runtime directories...")
    enable_cuda_runtime(verbose=True)

    print("\n2. Loading shared Monarch embedding model...")
    load_model()
    embedder = get_model()
    model_dir = _model_dir(embedder)

    if model_dir is None:
        print("Could not locate the cached model directory from FastEmbed internals.")
        sys.exit(1)

    onnx_files = list(model_dir.rglob("*.onnx"))
    if not onnx_files:
        print("Could not find an ONNX file under the cached model directory.")
        sys.exit(1)

    target_onnx = str(onnx_files[0])
    print(f"   Found: {Path(target_onnx).name}")

    print("\n3. Attempting direct CUDA binding...")
    try:
        session = ort.InferenceSession(
            target_onnx,
            providers=["CUDAExecutionProvider"],
        )
        print("\n" + "=" * 60)
        print(f"Session providers: {session.get_providers()}")
        print("=" * 60)
    except Exception as exc:
        print("\n" + "!" * 60)
        print("Direct CUDA binding failed:")
        print("!" * 60)
        print(exc)
        print("!" * 60)


if __name__ == "__main__":
    main()
