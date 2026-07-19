"""Diagnostic: verify which ONNX Runtime provider FastEmbed is using."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monarch.embeddings import load_model, providers  # noqa: E402
from monarch.embeddings.cuda import enable_cuda_runtime  # noqa: E402


def main() -> None:
    print("1. Enabling CUDA runtime directories...")
    discovered_cuda = enable_cuda_runtime(verbose=True)
    print(f"   CUDA runtime directories discovered: {discovered_cuda}")

    print("\n2. Loading shared Monarch embedding model...")
    load_model()
    bound_providers = providers()

    print("\n" + "=" * 60)
    print(f"BOUND ENGINES: {bound_providers}")
    print("=" * 60)

    if "CUDAExecutionProvider" in bound_providers:
        print("\nSUCCESS: CUDAExecutionProvider is active.")
    else:
        print("\nWARNING: model is not bound to CUDA.")


if __name__ == "__main__":
    main()
