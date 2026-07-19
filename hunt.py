"""Diagnostic: locate CUDA DLLs and test the shared embedding model."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monarch.embeddings import load_model, providers  # noqa: E402
from monarch.embeddings.cuda import enable_cuda_runtime  # noqa: E402


def main() -> None:
    print("1. Scanning Python environment for CUDA runtime DLLs...")
    discovered_cuda = enable_cuda_runtime(verbose=True)

    if not discovered_cuda:
        print("\nNo pip-installed NVIDIA runtime directories were found.")
        print(
            "Install CUDA runtime wheels such as "
            "nvidia-cublas-cu12 and nvidia-cudnn-cu12 if GPU binding is required."
        )

    print("\n2. Attempting shared FastEmbed binding...")
    load_model()
    locked_engines = providers()

    print("\n" + "=" * 60)
    print(f"FINAL BOUND ENGINES: {locked_engines}")
    print("=" * 60)

    if "CUDAExecutionProvider" in locked_engines:
        print("SUCCESS: CUDAExecutionProvider is active.")
    else:
        print("WARNING: model is not bound to CUDA.")


if __name__ == "__main__":
    main()
