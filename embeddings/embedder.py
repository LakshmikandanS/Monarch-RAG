import os
import sys
from pathlib import Path

# =====================================================================
# 1. BARE-METAL RTX 5060 CUDA BOOTSTRAP (Must sit at absolute top)
# =====================================================================
_venv_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
_nvidia_bins = [str(p.resolve()) for p in _venv_nvidia.glob("*/bin")]

if _nvidia_bins:
    # Win32 LoadLibraryA only checks %PATH%; it ignores add_dll_directory
    os.environ["PATH"] = ";".join(_nvidia_bins) + ";" + os.environ.get("PATH", "")
    for _bin in _nvidia_bins:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_bin)
            except Exception:
                pass
# =====================================================================

import time
from fastembed import TextEmbedding
from tqdm import tqdm

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Doubled from 64. Your RTX 5060 has 8GB of VRAM; 
# feeding it 128 chunks at a time keeps the CUDA cores fed.
BATCH_SIZE = 128  


def load_embedding_model():
    print(f"[Init] Waking up RTX 5060 & loading {MODEL_NAME}...")
    start = time.time()

    model = TextEmbedding(
        model_name=MODEL_NAME, providers=["CUDAExecutionProvider"]
    )

    # --- THE ANTI-GASLIGHT AUDIT ---
    # Dig into the underlying C++ session. If Windows handed us a CPU, kill the program.
    raw_session = getattr(model.model, "model", None) or getattr(model.model, "_model", None)
    active_engines = raw_session.get_providers() if raw_session else ["Unknown"]

    if "CUDAExecutionProvider" not in active_engines:
        raise RuntimeError(
            f"\n[FATAL ERROR] ONNX silently fell back to CPU! Active engine: {active_engines}\n"
            "The C++ runtime cannot read the NVIDIA DLLs. Check your environment PATH."
        )

    print(f"[Init] GPU Model locked to {active_engines[0]} in {time.time() - start:.2f}s")
    return model


def embed_documents(documents, model):
    payloads = []

    for doc in documents:
        meta = doc.get("metadata", {})

        raw_name = (
            meta.get("file_name", "Document")
            .replace(".html", "")
            .replace("_", " ")
        )
        full_path = meta.get("section", "")
        specific_sec = full_path.split(" > ")[-1] if full_path else ""

        prefix = f"[{raw_name} - {specific_sec}]\n" if specific_sec else f"[{raw_name}]\n"
        payloads.append(f"{prefix}{doc['content']}".strip())

    if not payloads:
        return []

    return list(
        tqdm(
            model.embed(payloads, batch_size=BATCH_SIZE),
            total=len(payloads),
            desc=f"GPU Vectorizing ({len(payloads)} chunks)",
            unit="chunk",
        )
    )


def embed_query(query, model):
    """
    Applies the mandatory Asymmetric Task Instruction.
    """
    bge_instruction = "Represent this sentence for searching relevant passages: "
    asymmetric_query = f"{bge_instruction}{query}"

    return list(model.embed([asymmetric_query]))[0]