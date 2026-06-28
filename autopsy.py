import os
import site

# --- THE SYSTEM ERROR 126 SILENCER ---
# This forces Windows to read the pip-installed NVIDIA DLLs into memory 
# before the ONNX C++ engine attempts to initialize.
for site_pkg in site.getsitepackages():
    for nvidia_submod in ["cublas", "cudnn", "cuda_runtime"]:
        dll_dir = os.path.join(site_pkg, "nvidia", nvidia_submod, "bin")
        if os.path.exists(dll_dir):
            os.add_dll_directory(dll_dir)
# -------------------------------------

from pathlib import Path
import onnxruntime as ort
from fastembed import TextEmbedding

print("1. Locating your cached BGE-Base model weights...")
embedder = TextEmbedding("BAAI/bge-base-en-v1.5")
model_dir = embedder.model._model_dir

onnx_files = list(Path(model_dir).rglob("*.onnx"))
if not onnx_files:
    print("❌ Critical: Could not find the .onnx file on your disk.")
    exit()

target_onnx = str(onnx_files[0])
print(f"   Found: {Path(target_onnx).name}")

print("\n2. Attempting bare-metal CUDA binding (Zero safety net)...")
try:
    # Notice the list: ONLY CUDA. If a single DLL is unreadable, 
    # C++ will refuse to catch the exception and scream the exact file at us.
    session = ort.InferenceSession(target_onnx, providers=["CUDAExecutionProvider"])
    
    print("\n" + "="*60)
    print("🎉 HOLY SHIT, IT ACCEPTED IT.")
    print(f"Engines officially locked to session: {session.get_providers()}")
    print("="*60)

except Exception as e:
    print("\n" + "!"*60)
    print("💀 THE EXACT C++ CRASH REASON WE'VE BEEN LOOKING FOR:")
    print("!"*60)
    print(e)
    print("!"*60)