import os
import sys
from pathlib import Path

print("1. Physically scanning your .venv for cublasLt64_12.dll...")
venv_root = Path(sys.prefix)

cublas_files = list(venv_root.rglob("cublasLt64_12.dll"))

if not cublas_files:
    print("\n❌ FOUND THE ISSUE: The DLL is physically missing from your disk.")
    print("Run this exact command in your terminal right now:")
    print("  .\\.venv\\Scripts\\python.exe -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
    sys.exit()

cublas_dir = str(cublas_files[0].parent.resolve())
print(f"   ✔ Found it sitting inside: {cublas_dir}")
os.add_dll_directory(cublas_dir)

# Grab cuDNN while we are here
cudnn_files = list(venv_root.rglob("cudnn*.dll"))
if cudnn_files:
    cudnn_dir = str(cudnn_files[0].parent.resolve())
    print(f"   ✔ Found cuDNN sitting inside: {cudnn_dir}")
    os.add_dll_directory(cudnn_dir)

print("\n2. Attempting FastEmbed GPU Binding...")
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-base-en-v1.5", providers=["CUDAExecutionProvider"])
locked_engines = model.model._model.session.get_providers()

print("\n" + "="*60)
print(f"FINAL BOUND ENGINES: {locked_engines}")
print("="*60)

if "CUDAExecutionProvider" in locked_engines:
    print("🚀 SUCCESS. The RTX 5060 is holding the model.")
else:
    print("💀 STILL CPU. Windows rejected the injected path.")