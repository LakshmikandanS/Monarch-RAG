import os
import sys
from pathlib import Path

print("1. Hijacking Windows OS %PATH% variable...")
venv_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"

# Harvest every single 'bin' folder inside the pip nvidia packages
nvidia_bins = [str(p.resolve()) for p in venv_nvidia.glob("*/bin")]

if not nvidia_bins:
    print("❌ Critical: Could not locate the unpacked nvidia/ folders.")
    sys.exit()

# THE SLEDGEHAMMER: Force them to the absolute front of the OS PATH
os.environ["PATH"] = ";".join(nvidia_bins) + ";" + os.environ.get("PATH", "")

# Call add_dll_directory too just to cover 100% of the bases
for bin_path in nvidia_bins:
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(bin_path)

print(f"   ✔ Injected {len(nvidia_bins)} NVIDIA directories into master OS PATH.")

print("\n2. Igniting ONNX Engine...")
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-base-en-v1.5", providers=["CUDAExecutionProvider"])

# Dig out the bare C++ session safely
session = getattr(model.model, "model", None) or getattr(model.model, "_model", None)
bound_providers = session.get_providers()

print("\n" + "=" * 60)
print(f"OFFICIALLY BOUND ENGINES: {bound_providers}")
print("=" * 60)

if "CUDAExecutionProvider" in bound_providers:
    print("\n🏆 WE WIN. THE RTX 5060 HAS TAKEN THE MODEL.")
else:
    print("\n💀 STILL CPU. Windows has defeated us.")