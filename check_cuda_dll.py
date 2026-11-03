import os
import site
import ctypes
from pathlib import Path

sp = Path(site.getsitepackages()[-1])

cuda_dir = sp / "nvidia" / "cu13" / "bin" / "x86_64"
cudnn_dir = sp / "nvidia" / "cudnn" / "bin"

handles = []
for d in [cuda_dir, cudnn_dir]:
    if d.exists():
        handles.append(os.add_dll_directory(str(d)))

print("=== CUDA DLL files ===")

for name in [
    "cublasLt64_13.dll",
    "cublas64_13.dll",
    "cudart64_13.dll",
    "cudnn64_9.dll",
]:
    files = list((sp / "nvidia").rglob(name))
    print(f"\n{name}:")
    for f in files:
        print(" ", f)

        try:
            ctypes.WinDLL(str(f))
            print("  LOAD SUCCESS")
        except OSError as e:
            print("  LOAD FAILED:", e)
