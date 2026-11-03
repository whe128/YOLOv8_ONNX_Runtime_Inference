import os
import site
import ctypes
from pathlib import Path

sp = Path(site.getsitepackages()[-1])

# DLL 搜索目录
dll_dirs = [
    sp / "nvidia" / "cu13" / "bin",
    sp / "nvidia" / "cu13" / "bin" / "x86_64",
    sp / "nvidia" / "cudnn" / "bin",
]

handles = []

for d in dll_dirs:
    if d.exists():
        handles.append(os.add_dll_directory(str(d)))
        print("Registered:", d)

# 显式加载 CUDA 依赖，并保持引用
dll_objects = []

dll_names = [
    "cublasLt64_13.dll",
    "cublas64_13.dll",
    "cudart64_13.dll",
    "cudnn64_9.dll",
]

for name in dll_names:
    files = list((sp / "nvidia").rglob(name))

    if files:
        try:
            obj = ctypes.WinDLL(str(files[0]))
            dll_objects.append(obj)
            print("[OK]", name)
        except OSError as e:
            print("[FAIL]", name, e)

# 测试 ONNX Runtime CUDA Provider DLL
provider_dll = (
    sp / "onnxruntime" / "capi" /
    "onnxruntime_providers_cuda.dll"
)

print("\nTesting:", provider_dll)

try:
    provider = ctypes.WinDLL(str(provider_dll))
    print("[SUCCESS] CUDA Provider DLL loaded!")
except OSError as e:
    print("[FAILED] CUDA Provider DLL:", e)
    print("Windows error code:", e.winerror)

# 测试实际 Session
import onnxruntime as ort

print("\nONNX Runtime:", ort.__version__)

session = ort.InferenceSession(
    "models/yolov8n.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print("Session providers:", session.get_providers())
