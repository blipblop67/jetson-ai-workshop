"""
CUDA EXAMPLE 1 - HELLO, GPU!
============================
Asks the Jetson: "Do you have a GPU, and what is it like?"

    python3 01_hello_gpu.py

CUDA is NVIDIA's way of letting programs use the GPU. We use it through
PyTorch, which talks to CUDA for us.
"""

import os
import torch

# CUDA cores per "streaming multiprocessor" (SM) for common NVIDIA chips
CORES_PER_SM = {(5, 3): 128, (6, 2): 128, (7, 2): 64, (8, 7): 128, (11, 0): 128}

print("=" * 52)
print(" HELLO GPU")
print("=" * 52)
print(f"PyTorch version      : {torch.__version__}")
print(f"CPU cores            : {os.cpu_count()}")

if not torch.cuda.is_available():
    print("\nNo CUDA GPU found!")
    print("  - On the Jetson, check that PyTorch was installed from the Jetson AI Lab index")
    print("  - Check CUDA itself with:  nvcc --version")
    raise SystemExit(1)

p = torch.cuda.get_device_properties(0)
cc = (p.major, p.minor)
cores = CORES_PER_SM.get(cc, 0) * p.multi_processor_count
print(f"CUDA version         : {torch.version.cuda}")
print(f"GPU name             : {p.name}")
print(f"Compute capability   : {p.major}.{p.minor}   (Orin = 8.7)")
print(f"Multiprocessors (SM) : {p.multi_processor_count}")
if cores:
    print(f"CUDA cores           : {cores}   <- tiny workers that run at the same time")
print(f"GPU memory           : {p.total_memory / 1e9:.1f} GB  (on Jetson the CPU and GPU SHARE this memory)")

# ---- our first GPU calculation ----
print("\nFirst GPU calculation:")
a = torch.tensor([1.0, 2.0, 3.0], device="cuda")   # numbers stored on the GPU
b = torch.tensor([10.0, 20.0, 30.0], device="cuda")
c = a + b                                          # the addition runs on the GPU
print(f"  {a.tolist()} + {b.tolist()} = {c.tolist()}")
print(f"  This result lives on: {c.device}")
print("\nYour Jetson GPU is ready!  Next: python3 02_cpu_vs_gpu_add.py")
