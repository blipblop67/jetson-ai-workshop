"""
CUDA EXAMPLE 3 - THE MATRIX MULTIPLICATION RACE
===============================================
Neural networks are mostly ONE operation repeated millions of times:
matrix multiplication (multiply grids of numbers together).
This is exactly what GPUs were built for. Let's race!

    python3 03_matrix_race.py

GFLOPS = billions of calculations per second (higher = faster).
"""

import time
import numpy as np
import torch

GPU = torch.cuda.is_available()
SIZES = [256, 512, 1024, 2048, 4096]
CPU_LIMIT_S = 8.0          # skip bigger CPU runs once one takes longer than this


def bench(fn, sync):
    fn()                                       # warm-up run
    if sync: torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        fn()
    if sync: torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


print("Multiplying two N x N grids of numbers (a neural-network-style job)\n")
print(f"{'N':>6} | {'calculations':>14} | {'CPU time':>9} | {'GPU time':>9} | {'GPU speed-up':>12}")
print("-" * 64)

cpu_too_slow = False
for n in SIZES:
    flops = 2 * n ** 3                          # multiplies + adds needed
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)

    t_cpu = None
    if not cpu_too_slow:
        t_cpu = bench(lambda: a @ b, sync=False)
        cpu_too_slow = t_cpu > CPU_LIMIT_S

    t_gpu = None
    if GPU:
        ag, bg = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
        t_gpu = bench(lambda: ag @ bg, sync=True)

    cpu_s = f"{t_cpu * 1000:7.1f}ms" if t_cpu else "  skipped"
    gpu_s = f"{t_gpu * 1000:7.1f}ms" if t_gpu else "   no GPU"
    speed = f"{t_cpu / t_gpu:10.1f}x" if (t_cpu and t_gpu) else ""
    calc = f"{flops / 1e9:,.1f} billion" if flops >= 1e9 else f"{flops / 1e6:,.0f} million"
    print(f"{n:>6} | {calc:>14} | {cpu_s:>9} | {gpu_s:>9} | {speed:>12}")
    if t_gpu:
        last_gflops = flops / t_gpu / 1e9

if GPU:
    print(f"\nGPU speed on the biggest grid: about {last_gflops:,.0f} GFLOPS")

    # Bonus: half precision (FP16) uses the Tensor Cores - special AI hardware
    n = 4096
    ah = torch.rand(n, n, device="cuda", dtype=torch.float16)
    bh = torch.rand(n, n, device="cuda", dtype=torch.float16)
    t_h = bench(lambda: ah @ bh, sync=True)
    print(f"Bonus - same job in FP16 (Tensor Cores): {t_h * 1000:.1f} ms  "
          f"= about {2 * n ** 3 / t_h / 1e9:,.0f} GFLOPS")
    print("AI frameworks like TensorRT use these tricks to run models faster on Jetson.")

print("\nWhy does this matter? Training our CNN in Chapter 5 = millions of these")
print("multiplications. The GPU is the reason it takes minutes, not hours.")
print("Next: python3 04_edge_filter.py")
