"""
CUDA EXAMPLE 2 - CPU vs GPU: ADDING NUMBERS
===========================================
Adds two lists of numbers together: first on the CPU, then on the GPU.
We try small lists and HUGE lists. Who wins?

    python3 02_cpu_vs_gpu_add.py

The surprise: for SMALL jobs the CPU often wins, because sending work to the
GPU has a small start-up cost. For BIG jobs the GPU's many workers win.
"""

import time
import numpy as np
import torch

GPU = torch.cuda.is_available()
SIZES = [1_000, 100_000, 1_000_000, 10_000_000, 30_000_000]


def timeit(fn, repeats=5, sync=False):
    """Run fn a few times and return the fastest time in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        if sync:
            torch.cuda.synchronize()          # wait until the GPU is idle
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()          # wait until the GPU has FINISHED
        best = min(best, time.perf_counter() - t0)
    return best * 1000


print("Adding two lists of random numbers:  c = a + b\n")
print(f"{'numbers':>12} | {'CPU (ms)':>9} | {'GPU (ms)':>9} | winner")
print("-" * 52)

if GPU:                                       # warm-up: the very first GPU call is slow
    torch.ones(10, device="cuda") + 1
    torch.cuda.synchronize()

for n in SIZES:
    a_cpu = np.random.rand(n).astype(np.float32)
    b_cpu = np.random.rand(n).astype(np.float32)
    t_cpu = timeit(lambda: a_cpu + b_cpu)

    if GPU:
        a_gpu = torch.from_numpy(a_cpu).cuda()        # copy the numbers into GPU memory
        b_gpu = torch.from_numpy(b_cpu).cuda()
        t_gpu = timeit(lambda: a_gpu + b_gpu, sync=True)
        if t_cpu < t_gpu:
            winner = f"CPU  ({t_gpu / t_cpu:.1f}x faster)"
        else:
            winner = f"GPU  ({t_cpu / t_gpu:.1f}x faster)"
        print(f"{n:>12,} | {t_cpu:9.3f} | {t_gpu:9.3f} | {winner}")
    else:
        print(f"{n:>12,} | {t_cpu:9.3f} | {'no GPU':>9} |")

print("\nWhat did you notice?")
print("  - Small lists: the CPU is fine (or faster). Starting the GPU costs a little time.")
print("  - Big lists: thousands of GPU workers each add a few numbers at the SAME time.")
print("Next: python3 03_matrix_race.py")
