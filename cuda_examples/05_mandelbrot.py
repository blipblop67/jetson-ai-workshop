"""
CUDA EXAMPLE 5 - DRAW A FRACTAL: CPU vs GPU
===========================================
The Mandelbrot set is a famous pattern. Every pixel needs its OWN
calculation (up to 300 steps each), and no pixel depends on another.
That's perfect for a GPU: thousands of workers, one pixel each.

    python3 05_mandelbrot.py              # draws mandelbrot.png
    python3 05_mandelbrot.py --zoom 50    # zoom in (try 10, 100, 1000)
"""

import argparse
import time

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=1200)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--zoom", type=float, default=1.0)
args = ap.parse_args()

# the interesting spot we zoom into
CX, CY = -0.743643887, 0.131825904


def mandelbrot(lib, device=None):
    """Same code for CPU (numpy) and GPU (torch): count steps until each point 'escapes'."""
    span = 3.0 / args.zoom
    if lib == "numpy":
        xs = np.linspace(CX - span * 0.75, CX + span * 0.75, args.width, dtype=np.float32)
        ys = np.linspace(CY - span * 0.5, CY + span * 0.5, args.height, dtype=np.float32)
        cr, ci = np.meshgrid(xs, ys)
        zr, zi = np.zeros_like(cr), np.zeros_like(ci)
        count = np.zeros(cr.shape, np.int32)
        alive = np.ones(cr.shape, bool)                              # points that haven't escaped yet
        for _ in range(args.steps):
            zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci           # z = z*z + c
            alive &= (zr * zr + zi * zi) < 4.0                          # once out, always out
            count += alive
            zr = np.where(alive, zr, 0.0); zi = np.where(alive, zi, 0.0)  # stop escaped points growing
        return count
    xs = torch.linspace(CX - span * 0.75, CX + span * 0.75, args.width, device=device)
    ys = torch.linspace(CY - span * 0.5, CY + span * 0.5, args.height, device=device)
    ci, cr = torch.meshgrid(ys, xs, indexing="ij")
    zr, zi = torch.zeros_like(cr), torch.zeros_like(ci)
    count = torch.zeros(cr.shape, dtype=torch.int32, device=device)
    alive = torch.ones(cr.shape, dtype=torch.bool, device=device)
    for _ in range(args.steps):
        zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
        alive &= (zr * zr + zi * zi) < 4.0
        count += alive
        zr = torch.where(alive, zr, 0.0); zi = torch.where(alive, zi, 0.0)
    if device == "cuda":
        torch.cuda.synchronize()
    return count.cpu().numpy()


pixels = args.width * args.height
print(f"Drawing a {args.width}x{args.height} fractal = {pixels:,} pixels x {args.steps} steps each\n")

t0 = time.perf_counter(); img = mandelbrot("numpy"); t_cpu = time.perf_counter() - t0
print(f"CPU : {t_cpu:6.2f} s")

if torch.cuda.is_available():
    mandelbrot("torch", "cuda")                       # warm-up
    t0 = time.perf_counter(); img = mandelbrot("torch", "cuda"); t_gpu = time.perf_counter() - t0
    print(f"GPU : {t_gpu:6.2f} s   ->  {t_cpu / t_gpu:.0f}x faster!")
else:
    print("GPU : not available")

# colour it in and save
norm = (np.sqrt(img.astype(np.float32) / args.steps) * 255).astype(np.uint8)   # sqrt = brighter detail
coloured = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
coloured[img == args.steps] = 0                         # points that never escaped = black
cv2.imwrite("mandelbrot.png", coloured)
print("\nSaved: mandelbrot.png  (open it!)")
print("Try zooming in:  python3 05_mandelbrot.py --zoom 100")
