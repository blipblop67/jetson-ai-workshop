"""
CUDA EXAMPLE 4 - LIVE CAMERA FILTERS: CPU vs GPU
================================================
Applies image filters to the live camera, using the SAME kind of maths a CNN
uses (convolution: sliding a small grid over the image).

    python3 04_edge_filter.py              # USB camera
    python3 04_edge_filter.py --csi        # CSI camera
    python3 04_edge_filter.py --benchmark  # no camera: just time CPU vs GPU

Keys (click the window first)
    G   : switch between CPU and GPU  (watch the FPS!)
    1   : edge detection  (finds outlines, like a CNN's first layer)
    2   : heavy blur      (a big 15x15 filter = lots of maths)
    3   : cartoon effect  (edges + colours)
    Q   : quit
"""

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

GPU = torch.cuda.is_available()

# Sobel filters: tiny 3x3 grids that respond to horizontal / vertical edges
SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
SOBEL_Y = SOBEL_X.t()


def make_filters(device):
    edge = torch.stack([SOBEL_X, SOBEL_Y]).unsqueeze(1).to(device)      # (2,1,3,3)
    blur = (torch.ones(3, 1, 15, 15) / 225.0).to(device)                # 15x15 average, per colour
    soft = (torch.ones(3, 1, 7, 7) / 49.0).to(device)                   # 7x7 average (cartoon)
    return edge, blur, soft


def apply(frame_bgr, mode, device, filters):
    """Run the filter on 'device' (cpu or cuda) and return a BGR image."""
    edge_k, blur_k, soft_k = filters
    x = torch.from_numpy(frame_bgr).to(device).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1,3,H,W)
    if mode in (1, 3):
        gray = x.mean(dim=1, keepdim=True)                       # (1,1,H,W)
        g = F.conv2d(gray, edge_k, padding=1)                    # slide the Sobel grids
        mag = torch.sqrt(g[:, 0:1] ** 2 + g[:, 1:2] ** 2).clamp(0, 1)
        if mode == 1:
            out = mag.repeat(1, 3, 1, 1)
        else:
            smooth = F.conv2d(x, soft_k, padding=3, groups=3)    # 7x7 blur
            colours = torch.round(smooth * 5) / 5                # fewer colours = cartoon look
            out = colours * (mag < 0.25).float()                 # draw dark outlines
    else:
        out = F.conv2d(x, blur_k, padding=7, groups=3)           # 15x15 blur on each colour
    out = (out.clamp(0, 1) * 255).byte().squeeze(0).permute(1, 2, 0)
    return np.ascontiguousarray(out.cpu().numpy())        # OpenCV needs a normal (contiguous) array


def open_camera(cam, csi, w=1280, h=720):
    if csi:
        pipe = (f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={w}, height={h}, framerate=30/1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1")
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise SystemExit("[ERROR] Camera not found. Try --cam 1, --csi, or --benchmark (no camera)")
    return cap


def benchmark():
    """No camera needed: time each filter on a 1280x720 test image."""
    frame = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
    names = {1: "edge detection", 2: "heavy blur 15x15", 3: "cartoon"}
    devices = ["cpu"] + (["cuda"] if GPU else [])
    filt = {d: make_filters(d) for d in devices}
    print("Filter timing on a 1280x720 image (frames per second, higher = better)\n")
    print(f"{'filter':18s} | {'CPU fps':>8} | {'GPU fps':>8} | speed-up")
    print("-" * 52)
    for mode, name in names.items():
        fps = {}
        for d in devices:
            apply(frame, mode, d, filt[d])                       # warm-up
            if d == "cuda": torch.cuda.synchronize()
            t0, n = time.perf_counter(), 0
            while time.perf_counter() - t0 < 2.0:                # run for 2 seconds
                apply(frame, mode, d, filt[d]); n += 1
            fps[d] = n / (time.perf_counter() - t0)
        g = fps.get("cuda")
        print(f"{name:18s} | {fps['cpu']:8.1f} | " + (f"{g:8.1f}" if g else f"{'-':>8}") + " | "
              + (f"{g / fps['cpu']:.1f}x" if g else "no GPU"))


def live(args):
    cap = open_camera(args.cam, args.csi)
    device = "cuda" if GPU else "cpu"
    filters = {d: make_filters(d) for d in (["cpu", "cuda"] if GPU else ["cpu"])}
    mode, fps, t_prev = 1, 0.0, time.time()
    names = {1: "EDGES", 2: "HEAVY BLUR", 3: "CARTOON"}
    print("G = switch CPU/GPU, 1/2/3 = filter, Q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = apply(frame, mode, device, filters[device])
        now = time.time(); fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6); t_prev = now
        h = out.shape[0]
        cv2.rectangle(out, (0, 0), (470, 80), (0, 0, 0), -1)
        col = (0, 255, 0) if device == "cuda" else (0, 165, 255)
        cv2.putText(out, f"{device.upper().replace('CUDA', 'GPU')}  |  {names[mode]}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.putText(out, f"FPS: {fps:5.1f}", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(out, "G: CPU/GPU   1 2 3: filter   Q: quit", (12, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        small = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))   # original in the corner
        out[h - small.shape[0] - 40:h - 40, -small.shape[1] - 10:-10] = small
        cv2.imshow("CUDA Example 4 - Filters (Q to quit)", out)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27): break
        if k == ord("g") and GPU:
            device = "cpu" if device == "cuda" else "cuda"; fps = 0.0
            print(f"  now running on: {device.upper()}")
        if k in (ord("1"), ord("2"), ord("3")):
            mode = k - ord("0")
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--csi", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="no camera: time CPU vs GPU")
    a = ap.parse_args()
    benchmark() if a.benchmark else live(a)
