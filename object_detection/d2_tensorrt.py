"""
DETECTION 2 - SPEED IT UP WITH TENSORRT
=======================================
TensorRT is NVIDIA's optimiser for running AI models on the GPU. It turns
the PyTorch model into an "engine" built for YOUR Jetson: it fuses layers
together, uses FP16 (half precision) and picks the fastest GPU code.
This is how real Jetson products deploy their models.

    python3 d2_tensorrt.py                 # build yolo26n.engine (a few minutes, once) + race
    python3 d2_tensorrt.py --race-only     # engine already built: just race

Then use it live:  python3 d1_detect.py --model yolo26n.engine
"""

import argparse
import os
import time

import cv2
import numpy as np

from od_common import DEFAULT_MODEL, open_source, load_model


def test_frames(n=20):
    """Grab real frames from the camera if there is one, otherwise make test images."""
    try:
        cap = open_source("0")
        frames = []
        for _ in range(n):
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
        if frames:
            print(f"[INFO] Using {len(frames)} camera frames for the race")
            return frames
    except SystemExit:
        pass
    print("[INFO] No camera - using test images for the race")
    return [np.random.randint(0, 255, (720, 1280, 3), np.uint8) for _ in range(n)]


def race(model, frames, label, runs=100):
    for f in frames[:5]:                                  # warm-up
        model.predict(f, verbose=False)
    t0 = time.time()
    for i in range(runs):
        model.predict(frames[i % len(frames)], verbose=False)
    ms = (time.time() - t0) / runs * 1000
    print(f"  {label:28s} {ms:7.1f} ms per frame  = {1000 / ms:6.1f} FPS")
    return ms


ONNX_FIX = ('pip install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.82" onnxruntime-gpu "numpy<2" \\\n'
            '    --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/')


def check_export_tools():
    """TensorRT export goes PyTorch -> ONNX -> TensorRT, so the ONNX tools must be installed first.
    (onnxruntime-gpu for Jetson is only on the Jetson AI Lab index, not on normal PyPI.)"""
    missing = []
    for mod, name in [("onnx", "onnx"), ("onnxslim", "onnxslim"), ("onnxruntime", "onnxruntime-gpu"), ("tensorrt", "tensorrt")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(name)
    if "tensorrt" in missing:
        raise SystemExit("[ERROR] Python can't see TensorRT. Create the venv with --system-site-packages "
                         "(TensorRT comes with JetPack).")
    if missing:
        raise SystemExit(f"[ERROR] Missing export tools: {', '.join(missing)}\nInstall them once (inside the venv):\n  {ONNX_FIX}")


def main():
    ap = argparse.ArgumentParser(description="Export YOLO to TensorRT and compare speed")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--race-only", action="store_true")
    args = ap.parse_args()

    engine = os.path.splitext(args.model)[0] + ".engine"
    pt_model = load_model(args.model)

    if not args.race_only or not os.path.exists(engine):
        check_export_tools()
        print(f"\n[STEP 1] Building the TensorRT engine (FP16). "
              "This takes a few minutes the first time - perfect time for a question!")
        t0 = time.time()
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            fp16 = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}   # newer Ultralytics renamed it
            out = pt_model.export(format="engine", imgsz=640, device=0, **fp16)
        except Exception as e:
            raise SystemExit(f"[ERROR] TensorRT export failed: {e}\n"
                             "  Run this on the Jetson, with the venv created using --system-site-packages "
                             "(so Python can see TensorRT).")
        engine = str(out)
        print(f"[DONE] {engine} built in {time.time() - t0:.0f} s")

    print("\n[STEP 2] The race: PyTorch vs TensorRT")
    frames = test_frames()
    t_pt = race(pt_model, frames, f"PyTorch  ({os.path.basename(args.model)})")
    t_trt = race(load_model(engine), frames, f"TensorRT ({os.path.basename(engine)})")
    print(f"\n  TensorRT is {t_pt / t_trt:.1f}x faster on this Jetson.")
    print(f"\nUse it:  python3 d1_detect.py --model {engine}")


if __name__ == "__main__":
    main()
