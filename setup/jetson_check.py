"""
JETSON HEALTH CHECK
===================
Checks that your Jetson Orin Nano Super is set up and configured for the
workshop, and tells you how to fix anything that isn't.

    python3 jetson_check.py

Checks: model, JetPack/L4T version, power mode, memory, disk, temperature,
CUDA, PyTorch (GPU), OpenCV, TensorRT, Ultralytics YOLO and cameras.
"""

import glob
import os
import re
import shutil
import subprocess

OK, WARN, BAD = "✅", "⚠️ ", "❌"
tips = []


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def read(path):
    try:
        return open(path).read().strip().replace("\x00", "")
    except Exception:
        return ""


def line(status, label, value):
    print(f" {status} {label:<16} {value}")


print("=" * 60)
print(" JETSON HEALTH CHECK")
print("=" * 60)

# ---------------- hardware & software versions ----------------
model = read("/proc/device-tree/model")
line(OK if "Orin" in model else WARN, "Device", model or "not a Jetson (or unknown)")

rel = read("/etc/nv_tegra_release")
m = re.search(r"R(\d+).*REVISION: ([\d.]+)", rel)
# L4T (Jetson Linux) version -> JetPack version (newest first; first match wins)
L4T_TO_JETPACK = [("36.5.2", "6.2.3"), ("36.5", "6.2.2"), ("36.4.4", "6.2.1"), ("36.4.3", "6.2"),
                  ("36.4", "6.1"), ("36.3", "6.0"), ("36.2", "6.0 DP"), ("35.6", "5.1.4"),
                  ("35.5", "5.1.3"), ("35.4", "5.1.2"), ("35.", "5.x")]
if m:
    l4t = f"{m.group(1)}.{m.group(2)}"
    # 1st choice: ask the installed JetPack package directly (e.g. "6.2.2+b24")
    jp = run("dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null").split("+")[0]
    if not jp:   # 2nd choice: work it out from the L4T number
        jp = next((v for k, v in L4T_TO_JETPACK if l4t.startswith(k)), "unknown - newer than this script")
    line(OK, "L4T / JetPack", f"L4T {l4t}  (JetPack {jp})")
else:
    line(WARN, "L4T / JetPack", "unknown")

# ---------------- power mode ----------------
pm = run("nvpmodel -q 2>/dev/null") or run("sudo -n nvpmodel -q 2>/dev/null")
mode = re.search(r"Power Mode:\s*(\S+)", pm)
if mode:
    name = mode.group(1)
    good = "MAXN" in name.upper()
    line(OK if good else WARN, "Power mode", name)
    if not good:
        tips.append("Fastest mode: sudo nvpmodel -m 2   (MAXN SUPER; needs the Super flashing config)")
else:
    line(WARN, "Power mode", "could not read (try: sudo nvpmodel -q)")

# ---------------- memory, disk, temperature ----------------
mem = read("/proc/meminfo")
tot = re.search(r"MemTotal:\s+(\d+)", mem); avail = re.search(r"MemAvailable:\s+(\d+)", mem)
swap = re.search(r"SwapTotal:\s+(\d+)", mem)
if tot and avail:
    t, a = int(tot.group(1)) / 1e6, int(avail.group(1)) / 1e6
    line(OK if a > 2 else WARN, "Memory", f"{a:.1f} GB free of {t:.1f} GB (shared by CPU + GPU)")
    if a <= 2:
        tips.append("Low memory: close the browser/other apps before training")
if swap:
    line(OK, "Swap", f"{int(swap.group(1)) / 1e6:.1f} GB")
du = shutil.disk_usage(os.path.expanduser("~"))
line(OK if du.free > 5e9 else WARN, "Disk", f"{du.free / 1e9:.0f} GB free")
if du.free <= 5e9:
    tips.append("Low disk space: models and photos need a few GB")
temps = []
for z in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
    tp, v = read(z + "/type"), read(z + "/temp")
    if v.lstrip("-").isdigit() and int(v) > 0:
        temps.append((tp, int(v) / 1000))
if temps:
    hot = max(temps, key=lambda x: x[1])
    line(OK if hot[1] < 80 else WARN, "Temperature", f"hottest {hot[0]} {hot[1]:.0f}°C")

# ---------------- CUDA & Python libraries ----------------
nvcc = run("nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null")
v = re.search(r"release ([\d.]+)", nvcc)
line(OK if v else WARN, "CUDA (nvcc)", f"CUDA {v.group(1)}" if v else "nvcc not found")
if not v:
    tips.append("Add CUDA to PATH: export PATH=/usr/local/cuda/bin:$PATH")

try:
    import numpy
    np2 = int(numpy.__version__.split(".")[0]) >= 2
    line(BAD if np2 else OK, "NumPy", numpy.__version__ + ("  (must be < 2 for the Jetson PyTorch build)" if np2 else ""))
    if np2:
        tips.append('NumPy 2 breaks PyTorch on Jetson ("Numpy is not available"): pip install "numpy<2" && pip uninstall -y opencv-python')
except ImportError:
    line(BAD, "NumPy", "not installed")

try:
    import torch
    gpu = torch.cuda.is_available()
    line(OK if gpu else BAD, "PyTorch", f"{torch.__version__}  GPU: {'YES - ' + torch.cuda.get_device_name(0) if gpu else 'NO'}")
    if not gpu:
        tips.append("Install GPU PyTorch: pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/")
except ImportError as e:
    if "libcudss" in str(e):     # newer Jetson PyTorch builds (2.10+) need NVIDIA cuDSS; 2.8.0 doesn't
        line(BAD, "PyTorch", "installed, but this version needs cuDSS (libcudss.so.0)")
        tips.append("Switch to PyTorch 2.8.0 (no cuDSS needed):  pip uninstall -y torch torchvision && "
                    "pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/")
    else:
        line(BAD, "PyTorch", "not found")
        tips.append("PyTorch: activate the venv first (source ~/ai_workshop/env/bin/activate). "
                    "Not installed yet? See the setup steps in ~/ai_workshop/README.md")

try:
    import cv2
    gst = "GStreamer:                   YES" in cv2.getBuildInformation() or re.search(r"GStreamer:\s+YES", cv2.getBuildInformation())
    line(OK, "OpenCV", f"{cv2.__version__}  (GStreamer/CSI camera: {'yes' if gst else 'no'})")
except ImportError:
    line(BAD, "OpenCV", "not installed")
    tips.append("OpenCV: create the venv with --system-site-packages to use the Jetson's built-in OpenCV")

try:
    import tensorrt
    line(OK, "TensorRT", tensorrt.__version__)
except ImportError:
    line(WARN, "TensorRT", "not found in Python (needed for the TensorRT speed-up)")
    tips.append("TensorRT comes with JetPack; create the venv with --system-site-packages to see it")

try:
    import ultralytics
    line(OK, "Ultralytics", ultralytics.__version__)
except ImportError:
    line(WARN, "Ultralytics", "not installed")
    tips.append('Ultralytics (Chapter 6): pip install ultralytics lap flask "numpy<2" && pip uninstall -y opencv-python')

missing = []
for mod in ("onnx", "onnxslim", "onnxruntime"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod if mod != "onnxruntime" else "onnxruntime-gpu")
line(WARN if missing else OK, "ONNX tools", ("missing: " + ", ".join(missing)) if missing else "ready (for the TensorRT export)")
if missing:
    tips.append('ONNX tools (Chapter 6, D2): pip install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.82" onnxruntime-gpu "numpy<2" '
                '--extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/')

# ---------------- cameras ----------------
cams = sorted(glob.glob("/dev/video*"))
line(OK if cams else WARN, "Cameras", ", ".join(cams) if cams else "none found")
if not cams:
    tips.append("Plug in the USB camera (or check the CSI ribbon cable), then: ls /dev/video*")

print("-" * 60)
import sys
if sys.prefix == sys.base_prefix:
    print("Note: you are NOT inside the workshop venv. Run:  source ~/ai_workshop/env/bin/activate\n")
if tips:
    print("To fix / improve:")
    for t in tips:
        print("  -", t)
else:
    print("All good - your Jetson is ready!")
print("\nMonitor live: jtop    (install once: sudo pip3 install -U jetson-stats && sudo reboot)")
