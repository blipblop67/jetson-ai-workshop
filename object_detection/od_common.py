"""
od_common.py - shared helpers for the object detection projects.
(You don't run this file directly.)
"""

import os
import time

import cv2

DEFAULT_MODEL = "yolo26n.pt"      # "n" = nano: the smallest, fastest YOLO, good for Jetson


def open_source(source="0", csi=False, width=1280, height=720):
    """Camera index ("0"), CSI camera (csi=True) or a video file path."""
    if csi:
        pipe = (f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, framerate=30/1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1")
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    elif str(source).isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv2.VideoCapture(source)                 # a video file
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open '{source}'. Camera: try --source 1 or --csi. "
                         "Check cameras with: ls /dev/video*")
    return cap


def load_model(path):
    """Load a YOLO model (.pt = PyTorch, .engine = TensorRT)."""
    if path.endswith(".engine") and not os.path.exists(path):
        raise SystemExit(f"[ERROR] {path} not found. Create it first:  python3 d2_tensorrt.py")
    from ultralytics import YOLO
    print(f"[INFO] Loading {path} (the first run downloads the model, about 6 MB)...")
    return YOLO(path, task="detect")


def class_ids(model, names):
    """['person', 'cup'] -> [0, 41]. None = every class."""
    if not names:
        return None
    lookup = {v.lower(): k for k, v in model.names.items()}
    ids, bad = [], []
    for n in names:
        n = n.lower().replace("_", " ")
        (ids.append(lookup[n]) if n in lookup else bad.append(n))
    if bad:
        raise SystemExit(f"[ERROR] Unknown class: {bad}\nThis model knows: {', '.join(model.names.values())}")
    return ids


def count_by_class(result):
    """{'person': 2, 'cup': 1} for one YOLO result."""
    counts = {}
    for c in result.boxes.cls.tolist():
        name = result.names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    return counts


class FPS:
    def __init__(self):
        self.t, self.fps = time.time(), 0.0

    def tick(self):
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.t, 1e-6) if self.fps else 1.0 / max(now - self.t, 1e-6)
        self.t = now
        return self.fps


def panel(img, lines, x=0, y=0, color=(255, 255, 255), title_color=(0, 255, 255)):
    """Black box with text lines in the top-left corner."""
    if not lines:
        return
    w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for l in lines) + 20
    cv2.rectangle(img, (x, y), (x + w, y + 12 + 26 * len(lines)), (0, 0, 0), -1)
    for i, l in enumerate(lines):
        cv2.putText(img, l, (x + 10, y + 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    title_color if i == 0 else color, 2)
