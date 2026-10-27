"""
DETECTION 1 - REAL-TIME OBJECT DETECTION WITH YOLO
==================================================
YOLO ("You Only Look Once") finds EVERY object in the image in one go and
draws a box around each one with a name and a confidence score.
It is pre-trained on the COCO dataset: 80 everyday things (person, car,
bottle, cup, cell phone, laptop, chair, dog, ...).

    python3 d1_detect.py                          # all 80 classes, USB camera
    python3 d1_detect.py --classes person cup     # only these classes
    python3 d1_detect.py --csi                    # CSI camera
    python3 d1_detect.py --source video.mp4       # a video file
    python3 d1_detect.py --image photo.jpg        # one photo (saved to detections/)
    python3 d1_detect.py --model yolo26n.engine   # the TensorRT version (see d2_tensorrt.py)

Keys:  +/- confidence threshold   S screenshot   Q quit
"""

import argparse
import os
import time

import cv2

from od_common import DEFAULT_MODEL, open_source, load_model, class_ids, count_by_class, FPS, panel


def main():
    ap = argparse.ArgumentParser(description="Live object detection with YOLO")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="yolo26n.pt (PyTorch) or yolo26n.engine (TensorRT)")
    ap.add_argument("--classes", nargs="+", help="only detect these, e.g. --classes person bottle")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold 0-1 (default 0.4)")
    ap.add_argument("--imgsz", type=int, default=640, help="image size the model sees (default 640)")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--csi", action="store_true", help="use a CSI camera")
    ap.add_argument("--image", nargs="+", help="detect in photo(s) instead of the camera")
    args = ap.parse_args()

    model = load_model(args.model)
    classes = class_ids(model, args.classes)

    # ---------------- photos ----------------
    if args.image:
        os.makedirs("detections", exist_ok=True)
        for p in args.image:
            r = model.predict(p, conf=args.conf, classes=classes, imgsz=args.imgsz, verbose=False)[0]
            out = os.path.join("detections", os.path.basename(p))
            cv2.imwrite(out, r.plot())
            print(f"{os.path.basename(p)}: {count_by_class(r) or 'nothing found'}  ->  {out}")
        return

    # ---------------- live ----------------
    cap = open_source(args.source, args.csi)
    fps, most = FPS(), {}
    print("[INFO] Running. +/- = confidence, S = screenshot, Q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        r = model.predict(frame, conf=args.conf, classes=classes, imgsz=args.imgsz, verbose=False)[0]
        infer_ms = (time.time() - t0) * 1000
        view = r.plot()                                   # boxes + labels drawn by Ultralytics
        counts = count_by_class(r)
        for k, v in counts.items():
            most[k] = max(most.get(k, 0), v)

        lines = [f"FPS {fps.tick():5.1f}   ({infer_ms:.0f} ms/frame)", f"confidence >= {args.conf:.2f}"]
        lines += [f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])][:8]
        panel(view, lines)
        cv2.imshow("Detection 1 - YOLO (Q to quit)", view)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("+"), ord("=")):
            args.conf = min(0.95, args.conf + 0.05)
        if k in (ord("-"), ord("_")):
            args.conf = max(0.05, args.conf - 0.05)
        if k == ord("s"):
            name = f"detect_{int(time.time())}.jpg"; cv2.imwrite(name, view); print(f"[INFO] Saved {name}")

    cap.release()
    cv2.destroyAllWindows()
    if most:
        print("\nMost seen at once:", ", ".join(f"{k} x{v}" for k, v in sorted(most.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
