"""
PROJECT 1 (CLASSIFICATION) - STEP 3: TEST & DEPLOY
==================================================
Loads p1_classifier.pth and recognises YOUR objects. Hold each object in the
green square and see if the AI gets it right!

Live camera mode (default)
--------------------------
    python3 p1_deploy.py
    python3 p1_deploy.py --csi              # CSI camera
  Shows the predicted object, confidence for every object and FPS, and
  prints an event log in the terminal each time a different object appears
  (a tiny example of what a deployed AI application does with predictions).
  Press 's' to save a screenshot, 'q' / ESC to quit.

Transfer-learning model?  python3 p1_deploy.py --model p1_classifier_resnet18.pth

Image mode (no camera needed)
-----------------------------
    python3 p1_deploy.py --image my_photo.jpg
    python3 p1_deploy.py --folder test_images/
"""

import argparse
import os
import time

import cv2
import torch

from common import get_device, open_camera, center_square, preprocess, load_checkpoint


def load_model(path, device):
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] {path} not found. Run p1_train.py first.")
    model, ckpt = load_checkpoint(path, device)          # works for cnn AND resnet18
    classes = ckpt["classes"]
    print(f"[INFO] Loaded {ckpt.get('arch', 'cnn')} model with objects: {classes}")
    return model, classes


@torch.no_grad()
def predict(model, frame_bgr, device):
    """Returns a list of probabilities (one per class)."""
    x = preprocess(frame_bgr).to(device)
    probs = torch.softmax(model(x), dim=1)[0]        # scores -> probabilities
    return probs.cpu().tolist()


def draw_results(view, classes, probs, threshold, fps=None):
    best = max(range(len(probs)), key=lambda i: probs[i])
    label = classes[best] if probs[best] >= threshold else "not sure"
    if label == "nothing":
        label = "no object"
    colour = (0, 255, 0) if probs[best] >= threshold else (0, 165, 255)

    panel_h = 60 + 28 * len(classes)
    cv2.rectangle(view, (0, 0), (360, panel_h), (0, 0, 0), -1)
    cv2.putText(view, f"{label} ({probs[best]:.0%})", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    for i, (c, p) in enumerate(zip(classes, probs)):
        y = 65 + 28 * i
        cv2.rectangle(view, (150, y - 15), (150 + int(190 * p), y + 3),
                      (0, 255, 0) if i == best else (120, 120, 120), -1)
        cv2.putText(view, f"{c[:12]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(view, f"{p:.0%}", (155, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if fps is not None:
        cv2.putText(view, f"FPS: {fps:.1f}", (view.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return label, probs[best]


def test_images(model, classes, device, paths, threshold):
    print(f"\n{'image':40s} prediction        confidence")
    print("-" * 70)
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{p:40s} [could not read]")
            continue
        square, _ = center_square(img)
        probs = predict(model, square, device)
        best = max(range(len(probs)), key=lambda i: probs[i])
        label = classes[best] if probs[best] >= threshold else "not sure"
        print(f"{os.path.basename(p)[:40]:40s} {label:17s} {probs[best]:.1%}")


def test_live(model, classes, device, args):
    cap = open_camera(args.cam, args.csi)
    fps, t_prev = 0.0, time.time()
    current, candidate, streak = None, None, 0     # for the event log
    seen = {}                                       # object -> times it appeared
    print("[INFO] Live test started. 's' = screenshot, 'q' = quit.")
    print("[INFO] Event log (a new line each time a different object appears):")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Could not read frame from camera")
            break

        square, (x0, y0, s) = center_square(frame)
        probs = predict(model, square, device)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))   # smoothed FPS
        t_prev = now

        view = frame.copy()
        cv2.rectangle(view, (x0, y0), (x0 + s - 1, y0 + s - 1), (0, 255, 0), 2)
        label, conf = draw_results(view, classes, probs, args.threshold, fps)

        # ---- "deployment": react when a NEW object stays in view for 10 frames ----
        candidate, streak = (label, streak + 1) if label == candidate else (label, 1)
        if streak == 10 and label != current and label != "not sure":
            current = label
            seen[label] = seen.get(label, 0) + 1
            print(f"  [{time.strftime('%H:%M:%S')}] {label.upper():15s} ({conf:.0%})  seen {seen[label]}x")
        cv2.imshow("Project 1 - Deploy (q to quit)", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            name = f"screenshot_{int(now)}.jpg"
            cv2.imwrite(name, view)
            print(f"[INFO] Saved {name}")

    cap.release()
    cv2.destroyAllWindows()
    if seen:
        print("\n=== Session summary ===")
        for k, v in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"  {k:15s} appeared {v} time(s)")


def main():
    ap = argparse.ArgumentParser(description="Test the trained CNN")
    ap.add_argument("--model", default="p1_classifier.pth")
    ap.add_argument("--image", nargs="+", help="one or more image files to classify")
    ap.add_argument("--folder", help="classify every image in this folder")
    ap.add_argument("--cam", type=int, default=0, help="USB camera index (default: 0)")
    ap.add_argument("--csi", action="store_true", help="use a CSI camera instead of USB")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="below this confidence show 'not sure' (default: 0.6)")
    args = ap.parse_args()

    device = get_device()
    model, classes = load_model(args.model, device)

    if args.image or args.folder:
        paths = list(args.image or [])
        if args.folder:
            paths += sorted(os.path.join(args.folder, f) for f in os.listdir(args.folder)
                            if f.lower().endswith((".jpg", ".jpeg", ".png")))
        test_images(model, classes, device, paths, args.threshold)
    else:
        test_live(model, classes, device, args)


if __name__ == "__main__":
    main()
