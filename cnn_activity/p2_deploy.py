"""
PROJECT 2 (REGRESSION) - STEP 3: TEST & DEPLOY
==============================================
Loads p2_tracker.pth and predicts WHERE your object is, live.

    python3 p2_deploy.py                    # live tracker
    python3 p2_deploy.py --csi              # CSI camera
    python3 p2_deploy.py --image photo.jpg  # predict on photos (saves marked copies)
    python3 p2_deploy.py --model p2_tracker_resnet18_finetune.pth   # transfer-learning model

Live controls
-------------
    G       : start the TARGET GAME - move your object onto the red target!
    S       : screenshot
    Q / ESC : quit

What you see: a crosshair where the CNN thinks the object is, a trail of the
last positions, the (x, y) numbers and which zone of the image it is in.
The terminal prints a line every time the object moves to a new zone - an
example of an application reacting to a regression output (like a robot
steering left/right, or a camera that follows a person).
"""

import argparse
import os
import random
import time
from collections import deque

import cv2
import torch

from common import get_device, open_camera, center_square, preprocess, load_checkpoint

ZONES_Y = ["top", "middle", "bottom"]
ZONES_X = ["left", "centre", "right"]


def load_model(path, device):
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] {path} not found. Run p2_train.py first.")
    model, ckpt = load_checkpoint(path, device, regression=True)   # cnn or resnet18
    print(f"[INFO] Loaded {ckpt.get('arch', 'cnn')} tracker for: {ckpt.get('object', 'object')}")
    return model, ckpt.get("object", "object")


@torch.no_grad()
def predict_xy(model, square_bgr, device):
    x, y = model(preprocess(square_bgr).to(device))[0].tolist()
    return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)


def zone(x, y):
    return f"{ZONES_Y[min(int(y * 3), 2)]}-{ZONES_X[min(int(x * 3), 2)]}"


def test_images(model, obj, device, paths):
    os.makedirs("p2_predictions", exist_ok=True)
    print(f"\n{'image':36s}   x      y     zone")
    print("-" * 64)
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{p:36s}   [could not read]"); continue
        sq, _ = center_square(img)
        x, y = predict_xy(model, sq, device)
        out = cv2.resize(sq, (480, 480))
        cv2.drawMarker(out, (int(x * 480), int(y * 480)), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.circle(out, (int(x * 480), int(y * 480)), 22, (0, 0, 255), 2)
        cv2.imwrite(os.path.join("p2_predictions", os.path.basename(p)), out)
        print(f"{os.path.basename(p)[:36]:36s} {x:5.2f}  {y:5.2f}   {zone(x, y)}")
    print("Marked copies saved in: p2_predictions/")


def test_live(model, obj, device, args):
    cap = open_camera(args.cam, args.csi)
    trail = deque(maxlen=20)
    sx = sy = None                     # smoothed position
    fps, t_prev = 0.0, time.time()
    last_zone = None
    game, score, target, hold, game_end = False, 0, None, 0, 0.0
    print("[INFO] Tracker running. G = target game, S = screenshot, Q = quit.")
    print("[INFO] Zone log:")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Could not read frame from camera"); break
        square, (x0, y0, s) = center_square(frame)
        x, y = predict_xy(model, square, device)

        # smooth the prediction a little so the crosshair doesn't jitter
        sx = x if sx is None else 0.6 * sx + 0.4 * x
        sy = y if sy is None else 0.6 * sy + 0.4 * y
        px, py = int(x0 + sx * s), int(y0 + sy * s)
        trail.append((px, py))

        z = zone(sx, sy)
        if z != last_zone:
            print(f"  [{time.strftime('%H:%M:%S')}] {obj} -> {z}")
            last_zone = z

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6)); t_prev = now

        view = frame.copy()
        cv2.rectangle(view, (x0, y0), (x0 + s - 1, y0 + s - 1), (0, 255, 0), 2)

        # ---- target game ----
        if game:
            if target is None:
                target = (random.uniform(0.15, 0.85), random.uniform(0.15, 0.85))
            tx, ty = int(x0 + target[0] * s), int(y0 + target[1] * s)
            r = int(0.09 * s)
            inside = (px - tx) ** 2 + (py - ty) ** 2 < r ** 2
            hold = hold + 1 if inside else 0
            cv2.circle(view, (tx, ty), r, (0, 0, 255), 3)
            cv2.circle(view, (tx, ty), int(r * min(hold, 8) / 8), (0, 0, 255), -1)
            if hold >= 8:                                  # held on target for 8 frames
                score += 1; target, hold = None, 0
            left = max(0, int(game_end - now))
            if left == 0:
                print(f"[GAME] Time's up! Score: {score}")
                game = False
            cv2.putText(view, f"SCORE {score}   TIME {left}s", (x0 + 10, y0 + s - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ---- tracker drawing ----
        for i in range(1, len(trail)):
            cv2.line(view, trail[i - 1], trail[i], (0, 200, 255), 2)
        cv2.drawMarker(view, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 36, 3)
        cv2.circle(view, (px, py), 18, (255, 0, 255), 2)

        cv2.rectangle(view, (0, 0), (300, 84), (0, 0, 0), -1)
        cv2.putText(view, f"{obj}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(view, f"x = {sx:.2f}   y = {sy:.2f}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(view, f"zone: {z}", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(view, f"FPS: {fps:.1f}", (view.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if not game:
            cv2.putText(view, "G = target game", (view.shape[1] - 190, view.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Project 2 - Tracker (q to quit)", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("g"):
            game, score, target, hold, game_end = True, 0, None, 0, now + args.game_time
            print(f"[GAME] Go! Hit as many targets as you can in {args.game_time}s")
        if key == ord("s"):
            name = f"tracker_{int(now)}.jpg"; cv2.imwrite(name, view); print(f"[INFO] Saved {name}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Live object tracker (regression)")
    ap.add_argument("--model", default="p2_tracker.pth")
    ap.add_argument("--image", nargs="+", help="predict on image files instead of the camera")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--csi", action="store_true")
    ap.add_argument("--game-time", type=int, default=45, help="seconds per game (default 45)")
    args = ap.parse_args()

    device = get_device()
    model, obj = load_model(args.model, device)
    if args.image:
        test_images(model, obj, device, args.image)
    else:
        test_live(model, obj, device, args)


if __name__ == "__main__":
    main()
