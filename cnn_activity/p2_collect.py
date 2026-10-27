"""
PROJECT 2 (REGRESSION) - STEP 1: COLLECT & LABEL
================================================
Classification answered "WHAT is it?". Regression answers "HOW MUCH / WHERE?"
with numbers. Here the CNN will learn WHERE your object is: an (x, y) position.

Pick ONE object (your water bottle, a ball, a toy...). Every photo needs a
label = the position of the object. You label by CLICKING on it:

    one click on the object  =  take the photo  +  save its (x, y) label

    python3 p2_collect.py                     # asks for the object name
    python3 p2_collect.py --object bottle --target 100
    python3 p2_collect.py --label my_phone_photos/    # label photos from a phone

Camera window controls
----------------------
    LEFT CLICK on the object : save photo + position
    U                        : undo the last photo
    Q / ESC                  : finish

Saved as:
    p2_dataset/images/bottle_001.jpg ...
    p2_dataset/labels.csv     (filename, x, y)   x, y go from 0.0 to 1.0
                                                 (0,0) = top-left, (1,1) = bottom-right

TIP: move the object to a NEW spot before every click and cover the WHOLE
green square (corners and edges too!). The 3x3 grid on screen shows which
areas still need photos.
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np

from common import open_camera, center_square, clean_name, load_image

GRID = 3          # coverage grid size (3x3)


class Labels:
    """Keeps p2_dataset/labels.csv and the images folder in sync."""
    def __init__(self, root):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.csv = os.path.join(root, "labels.csv")
        os.makedirs(self.img_dir, exist_ok=True)
        self.rows = []
        if os.path.exists(self.csv):
            with open(self.csv) as f:
                self.rows = [(r["filename"], float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]

    def save_all(self):
        with open(self.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "x", "y"])
            for r in self.rows:
                w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}"])

    def add(self, img_square, x, y, name):
        i = len(self.rows) + 1
        while True:
            fname = f"{name}_{i:03d}.jpg"
            if not os.path.exists(os.path.join(self.img_dir, fname)):
                break
            i += 1
        cv2.imwrite(os.path.join(self.img_dir, fname), img_square)
        self.rows.append((fname, x, y))
        self.save_all()

    def undo(self):
        if not self.rows:
            return None
        fname, _, _ = self.rows.pop()
        try:
            os.remove(os.path.join(self.img_dir, fname))
        except FileNotFoundError:
            pass
        self.save_all()
        return fname

    def coverage(self):
        g = np.zeros((GRID, GRID), int)
        for _, x, y in self.rows:
            g[min(int(y * GRID), GRID - 1), min(int(x * GRID), GRID - 1)] += 1
        return g


def ask_object(root):
    meta = os.path.join(root, "object.txt")
    if os.path.exists(meta):
        name = open(meta).read().strip()
        print(f"[INFO] Continuing with object: {name}")
        return name
    print("\n=== Which object will your AI track? ===")
    print("Pick ONE object you can move around easily (bottle, ball, toy, phone...).")
    name = ""
    while not name:
        name = clean_name(input("Object name: "))
    return name


def draw_coverage(view, cov, x, y, cell=26):
    """Small 3x3 heat-map: green = enough photos there, red = needs more."""
    target = max(5, cov.sum() // (GRID * GRID))
    cv2.rectangle(view, (x - 6, y - 24), (x + cell * GRID + 6, y + cell * GRID + 6), (40, 40, 40), -1)
    cv2.putText(view, "coverage", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for r in range(GRID):
        for c in range(GRID):
            n = cov[r, c]
            ratio = min(1.0, n / target)
            col = (0, int(200 * ratio), int(220 * (1 - ratio)))
            p0 = (x + c * cell, y + r * cell)
            cv2.rectangle(view, p0, (p0[0] + cell - 2, p0[1] + cell - 2), col, -1)
            cv2.putText(view, str(n), (p0[0] + 4, p0[1] + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def save_preview(labels, out="p2_dataset_preview.jpg", n=12, t=120):
    """Contact sheet with the labelled position drawn on each photo."""
    rows = labels.rows
    if not rows:
        return None
    picks = rows[:: max(1, len(rows) // n)][:n]
    tiles = []
    for fname, x, y in picks:
        img = cv2.imread(os.path.join(labels.img_dir, fname))
        if img is None:
            continue
        img = cv2.resize(img, (t, t))
        cv2.circle(img, (int(x * t), int(y * t)), 14, (255, 255, 255), 3)   # ring, so the object
        cv2.circle(img, (int(x * t), int(y * t)), 14, (0, 0, 255), 2)       # stays visible
        cv2.circle(img, (int(x * t), int(y * t)), 2, (0, 0, 255), -1)
        tiles.append(img)
    while len(tiles) % 6:
        tiles.append(np.full((t, t, 3), 230, np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    cv2.imwrite(out, sheet)
    return out


# --------------------------------------------------------------------------
# Mode A: live camera, click to capture + label
# --------------------------------------------------------------------------
def collect_live(labels, name, args):
    cap = open_camera(args.cam, args.csi)
    win = "Project 2 - Click on your object (Q to finish)"
    cv2.namedWindow(win)
    state = {"click": None}
    cv2.setMouseCallback(win, lambda ev, x, y, *_: state.update(click=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)

    flash_pt, flash_until = None, 0.0
    msg, msg_until = "", 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Could not read frame from camera")
            break
        square, (x0, y0, s) = center_square(frame)
        now = time.time()

        # ---- a click saves THIS frame with the click position as the label ----
        if state["click"] is not None:
            cx, cy = state["click"]; state["click"] = None
            if x0 <= cx < x0 + s and y0 <= cy < y0 + s:
                lx, ly = (cx - x0) / s, (cy - y0) / s          # normalise to 0..1
                labels.add(square, lx, ly, name)
                flash_pt, flash_until = (cx, cy), now + 0.6
            else:
                msg, msg_until = "Click INSIDE the green square", now + 1.5

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u"):
            f = labels.undo()
            msg, msg_until = (f"Removed {f}" if f else "Nothing to undo"), now + 1.5

        # ---- draw ----
        view = frame.copy()
        h, w = view.shape[:2]
        cv2.rectangle(view, (x0, y0), (x0 + s - 1, y0 + s - 1), (0, 255, 0), 2)
        for gx in range(1, GRID):                         # faint 3x3 guide lines
            cv2.line(view, (x0 + gx * s // GRID, y0), (x0 + gx * s // GRID, y0 + s), (0, 120, 0), 1)
            cv2.line(view, (x0, y0 + gx * s // GRID), (x0 + s, y0 + gx * s // GRID), (0, 120, 0), 1)
        for _, lx, ly in labels.rows[-8:]:                # recent labels as small dots
            cv2.circle(view, (int(x0 + lx * s), int(y0 + ly * s)), 4, (0, 200, 255), -1)
        if now < flash_until:
            cv2.circle(view, flash_pt, 14, (0, 0, 255), 3)
            cv2.drawMarker(view, flash_pt, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

        n = len(labels.rows)
        cv2.rectangle(view, (0, 0), (w, 70), (40, 40, 40), -1)
        cv2.putText(view, f"Tracking: {name.upper()}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(view, f"{n}/{args.target} labelled", (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        bw = w - 24
        cv2.rectangle(view, (12, 42), (12 + bw, 58), (90, 90, 90), -1)
        cv2.rectangle(view, (12, 42), (12 + int(bw * min(n, args.target) / args.target), 58), (0, 200, 0), -1)

        cv2.rectangle(view, (0, h - 56), (w, h), (40, 40, 40), -1)
        cv2.putText(view, "Move the object to a NEW spot, then CLICK on its centre", (12, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(view, "CLICK = photo + label | U undo | Q finish", (12, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        draw_coverage(view, labels.coverage(), w - 96, 104)
        if now < msg_until:
            cv2.rectangle(view, (30, h // 2 - 30), (w - 30, h // 2 + 12), (40, 40, 40), -1)
            cv2.putText(view, msg, (45, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if n == args.target and now < flash_until:
            cv2.putText(view, "Target reached! Keep going or press Q", (40, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win, view)

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Mode B: label photos taken with a phone (or any folder of images)
# --------------------------------------------------------------------------
def label_folder(labels, name, folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"[ERROR] No .jpg/.png photos found in {folder}")
    win = "Project 2 - Click on the object (S skip, Q quit)"
    cv2.namedWindow(win)
    state = {"click": None}
    cv2.setMouseCallback(win, lambda ev, x, y, *_: state.update(click=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    D = 480
    print(f"[INFO] {len(files)} photos to label. Click on the object in each one.")
    for i, f in enumerate(files):
        pil = load_image(os.path.join(folder, f))                 # handles phone rotation
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        sq, _ = center_square(img)
        sq = cv2.resize(sq, (D, D))
        while True:
            view = sq.copy()
            cv2.rectangle(view, (0, 0), (D, 30), (40, 40, 40), -1)
            cv2.putText(view, f"{i + 1}/{len(files)}  click the {name}  |  S skip  Q quit", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(win, view)
            key = cv2.waitKey(20) & 0xFF
            if state["click"] is not None:
                cx, cy = state["click"]; state["click"] = None
                labels.add(sq, cx / D, cy / D, name)
                break
            if key == ord("s"):
                break
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Collect and label (x, y) positions of one object")
    ap.add_argument("--object", help="object name (otherwise you are asked)")
    ap.add_argument("--target", type=int, default=80, help="number of labelled photos to aim for (default 80)")
    ap.add_argument("--data", default="p2_dataset", help="output folder (default: p2_dataset)")
    ap.add_argument("--label", metavar="FOLDER", help="label existing photos (e.g. from a phone) instead of the camera")
    ap.add_argument("--cam", type=int, default=0, help="USB camera index (default: 0)")
    ap.add_argument("--csi", action="store_true", help="use a CSI camera instead of USB")
    args = ap.parse_args()

    os.makedirs(args.data, exist_ok=True)
    name = clean_name(args.object) if args.object else ask_object(args.data)
    open(os.path.join(args.data, "object.txt"), "w").write(name)
    labels = Labels(args.data)

    if args.label:
        label_folder(labels, name, args.label)
    else:
        print("Camera starting... move your object, then CLICK on it in the camera window.")
        collect_live(labels, name, args)

    cov = labels.coverage()
    print(f"\n=== {len(labels.rows)} labelled photos of '{name}' ===")
    print("Coverage (photos per area of the image):")
    for r in cov:
        print("   " + " ".join(f"{v:4d}" for v in r))
    if cov.min() < 3 and len(labels.rows) >= 9:
        print("  -> Some areas have very few photos. The tracker will be weak there!")
    p = save_preview(labels)
    if p:
        print(f"Preview with your labels: {p}  (check each red ring sits on the object!)")
    print("Next: python3 p2_train.py")


if __name__ == "__main__":
    main()
