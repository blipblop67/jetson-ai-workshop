"""
PROJECT 1 (CLASSIFICATION) - STEP 1: COLLECT
============================================
Photograph your own objects.
YOU choose what the AI will recognise: grab 2-5 objects from your bag or
desk (phone, water bottle, pen, wallet, mouse, a toy ...).

The script asks for the name of each object, then guides you through taking
photos of each one, with tips on screen so you get a good mix of angles.

    python3 p1_collect.py                    # asks you for the object names
    python3 p1_collect.py --per-object 60    # take more photos per object
    python3 p1_collect.py --csi              # CSI camera instead of USB

Photos are saved like this (one folder per object):

    p1_dataset/
        water_bottle/   water_bottle_001.jpg ...
        phone/          ...
        nothing/        ...   <- empty table, so the AI can say "no object"

Camera window controls
----------------------
    SPACE : take ONE photo
    A     : AUTO mode on/off (3-2-1 countdown, then a photo every 0.4 s)
    N     : next object      B : back to previous object
    Q/ESC : finish

Took photos with your PHONE instead? Copy them into p1_dataset/<object_name>/
(JPG or PNG; on iPhone set Camera > Formats > "Most Compatible").
Running the script again adds more photos. Nothing is overwritten.
"""

import argparse
import os
import time

import cv2
import numpy as np

from common import open_camera, center_square, clean_name

# Tips shown on screen. They change as you take more photos of an object.
TIPS = [
    "Hold the object in the MIDDLE of the green square",
    "ROTATE it slowly - show every side",
    "Move it CLOSER, then FURTHER away",
    "Change the BACKGROUND (hand, table, paper, sleeve)",
    "Change the LIGHTING or tilt it at an angle",
]
NOTHING_TIP = "Point at the EMPTY table / background - no object!"


def ask_objects():
    """Ask the participant (in the terminal) which objects they will photograph."""
    print("\n=== What should your AI recognise? ===")
    print("Pick 2-5 objects you have with you (e.g. phone, water bottle, pen).")
    while True:
        try:
            n = int(input("How many objects? (2-5): ").strip())
            if 2 <= n <= 5:
                break
        except ValueError:
            pass
        print("  Please type a number from 2 to 5.")

    names = []
    while len(names) < n:
        raw = input(f"  Name of object {len(names) + 1}: ")
        name = clean_name(raw)
        if not name:
            print("    Name can't be empty.")
        elif name in names or name == "nothing":
            print("    That name is already used - pick another.")
        else:
            names.append(name)

    add_nothing = input("Add a 'nothing' class (empty background)? Recommended [Y/n]: ").strip().lower()
    if add_nothing in ("", "y", "yes"):
        names.append("nothing")
    return names


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def next_filename(folder, name):
    i = count_images(folder) + 1
    while True:
        path = os.path.join(folder, f"{name}_{i:03d}.jpg")
        if not os.path.exists(path):
            return path
        i += 1


def put(img, text, org, scale=0.7, color=(255, 255, 255), thick=2, box=False):
    if box:   # dark box behind text that sits on top of the camera image
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 10, y - th - 12), (x + tw + 10, y + 12), (40, 40, 40), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def save_preview(data_dir, objects, out="p1_dataset_preview.jpg", per_row=8, t=96):
    """Contact sheet: one row of thumbnails per object - a quick data check."""
    rows = []
    for obj in objects:
        folder = os.path.join(data_dir, obj)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        step = max(1, len(files) // per_row)
        picks = files[::step][:per_row]
        thumbs = []
        for f in picks:
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue
            sq, _ = center_square(img)
            thumbs.append(cv2.resize(sq, (t, t)))
        while len(thumbs) < per_row:
            thumbs.append(np.full((t, t, 3), 230, np.uint8))
        label = np.full((t, 170, 3), 255, np.uint8)
        cv2.putText(label, obj[:14], (8, t // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(label, f"{len(files)} photos", (8, t // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        rows.append(np.hstack([label] + thumbs))
    sheet = np.vstack(rows)
    cv2.imwrite(out, sheet)
    return out


def main():
    ap = argparse.ArgumentParser(description="Photograph your own objects for training")
    ap.add_argument("--objects", nargs="+", help="skip the questions, e.g. --objects phone pen nothing")
    ap.add_argument("--per-object", type=int, default=40, help="photo target per object (default 40)")
    ap.add_argument("--data", default="p1_dataset", help="output folder (default: p1_dataset)")
    ap.add_argument("--cam", type=int, default=0, help="USB camera index (default: 0)")
    ap.add_argument("--csi", action="store_true", help="use a CSI camera instead of USB")
    args = ap.parse_args()

    objects = [clean_name(o) for o in args.objects] if args.objects else ask_objects()
    for o in objects:
        os.makedirs(os.path.join(args.data, o), exist_ok=True)
    counts = {o: count_images(os.path.join(args.data, o)) for o in objects}

    print("\nObjects:", ", ".join(objects))
    print("Camera starting... click the camera window, then press SPACE to take photos.")
    cap = open_camera(args.cam, args.csi)

    idx = 0                    # which object we are photographing
    auto = False
    countdown_end = 0.0
    last_shot = 0.0
    flash = 0
    banner, banner_until = "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Could not read frame from camera")
            break
        obj = objects[idx]
        square, (x0, y0, s) = center_square(frame)
        now = time.time()

        # ---------------- keys ----------------
        key = cv2.waitKey(1) & 0xFF
        shoot = False
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            shoot = True
        elif key == ord("a"):
            auto = not auto
            countdown_end = now + 3 if auto else 0
        elif key == ord("n"):
            idx, auto = (idx + 1) % len(objects), False
        elif key == ord("b"):
            idx, auto = (idx - 1) % len(objects), False

        counting_down = auto and now < countdown_end
        if auto and not counting_down and now - last_shot > 0.4:
            shoot = True

        # ---------------- take the photo ----------------
        if shoot and now - last_shot > 0.15:
            folder = os.path.join(args.data, obj)
            cv2.imwrite(next_filename(folder, obj), square)
            counts[obj] += 1
            last_shot, flash = now, 2

            if counts[obj] == args.per_object:           # target reached -> next object
                auto = False
                remaining = [o for o in objects if counts[o] < args.per_object]
                if remaining:
                    idx = objects.index(remaining[0])
                    banner = f"Done! Next: {remaining[0].upper()}"
                else:
                    banner = "All objects done! Press Q to finish"
                banner_until = now + 2.5

        # ---------------- draw the screen ----------------
        view = frame.copy()
        h, w = view.shape[:2]
        col = (255, 255, 255) if flash else ((0, 200, 255) if counting_down else (0, 255, 0))
        cv2.rectangle(view, (x0, y0), (x0 + s - 1, y0 + s - 1), col, 3 if flash else 2)
        flash = max(0, flash - 1)

        # top panel: current object + progress
        cv2.rectangle(view, (0, 0), (w, 78), (40, 40, 40), -1)
        put(view, f"Object {idx + 1}/{len(objects)}: {obj.upper()}", (12, 30), 0.8, (0, 255, 255))
        n, target = counts[obj], args.per_object
        bar_w = w - 24
        cv2.rectangle(view, (12, 44), (12 + bar_w, 62), (90, 90, 90), -1)
        cv2.rectangle(view, (12, 44), (12 + int(bar_w * min(n, target) / target), 62), (0, 200, 0), -1)
        put(view, f"{n}/{target} photos", (w - 170, 30), 0.6)

        # tip at the bottom (changes every few photos)
        tip = NOTHING_TIP if obj == "nothing" else TIPS[min(n * len(TIPS) // max(target, 1), len(TIPS) - 1)]
        cv2.rectangle(view, (0, h - 60), (w, h), (40, 40, 40), -1)
        put(view, "TIP: " + tip, (12, h - 35), 0.55, (255, 255, 0))
        put(view, "SPACE photo | A auto | N next | B back | Q finish", (12, h - 10), 0.45, (200, 200, 200), 1)

        # side list of all objects
        cv2.rectangle(view, (w - 220, 84), (w, 92 + 24 * len(objects)), (40, 40, 40), -1)
        for i, o in enumerate(objects):
            mark = ">" if i == idx else " "
            c = (0, 255, 0) if counts[o] >= target else (255, 255, 255)
            put(view, f"{mark} {o[:12]}: {counts[o]}", (w - 210, 105 + 24 * i), 0.5, c, 1)

        if counting_down:
            put(view, str(int(countdown_end - now) + 1), (w // 2 - 20, h // 2 + 20), 2.5, (0, 200, 255), 5, box=True)
        elif auto:
            put(view, "AUTO", (20, 115), 0.8, (0, 0, 255), box=True)
        if now < banner_until:
            put(view, banner, (40, h // 2), 0.9, (0, 255, 0), 2, box=True)

        cv2.imshow("Project 1 - Collect (Q to finish)", view)

    cap.release()
    cv2.destroyAllWindows()

    # ---------------- summary ----------------
    print("\n=== Your dataset ===")
    for o in objects:
        warn = "   <-- take more photos!" if counts[o] < 30 else ""
        print(f"  {o:20s} {counts[o]:4d} photos{warn}")
    if len(objects) >= 2 and all(counts[o] > 0 for o in objects):
        print(f"Preview sheet saved: {save_preview(args.data, objects)}  (open it and check your photos!)")
    print(f"Photos are in: {os.path.abspath(args.data)}")
    print("Next: python3 p1_train.py")


if __name__ == "__main__":
    main()
