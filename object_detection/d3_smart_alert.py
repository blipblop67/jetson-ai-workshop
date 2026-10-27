"""
DETECTION 3 - A PRACTICAL APP: SMART COUNTER & ZONE ALARM
=========================================================
Detection on its own just draws boxes. A real product DOES something with
them. This app tracks objects (each one gets an ID number) and:

  COUNT mode : counts objects crossing a line   (shop entrance, conveyor, car park)
  ZONE  mode : raises an alarm when an object enters a zone   (danger area, restricted door)

Every event is saved to events.csv and the alarm saves a snapshot to alerts/.

    python3 d3_smart_alert.py --mode count                      # count people crossing the middle
    python3 d3_smart_alert.py --mode count --classes bottle cup --line 0.4
    python3 d3_smart_alert.py --mode zone                       # alarm when a person enters the zone

Keys:  ZONE mode: click 4 points to draw your own zone, R = reset zone
       S screenshot   Q quit
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from od_common import DEFAULT_MODEL, open_source, load_model, class_ids, FPS, panel


class EventLog:
    """Appends one row per event to a CSV file (open it in Excel later!)."""
    def __init__(self, path="events.csv"):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["time", "mode", "event", "object", "track_id", "detail"])

    def add(self, mode, event, obj, tid="", detail=""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([ts, mode, event, obj, tid, detail])
        print(f"  [{ts[11:]}] {event:10s} {obj} {('#' + str(tid)) if tid != '' else ''} {detail}")


def main():
    ap = argparse.ArgumentParser(description="Smart counter / zone alarm")
    ap.add_argument("--mode", choices=["count", "zone"], default="count")
    ap.add_argument("--classes", nargs="+", default=["person"], help="what to watch (default: person)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--line", type=float, default=0.5, help="COUNT: line position, 0-1 across the image")
    ap.add_argument("--horizontal", action="store_true", help="COUNT: horizontal line (objects moving up/down)")
    ap.add_argument("--cooldown", type=float, default=5.0, help="ZONE: seconds between saved snapshots")
    ap.add_argument("--source", default="0")
    ap.add_argument("--csi", action="store_true")
    args = ap.parse_args()

    model = load_model(args.model)
    classes = class_ids(model, args.classes)
    cap = open_source(args.source, args.csi)
    log, fps = EventLog(), FPS()
    os.makedirs("alerts", exist_ok=True)

    win = f"Detection 3 - {args.mode.upper()} (Q to quit)"
    cv2.namedWindow(win)
    zone_pts, clicks = None, []
    cv2.setMouseCallback(win, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)

    last_side = {}                      # COUNT: which side of the line each track ID was on
    count_in = count_out = 0
    inside_now, alarm, last_snap, alarm_start = set(), False, 0.0, 0.0
    log.add(args.mode, "START", ",".join(args.classes))
    print(f"[INFO] Watching for: {', '.join(args.classes)}   (events -> events.csv)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if zone_pts is None:            # default zone: the middle of the image
            zone_pts = np.array([[w * .3, h * .3], [w * .7, h * .3], [w * .7, h * .9], [w * .3, h * .9]], np.int32)
        if len(clicks) == 4:            # user drew a new zone
            zone_pts = np.array(clicks, np.int32); clicks.clear()
            log.add("zone", "NEW ZONE", "-", detail=str(zone_pts.tolist()))

        # track = detect + give every object an ID that stays the same between frames
        r = model.track(frame, persist=True, conf=args.conf, classes=classes, verbose=False,
                        tracker="bytetrack.yaml")[0]
        view = r.plot()
        boxes = r.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

        if args.mode == "count":
            # ---------- line crossing ----------
            line_col = (0, 255, 0) if time.time() - alarm_start < 0.4 else (0, 255, 255)   # flashes green on a crossing
            if args.horizontal:
                ly = int(h * args.line); cv2.line(view, (0, ly), (w, ly), line_col, 3)
            else:
                lx = int(w * args.line); cv2.line(view, (lx, 0), (lx, h), line_col, 3)
            for (x1, y1, x2, y2), c, tid in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), ids):
                if tid is None:
                    continue
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                side = (cy > ly) if args.horizontal else (cx > lx)
                if tid in last_side and side != last_side[tid]:
                    name = r.names[int(c)]
                    if side:
                        count_in += 1; log.add("count", "IN", name, tid, f"total in={count_in}")
                    else:
                        count_out += 1; log.add("count", "OUT", name, tid, f"total out={count_out}")
                    alarm_start = time.time()                          # makes the line flash
                last_side[tid] = side
                cv2.circle(view, (int(cx), int(cy)), 6, (0, 255, 255), -1)
            panel(view, [f"FPS {fps.tick():4.1f}", f"IN  : {count_in}", f"OUT : {count_out}",
                         f"NOW : {count_in - count_out} inside"])

        else:
            # ---------- zone alarm ----------
            now_inside = set()
            for (x1, y1, x2, y2), c, tid in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), ids):
                foot = (float((x1 + x2) / 2), float(y2))             # bottom-centre = where it stands
                if cv2.pointPolygonTest(zone_pts, foot, False) >= 0:
                    now_inside.add((tid, r.names[int(c)]))
                    cv2.circle(view, (int(foot[0]), int(foot[1])), 8, (0, 0, 255), -1)
            new_alarm = bool(now_inside)
            if new_alarm and not alarm:
                alarm_start = time.time()
                for tid, name in now_inside:
                    log.add("zone", "ALARM", name, tid if tid is not None else "", f"{len(now_inside)} in zone")
            if not new_alarm and alarm:
                log.add("zone", "CLEAR", "-", detail=f"alarm lasted {time.time() - alarm_start:.1f}s")
            alarm = new_alarm
            overlay = view.copy()
            cv2.fillPoly(overlay, [zone_pts], (0, 0, 255) if alarm else (0, 200, 0))
            view = cv2.addWeighted(overlay, 0.25, view, 0.75, 0)
            cv2.polylines(view, [zone_pts], True, (0, 0, 255) if alarm else (0, 255, 0), 3)
            for p in clicks:
                cv2.circle(view, p, 6, (255, 0, 255), -1)
            if alarm:
                cv2.putText(view, "!! ALARM !!", (w // 2 - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
                if time.time() - last_snap > args.cooldown:
                    snap = os.path.join("alerts", datetime.now().strftime("alert_%Y%m%d_%H%M%S.jpg"))
                    cv2.imwrite(snap, view); last_snap = time.time()
                    log.add("zone", "SNAPSHOT", "-", detail=snap)
            panel(view, [f"FPS {fps.tick():4.1f}", f"in zone: {len(now_inside)}",
                         "click 4 points = new zone", "R = reset zone"])

        cv2.imshow(win, view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            zone_pts, clicks[:] = None, []
        if k == ord("s"):
            name = f"smart_{int(time.time())}.jpg"; cv2.imwrite(name, view); print(f"[INFO] Saved {name}")

    log.add(args.mode, "STOP", "-", detail=f"in={count_in} out={count_out}" if args.mode == "count" else "")
    cap.release(); cv2.destroyAllWindows()
    print("\nAll events saved in events.csv" + ("  |  snapshots in alerts/" if args.mode == "zone" else ""))


if __name__ == "__main__":
    main()
