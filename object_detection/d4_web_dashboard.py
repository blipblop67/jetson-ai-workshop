"""
DETECTION 4 - DEPLOY IT: A LIVE WEB DASHBOARD
=============================================
Turns your Jetson into an AI camera that anyone on the same Wi-Fi can watch
from a phone or laptop browser. No monitor needed on the Jetson.

    python3 d4_web_dashboard.py                       # all classes
    python3 d4_web_dashboard.py --classes person car  # only these
    python3 d4_web_dashboard.py --model yolo26n.engine

Then open the address it prints, e.g.  http://192.168.1.23:5000

Pages:
    /        dashboard: live video + object counts
    /video   the raw live video stream (MJPEG)
    /stats   the numbers as JSON - other programs can read this (an API!)
"""

import argparse
import json
import socket
import threading
import time

import cv2
from flask import Flask, Response

from od_common import DEFAULT_MODEL, open_source, load_model, class_ids, count_by_class, FPS

PAGE = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jetson AI Camera</title><style>
body{font-family:sans-serif;background:#1e1e1e;color:#eee;margin:0;padding:16px}
h1{color:#76b900;font-size:22px;margin:0 0 12px}
img{width:100%;max-width:960px;border-radius:8px;border:2px solid #76b900}
.row{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
.card{background:#333;border-radius:8px;padding:10px 16px;min-width:110px}
.n{font-size:28px;font-weight:bold;color:#76b900}.l{font-size:13px;color:#bbb}
</style></head><body>
<h1>Jetson Orin Nano &middot; Live AI Camera</h1>
<img src="/video">
<div class="row" id="stats"></div>
<script>
async function tick(){
  const s = await (await fetch('/stats')).json();
  let h = `<div class="card"><div class="n">${s.fps}</div><div class="l">FPS</div></div>
           <div class="card"><div class="n">${s.total}</div><div class="l">objects now</div></div>`;
  for (const [k,v] of Object.entries(s.counts))
     h += `<div class="card"><div class="n">${v}</div><div class="l">${k}</div></div>`;
  document.getElementById('stats').innerHTML = h;
}
setInterval(tick, 1000); tick();
</script></body></html>"""

state = {"jpeg": None, "counts": {}, "fps": 0.0, "frames": 0, "started": time.strftime("%H:%M:%S")}
lock = threading.Lock()


def detector_loop(args):
    """Runs forever in the background: read camera -> detect -> store the latest result."""
    model = load_model(args.model)
    classes = class_ids(model, args.classes)
    cap = open_source(args.source, args.csi)
    fps = FPS()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1); continue
        r = model.predict(frame, conf=args.conf, classes=classes, verbose=False)[0]
        view = r.plot()
        ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with lock:
            state.update(jpeg=jpg.tobytes(), counts=count_by_class(r), fps=round(fps.tick(), 1),
                         frames=state["frames"] + 1)


def my_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))                 # no data is sent; just finds the Wi-Fi/LAN address
        ip = s.getsockname()[0]; s.close(); return ip
    except OSError:
        return "127.0.0.1"


app = Flask(__name__)


@app.route("/")
def index():
    return PAGE


@app.route("/stats")
def stats():
    with lock:
        c = dict(state["counts"])
        body = {"fps": state["fps"], "counts": c, "total": sum(c.values()),
                "frames": state["frames"], "started": state["started"]}
    return Response(json.dumps(body), mimetype="application/json")


@app.route("/video")
def video():
    def stream():
        last = None
        while True:
            with lock:
                jpg = state["jpeg"]
            if jpg is None or jpg is last:
                time.sleep(0.01); continue
            last = jpg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live AI camera web dashboard")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--classes", nargs="+")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--source", default="0")
    ap.add_argument("--csi", action="store_true")
    args = ap.parse_args()

    threading.Thread(target=detector_loop, args=(args,), daemon=True).start()
    print("\n" + "=" * 56)
    print(f"  Open on your phone/laptop (same Wi-Fi):  http://{my_ip()}:{args.port}")
    print(f"  Numbers for other programs (API):        http://{my_ip()}:{args.port}/stats")
    print("=" * 56 + "\n  Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
