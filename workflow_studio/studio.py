"""
AI WORKFLOW STUDIO - drag-and-drop AI projects on the Jetson Orin Nano
=====================================================================
    cd ~/ai_workshop && source env/bin/activate
    python3 workflow_studio/studio.py

Then open  http://localhost:8080  on the Jetson (or http://<jetson-ip>:8080 from a laptop).

Participants build projects by connecting pre-built blocks. Workflows are saved as plain
JSON settings (never code), so nothing can be installed or broken in the Python environment.
"""

import argparse
import json
import os
import re
import socket
import time

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory, abort

import engine

HERE = engine.HERE
PROJECTS = os.path.join(HERE, "projects")
TEMPLATES = os.path.join(HERE, "templates")
DEPLOYED = os.path.join(PROJECTS, "_deployed.json")      # the workflow that keeps running (survives restarts)
os.makedirs(PROJECTS, exist_ok=True)
PORT = 8080

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
runner = engine.Runner()


# Template groups (by task) - shown as headings in the Templates window
TASKS = [
    {"id": "count", "label": "🔢 Counting"},
    {"id": "safety", "label": "🚨 Safety & Security"},
    {"id": "inspect", "label": "🔍 Inspection & Sorting"},
    {"id": "time", "label": "⏱️ Time & Speed"},
]


def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9_\- ]", "", str(name)).strip().replace(" ", "_")
    return name[:60] or "my_project"


# ------------------------------------------------------------------ pages & catalogue
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/catalog")
def catalog():
    return jsonify({"blocks": engine.CATALOG, "order": engine.CATEGORY_ORDER, "colors": engine.CATEGORY_COLOR,
                    "block_order": list(engine.CATALOG), "coco": engine.COCO, "extra_names": engine.EXTRA_NAMES, "models": engine.list_models(),
                    "tasks": TASKS})


# ------------------------------------------------------------------ run control
@app.route("/api/validate", methods=["POST"])
def validate():
    _, _, problems = engine.validate(request.get_json(force=True))
    return jsonify({"problems": [{"node": n, "msg": m} for n, m in problems]})


def clear_deployment():
    if os.path.exists(DEPLOYED):
        os.remove(DEPLOYED)


@app.route("/api/run", methods=["POST"])
def run():
    problems = runner.start(request.get_json(force=True))
    if not problems:
        clear_deployment()            # a normal test run replaces a deployment
    return jsonify({"ok": not problems, "problems": [{"node": n, "msg": m} for n, m in problems]})


@app.route("/api/stop", methods=["POST"])
def stop():
    runner.stop()
    clear_deployment()
    return jsonify({"ok": True})


@app.route("/api/deploy", methods=["POST"])
def deploy():
    """Run the workflow as the Jetson's 'product': keeps running without the editor, resumes when the studio restarts."""
    wf = request.get_json(force=True)
    problems = runner.start(wf, deployed=True)
    if not problems:
        with open(DEPLOYED, "w") as f:
            json.dump(wf, f, indent=2)
    return jsonify({"ok": not problems, "problems": [{"node": n, "msg": m} for n, m in problems], **info_dict()})


@app.route("/api/deployed")
def deployed():
    if not os.path.exists(DEPLOYED):
        return jsonify({}), 404
    return send_from_directory(PROJECTS, os.path.basename(DEPLOYED))


def service_installed():
    return os.path.exists("/etc/systemd/system/ai_studio.service")


def info_dict():
    ip = my_ip()
    return {"ip": ip, "port": PORT, "view_url": f"http://{ip}:{PORT}/view", "boot": service_installed()}


@app.route("/api/info")
def info():
    return jsonify(info_dict())


@app.route("/view")
def view_page():
    """Simple live page for a laptop or phone: video, numbers, alerts. No editor."""
    return send_from_directory(app.static_folder, "view.html")


@app.route("/api/status")
def status():
    return jsonify(runner.status(int(request.args.get("alert", 0))))


STREAM_FPS = 15        # enough for a smooth view; more just fills the Wi-Fi and makes the picture lag


@app.route("/stream/<nid>")
def stream(nid):
    def gen():
        last, sent = None, 0.0
        while True:
            with runner.lock:
                jpg = runner.views.get(nid)
            wait = 1.0 / STREAM_FPS - (time.time() - sent)
            if jpg is None or jpg is last or wait > 0:
                time.sleep(0.2 if (not runner.running and jpg is None) else max(0.01, min(wait, 0.03)))
                continue
            last, sent = jpg, time.time()          # always the NEWEST frame: old ones are skipped, never queued
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/frame", methods=["POST"])
def frame():
    """One still image (for drawing zones): the live frame if running, else grab one from the source block."""
    img = runner.last_raw
    if img is None or not runner.running:
        wf = request.get_json(force=True) or {}
        src = next((n for n in wf.get("nodes", []) if engine.CATALOG.get(n["type"], {}).get("inputs", 1) == 0), None)
        if src is None:
            return jsonify({"error": "Add a Camera or Video block first."}), 400
        try:
            cap, _ = engine.open_source({"type": src["type"], "params": {**engine.defaults(src["type"]), **src.get("params", {})}})
            for _ in range(5):
                ok, img = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError("no image from the source")
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpg.tobytes(), mimetype="image/jpeg")


# ------------------------------------------------------------------ projects & templates
@app.route("/api/projects")
def list_projects():
    def items(folder):
        out = []
        for f in sorted(os.listdir(folder)):
            if f.endswith(".json") and not f.startswith("_"):
                try:
                    d = json.load(open(os.path.join(folder, f)))
                    out.append({"file": f, "name": d.get("name", f[:-5]), "description": d.get("description", ""),
                                "task": d.get("task", ""), "blocks": list(dict.fromkeys(n["type"] for n in d.get("nodes", [])))})
                except Exception:
                    pass
        return out
    return jsonify({"templates": items(TEMPLATES), "projects": items(PROJECTS)})


@app.route("/api/project/<kind>/<fname>")
def load_project(kind, fname):
    folder = {"template": TEMPLATES, "project": PROJECTS}.get(kind)
    if not folder:
        abort(404)
    return send_from_directory(folder, os.path.basename(fname))


@app.route("/api/project", methods=["POST"])
def save_project():
    wf = request.get_json(force=True)
    fname = safe_name(wf.get("name")) + ".json"
    with open(os.path.join(PROJECTS, fname), "w") as f:
        json.dump(wf, f, indent=2)
    return jsonify({"ok": True, "file": fname})


# ------------------------------------------------------------------ results
@app.route("/api/snapshots")
def snapshots():
    folder = os.path.join(engine.DATA, "snapshots")
    files = sorted((f for f in os.listdir(folder) if f.endswith(".jpg")), reverse=True)[:40]
    return jsonify(files)


@app.route("/snapshots/<fname>")
def snapshot_file(fname):
    return send_from_directory(os.path.join(engine.DATA, "snapshots"), os.path.basename(fname))


@app.route("/api/clips")
def clips():
    folder = os.path.join(engine.DATA, "clips")
    return jsonify(sorted((f for f in os.listdir(folder) if f.endswith(".mp4")), reverse=True)[:30])


@app.route("/clips/<fname>")
def clip_file(fname):
    return send_from_directory(os.path.join(engine.DATA, "clips"), os.path.basename(fname), as_attachment=True)


@app.route("/api/logs")
def logs():
    folder = os.path.join(engine.DATA, "logs")
    return jsonify(sorted(f for f in os.listdir(folder) if f.endswith(".csv")))


@app.route("/logs/<fname>")
def log_file(fname):
    return send_from_directory(os.path.join(engine.DATA, "logs"), os.path.basename(fname), as_attachment=True)


def my_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI Workflow Studio")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    PORT = args.port
    if os.path.exists(DEPLOYED):                      # resume the deployed project (e.g. after a reboot)
        try:
            wf = json.load(open(DEPLOYED))
            problems = runner.start(wf, deployed=True)
            print(f"  Deployed project '{wf.get('name')}' " + ("started." if not problems else f"NOT started: {problems}"))
        except Exception as e:
            print("  Could not resume the deployed project:", e)
    print("\n" + "=" * 60)
    print("  AI WORKFLOW STUDIO")
    print(f"  On this Jetson :  http://localhost:{args.port}")
    print(f"  From a laptop  :  http://{my_ip()}:{args.port}   (same Wi-Fi)")
    print(f"  Live page      :  http://{my_ip()}:{args.port}/view")
    print("=" * 60 + "\n  Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
