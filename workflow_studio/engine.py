"""
engine.py - the brain of AI Workflow Studio.

* CATALOG     : every block (module) the user can drag in, with its settings
* validate()  : checks a workflow before it runs and explains problems in plain words
* Runner      : runs a workflow frame by frame in a background thread

A workflow is plain data (no code), so participants can't break the Python environment:
    {"nodes": [{"id": "n1", "type": "camera", "params": {...}, "x": 0, "y": 0}, ...],
     "edges": [["n1", "n2"], ...]}
Every block has ONE input (except sources) and ONE output; an output can feed several blocks.
"""

import csv
import json
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSHOP = os.path.dirname(HERE)                                  # ~/ai_workshop
DATA = os.path.join(HERE, "data")                                 # snapshots, logs, videos, clips
MODELS = os.path.join(HERE, "models")                             # your own custom YOLO models (.pt / .engine)
for d in ("snapshots", "logs", "videos", "clips"):
    os.makedirs(os.path.join(DATA, d), exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

# ----------------------------------------------------------------------------------------------
# The block catalogue. type -> definition. Param kinds: text, number, bool, select, classes, zone
# "dynamic" selects are filled by the server (models, videos, ...).
# ----------------------------------------------------------------------------------------------
CATALOG = {
    # ---------------- SOURCES ----------------
    "camera": {"cat": "Source", "label": "Camera", "icon": "📷", "inputs": 0,
               "help": "Live video from the USB or CSI camera.",
               "params": [{"key": "source", "label": "Camera number", "kind": "text", "default": "0"},
                          {"key": "csi", "label": "CSI camera (ribbon cable)", "kind": "bool", "default": False},
                          {"key": "resolution", "label": "Resolution", "kind": "select", "default": "1280x720",
                           "options": ["640x480", "1280x720", "1920x1080"]}]},
    "video": {"cat": "Source", "label": "Video File", "icon": "🎞️", "inputs": 0,
              "help": "Plays a video from workflow_studio/data/videos (great for testing without a camera).",
              "params": [{"key": "file", "label": "Video", "kind": "select", "default": "", "dynamic": "videos"},
                         {"key": "loop", "label": "Loop forever", "kind": "bool", "default": True}]},
    "ipcam": {"cat": "Source", "label": "Phone / CCTV Camera", "icon": "📱", "inputs": 0,
              "help": "A network camera: a phone running an IP-camera app (e.g. 'IP Webcam' on Android: "
                      "http://<phone-ip>:8080/video) or a CCTV camera's RTSP address (rtsp://user:pass@ip:554/...).",
              "params": [{"key": "url", "label": "Camera address (URL)", "kind": "text", "default": "http://192.168.1.50:8080/video"}]},
    # ---------------- VISION AI ----------------
    "detect": {"cat": "Vision AI", "label": "Object Detector", "icon": "🎯",
               "help": "YOLO finds every object (80 everyday classes) and draws boxes. Turn on tracking to give "
                       "each object an ID (needed by the Line Counter).",
               "params": [{"key": "model", "label": "Model", "kind": "select", "default": "yolo26n.pt", "dynamic": "yolo"},
                          {"key": "classes", "label": "Only these objects (empty = all)", "kind": "classes", "default": []},
                          {"key": "conf", "label": "Confidence (0-1)", "kind": "number", "default": 0.4, "min": 0.05, "max": 0.95, "step": 0.05},
                          {"key": "track", "label": "Tracking (object IDs)", "kind": "bool", "default": False}]},
    "classify": {"cat": "Vision AI", "label": "My Classifier", "icon": "🧠",
                 "help": "Your own Project 1 model (Chapter 5) looks at the centre square and names what it sees.",
                 "params": [{"key": "model", "label": "Model file", "kind": "select", "default": "", "dynamic": "classifiers"},
                            {"key": "threshold", "label": "Minimum confidence", "kind": "number", "default": 0.6, "min": 0.1, "max": 0.99, "step": 0.05}]},
    "pose": {"cat": "Vision AI", "label": "Pose Detector", "icon": "🧍",
             "help": "Finds people and their body points (head, shoulders, wrists, hips, knees). Use it with Pose "
                     "Condition for hand-raise counting, fall detection and posture projects.",
             "params": [{"key": "model", "label": "Model", "kind": "select", "default": "yolo26n-pose.pt", "dynamic": "pose"},
                        {"key": "conf", "label": "Confidence (0-1)", "kind": "number", "default": 0.4, "min": 0.05, "max": 0.95, "step": 0.05},
                        {"key": "track", "label": "Tracking (person IDs)", "kind": "bool", "default": False}]},
    "color": {"cat": "Vision AI", "label": "Colour Detector", "icon": "🌈",
              "help": "Finds objects by colour (no AI training needed). Each coloured object becomes an 'object' "
                      "named after its colour (red, green, ...), so Zone, Line Counter and Count Condition work with it. "
                      "Classic for sorting conveyors and fruit ripeness. Works best on a plain background with even light.",
              "params": [{"key": "colors", "label": "Colours to find", "kind": "multi", "default": ["red", "green", "blue"],
                          "options": ["red", "orange", "yellow", "green", "blue", "purple", "black", "white"]},
                         {"key": "min_area", "label": "Ignore blobs smaller than (% of image)", "kind": "number", "default": 0.5, "min": 0.01, "max": 50, "step": 0.1}]},
    "motion": {"cat": "Vision AI", "label": "Motion Detector", "icon": "👋",
               "help": "Finds anything that moves (compares each frame with the background). Very fast. Each moving "
                       "area becomes an object called 'motion'. Good first stage of a security camera.",
               "params": [{"key": "min_area", "label": "Ignore movement smaller than (% of image)", "kind": "number", "default": 1, "min": 0.05, "max": 50, "step": 0.1},
                          {"key": "sensitivity", "label": "Sensitivity", "kind": "select", "default": "medium", "options": ["low", "medium", "high"]}]},
    "measure": {"cat": "Vision AI", "label": "Measure Size", "icon": "📏",
                "help": "Measures objects on a plain background (top-down camera) in millimetres and marks each one "
                        "'ok' or 'reject' using your tolerance. Calibrate: measure how many mm the camera view is wide. "
                        "Use Count Condition with 'reject' to trigger.",
                "params": [{"key": "view_mm", "label": "Width of the camera view (mm)", "kind": "number", "default": 300, "min": 10, "max": 10000, "step": 1},
                           {"key": "min_mm", "label": "OK if length is at least (mm)", "kind": "number", "default": 40, "min": 0, "max": 10000, "step": 0.5},
                           {"key": "max_mm", "label": "OK if length is at most (mm)", "kind": "number", "default": 60, "min": 0, "max": 10000, "step": 0.5},
                           {"key": "background", "label": "Background", "kind": "select", "default": "light (objects darker)",
                            "options": ["light (objects darker)", "dark (objects lighter)"]},
                           {"key": "min_area", "label": "Ignore blobs smaller than (% of image)", "kind": "number", "default": 0.3, "min": 0.01, "max": 50, "step": 0.1}]},
    # ---------------- LOGIC ----------------
    "keep": {"cat": "Logic", "label": "Keep Objects", "icon": "🧹",
             "help": "Keeps only the chosen objects and removes everything else.",
             "params": [{"key": "classes", "label": "Objects to keep", "kind": "classes", "default": ["person"]}]},
    "zone": {"cat": "Logic", "label": "Zone", "icon": "⬛",
             "help": "Keeps only objects standing inside an area you draw on the camera image.",
             "params": [{"key": "points", "label": "Zone shape", "kind": "zone",
                         "default": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.9], [0.3, 0.9]]},
                        {"key": "name", "label": "Zone name", "kind": "text", "default": "Zone A"}]},
    "line": {"cat": "Logic", "label": "Line Counter", "icon": "↔️",
             "help": "Counts objects crossing a line (IN / OUT). Needs tracking (switched on automatically).",
             "params": [{"key": "direction", "label": "Line", "kind": "select", "default": "vertical",
                         "options": ["vertical", "horizontal"]},
                        {"key": "position", "label": "Position (0-1)", "kind": "number", "default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05},
                        {"key": "per_class", "label": "Also count each object type (car, bus, ...)", "kind": "bool", "default": False}]},
    "count_if": {"cat": "Logic", "label": "Count Condition", "icon": "🔢",
                 "help": "Triggers the actions after it when the number of objects matches, e.g. 'person >= 3'.",
                 "params": [{"key": "object", "label": "Object (empty = any)", "kind": "classes", "default": []},
                            {"key": "op", "label": "Rule", "kind": "select", "default": ">=", "options": [">=", ">", "==", "<=", "<"]},
                            {"key": "value", "label": "Number", "kind": "number", "default": 1, "min": 0, "max": 100, "step": 1},
                            {"key": "hold", "label": "Must stay true for (seconds)", "kind": "number", "default": 0.5, "min": 0, "max": 30, "step": 0.5}]},
    "label_if": {"cat": "Logic", "label": "Classifier Condition", "icon": "🏷️",
                 "help": "Triggers when My Classifier sees a chosen label.",
                 "params": [{"key": "label", "label": "Label", "kind": "text", "default": ""},
                            {"key": "hold", "label": "Must stay true for (seconds)", "kind": "number", "default": 0.5, "min": 0, "max": 30, "step": 0.5}]},
    "pose_if": {"cat": "Logic", "label": "Pose Condition", "icon": "🙋",
                "help": "Triggers when enough people are in a pose: hand raised, both hands up, or lying down (fall).",
                "params": [{"key": "pose", "label": "Pose", "kind": "select", "default": "hand raised",
                            "options": ["hand raised", "both hands up", "lying down (fall)"]},
                           {"key": "value", "label": "At least this many people", "kind": "number", "default": 1, "min": 1, "max": 100, "step": 1},
                           {"key": "hold", "label": "Must stay true for (seconds)", "kind": "number", "default": 1, "min": 0, "max": 60, "step": 0.5}]},
    "dwell": {"cat": "Logic", "label": "Time in Area", "icon": "⏳",
              "help": "Times how long each object stays (put it after a Zone). Triggers when someone stays longer than "
                      "the limit: queue waiting time, loitering, parking overstay. Turns on tracking automatically.",
              "params": [{"key": "limit", "label": "Trigger after (seconds)", "kind": "number", "default": 60, "min": 1, "max": 86400, "step": 1}]},
    "occupancy": {"cat": "Logic", "label": "Occupancy", "icon": "🅿️",
                  "help": "Compares the number of objects with a capacity and shows Occupied / Free: parking bays, "
                          "lab seats, classroom capacity. Can trigger when full or when empty.",
                  "params": [{"key": "object", "label": "Object (empty = any)", "kind": "classes", "default": []},
                             {"key": "capacity", "label": "Capacity", "kind": "number", "default": 10, "min": 1, "max": 1000, "step": 1},
                             {"key": "when", "label": "Trigger", "kind": "select", "default": "when full",
                              "options": ["when full", "when empty", "never (just show)"]},
                             {"key": "hold", "label": "Must stay true for (seconds)", "kind": "number", "default": 2, "min": 0, "max": 3600, "step": 1}]},
    "speed": {"cat": "Logic", "label": "Speed Check", "icon": "🚗",
              "help": "Measures speed between two lines a known distance apart (measure it on the road). Triggers when "
                      "an object is faster than the limit. Turns on tracking automatically.",
              "params": [{"key": "direction", "label": "Objects move", "kind": "select", "default": "up / down",
                          "options": ["up / down", "left / right"]},
                         {"key": "line1", "label": "Line 1 position (0-1)", "kind": "number", "default": 0.35, "min": 0.02, "max": 0.98, "step": 0.01},
                         {"key": "line2", "label": "Line 2 position (0-1)", "kind": "number", "default": 0.75, "min": 0.02, "max": 0.98, "step": 0.01},
                         {"key": "metres", "label": "Real distance between the lines (m)", "kind": "number", "default": 10, "min": 0.5, "max": 500, "step": 0.5},
                         {"key": "limit", "label": "Speed limit (km/h)", "kind": "number", "default": 30, "min": 1, "max": 300, "step": 1}]},
    "schedule": {"cat": "Logic", "label": "Time Schedule", "icon": "🕒",
                 "help": "Only lets the trigger through between two clock times, e.g. 18:00 to 07:00 for an "
                         "after-hours intruder alarm.",
                 "params": [{"key": "start", "label": "Active from (HH:MM)", "kind": "text", "default": "18:00"},
                            {"key": "end", "label": "Active until (HH:MM)", "kind": "text", "default": "07:00"}]},
    "cooldown": {"cat": "Logic", "label": "Cooldown", "icon": "⏱️",
                 "help": "While the condition stays true, lets it through at most once every N seconds (reminders, not spam).",
                 "params": [{"key": "seconds", "label": "Seconds", "kind": "number", "default": 10, "min": 1, "max": 3600, "step": 1}]},
    # ---------------- EFFECTS ----------------
    "blur": {"cat": "Effects", "label": "Privacy Blur", "icon": "🌫️",
             "help": "Blurs the chosen detected objects (e.g. people) for privacy.",
             "params": [{"key": "classes", "label": "Blur these", "kind": "classes", "default": ["person"]}]},
    "filter": {"cat": "Effects", "label": "Image Filter", "icon": "🎨",
               "help": "Changes the look of the video.",
               "params": [{"key": "type", "label": "Filter", "kind": "select", "default": "edges",
                           "options": ["edges", "grayscale", "blur", "cartoon", "night vision"]}]},
    "heatmap": {"cat": "Effects", "label": "Heatmap", "icon": "🔥",
                "help": "Colours the places where objects stand most often: customer footfall in a shop, busy spots "
                        "in a workshop or corridor.",
                "params": [{"key": "fade", "label": "Forget old positions", "kind": "select", "default": "slowly",
                            "options": ["never", "slowly", "quickly"]},
                           {"key": "opacity", "label": "Opacity (0-1)", "kind": "number", "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.1}]},
    # ---------------- OUTPUTS ----------------
    "view": {"cat": "Output", "label": "Live View", "icon": "🖥️",
             "help": "Shows the video with boxes, zones and counts in the Live tab.",
             "params": [{"key": "title", "label": "Title", "kind": "text", "default": "Live"},
                        {"key": "labels", "label": "Show labels", "kind": "bool", "default": True}]},
    "alert": {"cat": "Output", "label": "Screen Alert", "icon": "🚨",
              "help": "When triggered: red banner on the video, pop-up and sound in the browser. "
                      "Placeholders: {object} {count} {label} {reason}.",
              "params": [{"key": "message", "label": "Message", "kind": "text", "default": "Alert!"},
                         {"key": "sound", "label": "Play sound", "kind": "bool", "default": True}]},
    "snapshot": {"cat": "Output", "label": "Save Snapshot", "icon": "📸",
                 "help": "When triggered: saves a photo to the Snapshots tab.",
                 "params": [{"key": "every", "label": "At most one every (seconds)", "kind": "number", "default": 5, "min": 1, "max": 3600, "step": 1}]},
    "log": {"cat": "Output", "label": "Event Log (CSV)", "icon": "📝",
            "help": "Writes rows to a CSV file you can open in Excel: on every trigger, or every N seconds.",
            "params": [{"key": "mode", "label": "Write a row", "kind": "select", "default": "on trigger",
                        "options": ["on trigger", "every N seconds"]},
                       {"key": "seconds", "label": "N seconds", "kind": "number", "default": 10, "min": 1, "max": 3600, "step": 1},
                       {"key": "file", "label": "File name", "kind": "text", "default": "events.csv"}]},
    "record": {"cat": "Output", "label": "Record Clip", "icon": "🎬",
               "help": "When triggered: saves a short video clip (including a few seconds BEFORE the trigger) to the "
                       "Snapshots tab. Evidence for security and safety projects.",
               "params": [{"key": "seconds", "label": "Clip length after the trigger (s)", "kind": "number", "default": 8, "min": 2, "max": 120, "step": 1},
                          {"key": "before", "label": "Seconds before the trigger", "kind": "number", "default": 3, "min": 0, "max": 10, "step": 1}]},
    "webhook": {"cat": "Output", "label": "Webhook", "icon": "🔗",
                "help": "Sends all counts and metrics as JSON to a URL (Node-RED, your own server...).",
                "params": [{"key": "url", "label": "URL", "kind": "text", "default": "http://"},
                           {"key": "mode", "label": "Send", "kind": "select", "default": "on trigger", "options": ["on trigger", "every N seconds"]},
                           {"key": "every", "label": "At most one every (seconds)", "kind": "number", "default": 5, "min": 1, "max": 3600, "step": 1}]},
    "sheets": {"cat": "Output", "label": "Google Sheets", "icon": "📊",
               "help": "Adds a row to a Google Sheet (attendance, traffic survey, footfall...). One-time setup: paste "
                       "extras/google_sheets.gs into the sheet's Apps Script and deploy it as a web app; paste its URL here. "
                       "Needs internet.",
               "params": [{"key": "url", "label": "Apps Script web app URL", "kind": "text", "default": ""},
                          {"key": "mode", "label": "Add a row", "kind": "select", "default": "every N seconds", "options": ["on trigger", "every N seconds"]},
                          {"key": "every", "label": "N seconds (minimum gap)", "kind": "number", "default": 60, "min": 5, "max": 86400, "step": 1}]},
    "mqtt": {"cat": "Output", "label": "MQTT (IoT)", "icon": "📡",
             "help": "Publishes JSON to an MQTT broker for IoT dashboards (Node-RED, Home Assistant, MQTT phone apps, "
                     "other devices). Use a broker in your lab (e.g. Mosquitto) or a public test broker.",
             "params": [{"key": "host", "label": "Broker address", "kind": "text", "default": "broker.hivemq.com"},
                        {"key": "port", "label": "Port", "kind": "number", "default": 1883, "min": 1, "max": 65535, "step": 1},
                        {"key": "topic", "label": "Topic", "kind": "text", "default": "polimas/jetson1/ai"},
                        {"key": "message", "label": "Message (empty = all counts as JSON)", "kind": "text", "default": ""},
                        {"key": "mode", "label": "Send", "kind": "select", "default": "every N seconds", "options": ["on trigger", "every N seconds"]},
                        {"key": "every", "label": "N seconds (minimum gap)", "kind": "number", "default": 5, "min": 1, "max": 3600, "step": 1}]},
    "telegram": {"cat": "Output", "label": "Telegram Message", "icon": "✈️",
                 "help": "When triggered: sends a message with a photo to Telegram (needs internet, a bot token and a "
                         "chat ID). Placeholders: {object} {count} {label} {reason}.",
                 "params": [{"key": "token", "label": "Bot token", "kind": "text", "default": ""},
                            {"key": "chat", "label": "Chat ID", "kind": "text", "default": ""},
                            {"key": "message", "label": "Message", "kind": "text", "default": "Alert from Jetson!"},
                            {"key": "every", "label": "At most one every (seconds)", "kind": "number", "default": 30, "min": 5, "max": 3600, "step": 1}]},
}
CATEGORY_ORDER = ["Source", "Vision AI", "Logic", "Effects", "Output"]
CATEGORY_COLOR = {"Source": "#2f80ed", "Vision AI": "#76b900", "Logic": "#f2994a", "Effects": "#9b51e0", "Output": "#eb5757"}
PRODUCERS = {"detect", "pose", "color", "motion", "measure"}            # blocks that create "objects"
NEEDS_DETECTIONS = {"keep", "zone", "line", "count_if", "blur", "dwell", "occupancy", "speed", "heatmap"}
NEEDS_TRACKING = {"line", "dwell", "speed"}
SOURCES_HELP = "Camera, Phone / CCTV Camera or Video File"
# names that blocks other than YOLO can produce (offered when typing object names)
EXTRA_NAMES = ["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "motion", "ok", "reject"]


def defaults(node_type):
    return {p["key"]: p["default"] for p in CATALOG[node_type]["params"]}


def _hhmm(text):
    try:
        h, m = (int(v) for v in str(text).strip().split(":"))
        return h * 60 + m if 0 <= h < 24 and 0 <= m < 60 else None
    except ValueError:
        return None


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------
def validate(wf):
    """Returns (ordered_node_ids, parents, problems). problems = [(node_id or None, message)]."""
    nodes = {n["id"]: n for n in wf.get("nodes", [])}
    problems, parents, children = [], {}, {nid: [] for nid in nodes}
    for a, b in wf.get("edges", []):
        if a in nodes and b in nodes:
            if b in parents:
                problems.append((b, "has more than one input. Each block takes ONE input."))
            parents[b] = a
            children[a].append(b)
    for nid, n in nodes.items():
        if n["type"] not in CATALOG:
            problems.append((nid, f"unknown block type '{n['type']}'"))
    sources = [nid for nid, n in nodes.items() if CATALOG.get(n["type"], {}).get("inputs", 1) == 0]
    if not nodes:
        return [], {}, [(None, "The workflow is empty. Drag a Camera block onto the canvas to start.")]
    if len(sources) != 1:
        problems.append((None, f"Use exactly ONE source block ({SOURCES_HELP})." if sources else
                               f"Add a source block ({SOURCES_HELP}) to start the workflow."))
        return [], parents, problems
    # walk from the source (breadth first) -> run order
    order, seen = [], set()
    queue = deque(sources)
    while queue:
        nid = queue.popleft()
        if nid in seen:
            problems.append((nid, "is part of a loop. Workflows must flow in one direction."))
            continue
        seen.add(nid); order.append(nid)
        queue.extend(children[nid])
    for nid in nodes:
        if nid not in seen:
            problems.append((nid, "is not connected to the source. Connect its input or delete it."))
    if not any(nodes[n]["type"] == "view" for n in order):
        problems.append((None, "Add a Live View block so you can see the result."))
    for nid in order:
        t = nodes[nid]["type"]
        p = {**defaults(t), **nodes[nid].get("params", {})}
        up, q = [], parents.get(nid)
        while q:
            up.append(nodes[q]["type"]); q = parents.get(q)
        if t in NEEDS_DETECTIONS and not PRODUCERS.intersection(up):
            problems.append((nid, "needs an Object Detector (or Pose / Colour / Motion / Measure) before it."))
        if t == "label_if" and "classify" not in up:
            problems.append((nid, "needs My Classifier before it."))
        if t == "pose_if" and "pose" not in up:
            problems.append((nid, "needs a Pose Detector before it."))
        if t == "classify" and not p.get("model"):
            problems.append((nid, "choose your Project 1 model file (train one with p1_train.py first)."))
        if t == "video" and not p.get("file"):
            problems.append((nid, "choose a video (put .mp4 files in workflow_studio/data/videos)."))
        if t == "ipcam" and not str(p.get("url", "")).strip().lower().startswith(("http://", "https://", "rtsp://")):
            problems.append((nid, "enter the camera address, starting with http:// or rtsp://"))
        if t == "telegram" and not (p.get("token") and p.get("chat")):
            problems.append((nid, "fill in the bot token and chat ID."))
        url = str(p.get("url", "")).strip().lower()
        if t in ("webhook", "sheets") and (not url.startswith(("http://", "https://")) or url in ("http://", "https://")):
            problems.append((nid, "enter the URL (starting with https://)."))
        if t == "mqtt" and not (str(p.get("host", "")).strip() and str(p.get("topic", "")).strip()):
            problems.append((nid, "fill in the broker address and topic."))
        if t == "measure" and float(p.get("min_mm", 0)) > float(p.get("max_mm", 0)):
            problems.append((nid, "the minimum length is bigger than the maximum."))
        if t == "speed" and abs(float(p.get("line1", 0)) - float(p.get("line2", 0))) < 0.1:
            problems.append((nid, "put the two lines further apart (at least 0.1)."))
        if t == "schedule" and not all(_hhmm(p.get(k, "")) is not None for k in ("start", "end")):
            problems.append((nid, "write the times as HH:MM, e.g. 18:00."))
    return order, parents, problems


# ----------------------------------------------------------------------------------------------
# The data that flows through the blocks, one per frame
# ----------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, frame, t):
        self.frame, self.t = frame, t
        self.dets = []          # list of {"name", "conf", "box": [x1,y1,x2,y2], "id"}
        self.zones = []         # [(points_px, name, active)]
        self.lines = []         # [((x1,y1),(x2,y2), flash)]
        self.texts = []         # extra overlay lines
        self.trigger = False
        self.reason = ""
        self.label, self.label_conf = None, 0.0
        self.banner = None

    def copy(self):
        c = Ctx(self.frame, self.t)
        c.__dict__.update({k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()})
        return c

    def counts(self):
        out = {}
        for d in self.dets:
            out[d["name"]] = out.get(d["name"], 0) + 1
        return out


# ----------------------------------------------------------------------------------------------
# Shared resources (models are loaded once and reused between runs)
# ----------------------------------------------------------------------------------------------
_MODEL_CACHE = {}


def yolo_path(name):
    for folder in (MODELS, os.path.join(WORKSHOP, "object_detection"), HERE):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    return name                      # Ultralytics downloads official models by name


def get_yolo(name, task="detect"):
    key = ("yolo", name)
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO
        _MODEL_CACHE[key] = YOLO(yolo_path(name), task=task)
    return _MODEL_CACHE[key]


def get_classifier(path):
    key = ("cls", path)
    if key not in _MODEL_CACHE:
        sys.path.insert(0, os.path.join(WORKSHOP, "cnn_activity"))
        from common import load_checkpoint, get_device, preprocess, center_square   # noqa
        dev = get_device()
        model, ckpt = load_checkpoint(path, dev)
        _MODEL_CACHE[key] = (model, ckpt["classes"], dev, preprocess, center_square)
    return _MODEL_CACHE[key]


def list_models():
    found = set()
    for folder in (os.path.join(WORKSHOP, "object_detection"), HERE):
        if os.path.isdir(folder):
            found.update(f for f in os.listdir(folder) if f.endswith((".pt", ".engine")) and f.startswith("yolo"))
    custom = {f for f in os.listdir(MODELS) if f.endswith((".pt", ".engine"))}       # any name in models/
    yolo = {"yolo26n.pt"} | {f for f in found | custom if "pose" not in f and "-cls" not in f and "-seg" not in f}
    pose = {"yolo26n-pose.pt"} | {f for f in found | custom if "pose" in f}
    cls_dir = os.path.join(WORKSHOP, "cnn_activity")
    cls = sorted(f for f in os.listdir(cls_dir) if f.startswith("p1_classifier") and f.endswith(".pth")) \
        if os.path.isdir(cls_dir) else []
    vids = sorted(f for f in os.listdir(os.path.join(DATA, "videos")) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
    return {"yolo": sorted(yolo), "pose": sorted(pose), "classifiers": cls, "videos": vids}


# ----------------------------------------------------------------------------------------------
# Block implementations
# ----------------------------------------------------------------------------------------------
class _Fmt(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def fill(text, ctx, rt=None):
    """Replaces {object} {count} {label} {reason} {time} and any live metric, e.g. {IN}, in a message."""
    counts = ctx.counts()
    top = max(counts, key=counts.get) if counts else ""
    vals = _Fmt({**({k.replace(" ", "_"): v for k, v in rt.metrics.items()} if rt else {}),
                 "object": top, "count": len(ctx.dets), "label": ctx.label or "", "reason": ctx.reason,
                 "time": datetime.now().strftime("%H:%M:%S")})
    try:
        return str(text).format_map(vals)
    except (ValueError, IndexError):
        return str(text)


class CentroidTracker:
    """Gives IDs to objects from the Colour / Motion / Measure blocks (YOLO has its own tracker)."""
    def __init__(self, max_jump=0.12, max_missing=0.8):
        self.tracks, self.next_id, self.max_jump, self.max_missing = {}, 1, max_jump, max_missing

    def update(self, dets, t, w):
        free = dict(self.tracks)
        for d in dets:
            cx, cy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
            best, bd = None, self.max_jump * w
            for tid, (x, y, name, _) in free.items():
                dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
                if name == d["name"] and dist < bd:
                    best, bd = tid, dist
            if best is None:
                best, self.next_id = self.next_id, self.next_id + 1
            else:
                free.pop(best)
            d["id"] = best
            self.tracks[best] = (cx, cy, d["name"], t)
        self.tracks = {k: v for k, v in self.tracks.items() if t - v[3] <= self.max_missing}
        return dets


class Block:
    def __init__(self, nid, params, rt):
        self.id, self.p, self.rt = nid, {**{}, **params}, rt
        self.info = ""                       # short status shown on the block in the UI

    def process(self, ctx):
        return ctx

    def close(self):
        pass


class Detect(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.model = get_yolo(self.p.get("model") or "yolo26n.pt")
        names = {v.lower(): k for k, v in self.model.names.items()}
        self.classes = [names[c.lower()] for c in self.p.get("classes") or [] if c.lower() in names] or None
        self.track = bool(self.p.get("track")) or rt.needs_tracking

    def process(self, ctx):
        t0 = time.time()
        kw = dict(conf=float(self.p.get("conf", 0.4)), classes=self.classes, verbose=False)
        r = (self.model.track(ctx.frame, persist=True, tracker="bytetrack.yaml", **kw) if self.track
             else self.model.predict(ctx.frame, **kw))[0]
        b = r.boxes
        ids = b.id.int().tolist() if (b.id is not None) else [None] * len(b)
        ctx.dets = [{"name": r.names[int(c)], "conf": float(cf), "box": [float(v) for v in xy], "id": i}
                    for xy, c, cf, i in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist(), ids)]
        self.info = f"{len(ctx.dets)} objects · {(time.time() - t0) * 1000:.0f} ms"
        return ctx


class Classify(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        path = os.path.join(WORKSHOP, "cnn_activity", self.p["model"])
        self.model, self.classes, self.dev, self.prep, self.square = get_classifier(path)

    def process(self, ctx):
        import torch
        sq, _ = self.square(ctx.frame)
        with torch.no_grad():
            probs = torch.softmax(self.model(self.prep(sq).to(self.dev)), 1)[0].cpu().tolist()
        i = int(np.argmax(probs))
        ok = probs[i] >= float(self.p.get("threshold", 0.6))
        ctx.label, ctx.label_conf = (self.classes[i] if ok else "not sure"), probs[i]
        ctx.texts.append(f"{ctx.label} ({probs[i]:.0%})")
        self.info = f"{ctx.label} {probs[i]:.0%}"
        return ctx


class Keep(Block):
    def process(self, ctx):
        keep = {c.lower() for c in self.p.get("classes") or []}
        ctx.dets = [d for d in ctx.dets if d["name"].lower() in keep] if keep else ctx.dets
        self.info = f"{len(ctx.dets)} kept"
        return ctx


class Zone(Block):
    def process(self, ctx):
        h, w = ctx.frame.shape[:2]
        pts = np.array([[x * w, y * h] for x, y in self.p.get("points") or []], np.int32)
        if len(pts) < 3:
            return ctx
        inside = [d for d in ctx.dets
                  if cv2.pointPolygonTest(pts, ((d["box"][0] + d["box"][2]) / 2, d["box"][3]), False) >= 0]
        ctx.dets = inside
        ctx.zones.append((pts, self.p.get("name", "Zone"), bool(inside)))
        self.info = f"{len(inside)} inside"
        return ctx


class Line(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.side, self.n_in, self.n_out, self.flash = {}, 0, 0, 0.0
        self.per_class = {}

    def process(self, ctx):
        h, w = ctx.frame.shape[:2]
        pos, vert = float(self.p.get("position", 0.5)), self.p.get("direction") != "horizontal"
        crossed = False
        for d in ctx.dets:
            if d["id"] is None:
                continue
            c = (d["box"][0] + d["box"][2]) / 2 if vert else (d["box"][1] + d["box"][3]) / 2
            side = c > pos * (w if vert else h)
            if d["id"] in self.side and side != self.side[d["id"]]:
                crossed = True
                if self.p.get("per_class"):
                    self.per_class[d["name"]] = self.per_class.get(d["name"], 0) + 1
                    self.rt.metrics[d["name"]] = self.per_class[d["name"]]
                if side:
                    self.n_in += 1; self.rt.event("IN", f"{d['name']} #{d['id']}", self.id)
                else:
                    self.n_out += 1; self.rt.event("OUT", f"{d['name']} #{d['id']}", self.id)
            self.side[d["id"]] = side
        if crossed:
            self.flash = ctx.t; ctx.trigger, ctx.reason = True, "line crossed"
        p1, p2 = ((int(pos * w), 0), (int(pos * w), h)) if vert else ((0, int(pos * h)), (w, int(pos * h)))
        ctx.lines.append((p1, p2, ctx.t - self.flash < 0.4))
        ctx.texts.append(f"IN {self.n_in}   OUT {self.n_out}")
        self.rt.metrics["IN"], self.rt.metrics["OUT"] = self.n_in, self.n_out
        self.info = f"in {self.n_in} · out {self.n_out}"
        return ctx


class CountIf(Block):
    OPS = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "==": lambda a, b: a == b,
           "<=": lambda a, b: a <= b, "<": lambda a, b: a < b}

    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.since = None

    def process(self, ctx):
        objs = [o.lower() for o in self.p.get("object") or []]
        n = sum(1 for d in ctx.dets if not objs or d["name"].lower() in objs)
        cond = self.OPS.get(self.p.get("op", ">="), self.OPS[">="])(n, float(self.p.get("value", 1)))
        self.since = (self.since or ctx.t) if cond else None
        ok = cond and ctx.t - self.since >= float(self.p.get("hold", 0))
        what = "/".join(objs) or "objects"
        ctx.trigger, ctx.reason = ok, (f"{what} = {n} ({self.p.get('op')} {self.p.get('value')})" if ok else "")
        self.info = f"{what}: {n}" + ("  ✔ TRIGGER" if ok else "")
        return ctx


class LabelIf(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.since = None

    def process(self, ctx):
        cond = (ctx.label or "").lower() == str(self.p.get("label", "")).strip().lower()
        self.since = (self.since or ctx.t) if cond else None
        ok = cond and ctx.t - self.since >= float(self.p.get("hold", 0))
        ctx.trigger, ctx.reason = ok, (f"saw {ctx.label}" if ok else "")
        self.info = "✔ TRIGGER" if ok else f"waiting for '{self.p.get('label')}'"
        return ctx


class Cooldown(Block):
    """While the trigger is on, lets it through at most once every N seconds (a reminder, not spam)."""
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.last = -1e9

    def process(self, ctx):
        gap = float(self.p.get("seconds", 10))
        if ctx.trigger and ctx.t - self.last >= gap:
            self.last = ctx.t                      # pass this one (one frame) ...
        else:
            ctx.trigger = False                    # ... and hold back the rest until the gap is over
        left = gap - (ctx.t - self.last)
        self.info = f"next in {left:.0f}s" if left > 0 else "ready"
        return ctx


class Blur(Block):
    def process(self, ctx):
        names = {c.lower() for c in self.p.get("classes") or []}
        f = ctx.frame.copy()
        n = 0
        for d in ctx.dets:
            if not names or d["name"].lower() in names:
                x1, y1, x2, y2 = [int(max(0, v)) for v in d["box"]]
                if x2 > x1 and y2 > y1:
                    roi = f[y1:y2, x1:x2]
                    f[y1:y2, x1:x2] = cv2.resize(cv2.resize(roi, (8, 8)), (x2 - x1, y2 - y1),
                                                 interpolation=cv2.INTER_NEAREST)   # pixelate
                    n += 1
        ctx.frame = f
        self.info = f"{n} blurred"
        return ctx


class Filter(Block):
    def process(self, ctx):
        t, f = self.p.get("type", "edges"), ctx.frame
        if t == "edges":
            e = cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 80, 160)
            f = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
        elif t == "grayscale":
            f = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif t == "blur":
            f = cv2.GaussianBlur(f, (21, 21), 0)
        elif t == "cartoon":
            edges = cv2.adaptiveThreshold(cv2.medianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 7), 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            f = cv2.bitwise_and(cv2.bilateralFilter(f, 9, 75, 75), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        elif t == "night vision":
            g = cv2.equalizeHist(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            f = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
        ctx.frame = f
        self.info = t
        return ctx


class Alert(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.was, self.on_until = False, 0.0

    def process(self, ctx):
        msg = fill(self.p.get("message", "Alert!"), ctx, self.rt)
        if ctx.trigger and not self.was:
            self.rt.alert(msg, bool(self.p.get("sound", True)), ctx.reason)
            self.on_until = ctx.t + 2.0
            self.msg = msg
        self.was = ctx.trigger
        if ctx.trigger or ctx.t < self.on_until:
            ctx.banner = msg if ctx.trigger else getattr(self, "msg", msg)
        self.info = "ALERT" if ctx.banner else "quiet"
        return ctx


class Snapshot(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.last, self.n = -1e9, 0

    def process(self, ctx):
        if ctx.trigger and ctx.t - self.last >= float(self.p.get("every", 5)):
            self.last = ctx.t
            name = datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(DATA, "snapshots", name), render(ctx, True))
            self.n += 1
            self.rt.event("SNAPSHOT", name, self.id)
        self.info = f"{self.n} saved"
        return ctx


class Log(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        fname = os.path.basename(self.p.get("file") or "events.csv")
        self.path = os.path.join(DATA, "logs", fname if fname.endswith(".csv") else fname + ".csv")
        self.last, self.was, self.rows = -1e9, False, 0
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["time", "workflow", "reason", "total_objects", "counts", "label", "metrics"])

    def write(self, ctx):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.rt.name, ctx.reason or "periodic",
                                    len(ctx.dets), json.dumps(ctx.counts()), ctx.label or "",
                                    json.dumps(dict(self.rt.metrics))])
        self.rows += 1

    def process(self, ctx):
        if self.p.get("mode") == "every N seconds":
            if ctx.t - self.last >= float(self.p.get("seconds", 10)):
                self.last = ctx.t; self.write(ctx)
        elif ctx.trigger and not self.was:
            self.write(ctx)
        self.was = ctx.trigger
        self.info = f"{self.rows} rows"
        return ctx


class Sender(Block):
    """Base for blocks that send something over the network in the background (never blocks the video)."""
    needs_image = False
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.last, self.sent, self.err = -1e9, 0, ""

    def process(self, ctx):
        due = ctx.t - self.last >= float(self.p.get("every", 30))
        want = True if self.p.get("mode") == "every N seconds" else ctx.trigger
        if want and due:
            self.last = ctx.t
            if not ctx.reason and self.p.get("mode") == "every N seconds":
                ctx.reason = "periodic"
            img = render(ctx, True) if self.needs_image else None
            threading.Thread(target=self._safe_send, args=(ctx.copy(), img), daemon=True).start()
        self.info = f"{self.sent} sent" + (f" · {self.err}" if self.err else "")
        return ctx

    def payload(self, ctx):
        return {"workflow": self.rt.name, "time": datetime.now().isoformat(timespec="seconds"),
                "reason": ctx.reason, "total": len(ctx.dets), "counts": ctx.counts(), "label": ctx.label,
                "metrics": dict(self.rt.metrics)}

    def _safe_send(self, ctx, img):
        try:
            self.send(ctx, img); self.sent += 1; self.err = ""
            self.rt.event("SENT", CATALOG[self.kind]["label"], self.id)
        except Exception as e:
            self.err = str(e)[:60]
            self.rt.event("SEND FAILED", self.err, self.id)


class Webhook(Sender):
    kind = "webhook"

    def send(self, ctx, img):
        body = json.dumps(self.payload(ctx)).encode()
        req = urllib.request.Request(self.p["url"].strip(), data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=8).read()


class Sheets(Webhook):
    """Google Apps Script web app (extras/google_sheets.gs) appends one row per message."""
    kind = "sheets"

    def payload(self, ctx):
        d = super().payload(ctx)
        d["counts"], d["metrics"] = json.dumps(d["counts"]), json.dumps(d["metrics"])    # one cell each
        return d


def mqtt_publish(host, port, topic, message, client_id="jetson-studio", timeout=5):
    """Minimal MQTT 3.1.1 publish (QoS 0) using only the standard library - no extra packages to install."""
    import socket

    def s(text):
        b = text.encode()
        return len(b).to_bytes(2, "big") + b

    def packet(kind, body):
        n, rl = len(body), b""
        while True:                                   # "remaining length" encoding
            byte, n = n % 128, n // 128
            rl += bytes([byte | (128 if n else 0)])
            if not n:
                return bytes([kind]) + rl + body

    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.sendall(packet(0x10, s("MQTT") + bytes([4, 2]) + (30).to_bytes(2, "big") + s(client_id)))
        ack = sock.recv(4)
        if len(ack) < 4 or ack[0] != 0x20 or ack[3] != 0:
            raise RuntimeError("broker refused the connection")
        sock.sendall(packet(0x30, s(topic) + message.encode()))
        sock.sendall(bytes([0xE0, 0]))                # disconnect


class Mqtt(Sender):
    kind = "mqtt"

    def send(self, ctx, img):
        msg = fill(self.p["message"], ctx, self.rt) if str(self.p.get("message", "")).strip() \
            else json.dumps(self.payload(ctx))
        mqtt_publish(self.p["host"].strip(), self.p.get("port", 1883), self.p["topic"].strip(), msg,
                     client_id=f"jetson-{os.getpid()}-{self.id}")


class Telegram(Sender):
    kind = "telegram"
    needs_image = True

    def send(self, ctx, img):
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        boundary = "----jetson" + str(int(time.time()))
        caption = f"{fill(self.p.get('message', ''), ctx, self.rt)}\n{ctx.reason}"
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="chat_id"\r\n\r\n{self.p["chat"]}\r\n',
                 f'--{boundary}\r\nContent-Disposition: form-data; name="caption"\r\n\r\n{caption}\r\n',
                 f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="snap.jpg"\r\n'
                 f'Content-Type: image/jpeg\r\n\r\n']
        body = "".join(parts).encode() + jpg.tobytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{self.p['token']}/sendPhoto", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        urllib.request.urlopen(req, timeout=10).read()


# ---------------- extra Vision AI blocks ----------------
class Pose(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.model = get_yolo(self.p.get("model") or "yolo26n-pose.pt", task="pose")
        self.track = bool(self.p.get("track")) or rt.needs_tracking

    def process(self, ctx):
        t0 = time.time()
        kw = dict(conf=float(self.p.get("conf", 0.4)), verbose=False)
        r = (self.model.track(ctx.frame, persist=True, tracker="bytetrack.yaml", **kw) if self.track
             else self.model.predict(ctx.frame, **kw))[0]
        b = r.boxes
        ids = b.id.int().tolist() if (b.id is not None) else [None] * len(b)
        kp = r.keypoints.data.tolist() if r.keypoints is not None else [[] for _ in range(len(b))]
        ctx.dets = [{"name": "person", "conf": float(cf), "box": [float(v) for v in xy], "id": i, "kpts": k}
                    for xy, cf, i, k in zip(b.xyxy.tolist(), b.conf.tolist(), ids, kp)]
        self.info = f"{len(ctx.dets)} people · {(time.time() - t0) * 1000:.0f} ms"
        return ctx


# HSV ranges (OpenCV hue is 0-179)
COLOR_RANGES = {
    "red": [((0, 100, 70), (9, 255, 255)), ((170, 100, 70), (179, 255, 255))],
    "orange": [((10, 100, 80), (21, 255, 255))],
    "yellow": [((22, 90, 80), (34, 255, 255))],
    "green": [((35, 70, 50), (85, 255, 255))],
    "blue": [((86, 80, 50), (130, 255, 255))],
    "purple": [((131, 60, 50), (169, 255, 255))],
    "black": [((0, 0, 0), (179, 255, 50))],
    "white": [((0, 0, 200), (179, 40, 255))],
}


def blobs(mask, min_frac, name):
    """Contours of a binary mask -> objects."""
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) >= min_frac * w * h:
            x, y, bw, bh = cv2.boundingRect(c)
            out.append({"name": name, "conf": 1.0, "box": [x, y, x + bw, y + bh], "id": None, "contour": c})
    return out


class Color(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.tracker = CentroidTracker() if rt.needs_tracking else None
        self.kernel = np.ones((5, 5), np.uint8)

    def process(self, ctx):
        hsv = cv2.cvtColor(cv2.GaussianBlur(ctx.frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        found = []
        for name in self.p.get("colors") or []:
            mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in COLOR_RANGES.get(name, []))
            if isinstance(mask, int):
                continue
            mask = cv2.morphologyEx(np.clip(mask, 0, 255).astype(np.uint8), cv2.MORPH_OPEN, self.kernel)
            found += blobs(mask, float(self.p.get("min_area", 0.5)) / 100, name)
        for d in found:
            d.pop("contour", None)
        if self.tracker:
            self.tracker.update(found, ctx.t, ctx.frame.shape[1])
        ctx.dets = found
        if found:                      # the biggest coloured object also becomes the "label"
            big = max(found, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            ctx.label, ctx.label_conf = big["name"], 1.0
        c = ctx.counts()
        self.info = ", ".join(f"{k} {v}" for k, v in c.items()) or "no colour found"
        return ctx


class Motion(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        thr = {"low": 50, "medium": 25, "high": 12}.get(self.p.get("sensitivity"), 25)
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=thr, detectShadows=False)
        self.tracker = CentroidTracker(max_jump=0.2) if rt.needs_tracking else None
        self.n = 0

    def process(self, ctx):
        small = cv2.resize(ctx.frame, (320, int(320 * ctx.frame.shape[0] / ctx.frame.shape[1])))
        mask = self.bg.apply(small)
        self.n += 1
        if self.n < 15:                                # learn the background first
            self.info = "learning background..."
            ctx.dets = []
            return ctx
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.ones((9, 9), np.uint8))
        k = ctx.frame.shape[1] / 320
        found = blobs(mask, float(self.p.get("min_area", 1)) / 100, "motion")
        for d in found:
            d.pop("contour", None)
            d["box"] = [v * k for v in d["box"]]
        if self.tracker:
            self.tracker.update(found, ctx.t, ctx.frame.shape[1])
        ctx.dets = found
        pct = 100.0 * np.count_nonzero(mask) / mask.size
        self.rt.metrics["Motion %"] = round(float(pct), 1)
        self.info = f"{len(found)} moving areas · {pct:.1f}%"
        return ctx


class Measure(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.tracker = CentroidTracker() if rt.needs_tracking else None
        self.n_ok = self.n_rej = 0

    def process(self, ctx):
        h, w = ctx.frame.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        inv = cv2.THRESH_BINARY_INV if str(self.p.get("background", "")).startswith("light") else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, inv + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mm_px = float(self.p.get("view_mm", 300)) / w
        lo, hi = float(self.p.get("min_mm", 0)), float(self.p.get("max_mm", 1e9))
        found = []
        for d in blobs(mask, float(self.p.get("min_area", 0.3)) / 100, "part"):
            x1, y1, x2, y2 = d["box"]
            if x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1:
                continue                                       # touching the edge: not fully in view
            (_, _), (a, b), _ = cv2.minAreaRect(d.pop("contour"))
            length, width = max(a, b) * mm_px, min(a, b) * mm_px
            d["name"] = "ok" if lo <= length <= hi else "reject"
            d["tag"] = f"{length:.1f} x {width:.1f} mm"
            found.append(d)
        if self.tracker:
            self.tracker.update(found, ctx.t, w)
        ctx.dets = found
        c = ctx.counts()
        self.info = f"ok {c.get('ok', 0)} · reject {c.get('reject', 0)}"
        if found:
            ctx.texts.append("Size: " + ", ".join(d["tag"] for d in found[:3]))
        return ctx


# ---------------- extra Logic blocks ----------------
def _pt(k, i, min_conf=0.4):
    if k and len(k) > i and (len(k[i]) < 3 or k[i][2] >= min_conf):
        return k[i][0], k[i][1]
    return None


def pose_of(d):
    """Simple rules on the COCO body points: 0 nose, 5/6 shoulders, 9/10 wrists, 11/12 hips."""
    k = d.get("kpts") or []
    nose, ls, rs, lw, rw, lh, rh = (_pt(k, i) for i in (0, 5, 6, 9, 10, 11, 12))
    x1, y1, x2, y2 = d["box"]
    bh = max(1.0, y2 - y1)

    def up(wrist, shoulder):
        if not wrist:
            return False
        ref = nose[1] if nose else (shoulder[1] - 0.1 * bh if shoulder else None)
        return ref is not None and wrist[1] < ref

    left, right = up(lw, ls), up(rw, rs)
    lying = (x2 - x1) > 1.3 * bh
    if ls and rs and lh and rh:                            # torso direction is more reliable than the box
        sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
        hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
        lying = abs(hx - sx) > abs(hy - sy)
    return {"hand raised": left or right, "both hands up": left and right, "lying down (fall)": lying}


class PoseIf(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.since = None

    def process(self, ctx):
        want = self.p.get("pose", "hand raised")
        n = 0
        for d in ctx.dets:
            if d.get("kpts") is not None and pose_of(d).get(want):
                d["tag"] = want.split(" (")[0]
                n += 1
        cond = n >= int(self.p.get("value", 1))
        self.since = (self.since or ctx.t) if cond else None
        ok = cond and ctx.t - self.since >= float(self.p.get("hold", 1))
        ctx.trigger, ctx.reason = ok, (f"{n} people: {want}" if ok else "")
        self.rt.metrics[want.split(" (")[0].capitalize()] = n
        self.info = f"{want}: {n}" + ("  ✔ TRIGGER" if ok else "")
        return ctx


class Dwell(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.first, self.last, self.visits = {}, {}, []

    def process(self, ctx):
        limit = float(self.p.get("limit", 60))
        longest, over = 0.0, []
        for d in ctx.dets:
            if d["id"] is None:
                continue
            key = (d["name"], d["id"])
            self.first.setdefault(key, ctx.t)
            self.last[key] = ctx.t
            stay = ctx.t - self.first[key]
            d["tag"] = f"{stay:.0f}s"
            longest = max(longest, stay)
            if stay >= limit:
                over.append(f"{d['name']} #{d['id']}")
        for key in [k for k, v in self.last.items() if ctx.t - v > 2.0]:     # gone for 2 s = visit finished
            stay = self.last[key] - self.first[key]
            if stay >= 1:
                self.visits.append(stay)
                self.rt.event("VISIT", f"{key[0]} #{key[1]} stayed {stay:.0f}s", self.id)
            self.first.pop(key); self.last.pop(key)
        ctx.trigger, ctx.reason = bool(over), (f"{', '.join(over[:3])} over {limit:.0f}s" if over else "")
        self.rt.metrics["Here now"] = sum(1 for d in ctx.dets if d["id"] is not None)
        self.rt.metrics["Longest now (s)"] = int(longest)
        if self.visits:
            self.rt.metrics["Avg stay (s)"] = int(sum(self.visits) / len(self.visits))
        self.info = f"longest {longest:.0f}s" + ("  ✔ TRIGGER" if over else "")
        return ctx


class Occupancy(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.since = None

    def process(self, ctx):
        objs = [o.lower() for o in self.p.get("object") or []]
        n = sum(1 for d in ctx.dets if not objs or d["name"].lower() in objs)
        cap = int(self.p.get("capacity", 10))
        when = self.p.get("when", "when full")
        cond = (n >= cap) if when == "when full" else (n == 0) if when == "when empty" else False
        self.since = (self.since or ctx.t) if cond else None
        ok = cond and ctx.t - self.since >= float(self.p.get("hold", 2))
        ctx.trigger, ctx.reason = ok, ((f"full ({n}/{cap})" if n >= cap else "empty") if ok else "")
        self.rt.metrics["Occupied"], self.rt.metrics["Free"] = n, max(0, cap - n)
        ctx.texts.append(f"Occupied {n}/{cap}   Free {max(0, cap - n)}" + ("   FULL" if n >= cap else ""))
        self.info = f"{n}/{cap}" + ("  ✔ TRIGGER" if ok else "")
        return ctx


class Speed(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.side, self.t1, self.shown, self.speeds = {}, {}, {}, []
        self.until = -1.0

    def process(self, ctx):
        h, w = ctx.frame.shape[:2]
        vert = self.p.get("direction", "up / down") == "up / down"         # moving up/down -> horizontal lines
        size = h if vert else w
        a, b = sorted([float(self.p.get("line1", 0.35)) * size, float(self.p.get("line2", 0.75)) * size])
        limit, metres = float(self.p.get("limit", 30)), float(self.p.get("metres", 10))
        fired = []
        for d in ctx.dets:
            if d["id"] is None:
                continue
            c = (d["box"][1] + d["box"][3]) / 2 if vert else (d["box"][0] + d["box"][2]) / 2
            zone = 0 if c < a else (1 if c <= b else 2)                     # before / between / after the lines
            key = (d["name"], d["id"])
            prev = self.side.get(key)
            if prev is not None and prev != zone:
                if zone == 1:                                               # entered the measuring strip
                    self.t1[key] = (ctx.t, prev)
                elif key in self.t1 and self.t1[key][1] != zone:            # left on the other side
                    dt = ctx.t - self.t1.pop(key)[0]
                    if dt > 0.05:
                        kmh = metres / dt * 3.6
                        self.speeds.append(kmh)
                        self.shown[key] = (kmh, ctx.t)
                        fast = kmh > limit
                        self.rt.event("SPEED", f"{d['name']} #{d['id']}: {kmh:.0f} km/h" + ("  OVER LIMIT" if fast else ""), self.id)
                        if fast:
                            fired.append(f"{d['name']} #{d['id']} {kmh:.0f} km/h")
                            self.rt.metrics["Over limit"] = self.rt.metrics.get("Over limit", 0) + 1
            self.side[key] = zone
            if key in self.shown and ctx.t - self.shown[key][1] < 3:
                d["tag"] = f"{self.shown[key][0]:.0f} km/h"
        if fired:
            self.until, self.why = ctx.t + 1.0, ", ".join(fired)
        ok = ctx.t < self.until
        ctx.trigger, ctx.reason = ok, (f"speeding: {self.why}" if ok else "")
        for pos in (a, b):
            p1, p2 = ((0, int(pos)), (w, int(pos))) if vert else ((int(pos), 0), (int(pos), h))
            ctx.lines.append((p1, p2, ok))
        if self.speeds:
            self.rt.metrics["Avg km/h"] = round(sum(self.speeds) / len(self.speeds), 1)
            self.rt.metrics["Measured"] = len(self.speeds)
        self.info = (f"last {self.speeds[-1]:.0f} km/h" if self.speeds else "waiting for traffic") + ("  ✔ TRIGGER" if ok else "")
        return ctx


class Schedule(Block):
    def process(self, ctx):
        now = datetime.now()
        m, a, b = now.hour * 60 + now.minute, _hhmm(self.p.get("start")), _hhmm(self.p.get("end"))
        active = (a <= m < b) if a <= b else (m >= a or m < b)               # e.g. 18:00 -> 07:00 crosses midnight
        if not active:
            ctx.trigger, ctx.reason = False, ""
        self.info = "active" if active else f"sleeping until {self.p.get('start')}"
        return ctx


# ---------------- extra Effects / Output blocks ----------------
class Heatmap(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.acc = None

    def process(self, ctx):
        h, w = ctx.frame.shape[:2]
        sw, sh = 160, max(1, int(160 * h / w))
        if self.acc is None:
            self.acc = np.zeros((sh, sw), np.float32)
        fade = {"never": 1.0, "slowly": 0.998, "quickly": 0.98}.get(self.p.get("fade"), 0.998)
        self.acc *= fade
        for d in ctx.dets:
            x, y = (d["box"][0] + d["box"][2]) / 2 * sw / w, d["box"][3] * sh / h        # feet point
            cv2.circle(self.acc, (int(x), int(min(sh - 1, y))), 4, 1.0, -1)
        if self.acc.max() > 0:
            norm = cv2.GaussianBlur(self.acc, (0, 0), 3)
            norm = (255 * norm / norm.max()).astype(np.uint8)
            col = cv2.resize(cv2.applyColorMap(norm, cv2.COLORMAP_JET), (w, h))
            alpha = cv2.resize(norm, (w, h)).astype(np.float32)[..., None] / 255 * float(self.p.get("opacity", 0.5))
            ctx.frame = (ctx.frame * (1 - alpha) + col * alpha).astype(np.uint8)
        self.info = "building" if self.acc.max() == 0 else "on"
        return ctx


class Record(Block):
    def __init__(self, nid, params, rt):
        super().__init__(nid, params, rt)
        self.buf = deque()
        self.writer, self.until, self.n, self.name = None, 0.0, 0, ""

    def process(self, ctx):
        img = render(ctx, True)
        before = float(self.p.get("before", 3))
        if self.writer is None:
            self.buf.append((ctx.t, img))
            while self.buf and ctx.t - self.buf[0][0] > before:
                self.buf.popleft()
            if ctx.trigger:
                h, w = img.shape[:2]
                fps = max(5.0, min(30.0, self.rt.fps or 15.0))
                self.name = datetime.now().strftime("clip_%Y%m%d_%H%M%S.mp4")
                self.writer = cv2.VideoWriter(os.path.join(DATA, "clips", self.name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for _, f in self.buf:
                    self.writer.write(f)
                self.buf.clear()
                self.until = ctx.t + float(self.p.get("seconds", 8))
                self.rt.event("RECORDING", self.name, self.id)
        else:
            self.writer.write(img)
            if ctx.t >= self.until:
                self.close()
                self.n += 1
                self.rt.event("CLIP", self.name, self.id)
        self.info = f"recording {self.until - ctx.t:.0f}s" if self.writer else f"{self.n} clips"
        return ctx

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class View(Block):
    def process(self, ctx):
        img = render(ctx, bool(self.p.get("labels", True)))
        self.rt.publish(self.id, img)
        self.info = self.p.get("title", "Live")
        return ctx


BLOCKS = {"detect": Detect, "classify": Classify, "pose": Pose, "color": Color, "motion": Motion, "measure": Measure,
          "keep": Keep, "zone": Zone, "line": Line, "count_if": CountIf, "label_if": LabelIf, "pose_if": PoseIf,
          "dwell": Dwell, "occupancy": Occupancy, "speed": Speed, "schedule": Schedule, "cooldown": Cooldown,
          "blur": Blur, "filter": Filter, "heatmap": Heatmap, "view": View, "alert": Alert, "snapshot": Snapshot,
          "record": Record, "log": Log, "webhook": Webhook, "sheets": Sheets, "mqtt": Mqtt,
          "telegram": Telegram}

PALETTE = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49), (72, 249, 10),
           (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
           (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255), (82, 0, 133), (203, 56, 255)]


NAMED_COLORS = {"red": (40, 40, 230), "orange": (0, 140, 255), "yellow": (0, 220, 240), "green": (60, 190, 60),
                "blue": (230, 120, 30), "purple": (200, 60, 160), "black": (90, 90, 90), "white": (235, 235, 235),
                "ok": (60, 200, 60), "reject": (40, 40, 230), "motion": (0, 200, 255)}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (0, 5), (0, 6)]


def render(ctx, labels=True):
    """Draws boxes, zones, lines, texts and alert banner on a copy of the frame."""
    img = ctx.frame.copy()
    h, w = img.shape[:2]
    for pts, name, active in ctx.zones:
        over = img.copy()
        cv2.fillPoly(over, [pts], (0, 0, 255) if active else (0, 200, 0))
        img = cv2.addWeighted(over, 0.22, img, 0.78, 0)
        cv2.polylines(img, [pts], True, (0, 0, 255) if active else (0, 255, 0), 2)
        cv2.putText(img, name, tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for p1, p2, flash in ctx.lines:
        cv2.line(img, p1, p2, (0, 255, 0) if flash else (0, 255, 255), 3)
    for d in ctx.dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        col = NAMED_COLORS.get(d["name"])
        if col is None:
            r, g, b = PALETTE[sum(map(ord, d["name"])) % len(PALETTE)]
            col = (b, g, r)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        k = d.get("kpts")
        if k:
            for i, j in SKELETON:
                a, b2 = _pt(k, i, 0.4), _pt(k, j, 0.4)
                if a and b2:
                    cv2.line(img, (int(a[0]), int(a[1])), (int(b2[0]), int(b2[1])), (0, 255, 255), 2)
            for i in range(len(k)):
                q = _pt(k, i, 0.4)
                if q:
                    cv2.circle(img, (int(q[0]), int(q[1])), 3, (0, 0, 255), -1)
        if labels:
            name = d["name"] if d["conf"] >= 0.999 else f"{d['name']} {d['conf']:.2f}"
            txt = (f"#{d['id']} " if d["id"] is not None else "") + name + (f" | {d['tag']}" if d.get("tag") else "")
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), col, -1)
            cv2.putText(img, txt, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0) if sum(col) > 500 else (255, 255, 255), 1)
    lines = ctx.texts[:]
    counts = ctx.counts()
    if counts:
        lines.append(", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:5]))
    if lines:
        bw = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for l in lines) + 20
        cv2.rectangle(img, (0, 0), (bw, 10 + 26 * len(lines)), (0, 0, 0), -1)
        for i, l in enumerate(lines):
            cv2.putText(img, l, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if ctx.banner:
        cv2.rectangle(img, (0, h - 60), (w, h), (0, 0, 200), -1)
        cv2.putText(img, ctx.banner, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return img


# ----------------------------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------------------------
class LatestFrame:
    """Reads a live camera in a background thread and keeps ONLY the newest frame.
    Without this, OpenCV queues frames and the AI works on old ones: the video lags behind real life."""
    def __init__(self, cap):
        self.cap, self.frame, self.seq, self.taken, self.alive = cap, None, 0, 0, True
        self.cond = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05); continue
            with self.cond:
                self.frame, self.seq = f, self.seq + 1
                self.cond.notify_all()

    def read(self, timeout=5.0):
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > self.taken, timeout):
                return False, None
            self.taken = self.seq
            return True, self.frame

    def get(self, prop):
        return self.cap.get(prop)

    def set(self, prop, value):
        return self.cap.set(prop, value)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.alive = False
        time.sleep(0.1)
        self.cap.release()


def open_source(node):
    p = node["params"]
    if node["type"] == "video":
        cap = cv2.VideoCapture(os.path.join(DATA, "videos", p.get("file", "")))
        if not cap.isOpened():
            raise RuntimeError(f"Can't open video '{p.get('file')}'")
        return cap, bool(p.get("loop", True))
    if node["type"] == "ipcam":
        cap = cv2.VideoCapture(str(p.get("url", "")).strip())
        if not cap.isOpened():
            raise RuntimeError("Can't open the network camera. Check the address, that the phone/CCTV is on the "
                               "same network, and open the address in a browser to test it.")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return LatestFrame(cap), False
    w, h = (int(v) for v in str(p.get("resolution", "1280x720")).split("x"))
    if p.get("csi"):
        pipe = (f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={w}, height={h}, framerate=30/1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 sync=false")
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    else:
        src = int(p.get("source", "0")) if str(p.get("source", "0")).isdigit() else p.get("source")
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if isinstance(src, int) else cv2.VideoCapture(src)
        if not cap.isOpened() and isinstance(src, int):
            cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Can't open the camera. Check the cable, try camera number 1, or tick 'CSI camera'.")
    return LatestFrame(cap), False


# ----------------------------------------------------------------------------------------------
# The runner
# ----------------------------------------------------------------------------------------------
class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread, self.stop_flag = None, threading.Event()
        self.reset()

    def reset(self):
        self.running, self.error, self.name = False, "", ""
        self.fps, self.frames, self.started = 0.0, 0, None
        self.blocks, self.views, self.metrics = {}, {}, {}
        self.events, self.alerts, self.alert_seq = deque(maxlen=200), deque(maxlen=20), 0
        self.last_raw = None
        self.needs_tracking = False
        self.deployed, self.view_titles = False, {}

    # called by blocks
    def event(self, kind, detail, nid=""):
        self.events.append({"t": datetime.now().strftime("%H:%M:%S"), "kind": kind, "detail": detail, "node": nid})

    def alert(self, msg, sound, reason):
        self.alert_seq += 1
        self.alerts.append({"seq": self.alert_seq, "msg": msg, "sound": sound, "reason": reason})
        self.event("ALERT", f"{msg} ({reason})" if reason else msg)

    STREAM_WIDTH = 960          # the browser/phone stream is downsized: full HD over Wi-Fi causes lag

    def publish(self, nid, img):
        h, w = img.shape[:2]
        if w > self.STREAM_WIDTH:
            img = cv2.resize(img, (self.STREAM_WIDTH, int(h * self.STREAM_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.lock:
                self.views[nid] = jpg.tobytes()

    # control
    def start(self, wf, deployed=False):
        order, parents, problems = validate(wf)          # check first, so a bad workflow never stops a good one
        if problems:
            return problems
        self.stop()
        self.reset()
        self.name = wf.get("name", "workflow")
        self.deployed = deployed
        self.view_titles = {n["id"]: (n.get("params") or {}).get("title") or "Live"
                            for n in wf.get("nodes", []) if n.get("type") == "view"}
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, args=(wf, order, parents), daemon=True)
        self.thread.start()
        return []

    def stop(self):
        self.deployed = False
        if self.thread and self.thread.is_alive():
            self.stop_flag.set()
            self.thread.join(timeout=5)
        self.running = False

    def _run(self, wf, order, parents):
        nodes = {n["id"]: n for n in wf["nodes"]}
        cap = None
        try:
            self.running, self.started = True, time.time()
            self.event("START", self.name)
            self.needs_tracking = any(nodes[n]["type"] in NEEDS_TRACKING for n in order)
            for nid in order[1:]:
                n = nodes[nid]
                self.blocks[nid] = BLOCKS[n["type"]](nid, {**defaults(n["type"]), **n.get("params", {})}, self)
            cap, loop = open_source(nodes[order[0]])
            is_video = nodes[order[0]]["type"] == "video"
            frame_gap = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25) if is_video else 0   # play videos at real speed
            t_prev = time.time()
            while not self.stop_flag.is_set():
                if is_video:
                    wait = frame_gap - (time.time() - t_prev)
                    if wait > 0:
                        time.sleep(wait)
                ok, frame = cap.read()
                if not ok:
                    if loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                    self.event("END", "video finished"); break
                self.last_raw = frame
                now = time.time()
                out = {order[0]: Ctx(frame, now)}
                for nid in order[1:]:
                    parent_ctx = out[parents[nid]]
                    out[nid] = self.blocks[nid].process(parent_ctx.copy())
                self.frames += 1
                dt = now - t_prev; t_prev = now
                self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-6) if self.frames > 1 else 0.0
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.event("ERROR", self.error)
        finally:
            if cap is not None:
                cap.release()
            for b in self.blocks.values():
                b.close()
            self.running = False
            self.event("STOP", f"{self.frames} frames")

    def status(self, since_alert=0):
        return {"running": self.running, "error": self.error, "fps": round(self.fps, 1), "frames": self.frames,
                "name": self.name, "deployed": self.deployed,
                "view_titles": {v: self.view_titles.get(v, "Live") for v in self.views},
                "uptime": int(time.time() - self.started) if self.started and self.running else 0,
                "blocks": {nid: b.info for nid, b in self.blocks.items()},
                "views": list(self.views.keys()), "metrics": dict(self.metrics),
                "events": list(self.events)[-60:],
                "alerts": [a for a in self.alerts if a["seq"] > since_alert]}
