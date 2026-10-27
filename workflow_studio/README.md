# AI Workflow Studio: Day 2 "Build Your Own AI Project"

Drag blocks onto a canvas, connect them, press **▶ Run**. Participants build real edge-AI projects on the
Jetson **without writing code and without touching the Python environment**. Workflows are saved as plain
settings (JSON), never code, and every block is pre-built and pre-tested.

```
📷 Camera → 🎯 Object Detector → ⬛ Zone → 🔢 Count Condition → 🚨 Screen Alert → 🖥️ Live View
                                                              ↘ 🎬 Record Clip   ↘ 📊 Google Sheets
```

## Start it
```bash
bash ~/ai_workshop/workflow_studio/start_studio.sh
```
Open **http://localhost:8080** on the Jetson, or **http://\<jetson-ip\>:8080** from a laptop on the same Wi-Fi.
Only one workflow runs at a time per Jetson (it uses the camera).

## Run vs Deploy
* **▶ Run** = test it in the editor (Live tab).
* **🚀 Deploy** = make it the Jetson's "product". It keeps running after you close the editor, and a
  window shows a link + QR code to the **live page** (`http://<jetson-ip>:8080/view`): video, numbers,
  events and alerts (with sound) on any laptop or phone on the same Wi-Fi. The deployed project also resumes
  whenever the studio restarts. Stop it with **Stop deployment** (or ■ Stop in the editor).
* **Start after a reboot / power cut** (trainer, once per Jetson):
  ```bash
  bash ~/ai_workshop/workflow_studio/install_service.sh          # studio runs in the background from boot
  bash ~/ai_workshop/workflow_studio/install_service.sh remove   # undo
  ```
  With the service installed, `start_studio.sh` just prints the addresses (the studio is already running).

## Templates (Templates button), grouped by task

| Task | Templates | What it shows |
|---|---|---|
| 🔢 **Counting** | People Counter · Occupancy Monitor · Hand-Raise Counter | Line IN/OUT counts, Occupied / Free against a capacity (+ MQTT), pose-based counting |
| 🚨 **Safety & Security** | Safety Zone Alarm · After-Hours Intruder Alarm · Fall Detection | Draw a zone, time schedule + video clip, pose fall detection |
| 🔍 **Inspection & Sorting** | Part Size QC · Colour Sorting · My Object Sorter | Measure in mm with a tolerance, sort by colour, use your own Day 1 classifier |
| ⏱️ **Time & Speed** | Queue Wait-Time Monitor · Vehicle Speed Check | Time each person in an area, km/h between two lines |

Open one, change it (draw your zone, set the numbers, add outputs) and **Save** it as your own project.
Every other block (Heatmap, Motion, Privacy Blur, Google Sheets, Telegram ...) can be added to any template.

## The blocks (32)

| Group | Block | What it does |
|---|---|---|
| Source | 📷 Camera | USB or CSI camera |
| | 📱 Phone / CCTV Camera | Phone with an IP-camera app (`http://<phone-ip>:8080/video`) or CCTV RTSP stream |
| | 🎞️ Video File | Plays a video from `data/videos/` (test without a camera) |
| Vision AI | 🎯 Object Detector | YOLO26: 80 everyday objects, or **your own custom model** from `models/` |
| | 🧠 My Classifier | The participant's own Chapter 5 model (`cnn_activity/p1_classifier*.pth`) |
| | 🧍 Pose Detector | People + 17 body points (YOLO26 pose) |
| | 🌈 Colour Detector | Finds red / green / blue / ... objects, no AI training (sorting, ripeness) |
| | 👋 Motion Detector | Anything that moves (fast, no AI) |
| | 📏 Measure Size | Object length/width in **mm** on a plain background, marks `ok` / `reject` by tolerance |
| Logic | 🧹 Keep Objects | Keep only chosen objects |
| | ⬛ Zone | Keep objects inside an area you **draw on the camera image** |
| | ↔️ Line Counter | IN / OUT counting across a line, optionally per object type |
| | 🔢 Count Condition | Trigger when e.g. `person >= 3` for N seconds |
| | 🏷️ Classifier Condition | Trigger when My Classifier sees a label |
| | 🙋 Pose Condition | Hand raised · both hands up · lying down (fall) |
| | ⏳ Time in Area | Time each object stays; trigger over a limit (queues, loitering, overstay) |
| | 🅿️ Occupancy | Occupied / Free against a capacity; trigger when full or empty |
| | 🚗 Speed Check | km/h between two lines a known distance apart |
| | 🕒 Time Schedule | Only active between two clock times (e.g. 18:00 → 07:00) |
| | ⏱️ Cooldown | While the condition stays true, let it through once every N seconds |
| Effects | 🌫️ Privacy Blur | Pixelate people (or any object) |
| | 🎨 Image Filter | Edges, grayscale, blur, cartoon, night vision |
| | 🔥 Heatmap | Where objects stand most often |
| Output | 🖥️ Live View | Video with boxes, zones, counts in the **Live** tab |
| | 🚨 Screen Alert | Red banner + browser pop-up + sound |
| | 📸 Save Snapshot | Photos in the **Media** tab |
| | 🎬 Record Clip | Video clip including seconds *before* the trigger (Media tab) |
| | 📝 Event Log (CSV) | On every trigger or every N seconds, downloadable from the **Events** tab |
| | 📊 Google Sheets | Adds rows to a Google Sheet (setup: `extras/google_sheets.gs`) |
| | 📡 MQTT (IoT) | JSON or a custom message to an MQTT broker (Node-RED, Home Assistant, phone apps) |
| | 🔗 Webhook | POSTs counts as JSON to a URL |
| | ✈️ Telegram Message | Message + photo to Telegram (needs internet, bot token, chat ID) |

**Message placeholders** (Screen Alert, Telegram, MQTT): `{object}` `{count}` `{label}` `{reason}` `{time}`
and any live number, e.g. `{IN}`, `{Free}`, `{Occupied}`, `{Hand_raised}` (spaces become `_`).

**Rules (the Check button explains any problem):** exactly one source, one input per block (an output can
feed many blocks), a Live View somewhere, and an object-producing block (Detector, Pose, Colour, Motion or
Measure) before Zone, Line, Count, Time in Area, Occupancy, Speed, Keep, Blur and Heatmap.
Conditions set a *trigger*; alerts, snapshots, clips, logs and messages after them act only when triggered
(Sheets, MQTT, Webhook and Log can also send "every N seconds").

## Camera & internet notes
* **No extra hardware:** the only device used is the camera (no GPIO, no Arduino / microcontroller boards).
* **Phone camera:** Android "IP Webcam" app → *Start server* → use the address it shows + `/video`.
* **Google Sheets, Telegram and public MQTT brokers need internet.** For an offline lab, run a local broker
  (e.g. Mosquitto on a laptop) and use its IP address.
* **Custom models (PPE helmet/vest, fire, defects ...):** see `models/README.md`.

## Keep the environment safe (trainer)
```bash
bash ~/ai_workshop/setup/env_guard.sh lock      # before Day 2: baseline + read-only venv
bash ~/ai_workshop/setup/env_guard.sh check     # anything changed? + health check
bash ~/ai_workshop/setup/env_guard.sh restore   # put the exact versions back (internet), then lock again
bash ~/ai_workshop/setup/env_guard.sh unlock    # when you need to install something yourself
```
While locked, `pip install` and `pip uninstall` in the workshop environment fail with an error and nothing changes.
It can't stop `sudo pip`, so ask participants not to use `sudo`.
Every new block uses only what is already installed (OpenCV, Ultralytics, Python standard library), so the lock
doesn't need to be opened for any of them.

## Before Day 2 (trainer checklist)
1. Download the pose model once (with internet):
   `cd ~/ai_workshop/object_detection && python3 -c "from ultralytics import YOLO; YOLO('yolo26n-pose.pt')"`
2. Put a few test videos in `data/videos/` (a corridor, a road, a shop) for groups without a good camera view.
3. `bash ~/ai_workshop/setup/env_guard.sh lock`, start the studio, run one template with the real camera.

## Files
```
workflow_studio/
  studio.py          web server        engine.py   blocks, validation, runner
  install_service.sh run the studio (and the deployed project) on boot
  static/            the browser app: editor + view.html live page (Drawflow, QR code bundled, works offline)
  templates/*.json   11 starter workflows, grouped by task      projects/   saved projects
  models/            custom YOLO models (.pt / .engine)
  extras/            google_sheets.gs (Apps Script for the Google Sheets block)
  data/videos/       put .mp4 test videos here
  data/snapshots/    photos     data/clips/  video clips     data/logs/  CSV logs
```
To add a new block: add it to `CATALOG` and write a small class in `engine.py` (and add it to `BLOCKS`).
The palette and the settings forms update themselves.
