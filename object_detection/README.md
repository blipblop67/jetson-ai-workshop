# Chapter 6: Real-Time Object Detection & Deploying AI Apps

From "AI that sees" to "AI that does something": detect objects live with **YOLO**,
speed it up with **TensorRT**, turn it into **practical apps**, and **deploy** them so they run on their own.

| Step | Run | What you get |
|------|-----|--------------|
| D1 | `python3 d1_detect.py` | Live detection of 80 everyday objects: boxes, names, confidence, FPS |
| D2 | `python3 d2_tensorrt.py` | Builds a TensorRT engine and races it against PyTorch |
| D3 | `python3 d3_smart_alert.py --mode count` | Counts objects crossing a line, logged to `events.csv` |
| D3 | `python3 d3_smart_alert.py --mode zone` | Zone alarm with snapshots in `alerts/` and a log |
| D4 | `python3 d4_web_dashboard.py` | Live AI camera you can watch on any phone (+ a JSON API) |
| D5 | `bash install_service.sh` | Starts the dashboard automatically every time the Jetson boots |

## Setup (once)
```bash
cd ~/ai_workshop && source env/bin/activate      # the venv from Chapter 4
pip install ultralytics lap flask "numpy<2"     # numpy<2: the Jetson PyTorch build needs NumPy 1.x
pip uninstall -y opencv-python                  # use JetPack's own OpenCV (works with NumPy 1.x + CSI cameras)
pip install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.82" onnxruntime-gpu "numpy<2" \
    --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/   # for the TensorRT export (D2)
python3 -c "import torch; print(torch.cuda.is_available())"   # must still print True
cd object_detection
```
- If the last line prints `False`, Ultralytics replaced the Jetson PyTorch. Reinstall it:
  `pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/`
- **`No module named 'onnx'` / `No matching distribution found for onnxruntime-gpu` in D2?** TensorRT export goes
  PyTorch → ONNX → TensorRT. The Jetson build of `onnxruntime-gpu` is only on the Jetson AI Lab index, so install the
  ONNX tools with the `--extra-index-url` command above (Ultralytics' automatic install can't find it).
- **`RuntimeError: Numpy is not available`?** Something installed NumPy 2. Fix: `pip install "numpy<2"` and
  `pip uninstall -y opencv-python`. The Jetson PyTorch build and JetPack's OpenCV both need NumPy 1.x.
- **Trainer tip (no Wi-Fi?):** the first run downloads `yolo26n.pt` (about 6 MB). Run `python3 d1_detect.py --image any.jpg`
  once on each Jetson before the workshop. TensorRT engines are built *per device*, so run `d2_tensorrt.py` on each Jetson too.
- Fastest settings: `sudo nvpmodel -m 2` (MAXN SUPER) and `sudo jetson_clocks`.

## D1: Live detection
```bash
python3 d1_detect.py                          # everything
python3 d1_detect.py --classes person cup     # only some classes
python3 d1_detect.py --image photo.jpg        # photos -> detections/
python3 d1_detect.py --source video.mp4       # a video file
```
Keys: `+` / `-` change the confidence threshold, `S` takes a screenshot, `Q` quits.
YOLO is pre-trained on **COCO**: 80 everyday classes, for example person, bicycle, car, bus, truck, bottle, cup,
chair, laptop, cell phone, book, dog, cat, backpack and umbrella.

## D2: TensorRT
```bash
python3 d2_tensorrt.py                        # build yolo26n.engine (a few minutes, once) + race
python3 d1_detect.py --model yolo26n.engine   # use the fast version everywhere
```
PyTorch model → optimised for **this** GPU (FP16, fused layers) → `.engine` file. This is how Jetson products ship models.

## D3: Smart counter & zone alarm
Uses **tracking**, so every object keeps the same ID number from frame to frame.
```bash
python3 d3_smart_alert.py --mode count                       # people crossing the middle line
python3 d3_smart_alert.py --mode count --classes bottle cup --line 0.4 --horizontal
python3 d3_smart_alert.py --mode zone                        # alarm when a person stands in the zone
```
- **COUNT:** `IN` = crossing left→right (or top→bottom with `--horizontal`), `OUT` = the other way.
- **ZONE:** click 4 points on the video to draw your own zone, or press `R` to reset it.
  The alarm checks where the object **stands** (the bottom-centre of its box).
- Everything is logged to **`events.csv`** (open it in Excel). Zone alarms also save **snapshots** in `alerts/`.

## D4: Web dashboard
```bash
python3 d4_web_dashboard.py --classes person cup
```
Open the printed address (for example `http://192.168.1.23:5000`) on a phone on the same Wi-Fi.
`/stats` returns the counts as JSON, so other programs (a spreadsheet, Node-RED, a mobile app) can use the results.

## D5: Run on boot
```bash
bash install_service.sh           # install + start the dashboard as a service
journalctl -u ai_camera -f        # watch its log
bash install_service.sh remove    # uninstall
```

## Take it further
- **Detect your own objects:** label photos (for example with Roboflow or Label Studio), then
  `yolo detect train data=your_data.yaml model=yolo26n.pt epochs=50`, and use the new `best.pt` in any script above.
- **Send alerts:** add a Telegram, LINE or email message, or an MQTT publish, inside the `ALARM` part of `d3_smart_alert.py`.
- **Save results:** write `/stats` to a database every minute to build a footfall or usage report.
