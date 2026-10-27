# NVIDIA Jetson Orin Nano Super: Hands-On AI Workshop (code)

Copy this whole folder to the Jetson as `~/ai_workshop`.

| Folder | Chapter | What's inside |
|--------|---------|---------------|
| `setup/` | 3. Jetson Orin Nano Super | `jetson_check.py`: health check · `env_guard.sh`: lock / check / restore the environment |
| `cuda_examples/` | 4. CUDA | CPU vs GPU races, live GPU camera filters, your first CUDA kernel |
| `cnn_activity/` | 5. Build Your Own CNN | Train your own image classifier (+ transfer learning, + take-home regression project) |
| `object_detection/` | 6. Real-Time Object Detection | YOLO live detection, TensorRT, smart counter / zone alarm, web dashboard, run on boot |
| `workflow_studio/` | Day 2 | **AI Workflow Studio**: drag-and-drop AI projects in the browser (no code, no pip) |

## One-time setup
```bash
cd ~/ai_workshop
python3 setup/jetson_check.py                                   # what's ready, what to fix
sudo nvpmodel -m 2 && sudo jetson_clocks                         # MAXN SUPER: full speed

python3 -m venv --system-site-packages env                      # sees JetPack's OpenCV + TensorRT
source env/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
pip install matplotlib
pip install ultralytics lap flask "numpy<2"     # numpy<2: the Jetson PyTorch build needs NumPy 1.x
pip uninstall -y opencv-python                  # use JetPack's own OpenCV (works with NumPy 1.x + CSI cameras)
pip install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.82" onnxruntime-gpu "numpy<2" \
    --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/   # for the TensorRT export (D2)
python3 -c "import torch; print(torch.cuda.is_available())"     # must print True
```
**Why `torch==2.8.0`?** The newer PyTorch builds for Jetson (2.10, 2.11) also need NVIDIA's cuDSS library, which is not
part of JetPack. Without it you get `ImportError: libcudss.so.0`. Version 2.8.0 works with JetPack 6.x out of the box.
**Why `numpy<2` and uninstall `opencv-python`?** Ultralytics pulls in NumPy 2 and its own OpenCV, but the Jetson PyTorch
build only works with NumPy 1.x (otherwise: `RuntimeError: Numpy is not available`). JetPack's built-in OpenCV works with
NumPy 1.x and also supports CSI cameras.
Each folder has its own README with the details.

**Before the workshop (trainer):** run the setup above on every Jetson, then run each of these once so nothing needs to download on the day:
`python3 object_detection/d1_detect.py --image <any photo>`,
`python3 object_detection/d2_tensorrt.py` and
`python3 -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"`.

## Day 2: AI Workflow Studio
```bash
bash setup/env_guard.sh lock                 # trainer: protect the environment first
bash workflow_studio/start_studio.sh         # then open http://localhost:8080
```
11 templates grouped by task (Counting, Safety & Security, Inspection & Sorting, Time & Speed): people counting,
zone and intruder alarms, fall detection, size QC in mm, colour sorting, queue times and speed checks. Trainer prep (pose model download, test videos): see `workflow_studio/README.md`.
