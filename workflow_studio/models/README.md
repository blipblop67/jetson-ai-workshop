# Your own detection models

Put custom YOLO models here (`.pt`, or a TensorRT `.engine` built on this Jetson). They appear in the
**Object Detector** block's *Model* list automatically (reload the page). Pose models need "pose" in the file name.

Typical projects that need a custom model: helmet / safety vest (PPE) check, fire and smoke, plant disease spots,
PCB or weld defects, specific products on a shelf.

## Train one (after the workshop, takes 30 min to a few hours)
1. Collect and label photos (e.g. Roboflow, Label Studio or CVAT) and export in **YOLO** format.
2. On the Jetson (no install needed: training uses the packages already in the environment):
   ```bash
   source ~/ai_workshop/env/bin/activate
   yolo detect train data=/path/to/data.yaml model=yolo26n.pt epochs=50 imgsz=640 batch=8
   ```
3. Copy `runs/detect/train/weights/best.pt` here, e.g. as `ppe_helmet.pt`.
4. In the Object Detector block choose `ppe_helmet.pt` and type its class names (e.g. `helmet`, `no-helmet`)
   in Count Condition / Keep Objects.
