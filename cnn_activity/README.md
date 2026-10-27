# Chapter 5 Hands-On: Two CNN Projects on the Jetson Orin Nano

Both projects use photos **you take yourself** and follow the same AI workflow:

```
1. COLLECT  ->  2. TRAIN  ->  3. TEST & DEPLOY
```

| | Project 1: Classification | Project 2: Regression |
|---|---|---|
| Question | **WHAT** is in front of the camera? | **WHERE** is my object? |
| Your data | Photos of 2-5 of your own objects | Photos of 1 object, labelled by clicking on it |
| Output | A label (`mug`, `pen`, `phone`, `nothing`) | Two numbers: `x`, `y` (0.0 to 1.0) |
| Loss | Cross-Entropy | MSE (mean squared error) |
| Score | Accuracy (% correct) | Position error (% of image width) |
| Deploy | Live recognition + event log | Live tracker + target game + zone log |
| Transfer learning | `--model resnet18` (Activity 6) | `--model resnet18 --finetune` (bonus) |
| Scripts | `p1_collect.py`, `p1_train.py`, `p1_deploy.py` | `p2_collect.py`, `p2_train.py`, `p2_deploy.py` |

`common.py` holds the CNN and the camera helpers shared by both projects.
Both projects use the **same CNN**. Only the last layer and the loss change.
**Activity 6** retrains with a pre-trained ResNet-18 (transfer learning) and compares the results.

---

## 0. Setup (done in Chapter 4)

The Python environment is the same one used for the Chapter 4 CUDA examples. If it's already done, just run:
```bash
cd ~/ai_workshop && source env/bin/activate && cd cnn_activity
```
Otherwise, set it up first (tested with JetPack 6.2 / CUDA 12.6):
```bash
mkdir ~/ai_workshop && cd ~/ai_workshop
# copy the cuda_examples and cnn_activity folders here
python3 -m venv --system-site-packages env     # can still see the Jetson's built-in OpenCV
source env/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
pip install matplotlib
python3 -c "import torch; print(torch.cuda.is_available())"   # should print True
cd cnn_activity
```
If `import cv2` fails inside the venv: `pip install opencv-python` (USB webcams only).
Use a USB webcam, or a CSI camera by adding `--csi` to any script.

---

# Project 1: Classification ("What is it?")

### Step 1: Collect

Grab **2-5 objects** you have with you. Objects that look clearly different are easiest.

```bash
python3 p1_collect.py
```
The script asks:
```
How many objects? (2-5): 3
  Name of object 1: water bottle
  Name of object 2: phone
  Name of object 3: pen
Add a 'nothing' class (empty background)? Recommended [Y/n]: y
```
The camera then guides you through each object, showing its name, a progress bar (40 photos) and a tip that changes as you go.

| Key | Action |
|-----|--------|
| `SPACE` | Take one photo |
| `A` | Auto mode: 3-2-1, then a photo every 0.4 s (keep moving the object!) |
| `N` / `B` | Next / previous object |
| `Q` | Finish. Saves `p1_dataset_preview.jpg` so you can check your photos |

**Phone photos:** copy them into `p1_dataset/<object_name>/` (JPG/PNG; on iPhone set
*Settings > Camera > Formats > Most Compatible*). Portrait and landscape both work.

### Step 2: Train
```bash
python3 p1_train.py              # 15 epochs
python3 p1_train.py --epochs 25
```
Prints training and validation accuracy for each epoch. At the end it prints a **confusion matrix**, showing which objects get mixed up, and the accuracy for each object. Saves `p1_classifier.pth`, `p1_training_curves.png` and `p1_confusion_matrix.png`.

### Step 3: Test & deploy
```bash
python3 p1_deploy.py                      # live camera
python3 p1_deploy.py --image photo.jpg    # photos
```
Hold each object in the green square. The screen shows the prediction and a confidence bar for every object.
The **terminal prints an event log** every time a different object appears. This is how a deployed app
would react to predictions (open a gate, count items, sort objects...).

---

# Project 2: Regression ("Where is it?")

### Step 1: Collect & label
Pick **one** object you can move around easily.

```bash
python3 p2_collect.py
```
**One click on the object = take the photo + save its (x, y) position as the label.**

| Action | What it does |
|--------|--------------|
| Left click on the object | Save photo + position |
| `U` | Undo the last photo |
| `Q` | Finish. Saves `p2_dataset_preview.jpg` with your labels drawn on |

- Move the object to a **new spot before every click**.
- Watch the **3x3 coverage grid**: get every square green, including the corners and edges.
- Aim for **80+ labelled photos**.

Labels are saved in `p2_dataset/labels.csv` as numbers from 0 to 1: `(0, 0)` = top-left, `(1, 1)` = bottom-right.

**Phone photos:** `python3 p2_collect.py --label my_photos/` shows each photo; click on the object (`S` = skip).

### Step 2: Train
```bash
python3 p2_train.py              # 30 epochs
```
Shows the MSE loss, the **position error** (% of image width) and the share of predictions **within 10%**.
Data augmentation shifts and flips the images *and moves the labels with them*.
Saves `p2_tracker.pth` and `p2_training_curves.png`.

### Step 3: Test & deploy
```bash
python3 p2_deploy.py                      # live tracker
python3 p2_deploy.py --image photo.jpg    # photos (marked copies saved in p2_predictions/)
```
A crosshair follows your object and leaves a trail, with the x, y values and zone shown on screen.
The terminal logs every zone change, the way a robot might steer left or right.
Press **`G`** for the **target game**: hold your object on each red target. How many can you hit in 45 s?

---

# Activity 6: Transfer Learning

Instead of starting from random weights, start from a **ResNet-18 that already learned to see**
from 1.2 million ImageNet photos, and teach it only what is new: your objects.
Use the **same Project 1 photos** and compare three ways of training:

```bash
python3 p1_train.py                              # 1. our CNN, from scratch
python3 p1_train.py --model resnet18             # 2. transfer: pre-trained layers frozen, train only the new last layer
python3 p1_train.py --model resnet18 --finetune  # 3. transfer: fine-tune every layer (small learning rate)
```
Every run is added to **`p1_results.csv`**, and a comparison table is printed after each run:
```
model              trainable_params  epochs  best_val_acc  epoch_reached_90%  train_time
cnn                914,308           15      ...
resnet18           2,052             15      ...
resnet18_finetune  11,178,564        15      ...
```
(Trainable weights shown for 4 objects. Your numbers depend on how many objects you have.)

Deploy any of them:
```bash
python3 p1_deploy.py --model p1_classifier_resnet18.pth
```

Discuss:
- Why can ~2,000 trainable weights beat ~914,000?
- Which model reached 90% in the fewest epochs?
- **Experiment:** `--model resnet18 --no-pretrained` uses the same network with *random* weights. Without the ImageNet knowledge, how much worse is it?
- **Bonus:** transfer learning works for the tracker too: `python3 p2_train.py --model resnet18 --finetune`, then `python3 p2_deploy.py --model p2_tracker_resnet18_finetune.pth`.

**Trainer tip (no Wi-Fi at the venue?)** The first ResNet-18 run downloads a 45 MB weights file.
Download it on each Jetson before the workshop:
```bash
python3 -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"
```
It is cached in `~/.cache/torch/hub/checkpoints/` and works offline afterwards.

---

## The CNN (`common.py`)
```
Input 3x128x128
 -> 4 x [Conv 3x3 + BatchNorm + ReLU + MaxPool]   (32, 64, 128, 256 filters)
 -> AvgPool to 4x4 -> Flatten (4096)
 -> Fully connected 128 + ReLU + Dropout
 -> Project 1: N outputs (one per object), Softmax gives the probabilities
    Project 2: 2 outputs (x, y), Sigmoid keeps them between 0 and 1
```

## Troubleshooting
| Problem | Fix |
|---------|-----|
| `Could not open the camera` | `ls /dev/video*` and try `--cam 1`. For a CSI camera use `--csi` |
| `CUDA not available` | You installed CPU-only PyTorch. Reinstall from the Jetson AI Lab index (step 0) |
| NumPy version error when importing torch | `pip install "numpy<2"` |
| Window doesn't open over SSH | Run on the Jetson desktop, or use `ssh -X` |
| Out of memory while training | Add `--batch 8` |
| P1 always predicts the same object | Unbalanced photo counts or too few photos. Take more |
| P2 crosshair stuck in the middle | Too few labels or poor coverage. Label more photos in the red grid areas |
| Phone photos not picked up | Use .jpg/.png (not HEIC) |

## Challenges
1. **P1: Add another object.** Photograph a new object and retrain. Does the accuracy drop?
2. **P1: Look-alikes.** Try two similar objects. Can the AI tell them apart? Does transfer learning do better?
3. **P2: Fewer labels.** Train with 30 labels, then 100. How does the position error change?
4. **Both: New environment.** Test in another room or under different lighting. Why does it get better or worse?
