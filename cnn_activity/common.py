"""
common.py - Shared pieces used by all three activity scripts.

  * SimpleCNN       : the convolutional neural network we train
  * get_device()    : picks the Jetson GPU (CUDA) if available, otherwise CPU
  * open_camera()   : opens a USB webcam or a CSI camera (Raspberry Pi / IMX219)
  * preprocess()    : turns an OpenCV frame into a tensor the CNN understands

You do not need to run this file directly.
"""

import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as TF

# Image size the CNN works with (pixels). Small = fast training on the Jetson.
IMG_SIZE = 128

# Standard ImageNet colour normalisation - keeps pixel values in a nice range.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# The CNN
# --------------------------------------------------------------------------
def conv_block(in_ch, out_ch):
    """Conv -> BatchNorm -> ReLU -> MaxPool.  Each block halves the image size."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # learn 3x3 filters
        nn.BatchNorm2d(out_ch),                              # stabilise training
        nn.ReLU(inplace=True),                               # non-linearity
        nn.MaxPool2d(2),                                     # downsample by 2
    )


class SimpleCNN(nn.Module):
    """
    A small CNN for image classification.

    Input : 3 x 128 x 128 colour image
    Block1: 32 filters  -> 32 x 64 x 64   (edges, colours)
    Block2: 64 filters  -> 64 x 32 x 32   (corners, textures)
    Block3: 128 filters -> 128 x 16 x 16  (parts, shapes)
    Block4: 256 filters -> 256 x 8 x 8    (whole-object patterns)
    Average pool to 4x4 -> 256 x 4 x 4 = 4096 numbers (Flatten)
    Fully connected     -> 128 neurons
    Fully connected     -> one score per class
    """

    def __init__(self, num_classes, regression=False):
        """num_classes = number of outputs.
        regression=True adds a Sigmoid so every output stays between 0 and 1
        (used in Project 2, where the outputs are the x and y position)."""
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)      # shrink feature maps to 4x4
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # 256 x 4 x 4 -> 4096 numbers
            nn.Linear(256 * 4 * 4, 128),         # fully connected layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                     # randomly switch off neurons -> less overfitting
            nn.Linear(128, num_classes),         # one output score per class
        )
        if regression:
            self.classifier.append(nn.Sigmoid())  # squash outputs into 0..1

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[INFO] CUDA not available - using CPU (training will be slower)")
    return torch.device("cpu")


def csi_pipeline(width=1280, height=720, fps=30, flip=0):
    """GStreamer pipeline for a CSI camera (e.g. Raspberry Pi Camera v2 / IMX219)."""
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def open_camera(cam_index=0, use_csi=False, width=640, height=480):
    """Open the camera and return an OpenCV VideoCapture object."""
    if use_csi:
        cap = cv2.VideoCapture(csi_pipeline(), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():                       # fall back to OpenCV default backend
            cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise SystemExit(
            "[ERROR] Could not open the camera.\n"
            "  - USB webcam: check it is plugged in, try --cam 1 (list with: ls /dev/video*)\n"
            "  - CSI camera: add --csi (needs OpenCV built with GStreamer)"
        )
    return cap


def center_square(frame):
    """Crop the largest centred square from a frame (what the CNN 'sees')."""
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s], (x0, y0, s)


def clean_name(name):
    """'My Water Bottle' -> 'my_water_bottle' (safe to use as a folder name)."""
    out = "".join(c if c.isalnum() else "_" for c in name.strip().lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def load_image(path):
    """Open any photo (Jetson camera OR phone) the right way up, as RGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)        # phones store rotation in EXIF
    return img.convert("RGB")


class SquareCrop:
    """Crop the largest centred square, the same view the camera scripts use.
    Makes phone photos (portrait / landscape) match the Jetson camera images."""
    def __call__(self, img):
        w, h = img.size
        s = min(w, h)
        return TF.center_crop(img, [s, s])


# Transform used at test time (no random augmentation)
eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def preprocess(frame_bgr):
    """OpenCV BGR frame -> normalised 1x3xHxW tensor."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return eval_transform(rgb).unsqueeze(0)


# --------------------------------------------------------------------------
# Transfer learning: a ResNet-18 pre-trained on ImageNet (1.2 million photos)
# --------------------------------------------------------------------------
ARCHS = ["cnn", "resnet18"]


def build_model(arch, num_outputs, regression=False, pretrained=True, freeze=True):
    """
    arch="cnn"      : our SimpleCNN, every weight starts RANDOM (trained from scratch)
    arch="resnet18" : TRANSFER LEARNING - ResNet-18 that already learned edges,
                      textures and shapes from ImageNet. We swap its last layer
                      for a new one sized for OUR task.
        freeze=True : keep the pre-trained layers fixed, train ONLY the new last layer
        freeze=False: fine-tune: train every layer (slower, often a bit better)
    """
    if arch == "cnn":
        return SimpleCNN(num_outputs, regression=regression)
    if arch != "resnet18":
        raise ValueError(f"unknown model '{arch}', choose from {ARCHS}")

    from torchvision import models
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        net = models.resnet18(weights=weights)
    except Exception as e:                       # usually: no internet for the download
        raise SystemExit(
            "[ERROR] Could not download the pre-trained ResNet-18 weights (needs internet once).\n"
            "  Trainer tip: before the workshop run\n"
            "    python3 -c \"import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')\"\n"
            "  It is saved in ~/.cache/torch/hub/checkpoints/ and works offline afterwards.\n"
            f"  ({e})")
    if freeze:
        for p in net.parameters():
            p.requires_grad = False              # pre-trained layers stay as they are
    head = [nn.Linear(net.fc.in_features, num_outputs)]   # NEW last layer (trainable)
    if regression:
        head.append(nn.Sigmoid())
    net.fc = nn.Sequential(*head)
    return net


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint(path, device, regression=False):
    """Rebuild whichever model was trained (cnn or resnet18) and load its weights."""
    ckpt = torch.load(path, map_location=device)
    n_out = 2 if regression else len(ckpt["classes"])
    model = build_model(ckpt.get("arch", "cnn"), n_out, regression=regression, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def set_mode(model, training):
    """model.train()/eval(), but frozen pre-trained layers stay in eval mode so
    their BatchNorm statistics (learned on ImageNet) are not overwritten."""
    model.train(training)
    if training:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
                m.eval()


def run_name(arch, finetune, pretrained=True):
    """Short tag used in file names and the results table."""
    if arch == "cnn":
        return "cnn"
    tag = "resnet18_finetune" if finetune else "resnet18"
    return tag if pretrained else tag + "_nopretrain"


def log_result(csv_path, row):
    """Append one training run to a CSV and print every run so far (for comparing)."""
    import csv, os
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    keys = list(rows[0])
    widths = [max(len(k), *(len(r[k]) for r in rows)) for k in keys]
    print("\n=== All your training runs (" + csv_path + ") ===")
    print("  " + "  ".join(k.ljust(w) for k, w in zip(keys, widths)))
    for r in rows:
        print("  " + "  ".join(r[k].ljust(w) for k, w in zip(keys, widths)))
