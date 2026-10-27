"""
PROJECT 2 (REGRESSION) - STEP 2: TRAIN
======================================
Trains the SAME CNN as Project 1, with two changes that turn it from
classification into REGRESSION:

    Project 1 (classification)          Project 2 (regression)
    output = one score per object       output = 2 numbers: x and y (0..1, Sigmoid)
    loss   = Cross-Entropy              loss   = MSE (mean squared error)
    metric = accuracy (% correct)       metric = position error (% of image width)

Outputs
-------
    p2_tracker.pth           : best model (lowest validation error)
    p2_training_curves.png   : loss & error graph for every epoch

Usage
-----
    python3 p2_train.py                 # 30 epochs
    python3 p2_train.py --epochs 50

Transfer learning
-----------------
    python3 p2_train.py --model resnet18 --finetune    # recommended for regression
    python3 p2_train.py --model resnet18               # frozen: only the last layer learns
Every run is added to p2_results.csv and a comparison table is printed.
"""

import argparse
import csv
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from common import (get_device, IMG_SIZE, MEAN, STD, SquareCrop, load_image,
                    build_model, count_params, set_mode, run_name, log_result, ARCHS)


class PositionDataset(Dataset):
    """Images + (x, y) labels. Augmentation moves the LABEL together with the image."""

    def __init__(self, root, rows, augment):
        self.img_dir = os.path.join(root, "images")
        self.rows, self.augment = rows, augment
        self.square = SquareCrop()
        self.jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        self.norm = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fname, x, y = self.rows[i]
        img = self.square(load_image(os.path.join(self.img_dir, fname)))

        if self.augment:
            # random shift: resize a bit bigger, then crop -> the object moves, so the label moves too
            big = IMG_SIZE + 16
            img = img.resize((big, big))
            ox, oy = random.randint(0, big - IMG_SIZE), random.randint(0, big - IMG_SIZE)
            img = img.crop((ox, oy, ox + IMG_SIZE, oy + IMG_SIZE))
            x, y = (x * big - ox) / IMG_SIZE, (y * big - oy) / IMG_SIZE
            if random.random() < 0.5:                       # mirror left-right -> x flips
                img = TF.hflip(img)
                x = 1.0 - x
            img = self.jitter(img)                          # colours don't change the position
        else:
            img = img.resize((IMG_SIZE, IMG_SIZE))

        target = torch.tensor([x, y], dtype=torch.float32).clamp(0.0, 1.0)
        return self.norm(TF.to_tensor(img)), target


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    set_mode(model, training)
    tot_loss, tot_err, hits, n = 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)                              # (batch, 2) -> x, y
            loss = criterion(preds, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            dist = torch.linalg.norm(preds.detach() - targets, dim=1)   # distance in image widths
            tot_loss += loss.item() * len(targets)
            tot_err += dist.sum().item()
            hits += (dist < 0.10).sum().item()               # within 10% of the image width
            n += len(targets)
    return tot_loss / n, tot_err / n, hits / n


def main():
    ap = argparse.ArgumentParser(description="Train a CNN to predict the (x, y) position of your object")
    ap.add_argument("--data", default="p2_dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="default 0.001, or 0.0001 with --finetune")
    ap.add_argument("--model", choices=ARCHS, default="cnn", help="cnn = from scratch, resnet18 = transfer learning")
    ap.add_argument("--finetune", action="store_true", help="resnet18: train ALL layers")
    ap.add_argument("--no-pretrained", action="store_true", help="resnet18 with RANDOM weights (experiment)")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = run_name(args.model, args.finetune, not args.no_pretrained)
    args.out = args.out or ("p2_tracker.pth" if tag == "cnn" else f"p2_tracker_{tag}.pth")
    curves = "p2_training_curves.png" if tag == "cnn" else f"p2_training_curves_{tag}.png"
    if args.lr is None:
        args.lr = 0.0001 if args.finetune else 0.001

    random.seed(42); torch.manual_seed(42)
    device = get_device()

    csv_path = os.path.join(args.data, "labels.csv")
    if not os.path.exists(csv_path):
        raise SystemExit("[ERROR] No labels found. Run p2_collect.py first.")
    with open(csv_path) as f:
        rows = [(r["filename"], float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]
    obj_file = os.path.join(args.data, "object.txt")
    obj = open(obj_file).read().strip() if os.path.exists(obj_file) else "object"
    if len(rows) < 20:
        raise SystemExit(f"[ERROR] Only {len(rows)} labelled photos. Collect at least 20 (80+ is better).")

    random.shuffle(rows)
    n_val = max(4, int(len(rows) * args.val_split))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    train_loader = DataLoader(PositionDataset(args.data, train_rows, True), batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(PositionDataset(args.data, val_rows, False), batch_size=args.batch, num_workers=2)
    print(f"[INFO] Object       : {obj}")
    print(f"[INFO] Labelled     : {len(rows)} photos (train {len(train_rows)}, validation {len(val_rows)})")

    model = build_model(args.model, 2, regression=True, pretrained=not args.no_pretrained,
                        freeze=not args.finetune).to(device)        # 2 outputs (x, y) + Sigmoid
    total, trainable = count_params(model)
    print(f"[INFO] Model        : {tag}   ({total:,} parameters, {trainable:,} trainable)")
    criterion = nn.MSELoss()                          # regression loss
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    hist = {"train_loss": [], "val_loss": [], "train_err": [], "val_err": []}
    best, t_start = math.inf, time.time()
    print("\nEpoch | Train loss | Val loss | Val error | Within 10% | Time")
    print("-" * 64)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_err, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_err, va_hit = run_epoch(model, val_loader, criterion, device)
        sched.step()
        for k, v in zip(hist, (tr_loss, va_loss, tr_err, va_err)):
            hist[k].append(v)
        flag = ""
        if va_err < best:
            best = va_err
            torch.save({"state_dict": model.state_dict(), "task": "regression", "object": obj,
                        "img_size": IMG_SIZE, "arch": args.model}, args.out)
            flag = "  * saved"
        print(f"{ep:5d} | {tr_loss:10.4f} | {va_loss:8.4f} | {va_err * 100:7.1f}%  | {va_hit:9.0%}  | {time.time() - t0:4.1f}s{flag}")

    print(f"\n[DONE] Best validation error: {best * 100:.1f}% of the image width")
    print(f"       (on a 480-pixel camera image that is about {int(best * 480)} pixels off on average)")
    print(f"[DONE] Model saved to: {args.out}")

    ep = range(1, args.epochs + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, hist["train_loss"], label="train"); a1.plot(ep, hist["val_loss"], label="validation")
    a1.set_title("MSE loss (lower is better)"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(ep, [e * 100 for e in hist["train_err"]], label="train")
    a2.plot(ep, [e * 100 for e in hist["val_err"]], label="validation")
    a2.set_title("Position error, % of image width (lower is better)"); a2.set_xlabel("epoch"); a2.legend()
    fig.suptitle(f"Project 2 - {tag}")
    fig.tight_layout(); fig.savefig(curves, dpi=120)
    print(f"[DONE] Training graph saved to: {curves}")
    log_result("p2_results.csv", {
        "model": tag, "trainable_params": f"{trainable:,}", "epochs": str(args.epochs),
        "best_val_error": f"{best * 100:.1f}%", "train_time": f"{time.time() - t_start:.0f}s"})
    print("\nNext: python3 p2_deploy.py" + ("" if args.out == "p2_tracker.pth" else f" --model {args.out}"))


if __name__ == "__main__":
    main()
