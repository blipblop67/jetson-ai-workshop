"""
PROJECT 1 (CLASSIFICATION) - STEP 2: TRAIN
==========================================
Reads the photos of YOUR objects from Step 1 (Jetson camera and/or phone
photos), splits them into a TRAINING set (80%) and a VALIDATION set (20%),
and trains the SimpleCNN from common.py. Each folder name = one object.

Outputs
-------
    p1_classifier.pth      : the best trained model (highest validation accuracy)
    p1_training_curves.png : loss & accuracy graph for every epoch

Usage examples
--------------
    python3 p1_train.py                      # defaults: 15 epochs
    python3 p1_train.py --epochs 25 --lr 0.0005
    python3 p1_train.py --no-flip            # use if left/right matters (e.g. arrows)

Transfer learning (same photos, compare the results!)
-----------------------------------------------------
    python3 p1_train.py --model resnet18               # pre-trained ResNet-18, train only the last layer
    python3 p1_train.py --model resnet18 --finetune    # pre-trained ResNet-18, train every layer
Every run is added to p1_results.csv and a comparison table is printed.
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")              # save graphs to file (works without a display)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import (get_device, IMG_SIZE, MEAN, STD, SquareCrop, load_image,
                    build_model, count_params, set_mode, run_name, log_result, ARCHS)


def build_transforms(flip=True):
    # Data augmentation: small random changes so the model sees "new" images every epoch
    aug = [
        SquareCrop(),                                   # same view as the camera
        transforms.Resize(IMG_SIZE + 32),               # shrink big phone photos first
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ]
    if flip:
        aug.insert(3, transforms.RandomHorizontalFlip())
    train_tf = transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    val_tf = transforms.Compose([
        SquareCrop(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise just evaluates."""
    training = optimizer is not None
    set_mode(model, training)        # (keeps frozen pre-trained layers frozen)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)                 # forward pass
            loss = criterion(outputs, labels)       # how wrong are we?

            if training:
                optimizer.zero_grad()
                loss.backward()                     # backpropagation
                optimizer.step()                    # update the weights

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser(description="Train a CNN on your collected images")
    ap.add_argument("--data", default="p1_dataset", help="dataset folder (default: p1_dataset)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16, help="batch size")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (default 0.001, or 0.0001 with --finetune)")
    ap.add_argument("--model", choices=ARCHS, default="cnn", help="cnn = from scratch, resnet18 = transfer learning")
    ap.add_argument("--finetune", action="store_true", help="resnet18: train ALL layers instead of only the last one")
    ap.add_argument("--no-pretrained", action="store_true", help="resnet18 with RANDOM weights (experiment: is pre-training what helps?)")
    ap.add_argument("--val-split", type=float, default=0.2, help="fraction kept for validation")
    ap.add_argument("--no-flip", action="store_true", help="disable horizontal flip augmentation")
    ap.add_argument("--out", default=None, help="model file (default depends on --model)")
    args = ap.parse_args()
    tag = run_name(args.model, args.finetune, not args.no_pretrained)
    args.out = args.out or ("p1_classifier.pth" if tag == "cnn" else f"p1_classifier_{tag}.pth")
    curves = "p1_training_curves.png" if tag == "cnn" else f"p1_training_curves_{tag}.png"
    if args.lr is None:
        args.lr = 0.0001 if args.finetune else 0.001

    torch.manual_seed(42)
    device = get_device()

    # ---------------- 1. Load the data ----------------
    train_tf, val_tf = build_transforms(flip=not args.no_flip)
    train_full = datasets.ImageFolder(args.data, transform=train_tf, loader=load_image)  # folder = label
    val_full = datasets.ImageFolder(args.data, transform=val_tf, loader=load_image)
    classes = train_full.classes

    n = len(train_full)
    if n < 10 or len(classes) < 2:
        raise SystemExit("[ERROR] Need at least 2 classes and some images. Run p1_collect.py first.")

    perm = torch.randperm(n).tolist()
    n_val = max(1, int(n * args.val_split))
    val_set = Subset(val_full, perm[:n_val])
    train_set = Subset(train_full, perm[n_val:])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=2)

    print(f"[INFO] Objects      : {classes}")
    print(f"[INFO] Total images : {n}  (train {len(train_set)}, validation {len(val_set)})")
    for i, c in enumerate(classes):
        k = train_full.targets.count(i)
        print(f"         {c:20s} {k} images" + ("   <-- very few, take more photos!" if k < 20 else ""))

    # ---------------- 2. Build the model ----------------
    model = build_model(args.model, len(classes), pretrained=not args.no_pretrained,
                        freeze=not args.finetune).to(device)
    total, trainable = count_params(model)
    print(f"[INFO] Model        : {tag}   ({total:,} parameters, {trainable:,} trainable)")
    if args.model == "resnet18" and not args.finetune:
        print("[INFO] Transfer learning: pre-trained layers are FROZEN, only the new last layer learns")

    criterion = nn.CrossEntropyLoss()                                # classification loss
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # ---------------- 3. Train ----------------
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, first90, t_start = 0.0, None, time.time()
    print("\nEpoch | Train loss | Train acc | Val loss | Val acc | Time")
    print("-" * 60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)
        if first90 is None and va_acc >= 0.9:
            first90 = epoch

        flag = ""
        if va_acc >= best_acc:                 # keep the best model only
            best_acc = va_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "img_size": IMG_SIZE, "arch": args.model}, args.out)
            flag = "  * saved"
        print(f"{epoch:5d} | {tr_loss:10.4f} | {tr_acc:8.1%} | {va_loss:8.4f} | "
              f"{va_acc:6.1%} | {time.time() - t0:4.1f}s{flag}")

    train_time = time.time() - t_start
    print(f"\n[DONE] Best validation accuracy: {best_acc:.1%}")
    print(f"[DONE] Model saved to: {args.out}")

    # ---------------- 4. Confusion matrix on the validation set ----------------
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    set_mode(model, False)
    k = len(classes)
    cm = [[0] * k for _ in range(k)]
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device)).argmax(1).cpu()
            for t, p in zip(labels.tolist(), preds.tolist()):
                cm[t][p] += 1
    total_val = sum(map(sum, cm))
    print("\n" + "=" * 62)
    print(" CONFUSION MATRIX  (validation photos the model never trained on)")
    print("   rows = TRUE object, columns = what the model PREDICTED")
    print("   numbers on the diagonal are correct; anything else is a mix-up")
    print("=" * 62)
    w = max(max(len(c) for c in classes) + 2, 18)
    print("true \\ predicted".ljust(w) + "".join(f"{c[:9]:>10s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<{w}s}" + "".join((f"[{v}]" if i == j else str(v)).rjust(10) for j, v in enumerate(cm[i])))
    print("-" * 62)
    for i, c in enumerate(classes):
        n_i = sum(cm[i])
        acc = cm[i][i] / n_i if n_i else 0.0
        worst = max(((cm[i][j], classes[j]) for j in range(k) if j != i), default=(0, ""))
        note = f"   (mixed up with {worst[1]} x{worst[0]})" if worst[0] else ""
        print(f"  {c:<{w}s} {cm[i][i]:3d}/{n_i:<3d} correct = {acc:6.1%}{note}")
    print(f"  Overall: {sum(cm[i][i] for i in range(k))}/{total_val} correct")

    cm_png = "p1_confusion_matrix.png" if tag == "cnn" else f"p1_confusion_matrix_{tag}.png"
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * k, 1.2 + 1.0 * k))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(k)); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix - {tag}")
    top = max(max(r) for r in cm) or 1
    for i in range(k):
        for j in range(k):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=12,
                    color="white" if cm[i][j] > top / 2 else "black")
    fig.tight_layout(); fig.savefig(cm_png, dpi=120); plt.close(fig)
    print(f"[DONE] Confusion matrix picture saved to: {cm_png}")

    # ---------------- 5. Plot training curves ----------------
    ep = range(1, args.epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ep, history["train_loss"], label="train")
    ax1.plot(ep, history["val_loss"], label="validation")
    ax1.set_title("Loss (lower is better)"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(ep, [a * 100 for a in history["train_acc"]], label="train")
    ax2.plot(ep, [a * 100 for a in history["val_acc"]], label="validation")
    ax2.set_title("Accuracy % (higher is better)"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.suptitle(f"Project 1 - {tag}")
    fig.tight_layout()
    fig.savefig(curves, dpi=120)
    print(f"[DONE] Training graph saved to: {curves}")

    log_result("p1_results.csv", {
        "model": tag, "trainable_params": f"{trainable:,}", "epochs": str(args.epochs),
        "best_val_acc": f"{best_acc:.1%}", "epoch_reached_90%": str(first90 or "-"),
        "train_time": f"{train_time:.0f}s"})
    print("\nNext: python3 p1_deploy.py" + ("" if args.out == "p1_classifier.pth" else f" --model {args.out}"))


if __name__ == "__main__":
    main()
