"""
method2_cnn.py — Phương pháp 2: CNN from scratch
Tiền xử lý: Resize 48x48 + Normalize (không CLAHE/Noise cho CNN nhỏ)

Chạy: python method2_cnn.py --data data/archive
"""

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import NUM_CLASSES, EMOTION_LABELS
from utils import evaluate, plot_history, plot_confusion_matrix, print_report


# ── Transform đơn giản cho CNN nhỏ ───────────────────────────────────────────
def get_transforms_cnn():
    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507], std=[0.255]),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507], std=[0.255]),
    ])
    return train_tf, val_tf


def get_dataloaders_cnn(data_dir, batch_size=64, val_split=0.1):
    train_tf, val_tf = get_transforms_cnn()
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    targets    = np.array(full_train.targets)
    indices    = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=targets, random_state=42
    )
    val_dataset  = datasets.ImageFolder(train_dir, transform=val_tf)
    train_subset = Subset(full_train,  train_idx)
    val_subset   = Subset(val_dataset, val_idx)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_tf)

    loaders = {
        "train": DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False),
        "val":   DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False),
        "test":  DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False),
    }
    print(f"[dataset] train={len(train_subset)} | val={len(val_subset)} | test={len(test_dataset)}")
    return loaders, targets[train_idx]


# ── Kiến trúc CNN ─────────────────────────────────────────────────────────────
class FER_CNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 256),     nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Class weights ──────────────────────────────────────────────────────────────
def compute_class_weights(train_targets, device):
    counts  = np.bincount(train_targets)
    weights = len(train_targets) / (len(counts) * counts)
    weights = np.sqrt(weights)
    weights = weights / weights.sum() * len(counts)
    print("\n[Class weights trong Loss]")
    for name, w, c in zip(EMOTION_LABELS.values(), weights, counts):
        print(f"  {name:10s}: count={c:5d} → weight={w:.3f}")
    return torch.FloatTensor(weights).to(device)


# ── Train loop ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def train(model, loaders, optimizer, criterion, scheduler=None, epochs=50, device="cpu"):
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0
    print(f"\n{'='*55}\n  Training CNN | device={device} | epochs={epochs}\n{'='*55}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        if scheduler:
            scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "best_cnn.pth")

        print(f"  Epoch {epoch:3d}/{epochs} | train={tr_acc:.4f} | val={vl_acc:.4f}")

    print(f"\n  Best val accuracy: {best_val_acc:.4f}")
    return history


# ── Main ───────────────────────────────────────────────────────────────────────
def main(data_dir: str, epochs: int = 50, batch_size: int = 64):
    print("=" * 55)
    print("  Phương pháp 2: CNN (from scratch)")
    print("  Tiền xử lý: Resize+Normalize+Aug+ClassWeights")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cpu":
        print("  Lưu ý: Train trên CPU sẽ mất ~2-4 giờ cho 50 epoch.")
        print("  Gợi ý: dùng Google Colab (GPU miễn phí) nếu cần nhanh hơn.\n")

    # Load data
    loaders, train_targets = get_dataloaders_cnn(data_dir, batch_size=batch_size)

    # Class weights
    class_weights = compute_class_weights(train_targets, device)

    # Model
    model = FER_CNN(NUM_CLASSES).to(device)
    print(f"\n  Số tham số: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Train
    t0 = time.time()
    history = train(model, loaders, optimizer, criterion,
                    scheduler=scheduler, epochs=epochs, device=device)
    train_time = time.time() - t0

    # Load best → evaluate
    os.makedirs("saved_models", exist_ok=True)
    if os.path.exists("best_cnn.pth"):
        model.load_state_dict(torch.load("best_cnn.pth", map_location=device))
        os.replace("best_cnn.pth", "saved_models/model_cnn.pth")
        print("\n  Checkpoint tốt nhất: saved_models/model_cnn.pth")

    _, test_acc, y_pred, y_true = evaluate(
        model, loaders["test"], nn.CrossEntropyLoss(), device
    )

    print_report(y_true, y_pred, method_name="CNN")
    plot_history(history, method_name="CNN",
                 save_path="saved_models/history_cnn.png")
    plot_confusion_matrix(y_true, y_pred, method_name="CNN",
                          save_path="saved_models/cm_cnn.png")
    np.save("saved_models/result_cnn.npy", {
        "accuracy": test_acc, "train_time_min": train_time/60,
        "y_pred": y_pred, "y_true": y_true, "history": history,
    })

    # Export TorchScript
    model.eval()
    scripted = torch.jit.trace(model, torch.randn(1, 1, 48, 48).to(device))
    scripted.save("saved_models/model_cnn_scripted.pt")

    print(f"\n  ✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  ✓ Train time  : {train_time/60:.1f} phút")
    print(f"  ✓ TorchScript : saved_models/model_cnn_scripted.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args.data, args.epochs, args.batch_size)