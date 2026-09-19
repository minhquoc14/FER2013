"""
utils.py — Các hàm dùng chung: train loop, evaluate, visualize.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EMOTION_LABELS


# ── Training loop (PyTorch) ────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train(model, loaders, optimizer, criterion, scheduler=None,
          epochs=30, device="cpu", method_name="CNN"):
    """
    Vòng lặp train đầy đủ, trả về history để vẽ đồ thị.
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'='*55}")
    print(f"  Training: {method_name}  |  device={device}  |  epochs={epochs}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, loaders["val"], criterion, device)

        if scheduler:
            scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        # Lưu checkpoint tốt nhất
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), f"best_{method_name.lower().replace(' ', '_')}.pth")

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | "
              f"{elapsed:.1f}s")

    print(f"\n  Best val accuracy: {best_val_acc:.4f}")
    return history


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_history(history, method_name="CNN", save_path=None):
    """Vẽ đường cong loss và accuracy qua các epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training History — {method_name}", fontsize=13)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, method_name="CNN", save_path=None):
    """Vẽ confusion matrix chuẩn hóa."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    labels = list(EMOTION_LABELS.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {method_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_report(y_true, y_pred, method_name="CNN"):
    """In classification report chi tiết theo từng lớp cảm xúc."""
    labels = list(EMOTION_LABELS.values())
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  Results: {method_name}")
    print(f"{'='*55}")
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print()
    print(classification_report(y_true, y_pred, target_names=labels))


# ── So sánh nhiều phương pháp ──────────────────────────────────────────────────
def plot_comparison(results: dict, save_path=None):
    """
    Vẽ bar chart so sánh accuracy và thời gian train.

    results = {
        "LBP + SVM":       {"accuracy": 0.48, "train_time_min": 2},
        "CNN":             {"accuracy": 0.63, "train_time_min": 90},
        "Pretrained CNN":  {"accuracy": 0.71, "train_time_min": 45},
    }
    """
    methods = list(results.keys())
    accuracies  = [v["accuracy"]       for v in results.values()]
    train_times = [v["train_time_min"] for v in results.values()]

    colors = ["#4C9EEB", "#F26419", "#2DC653"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("So sánh 3 phương pháp FER trên FER2013", fontsize=13)

    # Accuracy
    bars = axes[0].bar(methods, [a * 100 for a in accuracies], color=colors, width=0.5)
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Accuracy")
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=10)

    # Train time
    bars2 = axes[1].bar(methods, train_times, color=colors, width=0.5)
    axes[1].set_ylabel("Thời gian train (phút)")
    axes[1].set_title("Thời gian huấn luyện")
    for bar, t in zip(bars2, train_times):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{t} phút", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
