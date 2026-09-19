"""
train.py — ConvNeXt-Tiny + CBAM (v2)
Cải tiến:
  - Focal Loss thay CrossEntropy + WeightedSampler
  - Label Smoothing 0.05 thay vì 0.1
  - Mixup alpha=0.2
  - TTA x5
  - Tăng epochs: S1=10, S2=25, S3=40

Chạy: python train.py
"""

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_combined_dataloaders, RAFDBDataset, EMOTION_LABELS, NUM_CLASSES
from model import build_model


# ── Config ────────────────────────────────────────────────────────────────────
FER_DIR    = "data/fer2013"
RAFDB_DIR  = "data/raf-db/DATASET"
BATCH_SIZE = 64
SAVE_DIR   = "saved_models"
EPOCHS_S1  = 10
EPOCHS_S2  = 25
EPOCHS_S3  = 40
TTA_TIMES  = 5


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: tập trung học class khó (Fear, Disgust).
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    gamma=2: standard, alpha=1: uniform class weight
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        n_classes  = inputs.size(1)
        smooth_val = self.label_smoothing / n_classes
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, smooth_val)
            smooth_targets.scatter_(1, targets.unsqueeze(1),
                                    1.0 - self.label_smoothing + smooth_val)

        log_prob = F.log_softmax(inputs, dim=1)
        prob     = torch.exp(log_prob)

        # Focal weight
        pt          = (prob * smooth_targets).sum(dim=1)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = -(smooth_targets * log_prob).sum(dim=1)
        return (focal_weight * loss).mean()


# ── Mixup ─────────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss/total, correct/total, np.array(all_preds), np.array(all_labels)


# ── TTA ───────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_tta(model, loader, device):
    """TTA: average softmax over multiple passes với dropout ON."""
    model.train()  # giữ dropout ON để tạo stochastic prediction
    all_probs, all_labels = None, None

    for t in range(TTA_TIMES):
        print(f"  TTA {t+1}/{TTA_TIMES}...")
        probs_t, labels_t = [], []
        for images, labels in loader:
            images = images.to(device)
            with torch.no_grad():
                out = torch.softmax(model(images), dim=1)
            probs_t.append(out.cpu().numpy())
            labels_t.extend(labels.numpy())
        probs_t = np.concatenate(probs_t, axis=0)
        if all_probs is None:
            all_probs  = probs_t
            all_labels = np.array(labels_t)
        else:
            all_probs += probs_t

    all_probs /= TTA_TIMES
    y_pred = all_probs.argmax(axis=1)
    acc    = accuracy_score(all_labels, y_pred)
    return acc, y_pred, all_labels


# ── Train epoch ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_mixup:
            mx, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)
            out  = model(mx)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
            out  = model(images)
            loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss/total, correct/total


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Training History — ConvNeXt + CBAM v2",
                 fontsize=14, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=2, linestyle="--")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend()
    axes[1].spines[["top","right"]].set_visible(False)
    for ax in axes:
        ax.axvline(x=EPOCHS_S1,             color="gray", linestyle=":", alpha=0.7)
        ax.axvline(x=EPOCHS_S1 + EPOCHS_S2, color="gray", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    labels = list(EMOTION_LABELS.values())
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved: {save_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Method 4 v2: ConvNeXt-Tiny + CBAM")
    print("  Data  : FER2013 + RAF-DB")
    print("  Loss  : Focal Loss (gamma=2)")
    print("  Mixup : alpha=0.2 (giai đoạn 2+3)")
    print(f"  TTA   : x{TTA_TIMES}")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    loaders, train_targets = get_combined_dataloaders(
        fer_dir=FER_DIR, rafdb_dir=RAFDB_DIR,
        batch_size=BATCH_SIZE,
    )

    # Focal Loss
    criterion = FocalLoss(gamma=2.0, alpha=1.0, label_smoothing=0.05)
    print("\n  Loss: Focal Loss (gamma=2, label_smoothing=0.05)")

    # Build model
    print("\n  Build ConvNeXt-Tiny + CBAM...")
    model = build_model(NUM_CLASSES, freeze_backbone=True).to(device)

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0
    t0 = time.time()

    def run_stage(name, epochs, optimizer, scheduler, use_mixup=False):
        nonlocal best_val_acc
        print(f"\n{'='*55}\n  {name} — {epochs} epoch\n{'='*55}")
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device,
                use_mixup=use_mixup
            )
            vl_loss, vl_acc, _, _ = evaluate(
                model, loaders["val"], criterion, device
            )
            if scheduler: scheduler.step()
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(),
                           f"{SAVE_DIR}/model_convnext_cbam.pth")
            mixup_tag = " [Mixup]" if use_mixup else ""
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train={tr_acc:.4f} | val={vl_acc:.4f}{mixup_tag}")

    # Giai đoạn 1: Frozen, không Mixup
    opt1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=EPOCHS_S1, eta_min=1e-5)
    run_stage("Giai đoạn 1: Frozen backbone", EPOCHS_S1, opt1, sch1, use_mixup=False)

    # Giai đoạn 2: Unfreeze 2 stage + Mixup
    model.unfreeze_backbone(last_n_stages=2)
    opt2 = optim.AdamW([
        {"params": [p for n,p in model.backbone.named_parameters() if p.requires_grad], "lr": 1e-5},
        {"params": model.cbam.parameters(),       "lr": 5e-5},
        {"params": model.classifier.parameters(), "lr": 5e-5},
    ], weight_decay=1e-4)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS_S2, eta_min=1e-6)
    run_stage("Giai đoạn 2: Unfreeze 2 stage + Mixup", EPOCHS_S2, opt2, sch2, use_mixup=True)

    # Giai đoạn 3: Full fine-tune + Mixup
    model.unfreeze_all()
    opt3 = optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": 5e-6},
        {"params": model.cbam.parameters(),       "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 2e-5},
    ], weight_decay=1e-4)
    sch3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=EPOCHS_S3, eta_min=1e-7)
    run_stage("Giai đoạn 3: Fine-tune toàn bộ + Mixup", EPOCHS_S3, opt3, sch3, use_mixup=True)

    train_time = time.time() - t0
    print(f"\n  Best val accuracy: {best_val_acc:.4f}")

    # Evaluate Standard
    model.load_state_dict(
        torch.load(f"{SAVE_DIR}/model_convnext_cbam.pth", map_location=device)
    )
    print("\n[Evaluate] Standard...")
    _, test_acc, y_pred, y_true = evaluate(
        model, loaders["test"], nn.CrossEntropyLoss(), device
    )
    labels = list(EMOTION_LABELS.values())
    print(f"\n  Test Accuracy (Standard): {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(classification_report(y_true, y_pred, target_names=labels))
    plot_confusion_matrix(y_true, y_pred,
                          "Confusion Matrix — Standard",
                          f"{SAVE_DIR}/cm_standard.png")

    # Evaluate TTA
    print(f"\n[Evaluate] TTA x{TTA_TIMES}...")
    tta_acc, y_pred_tta, y_true_tta = evaluate_tta(model, loaders["test"], device)
    print(f"\n  Test Accuracy (TTA x{TTA_TIMES}): {tta_acc:.4f} ({tta_acc*100:.2f}%)")
    print(classification_report(y_true_tta, y_pred_tta, target_names=labels))
    plot_confusion_matrix(y_true_tta, y_pred_tta,
                          f"Confusion Matrix — TTA x{TTA_TIMES}",
                          f"{SAVE_DIR}/cm_tta.png")

    # Save
    plot_history(history, f"{SAVE_DIR}/history_convnext_cbam.png")
    np.save(f"{SAVE_DIR}/result_convnext_cbam.npy", {
        "accuracy_standard": test_acc,
        "accuracy_tta":      tta_acc,
        "train_time_min":    train_time / 60,
        "y_pred": y_pred, "y_true": y_true, "history": history,
    })

    # Export TorchScript
    model.eval()
    scripted = torch.jit.trace(model, torch.randn(1, 3, 192, 192).to(device))
    scripted.save(f"{SAVE_DIR}/model_convnext_cbam_scripted.pt")

    import shutil
    shutil.make_archive("/kaggle/working/saved_models", "zip",
                        "/kaggle/working/saved_models")

    print(f"\n  ✓ Standard : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  ✓ TTA x{TTA_TIMES}  : {tta_acc:.4f} ({tta_acc*100:.2f}%)")
    print(f"  ✓ Train time: {train_time/60:.1f} phút")
    print(f"  ✓ saved_models.zip saved!")


if __name__ == "__main__":
    main()
