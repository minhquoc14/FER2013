"""
method3_pretrained.py — Phương pháp 3: EfficientNet-B2 Fine-tune (v3)
Cải tiến so với v2:
  - Bỏ WeightedRandomSampler (gây bias quá mạnh)
  - Class weights nhẹ hơn (log thay vì sqrt)
  - Mixup bắt đầu muộn hơn (epoch 20 thay vì 5)
  - Warmup + Cosine LR decay

Chạy: python method3_pretrained.py --data data/archive
"""

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, Subset

from dataset import get_dataloaders, get_transforms, NUM_CLASSES, EMOTION_LABELS
from utils import evaluate, plot_history, plot_confusion_matrix, print_report


# ── Model: EfficientNet-B2 ────────────────────────────────────────────────────
def build_model(num_classes=7, freeze_backbone=True):
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ── Class weights nhẹ (log scale) ────────────────────────────────────────────
def compute_class_weights(train_idx, targets, device):
    """
    Dùng log scale thay vì sqrt để class weights nhẹ hơn.
    Tránh bias quá mạnh về class ít mẫu như Disgust.
    """
    train_targets = targets[train_idx]
    class_counts  = np.bincount(train_targets)
    # Log scale: nhẹ hơn sqrt, cân bằng hơn
    weights = np.log(len(train_targets) / (class_counts + 1))
    weights = np.clip(weights, 0.5, 3.0)   # giới hạn không quá mạnh
    weights = weights / weights.sum() * len(weights)
    cw_tensor = torch.FloatTensor(weights).to(device)

    print("\n[Class weights trong Loss (log scale)]")
    for name, c, w in zip(EMOTION_LABELS.values(), class_counts, weights):
        print(f"  {name:10s}: count={c:5d} → weight={w:.3f}")
    return cw_tensor


# ── Mixup ─────────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Train epoch ───────────────────────────────────────────────────────────────
def _train_epoch(model, loader, optimizer, criterion, device, use_mixup=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_mixup:
            mx, y_a, y_b, lam = mixup_data(images, labels)
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
    return total_loss / total, correct / total


# ── Train 2 giai đoạn ─────────────────────────────────────────────────────────
def train_two_stage(model, loaders, device, cw_tensor,
                    epochs_stage1=10, epochs_stage2=60):
    criterion    = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=0.1)
    history      = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0
    os.makedirs("saved_models", exist_ok=True)

    # ── Giai đoạn 1: Train classifier head ───────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Giai đoạn 1: Train head (frozen) — {epochs_stage1} epoch")
    print(f"{'='*55}")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    for epoch in range(1, epochs_stage1 + 1):
        tr_loss, tr_acc = _train_epoch(model, loaders["train"], optimizer,
                                       criterion, device, use_mixup=False)
        vl_loss, vl_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(vl_loss)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "saved_models/model_pretrained.pth")
        print(f"  Epoch {epoch:3d}/{epochs_stage1} | "
              f"train={tr_acc:.4f} | val={vl_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

    # ── Giai đoạn 2: Fine-tune + Mixup muộn + Warmup Cosine ──────────────────
    print(f"\n{'='*55}")
    print(f"  Giai đoạn 2: Fine-tune + Warmup + Mixup(ep>20) — {epochs_stage2} epoch")
    print(f"{'='*55}")
    for param in model.parameters():
        param.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params     = list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 5e-6},
        {"params": head_params,     "lr": 5e-5},
    ], weight_decay=1e-4)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(
            np.pi * (epoch - warmup) / (epochs_stage2 - warmup)
        ))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(1, epochs_stage2 + 1):
        # Mixup bắt đầu epoch 20 — sau warmup và lr đã ổn định
        use_mixup = epoch > 20
        tr_loss, tr_acc = _train_epoch(model, loaders["train"], optimizer,
                                       criterion, device, use_mixup=use_mixup)
        vl_loss, vl_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "saved_models/model_pretrained.pth")

        mixup_tag = " [Mixup ON]" if use_mixup else ""
        print(f"  Epoch {epoch:3d}/{epochs_stage2} | "
              f"train={tr_acc:.4f} | val={vl_acc:.4f}{mixup_tag}")

    print(f"\n  Best val accuracy: {best_val_acc:.4f}")
    return history


# ── Main ───────────────────────────────────────────────────────────────────────
def main(data_dir, epochs_s1=10, epochs_s2=60, batch_size=64):
    print("=" * 55)
    print("  Phương pháp 3: EfficientNet-B2 Fine-tune (v3)")
    print("  Bỏ Sampler + Class weights nhẹ + Mixup muộn")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cpu":
        print("  Lưu ý: Train trên CPU rất chậm. Dùng Kaggle/Colab GPU!\n")

    # Load data (không dùng sampler)
    loaders, train_idx, targets = get_dataloaders(
        data_dir, method="pretrained", batch_size=batch_size
    )

    # Class weights nhẹ (log scale)
    cw_tensor = compute_class_weights(train_idx, targets, device)

    # Build model
    print("\n  Build EfficientNet-B2...")
    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Tham số trainable: {trainable:,} / {total:,}")

    # Train
    t0 = time.time()
    history = train_two_stage(
        model, loaders, device, cw_tensor, epochs_s1, epochs_s2
    )
    train_time = time.time() - t0

    # Evaluate
    print("\n[Bước cuối] Đánh giá trên test set...")
    model.load_state_dict(
        torch.load("saved_models/model_pretrained.pth", map_location=device)
    )
    _, test_acc, y_pred, y_true = evaluate(
        model, loaders["test"], nn.CrossEntropyLoss(), device
    )

    print_report(y_true, y_pred, method_name="EfficientNet-B2 v3")
    plot_history(history, method_name="EfficientNet-B2 v3",
                 save_path="saved_models/history_pretrained.png")
    plot_confusion_matrix(y_true, y_pred,
                          method_name="EfficientNet-B2 v3",
                          save_path="saved_models/cm_pretrained.png")
    np.save("saved_models/result_pretrained.npy", {
        "accuracy":       test_acc,
        "train_time_min": train_time / 60,
        "y_pred":         y_pred,
        "y_true":         y_true,
        "history":        history,
    })

    # Export TorchScript
    model.eval()
    dummy    = torch.randn(1, 3, 224, 224).to(device)
    scripted = torch.jit.trace(model, dummy)
    scripted.save("saved_models/model_pretrained_scripted.pt")

    print(f"\n  ✓ TorchScript : saved_models/model_pretrained_scripted.pt")
    print(f"  ✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  ✓ Train time  : {train_time/60:.1f} phút")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--epochs_s1",  type=int, default=10)
    parser.add_argument("--epochs_s2",  type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args.data, args.epochs_s1, args.epochs_s2, args.batch_size)