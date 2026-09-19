"""
method4_cbam.py — Phương pháp 4: EfficientNet-B2 + CBAM Attention
==================================================================
Kiến trúc:
  EfficientNet-B2 (pretrained) → CBAM trên stage 4/5/6
  → Multi-Scale Fusion → Dropout → FC(7)

Kỹ thuật:
  - WeightedRandomSampler   : cân bằng lớp thiểu số (Disgust ~436 mẫu)
  - Label Smoothing (0.1)   : giảm ảnh hưởng nhãn nhiễu FER2013
  - Mixup (alpha=0.2)       : regularization
  - Two-stage training      : stage 1 frozen backbone, stage 2 fine-tune toàn bộ
  - Cosine LR + Warmup      : ổn định training

Chạy: python method4_cbam.py --data data/archive
"""

import os
import argparse
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import models

from dataset import get_dataloaders, EMOTION_LABELS, NUM_CLASSES


# ── Cấu hình ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
STAGE1_EPOCHS  = 15
STAGE2_EPOCHS  = 40
LR_HEAD        = 1e-3
LR_HEAD2       = 5e-4
LR_BACKBONE    = 5e-5
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.4
LABEL_SMOOTH   = 0.1
MIXUP_ALPHA    = 0.2
SAVE_PATH      = "saved_models/model_cbam.pth"


# ── 1. CBAM ───────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, C, ratio=16):
        super().__init__()
        mid = max(C // ratio, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, C, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        w = self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ca = ChannelAttention(C)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ── 2. Model ──────────────────────────────────────────────────────────────────
class EfficientNetB2_CBAM(nn.Module):
    """
    EfficientNet-B2 backbone + CBAM trên stage 4/5/6 + Multi-Scale Fusion.
    Stage channels: 4→56ch, 5→120ch, 6→208ch
    """
    def __init__(self, num_classes=7, dropout=0.4, freeze_until=3):
        super().__init__()
        base = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        self.features = base.features

        for i, blk in enumerate(self.features):
            if i <= freeze_until:
                for p in blk.parameters():
                    p.requires_grad = False

        self.cbam4 = CBAM(88)
        self.cbam5 = CBAM(120)
        self.cbam6 = CBAM(208)

        self.fuse = nn.Sequential(
            nn.Conv2d(88 + 120 + 208, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i == 4: f4 = x
            if i == 5: f5 = x
            if i == 6: f6 = x

        f4 = self.cbam4(f4)
        f5 = self.cbam5(f5)
        f6 = self.cbam6(f6)

        t  = (f6.shape[2], f6.shape[3])
        f4 = F.interpolate(f4, size=t, mode="bilinear", align_corners=False)
        f5 = F.interpolate(f5, size=t, mode="bilinear", align_corners=False)
        out = self.fuse(torch.cat([f4, f5, f6], dim=1))
        out = self.pool(out).flatten(1)
        return self.fc(self.dropout(out))


# ── 3. Mixup ──────────────────────────────────────────────────────────────────
def mixup(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ── 4. Train / Val một epoch ──────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, ya, yb, lam = mixup(imgs, labels, MIXUP_ALPHA)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = out.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (lam * (preds == ya).float() +
                       (1 - lam) * (preds == yb).float()).sum().item()
        n += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = n = 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out   = model(imgs)
        loss  = criterion(out, labels)
        preds = out.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        n          += imgs.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    return (total_loss / n, correct / n,
            torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy())


# ── 5. Weighted Sampler ───────────────────────────────────────────────────────
def make_sampler(targets, train_idx):
    t       = targets[train_idx]
    counts  = np.bincount(t, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / np.where(counts > 0, counts, 1.0)
    return WeightedRandomSampler(
        torch.from_numpy(weights[t]).float(), len(t), replacement=True
    )


# ── 6. Main ───────────────────────────────────────────────────────────────────
def main(data_dir: str):
    print("=" * 55)
    print("  Phương pháp 4: EfficientNet-B2 + CBAM")
    print("  Tiền xử lý: CLAHE + Resize 224x224 + ImageNet norm")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loaders, train_idx, targets = get_dataloaders(
        data_dir, method="pretrained", batch_size=BATCH_SIZE
    )
    sampler = make_sampler(targets, train_idx)
    loaders, _, _ = get_dataloaders(
        data_dir, method="pretrained",
        batch_size=BATCH_SIZE, sampler=sampler
    )

    print("[Phân phối class (train)]")
    counts = np.bincount(targets[train_idx], minlength=NUM_CLASSES)
    for name, c in zip(EMOTION_LABELS.values(), counts):
        print(f"  {name:10s}: {c:5d} ({c/counts.sum()*100:.1f}%)")

    # ── Model + Loss ──────────────────────────────────────────────────────────
    model     = EfficientNetB2_CBAM(NUM_CLASSES, DROPOUT, freeze_until=3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    total_p   = sum(p.numel() for p in model.parameters())
    train_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Total params: {total_p:,} | Trainable: {train_p:,}\n")

    best_val_acc = 0.0
    best_state   = None

    # ── Stage 1: Train head (backbone frozen) ─────────────────────────────────
    print(f"[Stage 1] Epoch 1–{STAGE1_EPOCHS}  (backbone frozen, train CBAM+FC)")
    opt1 = AdamW([p for p in model.parameters() if p.requires_grad],
                 lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    sch1 = SequentialLR(opt1, [
        LinearLR(opt1, start_factor=0.1, end_factor=1.0, total_iters=3),
        CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS - 3, eta_min=1e-6),
    ], milestones=[3])

    for ep in range(1, STAGE1_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, loaders["train"], criterion, opt1, device)
        vl_loss, vl_acc, _, _ = val_epoch(model, loaders["val"], criterion, device)
        sch1.step()
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())
        print(f"  Ep {ep:02d}/{STAGE1_EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")

    # ── Stage 2: Fine-tune toàn bộ ────────────────────────────────────────────
    print(f"\n[Stage 2] Epoch 1–{STAGE2_EPOCHS}  (unfreeze toàn bộ, fine-tune)")
    for p in model.parameters():
        p.requires_grad = True

    sampler2 = make_sampler(targets, train_idx)
    loaders2, _, _ = get_dataloaders(
        data_dir, method="pretrained",
        batch_size=BATCH_SIZE, sampler=sampler2
    )

    opt2 = AdamW([
        {"params": model.features.parameters(),  "lr": LR_BACKBONE},
        {"params": model.cbam4.parameters(),     "lr": LR_HEAD2},
        {"params": model.cbam5.parameters(),     "lr": LR_HEAD2},
        {"params": model.cbam6.parameters(),     "lr": LR_HEAD2},
        {"params": model.fuse.parameters(),      "lr": LR_HEAD2},
        {"params": model.fc.parameters(),        "lr": LR_HEAD2},
    ], weight_decay=WEIGHT_DECAY)
    sch2 = SequentialLR(opt2, [
        LinearLR(opt2, start_factor=0.1, end_factor=1.0, total_iters=2),
        CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS - 2, eta_min=1e-6),
    ], milestones=[2])

    model.load_state_dict(best_state)

    for ep in range(1, STAGE2_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, loaders2["train"], criterion, opt2, device)
        vl_loss, vl_acc, _, _ = val_epoch(model, loaders2["val"], criterion, device)
        sch2.step()
        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())
            flag = "  ✓ best"
        print(f"  Ep {ep:02d}/{STAGE2_EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
              f"({time.time()-t0:.0f}s){flag}")

    # ── Đánh giá test ─────────────────────────────────────────────────────────
    print("\n[Bước cuối] Đánh giá trên Test set...")
    model.load_state_dict(best_state)
    _, test_acc, y_pred, y_test = val_epoch(
        model, loaders2["test"], criterion, device
    )

    # In kết quả
    from sklearn.metrics import classification_report
    label_names = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    print(f"\n{'='*55}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=4))

    os.makedirs("saved_models", exist_ok=True)
    torch.save(best_state, SAVE_PATH)
    np.save("saved_models/result_cbam.npy", {
        "accuracy": test_acc, "y_pred": y_pred, "y_true": y_test,
    })
    print(f"  ✓ Model: {SAVE_PATH}")
    return test_acc, y_pred, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    main(args.data)
