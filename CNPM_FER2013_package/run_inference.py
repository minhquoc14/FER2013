"""
run_inference.py — Chạy inference để tạo file result .npy
Chạy: python run_inference.py --fer_dir data/archive

Yêu cầu:
  - saved_models/model_cbam.pth          (EfficientNet-B2 + CBAM)
  - saved_models/model_convnext_cbam.pth (ConvNeXt + CBAM)
  - data/archive/test/                   (FER2013 test set)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import EMOTION_LABELS, NUM_CLASSES


# ── Test loader dùng chung ────────────────────────────────────────────────────
def get_test_loader(fer_dir, batch_size=64):
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    test_ds = datasets.ImageFolder(
        os.path.join(fer_dir, "test"), transform=val_tf
    )
    loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    print(f"  [Test] FER2013 = {len(test_ds)} ảnh")
    return loader


# ══════════════════════════════════════════════════════════════════════════════
# Model 1: EfficientNet-B2 + CBAM (bạn kia — dùng ca/sa)
# ══════════════════════════════════════════════════════════════════════════════
class ChannelAttentionEff(nn.Module):
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


class SpatialAttentionEff(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAMEff(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ca = ChannelAttentionEff(C)  # key: cbam4.ca.*
        self.sa = SpatialAttentionEff()   # key: cbam4.sa.*

    def forward(self, x):
        return self.sa(self.ca(x))


class EfficientNetB2_CBAM(nn.Module):
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
        self.cbam4 = CBAMEff(88)
        self.cbam5 = CBAMEff(120)
        self.cbam6 = CBAMEff(208)
        self.fuse  = nn.Sequential(
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


# ══════════════════════════════════════════════════════════════════════════════
# Model 2: ConvNeXt-Tiny + CBAM (của bạn — dùng channel/spatial)
# ══════════════════════════════════════════════════════════════════════════════
class ChannelAttentionCnx(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg  = self.mlp(self.avg_pool(x))
        max_ = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttentionCnx(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAMCnx(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttentionCnx(in_channels, reduction)  # key: cbam.channel.*
        self.spatial = SpatialAttentionCnx(kernel_size)              # key: cbam.spatial.*

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ConvNeXtCBAM(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        convnext = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        self.backbone = convnext.features
        self.cbam = CBAMCnx(in_channels=768, reduction=16, kernel_size=7)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.gap(x)
        return self.classifier(x)


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images  = images.to(device)
        outputs = model(images)
        preds   = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    acc    = (y_pred == y_true).mean()
    return acc, y_pred, y_true


# ── Main ──────────────────────────────────────────────────────────────────────
def main(fer_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    label_names = list(EMOTION_LABELS.values())
    os.makedirs("saved_models", exist_ok=True)

    print("\n[Load test set]")
    test_loader = get_test_loader(fer_dir, batch_size=64)

    # ── Method 4: EfficientNet-B2 + CBAM ─────────────────────────────────────
    path_cbam = "saved_models/model_cbam.pth"
    if os.path.exists(path_cbam):
        print(f"\n{'='*55}")
        print(f"  Method 4: EfficientNet-B2 + CBAM")
        print(f"{'='*55}")
        model = EfficientNetB2_CBAM(num_classes=7, dropout=0.4).to(device)
        model.load_state_dict(torch.load(path_cbam, map_location=device))
        acc, y_pred, y_true = run_inference(model, test_loader, device)
        print(f"\n  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(classification_report(y_true, y_pred, target_names=label_names))
        np.save("saved_models/result_cbam.npy", {
            "accuracy": acc, "train_time_min": 0,
            "y_pred": y_pred, "y_true": y_true,
        })
        print(f"  ✓ Saved: saved_models/result_cbam.npy")
    else:
        print(f"\n[!] Không tìm thấy {path_cbam}")

    # ── Method 5: ConvNeXt-Tiny + CBAM ───────────────────────────────────────
    path_cnx = "saved_models/model_convnext_cbam.pth"
    if os.path.exists(path_cnx):
        print(f"\n{'='*55}")
        print(f"  Method 5: ConvNeXt-Tiny + CBAM")
        print(f"{'='*55}")
        model = ConvNeXtCBAM(num_classes=7).to(device)
        model.load_state_dict(torch.load(path_cnx, map_location=device))
        acc, y_pred, y_true = run_inference(model, test_loader, device)
        print(f"\n  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(classification_report(y_true, y_pred, target_names=label_names))
        np.save("saved_models/result_convnext_cbam.npy", {
            "accuracy": acc, "train_time_min": 526.3,
            "y_pred": y_pred, "y_true": y_true,
        })
        print(f"  ✓ Saved: saved_models/result_convnext_cbam.npy")
    else:
        print(f"\n[!] Không tìm thấy {path_cnx}")

    print(f"\n{'='*55}")
    print(f"  ✓ Xong! Chạy: python plot_results.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fer_dir", default="data/archive")
    args = parser.parse_args()
    main(args.fer_dir)