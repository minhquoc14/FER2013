"""
model.py — ConvNeXt-Tiny + CBAM Attention
Kiến trúc:
  - Backbone: ConvNeXt-Tiny pretrained ImageNet
  - CBAM: Channel Attention + Spatial Attention
  - Classifier Head: Linear(768→256→7)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ── CBAM: Channel Attention ────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Học xem kênh nào quan trọng.
    Dùng cả AvgPool và MaxPool để bắt đặc trưng toàn cục.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
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
        avg = self.mlp(self.avg_pool(x))
        max_ = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ── CBAM: Spatial Attention ────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Học xem vùng nào quan trọng (mắt, miệng, nếp nhăn).
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_, _ = x.max(dim=1, keepdim=True)
        feat = torch.cat([avg, max_], dim=1)
        scale = self.sigmoid(self.conv(feat))
        return x * scale


# ── CBAM Module ────────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Channel Attention → Spatial Attention (theo thứ tự trong paper).
    """
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ── ConvNeXt + CBAM ────────────────────────────────────────────────────────────
class ConvNeXtCBAM(nn.Module):
    """
    ConvNeXt-Tiny pretrained + CBAM + Classifier Head.

    Input:  (B, 3, 224, 224)
    Output: (B, 7)

    Kiến trúc:
        ConvNeXt-Tiny backbone (freeze giai đoạn 1)
        → CBAM (channel + spatial attention)
        → AdaptiveAvgPool2d
        → FC Head: 768 → 256 → 7
    """
    def __init__(self, num_classes: int = 7, freeze_backbone: bool = True):
        super().__init__()

        # Load ConvNeXt-Tiny pretrained
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

        # Lấy features (bỏ classifier gốc)
        self.backbone = convnext.features   # output: (B, 768, 7, 7) với input 224×224

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # CBAM attention — 768 channels (output của ConvNeXt-Tiny)
        self.cbam = CBAM(in_channels=768, reduction=16, kernel_size=7)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)    # (B, 768, 7, 7)
        x = self.cbam(x)        # (B, 768, 7, 7) — attention
        x = self.gap(x)         # (B, 768, 1, 1)
        x = self.classifier(x)  # (B, 7)
        return x

    def unfreeze_backbone(self, last_n_stages: int = 2):
        """
        Mở khóa N stage cuối của backbone để fine-tune.
        ConvNeXt-Tiny có 4 stages (index 0-7 trong features).
        last_n_stages=2: mở stage 6 và 7 (stage 3 và 4).
        """
        stages = list(self.backbone.children())
        unfreeze_from = len(stages) - last_n_stages * 2  # mỗi stage có 2 module
        for i, module in enumerate(stages):
            if i >= unfreeze_from:
                for param in module.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  [Unfreeze {last_n_stages} stages] trainable: {trainable:,} / {total:,}")

    def unfreeze_all(self):
        """Mở khóa toàn bộ backbone."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [Unfreeze ALL] trainable: {trainable:,}")


def build_model(num_classes: int = 7, freeze_backbone: bool = True) -> ConvNeXtCBAM:
    model = ConvNeXtCBAM(num_classes=num_classes, freeze_backbone=freeze_backbone)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  ConvNeXt-Tiny + CBAM")
    print(f"  Trainable: {trainable:,} / {total:,} params")
    return model


if __name__ == "__main__":
    # Test nhanh
    model = build_model(num_classes=7, freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"  Output shape: {out.shape}")  # (2, 7)
    print("  ✓ Model OK!")
