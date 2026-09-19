"""
check_pipeline.py — Kiểm tra pipeline tất cả 4 method
Chạy: python check_pipeline.py
"""

import torch
import numpy as np
from dataset import (get_dataloaders, get_combined_dataloaders,
                     EMOTION_LABELS)


def check_method123(fer_dir: str):
    print("\n" + "=" * 55)
    print("  Method 1-3: FER2013 Pipeline Check")
    print("=" * 55)

    # Method 1: LBP
    print("\n[1] LBP — Grayscale 48×48 + CLAHE:")
    loaders, _, _ = get_dataloaders(fer_dir, method="lbp", batch_size=64)
    images, labels = next(iter(loaders["train"]))
    print(f"    Batch shape : {images.shape}")
    print(f"    Pixel range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"    Labels      : {[EMOTION_LABELS[l.item()] for l in labels[:4]]}")

    # Method 2: CNN
    print("\n[2] CNN — Grayscale 48×48 + CLAHE + Aug:")
    loaders, _, _ = get_dataloaders(fer_dir, method="cnn", batch_size=64)
    images, labels = next(iter(loaders["train"]))
    print(f"    Batch shape : {images.shape}")
    print(f"    Pixel range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"    Labels      : {[EMOTION_LABELS[l.item()] for l in labels[:4]]}")

    # Method 3: EfficientNet
    print("\n[3] EfficientNet — RGB 224×224 + CLAHE + Aug:")
    loaders, _, _ = get_dataloaders(fer_dir, method="pretrained", batch_size=32)
    images, labels = next(iter(loaders["train"]))
    print(f"    Batch shape : {images.shape}")
    print(f"    Pixel range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"    Labels      : {[EMOTION_LABELS[l.item()] for l in labels[:4]]}")

    print("\n✓ Method 1-3 Pipeline OK!")


def check_method4(fer_dir: str, rafdb_dir: str):
    print("\n" + "=" * 55)
    print("  Method 4: FER2013 + RAF-DB Pipeline Check")
    print("=" * 55)

    loaders, train_targets = get_combined_dataloaders(
        fer_dir=fer_dir,
        rafdb_dir=rafdb_dir,
        batch_size=64,
        val_split=0.1,
    )

    print("\n[4] ConvNeXt — RGB 192×192 + CLAHE + Mixup:")
    images, labels = next(iter(loaders["train"]))
    print(f"    Batch shape : {images.shape}")
    print(f"    Pixel range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"    Labels      : {[EMOTION_LABELS[l.item()] for l in labels[:4]]}")

    print("\n[Test set]")
    images_t, labels_t = next(iter(loaders["test"]))
    print(f"    Batch shape : {images_t.shape}")

    print("\n✓ Method 4 Pipeline OK!")


def main():
    FER_DIR   = "data/fer2013"
    RAFDB_DIR = "data/raf-db/DATASET"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    check_method123(FER_DIR)
    check_method4(FER_DIR, RAFDB_DIR)

    print("\n" + "=" * 55)
    print("  ✓ Tất cả pipeline OK — sẵn sàng train!")
    print("=" * 55)


if __name__ == "__main__":
    main()
