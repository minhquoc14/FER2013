"""
dataset.py — FER2013 + RAF-DB Pipeline (tất cả 4 method)
Hỗ trợ:
  - Method 1: LBP + SVM (numpy)
  - Method 2: CNN (grayscale 48×48)
  - Method 3: EfficientNet-B2 (RGB 224×224)
  - Method 4: ConvNeXt + CBAM (FER2013 + RAF-DB, RGB 192×192)

Cấu trúc thư mục:
    data/
    ├── fer2013/
    │   ├── train/ (angry, disgust, fear, happy, neutral, sad, surprise)
    │   └── test/
    └── raf-db/
        └── DATASET/
            ├── train/ (1=surprise,2=fear,3=disgust,4=happy,5=sad,6=angry,7=neutral)
            └── test/
"""

import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import (Dataset, DataLoader, ConcatDataset,
                               Subset, WeightedRandomSampler)
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# ── Nhãn cảm xúc ──────────────────────────────────────────────────────────────
EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear",  3: "Happy",
    4: "Neutral", 5: "Sad",   6: "Surprise",
}
NUM_CLASSES = len(EMOTION_LABELS)

# RAF-DB: subfolder số → index FER2013
RAFDB_TO_FER = {
    "1": 6,  # Surprise
    "2": 2,  # Fear
    "3": 1,  # Disgust
    "4": 3,  # Happy
    "5": 5,  # Sad
    "6": 0,  # Angry
    "7": 4,  # Neutral
}


# ── CLAHE ─────────────────────────────────────────────────────────────────────
class ApplyCLAHE:
    """
    Contrast Limited Adaptive Histogram Equalization.
    clip_limit=2.0: tăng contrast mà không amplify noise.
    tile_grid=(8,8): xử lý theo vùng 8×8 pixel.
    """
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        arr = self.clahe.apply(arr)
        return Image.fromarray(arr, mode="L")


# ── RAF-DB Custom Dataset ──────────────────────────────────────────────────────
class RAFDBDataset(Dataset):
    """
    Load RAF-DB với subfolder 1-7, remap label về index FER2013.
    Dùng cho Method 4.
    """
    def __init__(self, root_dir: str, transform=None):
        self.samples   = []
        self.transform = transform
        for folder_name, label in RAFDB_TO_FER.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((
                        os.path.join(folder_path, fname), label
                    ))
        print(f"  [RAF-DB] {root_dir} → {len(self.samples)} ảnh")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def targets(self): return [s[1] for s in self.samples]


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(method: str = "cnn"):
    """
    Trả về (train_transform, val_transform).

    method:
        "lbp"        → Grayscale + Resize 48×48 + CLAHE
        "cnn"        → Grayscale + Resize 48×48 + CLAHE + Aug + Normalize
        "pretrained" → Grayscale + Resize 224×224 + CLAHE + RGB + Aug + Normalize
        "convnext"   → Grayscale + CLAHE + RGB + Resize 192×192 + Aug nhẹ + Normalize
    """
    clahe = ApplyCLAHE(clip_limit=2.0, tile_grid=(8, 8))

    # ── Method 1: LBP + SVM ───────────────────────────────────────────────────
    if method == "lbp":
        tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((48, 48)),
            clahe,
        ])
        return tf, tf

    # ── Method 2: CNN ─────────────────────────────────────────────────────────
    elif method == "cnn":
        train_tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((48, 48)),
            clahe,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)
            ),
            transforms.ColorJitter(brightness=0.0, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507], std=[0.255]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
        ])
        val_tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((48, 48)),
            clahe,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507], std=[0.255]),
        ])
        return train_tf, val_tf

    # ── Method 3: EfficientNet-B2 ─────────────────────────────────────────────
    elif method == "pretrained":
        train_tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224, 224)),
            clahe,
            transforms.Grayscale(3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        val_tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((224, 224)),
            clahe,
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return train_tf, val_tf

    # ── Method 4: ConvNeXt + CBAM ─────────────────────────────────────────────
    elif method == "convnext":
        train_tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((192, 192)),
            clahe,
            transforms.Grayscale(3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        val_tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((192, 192)),
            clahe,
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return train_tf, val_tf

    else:
        raise ValueError(
            f"method phải là 'lbp', 'cnn', 'pretrained', hoặc 'convnext'. Nhận: {method}"
        )


# ── DataLoader Method 1-3 (FER2013 only) ──────────────────────────────────────
def get_dataloaders(data_dir: str, method: str = "cnn",
                    batch_size: int = 64, val_split: float = 0.1,
                    sampler=None):
    """
    DataLoader cho Method 1, 2, 3 — chỉ dùng FER2013.

    Returns:
        loaders   : dict {"train", "val", "test"}
        train_idx : indices của train set
        targets   : toàn bộ nhãn train folder
    """
    train_tf, val_tf = get_transforms(method)
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
        "train": DataLoader(
            train_subset, batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=0, pin_memory=False,
        ),
        "val": DataLoader(
            val_subset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=False,
        ),
        "test": DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=False,
        ),
    }

    print(f"[dataset] method={method} | "
          f"train={len(train_subset)} | val={len(val_subset)} | test={len(test_dataset)}")
    print(f"[dataset] class mapping: {full_train.class_to_idx}")
    return loaders, train_idx, targets


# ── DataLoader Method 4 (FER2013 + RAF-DB) ────────────────────────────────────
def get_combined_dataloaders(fer_dir: str, rafdb_dir: str,
                              batch_size: int = 64,
                              val_split: float = 0.1):
    """
    DataLoader cho Method 4 — gộp FER2013 + RAF-DB.
    Test set: FER2013 test only (benchmark chuẩn).

    Returns:
        loaders      : dict {"train", "val", "test"}
        train_targets: nhãn của train set sau split
    """
    train_tf, val_tf = get_transforms("convnext")

    # FER2013
    fer_train = datasets.ImageFolder(
        os.path.join(fer_dir, "train"), transform=train_tf
    )
    fer_test = datasets.ImageFolder(
        os.path.join(fer_dir, "test"), transform=val_tf
    )
    print(f"[FER2013] train={len(fer_train)} | test={len(fer_test)}")

    # RAF-DB
    raf_train = RAFDBDataset(
        os.path.join(rafdb_dir, "train"), transform=train_tf
    )

    # Gộp train
    combined_train = ConcatDataset([fer_train, raf_train])
    all_targets    = np.concatenate([
        np.array(fer_train.targets),
        np.array(raf_train.targets)
    ])

    print(f"\n[Combined Train] FER2013({len(fer_train)}) + RAF-DB({len(raf_train)}) = {len(combined_train)}")
    print(f"[Phân phối class]")
    counts = np.bincount(all_targets, minlength=NUM_CLASSES)
    for name, c in zip(EMOTION_LABELS.values(), counts):
        print(f"  {name:10s}: {c:6,d} ({c/len(all_targets)*100:.1f}%)")

    # Stratified split
    indices = np.arange(len(combined_train))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split,
        stratify=all_targets, random_state=42
    )

    # Val với val_tf
    fer_train_val = datasets.ImageFolder(
        os.path.join(fer_dir, "train"), transform=val_tf
    )
    raf_train_val = RAFDBDataset(
        os.path.join(rafdb_dir, "train"), transform=val_tf
    )
    combined_val = ConcatDataset([fer_train_val, raf_train_val])

    train_subset = Subset(combined_train, train_idx)
    val_subset   = Subset(combined_val,   val_idx)

    # WeightedRandomSampler
    train_targets = all_targets[train_idx]
    class_counts  = np.bincount(train_targets, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(class_weights[train_targets]),
        num_samples=len(train_idx),
        replacement=True,
    )

    loaders = {
        "train": DataLoader(
            train_subset, batch_size=batch_size,
            sampler=sampler,
            num_workers=2, pin_memory=True,
        ),
        "val": DataLoader(
            val_subset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        ),
        "test": DataLoader(
            fer_test, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        ),
    }

    print(f"\n[Test] FER2013 test only = {len(fer_test)} ảnh")
    print(f"[DataLoader] train={len(train_subset)} | val={len(val_subset)} | test={len(fer_test)}")
    return loaders, train_targets


# ── Numpy loader cho Method 1: LBP + SVM ──────────────────────────────────────
def load_numpy(data_dir: str, val_split: float = 0.1):
    """
    Đọc ảnh FER2013, áp dụng CLAHE, trả về numpy arrays cho LBP+SVM.
    """
    tf, _ = get_transforms("lbp")

    def load_split(split_dir):
        ds = datasets.ImageFolder(split_dir, transform=tf)
        X, y = [], []
        for img, label in ds:
            X.append(np.array(img, dtype=np.uint8))
            y.append(label)
        return np.array(X), np.array(y)

    print("[numpy] Load train + CLAHE...")
    X_all, y_all = load_split(os.path.join(data_dir, "train"))

    idx = np.arange(len(y_all))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_split, stratify=y_all, random_state=42
    )
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val,   y_val   = X_all[val_idx],   y_all[val_idx]

    print("[numpy] Load test + CLAHE...")
    X_test, y_test = load_split(os.path.join(data_dir, "test"))

    print(f"[numpy] train={X_train.shape} | val={X_val.shape} | test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test
