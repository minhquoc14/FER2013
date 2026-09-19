"""
method1_lbp_svm.py — Phương pháp 1: LBP + SVM
Tiền xử lý: Resize 48x48 + Normalize histogram (density=True)
CLAHE bỏ vì làm giảm acc với LBP+SVM trên FER2013

Chạy: python method1_lbp_svm.py --data data/archive
"""

import os
import argparse
import time
import numpy as np
import joblib
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import EMOTION_LABELS
from utils import print_report, plot_confusion_matrix


LBP_RADIUS   = 2
LBP_N_POINTS = 16
LBP_METHOD   = "uniform"


def extract_lbp(image: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(image, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_features(X: np.ndarray, desc: str = "") -> np.ndarray:
    n, features = len(X), []
    for i, img in enumerate(X):
        features.append(extract_lbp(img))
        if (i + 1) % 2000 == 0 or (i + 1) == n:
            print(f"  [{desc}] {i+1}/{n} ảnh")
    return np.array(features)


def load_numpy_simple(data_dir: str, val_split: float = 0.1):
    """Load ảnh đơn giản: chỉ Grayscale + Resize 48x48, không CLAHE."""
    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
    ])

    def load_split(split_dir):
        ds = datasets.ImageFolder(split_dir, transform=tf)
        X, y = [], []
        for img, label in ds:
            X.append(np.array(img, dtype=np.uint8))
            y.append(label)
        return np.array(X), np.array(y)

    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    print("[numpy] Load train...")
    X_all, y_all = load_split(train_dir)
    idx = np.arange(len(y_all))
    train_idx, val_idx = train_test_split(idx, test_size=val_split, stratify=y_all, random_state=42)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val,   y_val   = X_all[val_idx],   y_all[val_idx]

    print("[numpy] Load test...")
    X_test, y_test = load_split(test_dir)
    print(f"[numpy] train={X_train.shape} | val={X_val.shape} | test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def main(data_dir: str):
    print("=" * 55)
    print("  Phương pháp 1: LBP + SVM")
    print("  Tiền xử lý: Grayscale + Resize + LBP histogram")
    print("=" * 55)

    X_train, y_train, X_val, y_val, X_test, y_test = load_numpy_simple(data_dir)

    print("\n[Phân phối class]")
    counts = np.bincount(y_train)
    for name, c in zip(EMOTION_LABELS.values(), counts):
        print(f"  {name:10s}: {c:5d} ({c/len(y_train)*100:.1f}%)")

    print("\n[Bước 1] Trích đặc trưng LBP...")
    t0 = time.time()
    X_train_feat = extract_features(X_train, "train")
    X_val_feat   = extract_features(X_val,   "val")
    X_test_feat  = extract_features(X_test,  "test")
    print(f"  Xong! {time.time()-t0:.1f}s | vector {X_train_feat.shape[1]} chiều")

    X_tv = np.vstack([X_train_feat, X_val_feat])
    y_tv = np.concatenate([y_train, y_val])

    print("\n[Bước 2] Train SVM (RBF)...")
    print("  Lưu ý: mất 5-15 phút trên CPU...")
    t1 = time.time()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf", C=10.0, gamma="scale",
            decision_function_shape="ovr",
            random_state=42, verbose=True,
        )),
    ])
    model.fit(X_tv, y_tv)
    train_time = time.time() - t1
    print(f"  Train xong! {train_time/60:.1f} phút")

    print("\n[Bước 3] Đánh giá...")
    y_pred = model.predict(X_test_feat)
    acc    = accuracy_score(y_test, y_pred)

    print_report(y_test, y_pred, method_name="LBP + SVM")
    os.makedirs("saved_models", exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, method_name="LBP + SVM",
                          save_path="saved_models/cm_lbp_svm.png")
    joblib.dump(model, "saved_models/model_lbp_svm.pkl")
    np.save("saved_models/result_lbp_svm.npy", {
        "accuracy": acc, "train_time_min": train_time/60,
        "y_pred": y_pred, "y_true": y_test,
    })
    print(f"\n  ✓ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  ✓ Model: saved_models/model_lbp_svm.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    main(args.data)