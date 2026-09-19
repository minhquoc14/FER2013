"""
plot_results.py — Vẽ biểu đồ so sánh kết quả 5 phương pháp FER
Đọc từ saved_models/*.npy

Chạy: python plot_results.py

Cấu trúc file .npy cần có:
  - accuracy       : float
  - train_time_min : float
  - y_pred         : np.array
  - y_true         : np.array
  - history        : dict (train_loss, val_loss, train_acc, val_acc) — không bắt buộc

Method 4 (EfficientNet+CBAM) dùng result_cbam.npy
Method 5 (ConvNeXt+CBAM)     dùng result_convnext_cbam.npy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ── Màu sắc 5 phương pháp ─────────────────────────────────────────────────────
COLORS = {
    "LBP + SVM":              "#888780",
    "CNN":                    "#534AB7",
    "EfficientNet-B2":        "#1D9E75",
    "EfficientNet-B2 + CBAM": "#E07B39",
    "ConvNeXt + CBAM":        "#D64045",
}

os.makedirs("results_plots", exist_ok=True)


# ── Load kết quả ──────────────────────────────────────────────────────────────
def load_results():
    results = {}

    files = {
        "LBP + SVM":              "saved_models/result_lbp_svm.npy",
        "CNN":                    "saved_models/result_cnn.npy",
        "EfficientNet-B2":        "saved_models/result_pretrained.npy",
        "EfficientNet-B2 + CBAM": "saved_models/result_cbam.npy",
        "ConvNeXt + CBAM":        "saved_models/result_convnext_cbam.npy",
    }

    print("=" * 55)
    print("  Load kết quả 5 phương pháp")
    print("=" * 55)

    for method, path in files.items():
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True).item()
            # Chuẩn hóa key accuracy_standard → accuracy
            if "accuracy_standard" in d and "accuracy" not in d:
                d["accuracy"] = d["accuracy_standard"]
            results[method] = d
            acc  = d.get("accuracy", 0) * 100
            time = d.get("train_time_min", 0)
            print(f"  ✓ {method:25s}: {acc:.2f}% | {time:.1f} phút")
        else:
            print(f"  ✗ {method:25s}: Không tìm thấy {path}")

    return results


# ── Tổng quan: Accuracy + Train time ─────────────────────────────────────────
def plot_overview(results):
    methods     = list(results.keys())
    accuracies  = [results[m]["accuracy"] * 100 for m in methods]
    train_times = [results[m].get("train_time_min", 0) for m in methods]
    colors      = [COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Kết quả tổng quan — 5 phương pháp FER2013",
                 fontsize=16, fontweight="bold", y=1.02)

    # Accuracy
    ax1 = axes[0]
    bars = ax1.bar(methods, accuracies, color=colors,
                   edgecolor="white", linewidth=0.8, width=0.55, zorder=3)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("So sánh Accuracy", fontsize=13, fontweight="bold")
    ax1.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.2,
                 f"{acc:.2f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Train time
    ax2 = axes[1]
    bars2 = ax2.bar(methods, train_times, color=colors,
                    edgecolor="white", linewidth=0.8, width=0.55, zorder=3)
    ax2.set_ylabel("Thời gian train (phút)", fontsize=12)
    ax2.set_title("Thời gian huấn luyện", fontsize=13, fontweight="bold")
    ax2.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    for bar, t in zip(bars2, train_times):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f"{t:.1f} phút",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = "results_plots/overview_5methods.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n✓ Saved: {path}")
    plt.show()


# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, method_name, save_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS,
                ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"Confusion Matrix — {method_name}",
                 fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {save_path}")
    plt.show()


# ── Training curve ────────────────────────────────────────────────────────────
def plot_training_curve(history, method_name, color, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Training History — {method_name}",
                 fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], label="Train", color=color, linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color=color,
                 linewidth=2, linestyle="--", alpha=0.7)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)
    axes[0].yaxis.grid(True, color="#e0e0e0"); axes[0].set_axisbelow(True)

    axes[1].plot(epochs, history["train_acc"], label="Train", color=color, linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color=color,
                 linewidth=2, linestyle="--", alpha=0.7)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend()
    axes[1].spines[["top","right"]].set_visible(False)
    axes[1].yaxis.grid(True, color="#e0e0e0"); axes[1].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {save_path}")
    plt.show()


# ── Per-class F1 so sánh 5 method ────────────────────────────────────────────
def plot_per_class_f1(results):
    methods = list(results.keys())
    n       = len(methods)
    x       = np.arange(len(EMOTION_LABELS))
    width   = 0.15
    offsets = np.linspace(-(n-1)/2*width, (n-1)/2*width, n)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    for i, method in enumerate(methods):
        d  = results[method]
        f1 = f1_score(d["y_true"], d["y_pred"], average=None, zero_division=0)
        ax.bar(x + offsets[i], f1 * 100, width,
               label=method, color=COLORS[method],
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, fontsize=11)
    ax.set_ylabel("F1-score (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Per-class F1-score — 5 phương pháp",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = "results_plots/perclass_f1_5methods.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results = load_results()

    if not results:
        print("\n[!] Không tìm thấy file kết quả nào!")
        return

    # Tổng quan
    print("\n[1] Vẽ tổng quan accuracy + train time...")
    plot_overview(results)

    # Confusion matrix + training curve từng method
    for method, d in results.items():
        safe_name = method.lower().replace(" ", "_").replace("+", "").replace("-", "")
        print(f"\n[2] Confusion Matrix — {method}...")
        plot_confusion_matrix(
            d["y_true"], d["y_pred"], method,
            f"results_plots/cm_{safe_name}.png"
        )
        if "history" in d and method != "LBP + SVM":
            print(f"[3] Training Curve — {method}...")
            plot_training_curve(
                d["history"], method,
                COLORS[method],
                f"results_plots/history_{safe_name}.png"
            )

    # Per-class F1
    if len(results) > 1:
        print("\n[4] Per-class F1-score so sánh 5 method...")
        plot_per_class_f1(results)

    print("\n✓ Tất cả biểu đồ đã lưu vào: results_plots/")


if __name__ == "__main__":
    main()