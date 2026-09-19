"""
plot_distribution.py — Vẽ biểu đồ phân phối class FER2013
Chạy: python plot_distribution.py --data data/archive
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.datasets as datasets
import torchvision.transforms as transforms

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_EMOJI  = ["😠", "🤢", "😨", "😊", "😐", "😢", "😲"]
COLORS = ["#E74C3C", "#2ECC71", "#9B59B6", "#F1C40F", "#3498DB", "#1ABC9C", "#E67E22"]


def count_classes(data_dir):
    tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((48, 48))])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=tf)

    train_counts = np.bincount(train_ds.targets, minlength=7)
    test_counts  = np.bincount(test_ds.targets,  minlength=7)
    return train_counts, test_counts


def plot_distribution(data_dir, save_path="class_distribution.png"):
    train_counts, test_counts = count_classes(data_dir)
    total = train_counts + test_counts

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0F1117")

    labels_with_emoji = [f"{e} {l}" for e, l in zip(EMOTION_EMOJI, EMOTION_LABELS)]
    x = np.arange(len(EMOTION_LABELS))

    # ── Plot 1: Train vs Test ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#1A1A2E")
    width = 0.35
    bars1 = ax1.bar(x - width/2, train_counts, width, label="Train",
                    color=[c + "CC" for c in COLORS], edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + width/2, test_counts,  width, label="Test",
                    color=COLORS, edgecolor="white", linewidth=0.5, alpha=0.7)

    # Số lượng trên mỗi bar
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom",
                 fontsize=8, color="white")
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom",
                 fontsize=8, color="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_with_emoji, rotation=15, ha="right",
                        fontsize=10, color="white")
    ax1.set_ylabel("Số lượng ảnh", color="white", fontsize=11)
    ax1.set_title("Phân phối Train / Test", color="white", fontsize=13, fontweight="bold")
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#444")
    ax1.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=10)
    ax1.set_ylim(0, max(train_counts) * 1.18)
    ax1.yaxis.set_tick_params(labelcolor="white")

    # ── Plot 2: Tổng + highlight mất cân bằng ─────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1A1A2E")
    bars = ax2.bar(x, total, color=COLORS, edgecolor="white", linewidth=0.5)

    # Highlight Disgust (class ít nhất)
    min_idx = np.argmin(total)
    max_idx = np.argmax(total)
    bars[min_idx].set_edgecolor("#FF4444")
    bars[min_idx].set_linewidth(2.5)
    bars[max_idx].set_edgecolor("#44FF44")
    bars[max_idx].set_linewidth(2.5)

    # Số lượng + phần trăm
    total_sum = total.sum()
    for i, bar in enumerate(bars):
        pct = total[i] / total_sum * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                 f"{int(bar.get_height()):,}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=8.5, color="white")

    # Annotation mất cân bằng
    ratio = total[max_idx] / total[min_idx]
    ax2.annotate(
        f"Mất cân bằng!\n{EMOTION_LABELS[max_idx]} gấp\n{ratio:.0f}× {EMOTION_LABELS[min_idx]}",
        xy=(min_idx, total[min_idx]),
        xytext=(min_idx + 2.5, total[min_idx] + 2500),
        fontsize=9, color="#FF6B6B",
        arrowprops=dict(arrowstyle="->", color="#FF6B6B", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2D1B1B", edgecolor="#FF4444"),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_with_emoji, rotation=15, ha="right",
                        fontsize=10, color="white")
    ax2.set_ylabel("Tổng số ảnh", color="white", fontsize=11)
    ax2.set_title("Phân phối tổng thể — Class Imbalance", color="white",
                  fontsize=13, fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444")
    ax2.set_ylim(0, max(total) * 1.25)
    ax2.yaxis.set_tick_params(labelcolor="white")

    # Legend cho highlight
    red_patch   = mpatches.Patch(edgecolor="#FF4444", facecolor="none",
                                  linewidth=2, label=f"Ít nhất: {EMOTION_LABELS[min_idx]}")
    green_patch = mpatches.Patch(edgecolor="#44FF44", facecolor="none",
                                  linewidth=2, label=f"Nhiều nhất: {EMOTION_LABELS[max_idx]}")
    ax2.legend(handles=[green_patch, red_patch], facecolor="#1A1A2E",
               labelcolor="white", fontsize=9)

    plt.suptitle("FER2013 — Phân phối Class", color="white",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0F1117", edgecolor="none")
    print(f"✓ Đã lưu: {save_path}")

    # In thống kê
    print("\n[Thống kê phân phối]")
    print(f"{'Class':12s} {'Train':>8s} {'Test':>7s} {'Total':>8s} {'%':>7s}")
    print("-" * 48)
    for i, name in enumerate(EMOTION_LABELS):
        print(f"{name:12s} {train_counts[i]:>8,} {test_counts[i]:>7,} "
              f"{total[i]:>8,} {total[i]/total_sum*100:>6.1f}%")
    print("-" * 48)
    print(f"{'Total':12s} {train_counts.sum():>8,} {test_counts.sum():>7,} "
          f"{total_sum:>8,} {'100.0%':>7s}")
    print(f"\nMất cân bằng: {EMOTION_LABELS[max_idx]} / {EMOTION_LABELS[min_idx]} "
          f"= {ratio:.1f}×")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Đường dẫn tới archive/ (chứa train/ và test/)")
    parser.add_argument("--save", default="class_distribution.png",
                        help="Đường dẫn lưu ảnh (mặc định: class_distribution.png)")
    args = parser.parse_args()
    plot_distribution(args.data, args.save)
