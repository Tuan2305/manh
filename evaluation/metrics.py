# metrics.py — Accuracy / Precision / Recall / F1 / Confusion Matrix / Classification Report

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from config import OUTPUT_DIR, LABEL_NAMES


# ── 1. Metrics tổng quan ──────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Tính Accuracy, Precision, Recall, F1 (macro + weighted). In + trả về dict."""
    metrics = {
        "Accuracy":             accuracy_score(y_true, y_pred),
        "Precision (macro)":    precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "Recall (macro)":       recall_score(   y_true, y_pred, average="macro",    zero_division=0),
        "F1-Score (macro)":     f1_score(       y_true, y_pred, average="macro",    zero_division=0),
        "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)":    recall_score(   y_true, y_pred, average="weighted", zero_division=0),
        "F1-Score (weighted)":  f1_score(       y_true, y_pred, average="weighted", zero_division=0),
    }

    print("\n" + "=" * 55)
    print("  BẢNG ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH FUZZY-LSTM")
    print("=" * 55)
    for name, val in metrics.items():
        bar = "█" * int(val * 30)
        print(f"  {name:<25} {val:.4f}  {bar}")
    print("=" * 55)

    return metrics


def plot_metrics_bar(metrics):
    """Vẽ và lưu biểu đồ thanh ngang cho các chỉ số hiệu suất."""
    save_path = os.path.join(OUTPUT_DIR, "1_metrics_bar.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    names  = list(metrics.keys())
    values = list(metrics.values())
    colors = [
        "#4fc3f7" if "Accuracy"  in n else
        "#81c784" if "Precision" in n else
        "#ffb74d" if "Recall"    in n else
        "#f06292"
        for n in names
    ]

    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.6)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                color="white", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Giá trị", color="white", fontsize=11)
    ax.set_title("Bảng Đánh Giá Hiệu Suất Mô Hình Fuzzy-LSTM",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    for label in ax.get_yticklabels():
        label.set_color("white")
    ax.axvline(x=0.7, color="#555", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0.701, -0.6, "Ngưỡng 0.70", color="#888", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Đã lưu: {save_path}")


# ── 2. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    """Vẽ confusion matrix dạng số lượng và phần trăm."""
    save_path = os.path.join(OUTPUT_DIR, "2_confusion_matrix.png")

    cm     = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Ma Trận Nhầm Lẫn (Confusion Matrix)",
                 color="white", fontsize=14, fontweight="bold", y=1.01)

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Số lượng mẫu", "Phần trăm (%)"],
    ):
        ax.set_facecolor("#1a1d27")
        sns.heatmap(data, annot=True, fmt=fmt,
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                    cmap="RdYlGn", linewidths=0.5, linecolor="#333", ax=ax,
                    annot_kws={"size": 10, "weight": "bold"})
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.set_xlabel("Nhãn Dự Đoán", color="white", fontsize=10)
        ax.set_ylabel("Nhãn Thực Tế",  color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Đã lưu: {save_path}")
    return cm


# ── 3. Classification Report ──────────────────────────────────────────────────

def save_classification_report(y_true, y_pred):
    """Lưu classification report dạng .txt và heatmap .png."""
    path_txt = os.path.join(OUTPUT_DIR, "3_classification_report.txt")
    path_img = os.path.join(OUTPUT_DIR, "3_classification_report.png")

    report_str  = classification_report(y_true, y_pred, target_names=LABEL_NAMES,
                                        zero_division=0, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=LABEL_NAMES,
                                        zero_division=0, output_dict=True)

    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    print(report_str)

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT - MÔ HÌNH FUZZY-LSTM\n")
        f.write("=" * 55 + "\n")
        f.write(report_str)
    print(f"  Đã lưu: {path_txt}")

    # Heatmap
    rows = LABEL_NAMES + ["macro avg", "weighted avg"]
    cols = ["precision", "recall", "f1-score"]
    data = [
        [report_dict[r][c] if r in report_dict else 0 for c in cols]
        for r in rows
    ]
    import pandas as pd
    df = pd.DataFrame(data, index=rows, columns=["Precision", "Recall", "F1-Score"])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    sns.heatmap(df, annot=True, fmt=".4f", cmap="Blues",
                linewidths=0.5, linecolor="#333", ax=ax,
                annot_kws={"size": 10, "weight": "bold"}, vmin=0, vmax=1)
    ax.set_title("Classification Report - Mô Hình Fuzzy-LSTM",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(path_img, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Đã lưu: {path_img}")

    return report_dict
