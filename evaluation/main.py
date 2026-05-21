"""
main.py — Điểm chạy chính của pipeline đánh giá mô hình Fuzzy-LSTM

Cách dùng:
    python main.py                          # chạy với dữ liệu mô phỏng
    python main.py --true  results/y_true.npy \
                   --pred  results/y_pred.npy \
                   --price data/tsla_test.csv  # chạy với dữ liệu thực
"""

import os
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config       import OUTPUT_DIR, LABEL_NAMES
from data_loader  import load_predictions
from metrics      import compute_metrics, plot_metrics_bar, plot_confusion_matrix, save_classification_report
from backtesting  import run_backtesting


# ── Tổng hợp kết quả → CSV ───────────────────────────────────────────────────

def save_summary(metrics, report_dict, backtest_results):
    save_path = os.path.join(OUTPUT_DIR, "5_evaluation_summary.csv")
    rows = []

    for k, v in metrics.items():
        rows.append({"Nhóm": "Metrics Tổng Quan", "Chỉ số": k, "Giá trị": round(v, 4)})

    for label in LABEL_NAMES:
        if label in report_dict:
            for metric in ["precision", "recall", "f1-score"]:
                rows.append({
                    "Nhóm":   f"Per-Class ({label})",
                    "Chỉ số": metric,
                    "Giá trị": round(report_dict[label][metric], 4),
                })

    for k, v in backtest_results.items():
        rows.append({"Nhóm": "Backtesting", "Chỉ số": k, "Giá trị": round(v, 4)})

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f" Đã lưu: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "=" * 55)
    print("  BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH FUZZY-LSTM")
    print("=" * 55)

    y_true, y_pred, prices = load_predictions(
        y_true_path  = args.true,
        y_pred_path  = args.pred,
        prices_path  = args.price,
    )

    print(f"\n  Tổng số mẫu test : {len(y_true)}")
    print(f"  Phân phối nhãn thực tế: "
          f"{dict(zip(LABEL_NAMES, [int(np.sum(y_true == i)) for i in range(5)]))}")

    print("\n  [1/4] Accuracy, Precision, Recall, F1...")
    metrics = compute_metrics(y_true, y_pred)
    plot_metrics_bar(metrics)

    print("\n  [2/4] Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred)

    print("\n  [3/4] Classification Report...")
    report_dict = save_classification_report(y_true, y_pred)

    print("\n  [4/4] Backtesting xu thế...")
    backtest_results = run_backtesting(y_true, y_pred, prices)

    print("\n  [5/5] Lưu tổng kết...")
    save_summary(metrics, report_dict, backtest_results)

    print("\n" + "=" * 55)
    print(f"  HOÀN THÀNH! Kết quả trong: ./{OUTPUT_DIR}/")
    print("  Các file đã tạo:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"     - {f:<45} ({size:,} bytes)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Fuzzy-LSTM")
    parser.add_argument("--true",  default=None, help="Đường dẫn file y_true.npy")
    parser.add_argument("--pred",  default=None, help="Đường dẫn file y_pred.npy")
    parser.add_argument("--price", default=None, help="Đường dẫn file giá CSV (cột: Date, Close)")
    args = parser.parse_args()
    main(args)
