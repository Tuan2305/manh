# 3_dataset.py — Tạo chuỗi thời gian (sequence) cho LSTM
#
# Input:  tsla_with_fuzzy.csv (từ bước 2)
# Output: X_train, X_val, X_test, y_train, y_val, y_test (.npy)

import pandas as pd
import numpy as np
import os

DATA_DIR  = "pipeline_data"
IN_PATH   = os.path.join(DATA_DIR, "tsla_with_fuzzy.csv")

SEQ_LEN   = 30       # dùng 30 phiên liên tiếp làm 1 mẫu đầu vào
TRAIN_PCT = 0.70     # 70% train
VAL_PCT   = 0.15     # 15% validation  → 15% test

# Các feature đưa vào LSTM: giá + chỉ báo kỹ thuật + fuzzy features
FEATURE_COLS = [
    "Close", "High", "Low", "Open", "Volume",
    "RSI", "ATR", "BB_width", "BB_pct",
    "sentiment", "fuzzy_label",            # đầu ra từ hệ thống mờ
]


def make_sequences(data: np.ndarray, labels: np.ndarray, seq_len: int):
    """Tạo cặp (X, y) dạng chuỗi trượt (sliding window)."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    print("\n" + "=" * 55)
    print("  [3/5] TẠO SEQUENCES CHO LSTM")
    print("=" * 55)

    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    print(f"  Đọc: {len(df):,} dòng")

    data   = df[FEATURE_COLS].values
    labels = df["label"].values.astype(int)

    X, y = make_sequences(data, labels, SEQ_LEN)
    print(f"  Tổng số sequences: {len(X):,}  shape: {X.shape}")

    # Chia tập
    n       = len(X)
    n_train = int(n * TRAIN_PCT)
    n_val   = int(n * VAL_PCT)

    X_train, y_train = X[:n_train],            y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],      y[n_train+n_val:]

    print(f"   Train     : {len(X_train):>5} mẫu")
    print(f"   Validation: {len(X_val):>5} mẫu")
    print(f"   Test      : {len(X_test):>5} mẫu")

    # Lưu
    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val",   X_val),   ("y_val",   y_val),
        ("X_test",  X_test),  ("y_test",  y_test),
    ]:
        path = os.path.join(DATA_DIR, f"{name}.npy")
        np.save(path, arr)
        print(f"  Đã lưu: {path}")

    print("=" * 55)


if __name__ == "__main__":
    main()
