# 3_dataset.py — Tổ chức dữ liệu chuỗi thời gian cho LSTM
#
# Input:  pipeline_data/tsla_with_fuzzy.csv  (từ bước 2)
# Output: X_train/val/test.npy, y_train/val/test.npy  (pipeline_data/)
#         pipeline_data/scaler.pkl
#
# Quy trình:
#   1. Đọc tsla_with_fuzzy.csv
#   2. Chia train/val/test theo thời gian (70/15/15) TRƯỚC khi normalize
#   3. Fit MinMaxScaler CHỈ trên tập train (tránh data leakage)
#   4. Transform val/test với cùng scaler
#   5. Tạo chuỗi sliding window (SEQ_LEN=30) → X shape: (N, 30, 16)

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SEQ_LEN, FEATURE_COLS, SCALE_COLS,
    PRICE_COLS, INDICATOR_COLS, SENTIMENT_COL, FUZZY_COLS,
)

DATA_DIR = "pipeline_data"
IN_PATH  = os.path.join(DATA_DIR, "tsla_with_fuzzy.csv")

TRAIN_PCT = 0.70
VAL_PCT   = 0.15


def make_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int):
    """
    Tạo cặp (X, y) theo sliding window.
    X[i] = features[i : i+seq_len]  shape (seq_len, n_features)
    y[i] = labels[i + seq_len]
    """
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    print("\n" + "=" * 55)
    print("  [3/4] TẠO SEQUENCES CHO LSTM")
    print("=" * 55)

    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    print(f"  Đọc: {len(df):,} dòng từ {IN_PATH}")
    print(f"  Tổng đặc trưng: {len(FEATURE_COLS)} chiều = "
          f"{len(PRICE_COLS)} OHLCV + {len(INDICATOR_COLS)} indicators + "
          f"{len(SENTIMENT_COL)} sentiment + {len(FUZZY_COLS)} fuzzy_score")

    # ── Chia theo thời gian TRƯỚC khi normalize ──────────────────────────────
    n       = len(df)
    n_train = int(n * TRAIN_PCT)
    n_val   = int(n * VAL_PCT)
    n_test  = n - n_train - n_val

    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train : n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val :].copy()

    print(f"\n  Chia tập (theo thời gian):")
    print(f"     Train: {len(train_df):>5} phiên  "
          f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")
    print(f"     Val  : {len(val_df):>5} phiên  "
          f"({val_df['Date'].min().date()} → {val_df['Date'].max().date()})")
    print(f"     Test : {len(test_df):>5} phiên  "
          f"({test_df['Date'].min().date()} → {test_df['Date'].max().date()})")

    # ── Normalize: fit CHỈ trên train, transform val/test cùng scaler ────────
    # Chỉ scale PRICE_COLS + INDICATOR_COLS (10 cột).
    # sentiment và fuzzy_0..4 đã ở [0,1] nên không cần scale thêm.
    scaler = MinMaxScaler()
    train_df[SCALE_COLS] = scaler.fit_transform(train_df[SCALE_COLS])
    val_df[SCALE_COLS]   = scaler.transform(val_df[SCALE_COLS])
    test_df[SCALE_COLS]  = scaler.transform(test_df[SCALE_COLS])

    scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\n  Scaler fit trên train only → {scaler_path}")

    # ── Tạo sliding window sequences ─────────────────────────────────────────
    def get_xy(split_df: pd.DataFrame):
        feats  = split_df[FEATURE_COLS].values.astype(np.float32)
        labels = split_df["label"].values.astype(np.int64)
        return make_sequences(feats, labels, SEQ_LEN)

    X_train, y_train = get_xy(train_df)
    X_val,   y_val   = get_xy(val_df)
    X_test,  y_test  = get_xy(test_df)

    print(f"\n  Sequences (SEQ_LEN={SEQ_LEN}):")
    print(f"     X_train shape: {X_train.shape}   y_train: {y_train.shape}")
    print(f"     X_val   shape: {X_val.shape}   y_val:   {y_val.shape}")
    print(f"     X_test  shape: {X_test.shape}   y_test:  {y_test.shape}")

    assert X_train.shape[2] == len(FEATURE_COLS), \
        f"Số chiều đặc trưng sai: {X_train.shape[2]} ≠ {len(FEATURE_COLS)}"

    # ── Phân phối nhãn ───────────────────────────────────────────────────────
    label_names = ["Giảm", "Giữ", "Tăng"]
    print(f"\n  Phân phối nhãn (train / val / test):")
    for i, name in enumerate(label_names):
        tc = int((y_train == i).sum()); vc = int((y_val == i).sum()); sc = int((y_test == i).sum())
        print(f"     {name:<6}: train={tc:>4} ({tc/len(y_train)*100:.1f}%)  "
              f"val={vc:>4} ({vc/len(y_val)*100:.1f}%)  "
              f"test={sc:>4} ({sc/len(y_test)*100:.1f}%)")

    # ── Lưu ──────────────────────────────────────────────────────────────────
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
