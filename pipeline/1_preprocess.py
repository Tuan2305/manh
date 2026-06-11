# 1_preprocess.py — Tiền xử lý dữ liệu: tính RSI, ATR, Bollinger Bands + gán nhãn xu thế

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os, sys, warnings
warnings.filterwarnings("ignore")

# ── Cấu hình ─────────────────────────────────────────────────────────────────
# Cho phép override HORIZON qua biến môi trường (từ run_pipeline.py --horizon)
sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_CSV_PATH, HORIZON as _CFG_HORIZON

RAW_PATH = RAW_CSV_PATH
HORIZON  = int(os.environ.get("HORIZON", _CFG_HORIZON))   # env var ưu tiên hơn config
OUT_DIR  = "pipeline_data"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_NAMES = ["Giảm", "Giữ", "Tăng"]



# ── 1. Đọc file CSV ───────────────────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    """
    Tự động nhận diện 2 định dạng CSV từ Yahoo Finance:

    Định dạng A — có 3 dòng header:
        Price,Close,High,Low,Open,Volume
        Ticker,TSLA,TSLA,TSLA,TSLA,TSLA
        Date,,,,,
        2020-01-02,28.684,...

    Định dạng B — không có header:
        2020-01-02,28.684,...
    """
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().lower()
    skiprows = 3 if first_line.startswith("price") or first_line.startswith("ticker") else 0

    df = pd.read_csv(
        path,
        skiprows=skiprows,
        header=None,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Date"],
        dayfirst=False,
    )
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date", "Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Đọc xong: {len(df):,} phiên  ({df['Date'].min().date()} → {df['Date'].max().date()})")
    return df


# ── 2. Chỉ báo kỹ thuật ───────────────────────────────────────────────────────
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    ma              = df["Close"].rolling(period).mean()
    sd              = df["Close"].rolling(period).std()
    df["BB_upper"]  = ma + std * sd
    df["BB_lower"]  = ma - std * sd
    df["BB_width"]  = df["BB_upper"] - df["BB_lower"]
    df["BB_pct"]    = (df["Close"] - df["BB_lower"]) / (df["BB_width"] + 1e-9)
    return df


# ── 3. Gán nhãn xu thế theo return N ngày tới ────────────────────────────────
def add_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Tính return trong `horizon` phiên tiếp theo, chia thành 3 nhãn:
        0: Giảm  (< -1.5%)
        1: Giữ   (-1.5% ~ +1.5%)
        2: Tăng  (> +1.5%)
    Ngưỡng ±1.5% phù hợp với biến động 1 ngày của TSLA.
    """
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1

    conditions = [
        future_ret < -0.015,
        future_ret < 0.015,
    ]
    choices = [0, 1]
    df["label"]  = np.select(conditions, choices, default=2)
    df["future_return"] = future_ret
    return df


# ── 4. Chuẩn hóa features ────────────────────────────────────────────────────
FEATURE_COLS = ["Close", "High", "Low", "Open", "Volume",
                "RSI", "ATR", "BB_width", "BB_pct"]

def normalize(df: pd.DataFrame):
    scaler = MinMaxScaler()
    df_feat = df[FEATURE_COLS].copy()
    df[FEATURE_COLS] = scaler.fit_transform(df_feat)
    return df, scaler


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    from config import HORIZON_LABEL
    horizon_desc = HORIZON_LABEL.get(HORIZON, f"{HORIZON} phiên")
    print("\n" + "=" * 55)
    print("  [1/5] TIỀN XỬ LÝ DỮ LIỆU")
    print(f"  Khung thời gian: {horizon_desc}")
    print("=" * 55)

    df = load_raw(RAW_PATH)

    print("   Tính RSI, ATR, Bollinger Bands...")
    df = add_rsi(df)
    df = add_atr(df)
    df = add_bollinger(df)
    df = add_labels(df, horizon=HORIZON)

    # Bỏ các dòng NaN do rolling window
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df, scaler = normalize(df)

    # Lưu
    out_path = os.path.join(OUT_DIR, "tsla_processed.csv")
    df.to_csv(out_path, index=False)

    print(f"  Đã lưu: {out_path}  ({len(df):,} dòng)")
    print(f"  Phân phối nhãn:")
    for i, name in enumerate(LABEL_NAMES):
        cnt = int((df["label"] == i).sum())
        pct = cnt / len(df) * 100
        print(f"     {name:<15} {cnt:>5} phiên  ({pct:.1f}%)")

    import joblib
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    print(f"  Đã lưu scaler: {OUT_DIR}/scaler.pkl")
    print("=" * 55)


if __name__ == "__main__":
    main()
