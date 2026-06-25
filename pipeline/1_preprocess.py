# 1_preprocess.py — Tiền xử lý dữ liệu: tính RSI, ATR, Bollinger Bands + gán nhãn xu thế
#
# Input:  TSLA.csv (dữ liệu ngày từ Yahoo Finance)
# Output: pipeline_data/tsla_processed.csv  (raw values, chưa normalize)
#
# Lưu ý: KHÔNG normalize ở bước này.
# MinMaxScaler chỉ được fit trên tập train ở bước 3_dataset.py để tránh data leakage.

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_CSV_PATH, HORIZON as _CFG_HORIZON, THRESHOLD, HORIZON_LABEL

HORIZON  = int(os.environ.get("HORIZON", _CFG_HORIZON))
OUT_DIR  = "pipeline_data"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_NAMES = ["Giảm", "Giữ", "Tăng"]


# ── 1. Đọc file CSV ───────────────────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    """
    Nhận diện tự động 2 định dạng CSV Yahoo Finance:
      Định dạng A (3 dòng header): Price/Ticker/Date
      Định dạng B (không header): 2010-06-29,...
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
    df.dropna(subset=["Date", "Close", "High", "Low", "Open", "Volume"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Đọc xong: {len(df):,} phiên  ({df['Date'].min().date()} → {df['Date'].max().date()})")
    return df


# ── 2. Chỉ báo kỹ thuật ───────────────────────────────────────────────────────
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI(14) — momentum chỉ số sức mạnh tương đối, range [0, 100]."""
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR(14) — Average True Range, đo biến động giá tuyệt đối ($)."""
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD Histogram = MACD - Signal. Phân biệt tốt nhất giữa lớp Giảm và các lớp khác."""
    ema_fast       = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow       = df["Close"].ewm(span=slow, adjust=False).mean()
    macd           = ema_fast - ema_slow
    signal_line    = macd.ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = macd - signal_line
    return df


def add_sentiment_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy cảm xúc thị trường từ momentum giá 5 ngày — thay thế Gemini API.
    ret_5d = (Close_t / Close_{t-5}) - 1
    Chuẩn hóa: -15% → 0.0 (tiêu cực), 0% → 0.5 (trung lập), +15% → 1.0 (tích cực)
    std ≈ 0.23 → có biến thiên thực sự, khác với fallback 0.5 bất biến trước đây.
    """
    ret_5d         = df["Close"].pct_change(5)
    df["sentiment"] = ((ret_5d.clip(-0.15, 0.15) + 0.15) / 0.30).fillna(0.5)
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands(20, 2):
      BB_upper  = MA20 + 2×std
      BB_middle = MA20
      BB_lower  = MA20 - 2×std
      BB_position = (Close - BB_lower) / (BB_upper - BB_lower)  ∈ [0,1] thông thường
    """
    ma = df["Close"].rolling(period).mean()
    sd = df["Close"].rolling(period).std()
    df["BB_upper"]    = ma + k * sd
    df["BB_middle"]   = ma
    df["BB_lower"]    = ma - k * sd
    width             = df["BB_upper"] - df["BB_lower"]
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (width + 1e-9)
    return df


# ── 3. Gán nhãn xu thế ───────────────────────────────────────────────────────
def add_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    """
    Tính % thay đổi giá trong `horizon` phiên tiếp theo.
    Ngưỡng θ = threshold (mặc định ±1.5%):
      label = 0 (Giảm)  nếu return < -θ
      label = 1 (Giữ)   nếu -θ ≤ return ≤ θ
      label = 2 (Tăng)  nếu return > θ
    """
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    conditions = [future_ret < -threshold, future_ret < threshold]
    choices    = [0, 1]
    df["label"]         = np.select(conditions, choices, default=2)
    df["future_return"] = future_ret
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    horizon_desc = HORIZON_LABEL.get(HORIZON, f"{HORIZON} phiên")
    print("\n" + "=" * 55)
    print("  [1/4] TIỀN XỬ LÝ DỮ LIỆU")
    print(f"  Khung thời gian: {horizon_desc}  |  θ = ±{THRESHOLD*100:.1f}%")
    print("=" * 55)

    df = load_raw(RAW_CSV_PATH)

    print("  Tinh RSI(14), ATR(14), Bollinger Bands(20), MACD(12/26/9), sentiment proxy...")
    df = add_rsi(df)
    df = add_atr(df)
    df = add_bollinger(df)
    df = add_macd(df)
    df = add_sentiment_proxy(df)
    df = add_labels(df, horizon=HORIZON, threshold=THRESHOLD)

    # Bỏ NaN do rolling window (max window = 20 của BB)
    before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Bỏ {before - len(df)} dòng NaN (rolling window) → còn {len(df):,} phiên")

    # Lưu với raw values (chưa normalize)
    out_path = os.path.join(OUT_DIR, "tsla_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"  Đã lưu: {out_path}")

    print(f"\n  Cột đầu ra: {list(df.columns)}")
    print(f"\n  Phân phối nhãn (θ=±{THRESHOLD*100:.1f}%, horizon={HORIZON}):")
    for i, name in enumerate(LABEL_NAMES):
        cnt = int((df["label"] == i).sum())
        pct = cnt / len(df) * 100
        print(f"     {name:<8} {cnt:>5} phiên  ({pct:.1f}%)")

    print("=" * 55)


if __name__ == "__main__":
    main()
