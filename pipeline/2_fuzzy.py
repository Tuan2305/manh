# 2_fuzzy.py — Hệ thống suy luận mờ (Fuzzy Inference System)
#
# Input:  pipeline_data/tsla_processed.csv  (từ bước 1, raw values)
# Output: pipeline_data/tsla_with_fuzzy.csv (thêm sentiment + fuzzy_label + fuzzy_0..4)
#
# Quy trình:
#   1. Lấy tin tức TSLA từ yfinance + phân tích cảm xúc bằng Gemini API (có cache)
#   2. Chạy hệ suy luận mờ IF-THEN với 4 biến đầu vào:
#        RSI (0-100), ATR_pct = ATR/Close*100 (%), BB_position (0-1), sentiment (0-1)
#   3. Sinh ra fuzzy_label (0-4) và fuzzy_score (vector 5 chiều, tổng = 1)

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = "pipeline_data"
IN_PATH  = os.path.join(DATA_DIR, "tsla_processed.csv")
OUT_PATH = os.path.join(DATA_DIR, "tsla_with_fuzzy.csv")

TREND_NAMES = ["Giam manh", "Giam nhe", "Binh thuong", "Tang nhe", "Tang manh"]


# ── Hàm thành viên (Membership Functions) ────────────────────────────────────

def trimf(x: float, a: float, b: float, c: float) -> float:
    """Hàm tam giác: tăng tuyến tính a→b, giảm tuyến tính b→c."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a + 1e-9)
    return (c - x) / (c - b + 1e-9)


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Hàm hình thang: a→b tăng, b→c bằng 1, c→d giảm."""
    if x <= a or x >= d:
        return 0.0
    if x <= b:
        return (x - a) / (b - a + 1e-9)
    if x <= c:
        return 1.0
    return (d - x) / (d - c + 1e-9)


# ── Fuzzify 4 biến đầu vào ────────────────────────────────────────────────────

def fuzzify_rsi(r: float) -> dict:
    """RSI thật trong [0, 100]. Cửa sổ 14 phiên."""
    return {
        "oversold":   trapmf(r,  0,  0, 30, 45),
        "neutral":    trimf( r, 30, 50, 70),
        "overbought": trapmf(r, 55, 70, 100, 100),
    }


def fuzzify_atr_pct(a: float) -> dict:
    """ATR_pct = ATR/Close*100 (%). TSLA thường: low<3%, medium 3-8%, high>7%."""
    return {
        "low":    trapmf(a, 0,  0,  2,  4),
        "medium": trimf( a, 3,  5,  8),
        "high":   trapmf(a, 6, 10, 20, 20),
    }


def fuzzify_bb_position(b: float) -> dict:
    """BB_position = (Close - BB_lower)/(BB_upper - BB_lower) ∈ [0,1]."""
    return {
        "lower":  trapmf(b, 0.0, 0.0, 0.35, 0.50),
        "middle": trimf( b, 0.30, 0.50, 0.70),
        "upper":  trapmf(b, 0.55, 0.65, 1.0, 1.0),
    }


def fuzzify_sentiment(s: float) -> dict:
    """sentiment ∈ [0,1]: 0=tiêu cực, 0.5=trung lập, 1=tích cực."""
    return {
        "negative": trapmf(s, 0.0, 0.0, 0.30, 0.50),
        "neutral":  trimf( s, 0.25, 0.50, 0.75),
        "positive": trapmf(s, 0.45, 0.65, 1.0, 1.0),
    }


# ── Bộ luật IF-THEN ───────────────────────────────────────────────────────────
# Mỗi luật: (sentiment_key, rsi_key, atr_key, bb_key) → label
# Label: 0=Giảm mạnh, 1=Giảm nhẹ, 2=Bình thường, 3=Tăng nhẹ, 4=Tăng mạnh

RULES = [
    # ── Tăng mạnh (4) ─────────────────────────────────────────────────────────
    # sentiment=pos AND RSI=overbought AND BB=upper → tín hiệu tăng mạnh nhất (Rule 1)
    ("positive", "overbought", "high",   "upper",  4),
    ("positive", "overbought", "medium", "upper",  4),
    ("positive", "neutral",    "high",   "upper",  4),
    ("positive", "overbought", "low",    "upper",  4),

    # ── Tăng nhẹ (3) ──────────────────────────────────────────────────────────
    # sentiment=pos AND ATR=high AND RSI=neutral (Rule 4 từ báo cáo)
    ("positive", "neutral",    "high",   "middle", 3),
    ("positive", "neutral",    "medium", "upper",  3),
    ("positive", "neutral",    "medium", "middle", 3),
    ("positive", "neutral",    "low",    "upper",  3),
    ("positive", "neutral",    "low",    "middle", 3),
    # Oversold + positive → kỳ vọng phục hồi
    ("positive", "oversold",   "high",   "lower",  3),
    ("positive", "oversold",   "medium", "lower",  3),
    ("positive", "oversold",   "low",    "lower",  3),
    ("positive", "oversold",   "medium", "middle", 3),
    # Neutral sentiment + oversold + BB lower → tín hiệu phục hồi kỹ thuật
    ("neutral",  "oversold",   "medium", "lower",  3),
    ("neutral",  "oversold",   "low",    "lower",  3),

    # ── Bình thường (2) ───────────────────────────────────────────────────────
    # sentiment=neutral AND ATR=low AND RSI=neutral (Rule 3 từ báo cáo)
    ("neutral",  "neutral",    "low",    "middle", 2),
    ("neutral",  "neutral",    "medium", "middle", 2),
    ("neutral",  "neutral",    "high",   "middle", 2),
    ("neutral",  "neutral",    "low",    "lower",  2),
    ("neutral",  "neutral",    "low",    "upper",  2),
    # negative + oversold → mixed signal, không rõ xu hướng
    ("negative", "oversold",   "low",    "lower",  2),
    ("negative", "oversold",   "low",    "middle", 2),
    # positive + oversold + high atr + overbought → quá mua, có thể điều chỉnh
    ("positive", "overbought", "high",   "middle", 2),

    # ── Giảm nhẹ (1) ──────────────────────────────────────────────────────────
    # sentiment=neg AND ATR=high AND RSI=neutral (Rule 5 từ báo cáo)
    ("negative", "neutral",    "high",   "middle", 1),
    ("negative", "neutral",    "medium", "middle", 1),
    ("negative", "neutral",    "low",    "middle", 1),
    ("negative", "neutral",    "medium", "upper",  1),
    ("negative", "neutral",    "low",    "upper",  1),
    # Overbought + neutral sentiment → pullback nhẹ
    ("neutral",  "overbought", "medium", "upper",  1),
    ("neutral",  "overbought", "low",    "upper",  1),
    ("neutral",  "overbought", "medium", "middle", 1),
    # negative + oversold + atr medium/high → downtrend chưa đảo chiều
    ("negative", "oversold",   "medium", "lower",  1),
    ("negative", "oversold",   "high",   "lower",  1),

    # ── Giảm mạnh (0) ─────────────────────────────────────────────────────────
    # sentiment=neg AND RSI=oversold AND BB=lower (Rule 2 từ báo cáo)
    ("negative", "oversold",   "high",   "middle", 0),
    ("negative", "neutral",    "high",   "lower",  0),
    ("negative", "overbought", "high",   "upper",  0),
    ("negative", "overbought", "medium", "upper",  0),
    ("negative", "overbought", "high",   "middle", 0),
    ("negative", "neutral",    "high",   "upper",  0),
]


# ── Hàm suy luận mờ chính ────────────────────────────────────────────────────

def fuzzy_inference(rsi: float, atr_pct: float, bb_pos: float,
                    sentiment: float) -> tuple:
    """
    Chạy suy luận mờ IF-THEN với 4 biến đầu vào.

    Returns:
        fuzzy_label (int 0-4): nhãn có firing strength cao nhất
        fuzzy_score (list[5]): vector mức kích hoạt đã chuẩn hóa (tổng = 1)
    """
    fr = fuzzify_rsi(rsi)
    fa = fuzzify_atr_pct(atr_pct)
    fb = fuzzify_bb_position(bb_pos)
    fs = fuzzify_sentiment(sentiment)

    firing = np.zeros(5)
    for s_key, r_key, a_key, b_key, label in RULES:
        strength = min(fs[s_key], fr[r_key], fa[a_key], fb[b_key])   # AND = min
        firing[label] = max(firing[label], strength)                   # OR  = max

    # Khi không có luật nào kích hoạt → mặc định Bình thường (label=2)
    if firing.max() == 0:
        firing[2] = 1.0

    fuzzy_label = int(np.argmax(firing))
    total = firing.sum()
    fuzzy_score = (firing / total).tolist()   # chuẩn hóa → tổng = 1

    return fuzzy_label, fuzzy_score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  [2/4] HE THONG SUY LUAN MO (FUZZY INFERENCE)")
    print("=" * 55)

    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    print(f"  Doc: {len(df):,} dong tu {IN_PATH}")

    # sentiment da duoc tinh o buoc 1 tu momentum gia 5 ngay
    if "sentiment" not in df.columns:
        raise ValueError("Khong tim thay cot 'sentiment' trong file. Chay lai buoc 1 truoc.")

    sent_mean = df["sentiment"].mean()
    sent_std  = df["sentiment"].std()
    print(f"  Sentiment proxy: mean={sent_mean:.3f}  std={sent_std:.3f}")

    # Tinh ATR_pct = ATR/Close * 100 (%) de fuzzify
    df["ATR_pct"] = df["ATR"] / (df["Close"] + 1e-9) * 100.0

    print(f"\n  Suy luan mo IF-THEN ({len(RULES)} luat, 4 bien dau vao)...")
    fuzzy_labels = []
    fuzzy_scores = []

    for _, row in df.iterrows():
        label, score = fuzzy_inference(
            rsi       = float(row["RSI"]),
            atr_pct   = float(row["ATR_pct"]),
            bb_pos    = float(row["BB_position"]),
            sentiment = float(row["sentiment"]),
        )
        fuzzy_labels.append(label)
        fuzzy_scores.append(score)

    df["fuzzy_label"] = fuzzy_labels

    scores_arr = np.array(fuzzy_scores)
    for i in range(5):
        df[f"fuzzy_{i}"] = scores_arr[:, i]

    df.drop(columns=["ATR_pct"], inplace=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"  Da luu: {OUT_PATH}")

    print(f"\n  Phan phoi fuzzy_label:")
    for i, name in enumerate(TREND_NAMES):
        cnt = int((df["fuzzy_label"] == i).sum())
        pct = cnt / len(df) * 100
        print(f"     {name:<15} {cnt:>5} phien  ({pct:.1f}%)")

    score_sum = scores_arr.sum(axis=1)
    print(f"\n  fuzzy_score tong moi hang: Min={score_sum.min():.4f}  Max={score_sum.max():.4f}  Mean={score_sum.mean():.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
