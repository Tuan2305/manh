# 2_fuzzy.py — Hệ thống suy luận mờ (Fuzzy Inference System)
#
# Input:  tsla_processed.csv (từ bước 1)
# Output: tsla_with_fuzzy.csv (thêm cột sentiment, fuzzy_label)
#
# Quy trình:
#   1. Lấy tin tức TSLA từ yfinance (có cache để tránh gọi lại)
#   2. Phân tích cảm xúc (sentiment) bằng Gemini API
#   3. Chạy hệ thống suy luận mờ IF-THEN để gán fuzzy_label (0-4)

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from google import genai

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import GEMINI_API_KEY

DATA_DIR        = "pipeline_data"
IN_PATH         = os.path.join(DATA_DIR, "tsla_processed.csv")
OUT_PATH        = os.path.join(DATA_DIR, "tsla_with_fuzzy.csv")
NEWS_CACHE_PATH = os.path.join(DATA_DIR, "news_sentiment_cache.csv")

TREND_NAMES = ["Giảm mạnh", "Giảm nhẹ", "Bình thường", "Tăng nhẹ", "Tăng mạnh"]

# ── Gemini setup ──────────────────────────────────────────────────────────────
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL   = "gemini-2.0-flash"


# ── Hàm thành viên (Membership Functions) ────────────────────────────────────

def trimf(x: float, a: float, b: float, c: float) -> float:
    """Hàm tam giác: tăng từ a→b, giảm từ b→c."""
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


# ── Fuzzify các biến đầu vào ──────────────────────────────────────────────────

def fuzzify_sentiment(s: float) -> dict:
    """s ∈ [0, 1]: 0 = tiêu cực, 0.5 = trung lập, 1 = tích cực."""
    return {
        "negative": trapmf(s, 0.0, 0.0, 0.30, 0.50),
        "neutral":  trimf( s, 0.25, 0.5, 0.75),
        "positive": trapmf(s, 0.45, 0.60, 1.0, 1.0),
    }


def fuzzify_volume(v: float) -> dict:
    """v ∈ [0, 1] sau MinMax scale."""
    return {
        "low":    trapmf(v, 0.0, 0.0, 0.20, 0.40),
        "medium": trimf( v, 0.25, 0.5, 0.75),
        "high":   trapmf(v, 0.45, 0.60, 1.0, 1.0),
    }


def fuzzify_rsi(r: float) -> dict:
    """r ∈ [0, 1] sau MinMax scale (tương ứng RSI 0-100)."""
    return {
        "oversold":   trapmf(r, 0.0, 0.0, 0.30, 0.45),
        "neutral":    trimf( r, 0.30, 0.5, 0.70),
        "overbought": trapmf(r, 0.55, 0.70, 1.0, 1.0),
    }


# ── Luật suy luận mờ IF-THEN ──────────────────────────────────────────────────
#
# Mỗi luật: (sentiment_key, volume_key, rsi_key) → trend_label
# Label: 0=Giảm mạnh, 1=Giảm nhẹ, 2=Bình thường, 3=Tăng nhẹ, 4=Tăng mạnh

RULES = [
    # --- Cảm xúc Tích cực ---
    ("positive", "high",   "overbought", 4),
    ("positive", "high",   "neutral",    4),
    ("positive", "medium", "overbought", 4),
    ("positive", "medium", "neutral",    3),
    ("positive", "low",    "neutral",    3),
    ("positive", "medium", "oversold",   3),
    ("positive", "high",   "oversold",   3),
    ("positive", "low",    "overbought", 3),
    ("positive", "low",    "oversold",   3),

    # --- Cảm xúc Trung lập ---
    ("neutral",  "medium", "neutral",    2),
    ("neutral",  "low",    "neutral",    2),
    ("neutral",  "high",   "neutral",    2),
    ("neutral",  "low",    "oversold",   3),
    ("neutral",  "medium", "oversold",   3),
    ("neutral",  "high",   "oversold",   2),
    ("neutral",  "low",    "overbought", 1),
    ("neutral",  "medium", "overbought", 1),
    ("neutral",  "high",   "overbought", 0),

    # --- Cảm xúc Tiêu cực ---
    ("negative", "high",   "overbought", 0),
    ("negative", "high",   "neutral",    0),
    ("negative", "medium", "overbought", 0),
    ("negative", "medium", "neutral",    1),
    ("negative", "low",    "neutral",    1),
    ("negative", "low",    "overbought", 1),
    ("negative", "medium", "oversold",   2),
    ("negative", "high",   "oversold",   1),
    ("negative", "low",    "oversold",   2),
]


def fuzzy_inference(sentiment: float, volume: float, rsi: float) -> int:
    """
    Chạy suy luận mờ, trả về nhãn xu thế (0-4) có firing strength cao nhất.
    Không cần defuzzify vì bài toán phân loại 5 nhãn định sẵn.
    """
    fs = fuzzify_sentiment(sentiment)
    fv = fuzzify_volume(volume)
    fr = fuzzify_rsi(rsi)

    firing = np.zeros(5)
    for s_key, v_key, r_key, label in RULES:
        strength = min(fs[s_key], fv[v_key], fr[r_key])   # AND = min
        firing[label] = max(firing[label], strength)        # OR  = max

    return int(np.argmax(firing))


# ── Thu thập tin tức từ yfinance ──────────────────────────────────────────────

def fetch_news_yfinance(ticker: str = "TSLA") -> dict:
    """
    Lấy tin tức gần đây của TSLA từ yfinance.
    Trả về dict {date_str: [title1, title2, ...]}

    Lưu ý: yfinance chỉ cung cấp tin tức vài ngày gần nhất.
    Các ngày lịch sử (2010-2023) sẽ không có tin tức → dùng neutral fallback.
    """
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news
        date_news: dict = {}

        for item in raw_news:
            if not isinstance(item, dict):
                continue

            # Tương thích nhiều phiên bản yfinance
            title     = (item.get("title")
                         or item.get("content", {}).get("title", ""))
            timestamp = (item.get("providerPublishTime")
                         or item.get("content", {}).get("pubDate", ""))

            if not title:
                continue

            # Chuyển timestamp → date string YYYY-MM-DD
            if isinstance(timestamp, (int, float)) and timestamp > 0:
                from datetime import datetime, timezone
                date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
            elif isinstance(timestamp, str) and len(timestamp) >= 10:
                date_str = timestamp[:10]
            else:
                continue

            date_news.setdefault(date_str, []).append(title.strip())

        total_titles = sum(len(v) for v in date_news.values())
        print(f"  yfinance: {total_titles} tiêu đề từ {len(date_news)} ngày")
        return date_news

    except Exception as exc:
        print(f"  [WARN] Không thể lấy tin tức từ yfinance: {exc}")
        return {}


# ── Phân tích cảm xúc bằng Gemini API ────────────────────────────────────────

def analyze_sentiment_gemini(headlines: list) -> float:
    """
    Gửi tối đa 5 tiêu đề tin tức lên Gemini, nhận về điểm cảm xúc [0.0, 1.0].
      0.0 = rất tiêu cực
      0.5 = trung lập
      1.0 = rất tích cực
    Trả về 0.5 nếu không có tin tức hoặc Gemini lỗi.
    """
    if not headlines:
        return 0.5

    sample = headlines[:5]
    headlines_text = "\n".join(f"- {h}" for h in sample)

    prompt = (
        "Bạn là chuyên gia phân tích tài chính. Hãy đánh giá cảm xúc chung "
        "của các tiêu đề tin tức sau đây về cổ phiếu Tesla (TSLA).\n\n"
        f"Tiêu đề:\n{headlines_text}\n\n"
        "Trả lời bằng MỘT số thực duy nhất trong khoảng [0.0, 1.0]:\n"
        "  0.0 = rất tiêu cực (tin xấu, kỳ vọng giá giảm mạnh)\n"
        "  0.5 = trung lập\n"
        "  1.0 = rất tích cực (tin tốt, kỳ vọng giá tăng mạnh)\n\n"
        "Không giải thích, chỉ trả về số."
    )

    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        raw = response.text.strip().split()[0]   # lấy token đầu tiên phòng model thêm chữ
        score = float(raw)
        return float(np.clip(score, 0.0, 1.0))
    except Exception as exc:
        print(f"  [WARN] Gemini error: {exc}")
        return 0.5


# ── Cache sentiment theo ngày ─────────────────────────────────────────────────

def load_sentiment_cache() -> dict:
    """Đọc cache CSV → {date_str: sentiment_score}."""
    if not os.path.exists(NEWS_CACHE_PATH):
        return {}
    df = pd.read_csv(NEWS_CACHE_PATH)
    df["Date"] = df["Date"].astype(str).str[:10]
    return dict(zip(df["Date"], df["sentiment"].astype(float)))


def save_sentiment_cache(cache: dict) -> None:
    """Lưu dict cache → CSV."""
    rows = [{"Date": k, "sentiment": v} for k, v in sorted(cache.items())]
    pd.DataFrame(rows).to_csv(NEWS_CACHE_PATH, index=False)


def build_sentiment_series(dates: pd.Series) -> pd.Series:
    """
    Với mỗi ngày trong `dates`:
      1. Nếu đã có trong cache → dùng luôn.
      2. Nếu có tin tức (yfinance) → gọi Gemini, lưu cache.
      3. Không có tin tức → 0.5 (trung lập), lưu cache.

    Trả về Series sentiment có cùng index với `dates`.
    """
    cache = load_sentiment_cache()
    date_strs = dates.dt.strftime("%Y-%m-%d").tolist()

    # Chỉ gọi yfinance khi còn ngày chưa có trong cache
    missing = [d for d in date_strs if d not in cache]
    if missing:
        print(f"  {len(missing)} ngày chưa có sentiment → lấy tin từ yfinance...")
        date_news = fetch_news_yfinance("TSLA")

        new_count = 0
        for date_str in missing:
            headlines = date_news.get(date_str, [])
            if headlines:
                score = analyze_sentiment_gemini(headlines)
                time.sleep(0.3)   # tránh rate-limit Gemini
            else:
                score = 0.5       # neutral fallback cho ngày không có tin
            cache[date_str] = score
            new_count += 1

        save_sentiment_cache(cache)
        has_news = sum(1 for d in missing if date_news.get(d))
        print(f"  Đã phân tích Gemini: {has_news} ngày có tin, "
              f"{new_count - has_news} ngày dùng neutral fallback")
        print(f"  Đã lưu cache: {NEWS_CACHE_PATH}")
    else:
        print(f"  Toàn bộ {len(date_strs)} ngày đã có trong cache.")

    return pd.Series([cache[d] for d in date_strs], index=dates.index, name="sentiment")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  [2/5] HỆ THỐNG SUY LUẬN MỜ (FUZZY INFERENCE)")
    print("=" * 55)

    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    print(f"  Đọc: {len(df):,} dòng từ {IN_PATH}")

    # Bước 1: Lấy sentiment thật từ Gemini + yfinance (có cache)
    print("\n  --- Bước 1: Phân tích cảm xúc tin tức (Gemini) ---")
    df["sentiment"] = build_sentiment_series(df["Date"])

    # Bước 2: Chạy suy luận mờ cho từng phiên
    print("\n  --- Bước 2: Suy luận mờ IF-THEN ---")
    fuzzy_labels = [
        fuzzy_inference(row["sentiment"], row["Volume"], row["RSI"])
        for _, row in df.iterrows()
    ]
    df["fuzzy_label"] = fuzzy_labels

    # Lưu kết quả
    df.to_csv(OUT_PATH, index=False)
    print(f"\n  Đã lưu: {OUT_PATH}")

    print(f"\n  Phân phối fuzzy_label:")
    for i, name in enumerate(TREND_NAMES):
        cnt = int((df["fuzzy_label"] == i).sum())
        pct = cnt / len(df) * 100
        print(f"     {name:<15} {cnt:>5} phiên  ({pct:.1f}%)")

    print("=" * 55)


if __name__ == "__main__":
    main()
