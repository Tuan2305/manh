"""
compare_models.py — So sánh Fuzzy-LSTM vs ARIMA vs ANFIS
Chạy từ thư mục evaluation/:
    python compare_models.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# ── Import config ──────────────────────────────────────────────────────────────
EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(EVAL_DIR, "..", "pipeline")

# evaluation/config.py trước, pipeline/config.py sau
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, EVAL_DIR)

from config import OUTPUT_DIR, LABEL_NAMES   # evaluation/config.py

# Đọc HORIZON từ pipeline/config.py qua importlib để tránh xung đột tên
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("pipeline_cfg", os.path.join(PIPELINE_DIR, "config.py"))
_pipe_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pipe_cfg)
HORIZON = getattr(_pipe_cfg, "HORIZON", 1)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "..", "pipeline", "pipeline_data")
MDL_DIR   = os.path.join(BASE, "..", "pipeline", "model_output")
PRICE_CSV = os.path.join(BASE, "..", "TSLA.csv")

SEQ_LEN   = 30
TRAIN_PCT = 0.70
VAL_PCT   = 0.15

# Threshold nhãn tương ứng horizon (khớp với 1_preprocess.py)
THRESHOLD_MAP = {1: 0.015, 5: 0.03, 20: 0.05, 60: 0.10}
THRESHOLD  = THRESHOLD_MAP.get(HORIZON, 0.03)
NUM_CLASSES = len(LABEL_NAMES)


# ── 1. Load dữ liệu ────────────────────────────────────────────────────────────
def load_all():
    X_train     = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train     = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test      = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test      = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    y_true_lstm = np.load(os.path.join(MDL_DIR, "y_true.npy"))
    y_pred_lstm = np.load(os.path.join(MDL_DIR, "y_pred.npy"))

    # Giá thực (chưa normalize) cho ARIMA
    df_price = pd.read_csv(
        PRICE_CSV, skiprows=2, header=0,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Date"],
    )
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        df_price[c] = pd.to_numeric(df_price[c], errors="coerce")
    df_price.dropna(subset=["Date", "Close"], inplace=True)
    df_price.sort_values("Date", inplace=True)
    df_price.reset_index(drop=True, inplace=True)

    return X_train, y_train, X_test, y_test, y_true_lstm, y_pred_lstm, df_price


# ── 2. Metrics helper ──────────────────────────────────────────────────────────
def metrics_row(y_true, y_pred, name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(   y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(       y_true, y_pred, average="macro", zero_division=0)
    print(f"  {name:<22} Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return {"Mo hinh": name, "Accuracy": acc, "Precision (macro)": prec,
            "Recall (macro)": rec, "F1-Score (macro)": f1}


# ── 3. ARIMA ───────────────────────────────────────────────────────────────────
def _price_to_label(ret):
    if NUM_CLASSES == 3:
        if ret < -THRESHOLD: return 0
        if ret <  THRESHOLD: return 1
        return 2
    else:  # 5 lớp
        if ret < -0.05: return 0
        if ret < -0.01: return 1
        if ret <  0.01: return 2
        if ret <  0.05: return 3
        return 4


def predict_arima(df_price, n_test):
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA

    closes = df_price["Close"].values
    n_seqs = len(closes) - SEQ_LEN
    n_tr   = int(n_seqs * TRAIN_PCT)
    n_vl   = int(n_seqs * VAL_PCT)

    test_start  = SEQ_LEN + n_tr + n_vl
    train_hist  = closes[:test_start].tolist()
    test_prices = closes[test_start:].tolist()

    n_test = min(n_test, len(test_prices) - HORIZON)
    print(f"  ARIMA(5,1,0): rolling {n_test} bước (HORIZON={HORIZON})...")

    preds = []
    history = train_hist.copy()

    try:
        fit = _ARIMA(history, order=(5, 1, 0)).fit()
    except Exception as e:
        print(f"  [WARN] ARIMA init fail: {e}. Dùng naive (Giữ).")
        return np.full(n_test, NUM_CLASSES // 2, dtype=int)

    for i in range(n_test):
        if i % 150 == 0 and i > 0:
            print(f"    [{i}/{n_test}]")
        try:
            fc = float(fit.forecast(steps=HORIZON)[-1])
        except Exception:
            fc = history[-1]

        current = test_prices[i]
        ret = (fc - current) / (abs(current) + 1e-9)
        preds.append(_price_to_label(ret))

        # Update: append giá thực, không refit toàn bộ (nhanh)
        new_val = test_prices[i] if i < len(test_prices) else history[-1]
        history.append(new_val)
        try:
            fit = fit.append([new_val], refit=False)
        except Exception:
            # Fallback: refit trên 300 điểm gần nhất
            try:
                fit = _ARIMA(history[-300:], order=(5, 1, 0)).fit()
            except Exception:
                pass  # giữ nguyên fit cũ

    return np.array(preds)


# ── 4. ANFIS (Fuzzy preprocessing + MLP) ──────────────────────────────────────
def _trimf(x, a, b, c):
    out = np.zeros_like(x, dtype=float)
    m1 = (x > a) & (x <= b);  m2 = (x > b) & (x < c)
    out[m1] = (x[m1] - a) / (b - a + 1e-9)
    out[m2] = (c - x[m2]) / (c - b + 1e-9)
    return out

def _trapmf(x, a, b, c, d):
    out = np.zeros_like(x, dtype=float)
    m1 = (x > a) & (x <= b); m2 = (x > b) & (x <= c); m3 = (x > c) & (x < d)
    out[m1] = (x[m1] - a) / (b - a + 1e-9)
    out[m2] = 1.0
    out[m3] = (d - x[m3]) / (d - c + 1e-9)
    return out

def anfis_features(X):
    """
    Trích xuất đặc trưng mờ từ sequences.
    X: (n, seq_len, n_features)  — Close=0,High=1,Low=2,Open=3,Vol=4,RSI=5,ATR=6,BBw=7,BBp=8
    """
    last   = X[:, -1, :]
    rsi    = last[:, 5]
    vol    = last[:, 4]
    bb_pct = last[:, 8]

    # Membership degrees
    rsi_os  = _trapmf(rsi, 0.0,  0.0,  0.30, 0.45)
    rsi_n   = _trimf( rsi, 0.30, 0.5,  0.70)
    rsi_ob  = _trapmf(rsi, 0.55, 0.70, 1.0,  1.0)
    vol_lo  = _trapmf(vol, 0.0,  0.0,  0.20, 0.40)
    vol_med = _trimf( vol, 0.25, 0.5,  0.75)
    vol_hi  = _trapmf(vol, 0.45, 0.60, 1.0,  1.0)
    bb_lo   = _trapmf(bb_pct, 0.0,  0.0,  0.20, 0.40)
    bb_mid  = _trimf( bb_pct, 0.25, 0.5,  0.75)
    bb_hi   = _trapmf(bb_pct, 0.55, 0.70, 1.0,  1.0)

    # Rule firing strengths (AND = min)
    rule_bull = np.minimum(rsi_os, vol_hi)   # oversold + high vol → Tăng
    rule_bear = np.minimum(rsi_ob, vol_hi)   # overbought + high vol → Giảm
    rule_neu  = np.minimum(rsi_n,  vol_med)

    # Momentum & statistics
    close_mom = X[:, -1, 0] - X[:, -5, 0]
    rsi_mom   = X[:, -1, 5] - X[:, -5, 5]
    close_std = X[:, :, 0].std(axis=1)
    rsi_mean  = X[:, :, 5].mean(axis=1)
    vol_mean  = X[:, :, 4].mean(axis=1)

    return np.column_stack([
        last,                                              # 9 raw
        rsi_os, rsi_n, rsi_ob,                           # RSI MF
        vol_lo, vol_med, vol_hi,                         # Volume MF
        bb_lo, bb_mid, bb_hi,                            # BB MF
        rule_bull, rule_bear, rule_neu,                  # Rules
        close_mom, rsi_mom, close_std, rsi_mean, vol_mean,  # Dynamics
    ])


def predict_anfis(X_train, y_train, X_test):
    print("  ANFIS: trích xuất đặc trưng mờ...")
    ft = anfis_features(X_train)
    fe = anfis_features(X_test)

    sc = StandardScaler()
    ft = sc.fit_transform(ft)
    fe = sc.transform(fe)

    print("  ANFIS: huấn luyện MLP(128→64→32)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        learning_rate_init=5e-4,
        verbose=False,
    )
    mlp.fit(ft, y_train)
    best = f"{mlp.best_loss_:.4f}" if mlp.best_loss_ is not None else "N/A"
    print(f"  ANFIS: hội tụ sau {mlp.n_iter_} iterations  (best val loss = {best})")
    return mlp.predict(fe)


# ── 5. Vẽ biểu đồ so sánh ─────────────────────────────────────────────────────
def plot_comparison(df_res):
    metrics  = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-Score (macro)"]
    xlabels  = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1-Score\n(macro)"]
    models   = df_res["Mo hinh"].tolist()
    colors   = ["#4fc3f7", "#ffb74d", "#81c784"]
    x        = np.arange(len(metrics))
    width    = 0.22

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for i, (mdl, clr) in enumerate(zip(models, colors)):
        row  = df_res[df_res["Mo hinh"] == mdl].iloc[0]
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=mdl, color=clr, alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008, f"{val:.3f}",
                    ha="center", va="bottom", color="white",
                    fontsize=8.5, fontweight="bold")

    rnd = 1 / NUM_CLASSES
    ax.axhline(y=rnd, color="#888", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.text(3.45, rnd + 0.01, f"Random ({rnd:.0%})", color="#999", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, color="white", fontsize=10)
    ax.set_ylabel("Giá trị", color="white", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"So Sánh Hiệu Suất: Fuzzy-LSTM vs ARIMA vs ANFIS  (HORIZON = {HORIZON} phiên)",
        color="white", fontsize=13, fontweight="bold", pad=15,
    )
    ax.legend(facecolor="#2a2d3a", labelcolor="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"6_model_comparison_H{HORIZON}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Da luu: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  SO SANH MO HINH: Fuzzy-LSTM  vs  ARIMA  vs  ANFIS")
    print(f"  HORIZON={HORIZON}  |  Threshold=+-{THRESHOLD*100:.1f}%  |  {NUM_CLASSES} lop: {LABEL_NAMES}")
    print("=" * 65)

    X_train, y_train, X_test, y_test, y_true_lstm, y_pred_lstm, df_price = load_all()

    dist = {LABEL_NAMES[i]: int((y_test == i).sum()) for i in range(NUM_CLASSES)}
    print(f"\n  Tap test: {len(y_test)} mau  |  Phan phoi: {dist}")

    results = {}

    # ── Fuzzy-LSTM ────────────────────────────────────────────────────────────
    print("\n[1/3] Fuzzy-LSTM (ket qua da co san)")
    n_lstm = min(len(y_true_lstm), len(y_pred_lstm))
    results["Fuzzy-LSTM"] = (y_true_lstm[:n_lstm], y_pred_lstm[:n_lstm])

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    print("\n[2/3] ARIMA(5,1,0) — rolling forecast")
    y_pred_arima = predict_arima(df_price, len(y_test))
    n_arima = min(len(y_test), len(y_pred_arima))
    results["ARIMA"] = (y_test[:n_arima], y_pred_arima[:n_arima])

    # ── ANFIS ─────────────────────────────────────────────────────────────────
    print("\n[3/3] ANFIS (Fuzzy preprocessing + MLP)")
    y_pred_anfis = predict_anfis(X_train, y_train, X_test)
    results["ANFIS"] = (y_test, y_pred_anfis)

    # ── Bảng tổng kết ─────────────────────────────────────────────────────────
    rows = []
    print("\n" + "=" * 65)
    print(f"  BANG SO SANH  (HORIZON = {HORIZON} phien)")
    print("=" * 65)
    for name, (yt, yp) in results.items():
        rows.append(metrics_row(yt, yp, name))

    df_res = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUT_DIR, f"6_model_comparison_H{HORIZON}.csv")
    df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  Da luu bang so sanh: {csv_path}")

    plot_comparison(df_res)

    # ── Classification report chi tiết ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CHI TIET TUNG LOP")
    print("=" * 65)
    for name, (yt, yp) in results.items():
        print(f"\n--- {name} ---")
        print(classification_report(yt, yp, target_names=LABEL_NAMES,
                                    zero_division=0, digits=4))

    # ── Gợi ý chạy multi-horizon ──────────────────────────────────────────────
    print("=" * 65)
    print("  GHI CHU: De so sanh theo tung khung thoi gian, thay doi")
    print("  HORIZON trong pipeline/config.py (1/5/20/60), chay lai")
    print("  pipeline, roi chay lai script nay de lay ket qua.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
