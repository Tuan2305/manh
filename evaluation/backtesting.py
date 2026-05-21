# backtesting.py — Backtesting xu thế: lợi nhuận, Sharpe Ratio, Max Drawdown

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import OUTPUT_DIR, LABEL_NAMES


def run_backtesting(y_true, y_pred, prices_df):
    """
    Chạy backtesting đơn giản dựa trên nhãn xu thế dự đoán.

    Chiến lược:
        Nhãn >= 3 (Tăng nhẹ / Tăng mạnh) → Mua  (+1)
        Nhãn <= 1 (Giảm nhẹ / Giảm mạnh) → Bán  (-1)
        Nhãn == 2 (Bình thường)           → Giữ  ( 0)
    """
    save_path = os.path.join(OUTPUT_DIR, "4_backtesting.png")

    n      = min(len(y_pred), len(prices_df) - 1)
    closes = prices_df["Close"].values

    real_returns = np.diff(closes[:n + 1]) / closes[:n]
    signals      = np.where(y_pred[:n] >= 3, 1, np.where(y_pred[:n] <= 1, -1, 0))

    strategy_returns = signals * real_returns
    strategy_cum     = np.cumprod(1 + strategy_returns)
    buy_hold_cum     = np.cumprod(1 + real_returns)

    def sharpe(rets, rf=0.0):
        excess = rets - rf / 252
        return (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() != 0 else 0

    def max_drawdown(cum):
        peak = np.maximum.accumulate(cum)
        return ((cum - peak) / peak).min()

    total_strat  = strategy_cum[-1] - 1
    total_bh     = buy_hold_cum[-1] - 1
    sharpe_strat = sharpe(strategy_returns)
    sharpe_bh    = sharpe(real_returns)
    mdd_strat    = max_drawdown(strategy_cum)
    mdd_bh       = max_drawdown(buy_hold_cum)

    # In bảng kết quả
    print("\n" + "=" * 55)
    print("  ẾT QUẢ BACKTESTING")
    print("=" * 55)
    print(f"  {'Chỉ số':<30} {'Fuzzy-LSTM':>10}  {'Buy & Hold':>10}")
    print(f"  {'-' * 50}")
    print(f"  {'Tổng lợi nhuận':<30} {total_strat:>9.2%}  {total_bh:>9.2%}")
    print(f"  {'Sharpe Ratio':<30} {sharpe_strat:>10.3f}  {sharpe_bh:>10.3f}")
    print(f"  {'Max Drawdown':<30} {mdd_strat:>9.2%}  {mdd_bh:>9.2%}")
    print("=" * 55)

    # ── Vẽ 3 biểu đồ ─────────────────────────────────────────────────────────
    dates     = prices_df["Date"].values[:n]
    buy_idx   = np.where(signals == 1)[0]
    sell_idx  = np.where(signals == -1)[0]

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)

    # Plot 1 — Giá + tín hiệu
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1a1d27")
    ax1.plot(dates, closes[:n], color="#90caf9", linewidth=1.2, label="Giá TSLA")
    ax1.scatter(dates[buy_idx],  closes[buy_idx],  marker="^", color="#66bb6a",
                s=40, zorder=5, label="Tín hiệu Mua", alpha=0.8)
    ax1.scatter(dates[sell_idx], closes[sell_idx], marker="v", color="#ef5350",
                s=40, zorder=5, label="Tín hiệu Bán", alpha=0.8)
    ax1.set_title("Giá TSLA và Tín Hiệu Giao Dịch từ Mô Hình",
                  color="white", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", facecolor="#2a2d3a", labelcolor="white", fontsize=8)
    ax1.tick_params(colors="white", labelsize=8)
    ax1.spines[:].set_color("#444")
    ax1.set_ylabel("Giá ($)", color="white")

    # Plot 2 — Lợi nhuận tích lũy
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1a1d27")
    ax2.plot(dates, strategy_cum, color="#ffd54f", linewidth=1.5,
             label=f"Fuzzy-LSTM ({total_strat:+.1%})")
    ax2.plot(dates, buy_hold_cum, color="#4fc3f7", linewidth=1.5, linestyle="--",
             label=f"Buy & Hold ({total_bh:+.1%})")
    ax2.axhline(y=1.0, color="#666", linestyle=":", linewidth=0.8)
    ax2.fill_between(dates, strategy_cum, 1, where=(strategy_cum >= 1),
                     alpha=0.15, color="#66bb6a")
    ax2.fill_between(dates, strategy_cum, 1, where=(strategy_cum < 1),
                     alpha=0.15, color="#ef5350")
    ax2.set_title("Hiệu Suất Tích Lũy (Cumulative Returns)",
                  color="white", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", facecolor="#2a2d3a", labelcolor="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.spines[:].set_color("#444")
    ax2.set_ylabel("Lợi nhuận tích lũy", color="white")

    # Plot 3 — Phân phối nhãn
    import numpy as _np
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#1a1d27")
    x     = _np.arange(5)
    width = 0.35
    tc = [_np.sum(y_true[:n] == i) for i in range(5)]
    pc = [_np.sum(y_pred[:n] == i) for i in range(5)]
    b1 = ax3.bar(x - width/2, tc, width, label="Thực tế",  color="#4fc3f7", alpha=0.8, edgecolor="none")
    b2 = ax3.bar(x + width/2, pc, width, label="Dự đoán",  color="#ffd54f", alpha=0.8, edgecolor="none")
    for bar in [*b1, *b2]:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(bar.get_height())), ha="center", va="bottom",
                 color="white", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(LABEL_NAMES, color="white", fontsize=9)
    ax3.set_title("Phân Phối Nhãn: Thực Tế vs Dự Đoán",
                  color="white", fontsize=11, fontweight="bold")
    ax3.legend(facecolor="#2a2d3a", labelcolor="white", fontsize=9)
    ax3.tick_params(colors="white", labelsize=8)
    ax3.spines[:].set_color("#444")
    ax3.set_ylabel("Số lượng mẫu", color="white")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Đã lưu: {save_path}")

    return {
        "total_return_strategy": total_strat,
        "total_return_bh":       total_bh,
        "sharpe_strategy":       sharpe_strat,
        "sharpe_bh":             sharpe_bh,
        "max_drawdown_strategy": mdd_strat,
        "max_drawdown_bh":       mdd_bh,
    }
