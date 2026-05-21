# data_loader.py — Load kết quả dự đoán và dữ liệu giá

import numpy as np
import pandas as pd


def load_tsla_prices(prices_path: str) -> pd.DataFrame:
    """
    Đọc file CSV giá TSLA từ Yahoo Finance với định dạng 2 dòng header đặc biệt:

        Dòng 0 (bỏ qua): Price  Close  High  Low   Open   Volume
        Dòng 1 (bỏ qua): Ticker TSLA   TSLA  TSLA  TSLA   TSLA
        Dòng 2+         : Date   giá    ...

    Ví dụ:
        Price      Close         High          Low           Open          Volume
        Ticker     TSLA          TSLA          TSLA          TSLA          TSLA
        Date
        1/2/2020   28.68400002   28.71333313   28.11400032   28.29999924   142981500

    Trả về DataFrame chuẩn với các cột:
        Date (datetime), Close, High, Low, Open, Volume (float)
    """
    # Bỏ 2 dòng header thừa (Price + Ticker), lấy dòng thứ 3 trở đi
    df = pd.read_csv(
        prices_path,
        skiprows=2,                          # bỏ dòng "Price..." và "Ticker..."
        header=0,                            # dòng đầu tiên còn lại = tên cột
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Date"],
        dayfirst=False,                      # định dạng M/D/YYYY
    )

    # Ép kiểu số cho các cột giá
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Date", "Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  📂 Đã load giá TSLA: {len(df):,} phiên "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    return df


def load_predictions(y_true_path=None, y_pred_path=None, prices_path=None):
    """
    Load nhãn thực tế, nhãn dự đoán và dữ liệu giá TSLA.

    Khi có model thực, truyền đường dẫn vào:
        load_predictions(
            y_true_path  = "results/y_true.npy",
            y_pred_path  = "results/y_pred.npy",
            prices_path  = "data/tsla.csv",   # file CSV từ Yahoo Finance
        )

    Nếu không truyền gì → tự động tạo dữ liệu mô phỏng để demo.
    """
    if y_true_path and y_pred_path:
        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
        prices = load_tsla_prices(prices_path) if prices_path else None
        return y_true, y_pred, prices

    # --- Dữ liệu mô phỏng (~72% accuracy) ---
    np.random.seed(42)
    n = 300
    y_true = np.random.randint(0, 5, n)

    noise_mask = np.random.rand(n) > 0.72
    y_pred = y_true.copy()
    y_pred[noise_mask] = np.random.randint(0, 5, noise_mask.sum())

    prices_array = 200 * np.cumprod(1 + np.random.normal(0, 0.02, n))
    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"Date": dates, "Close": prices_array})

    return y_true, y_pred, prices
