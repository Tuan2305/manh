# Hướng dẫn chạy dự án: Fuzzy-LSTM dự đoán xu thế giá cổ phiếu TSLA

## Tổng quan pipeline

```
TSLA.csv
   │
   ▼
[Bước 1] 1_preprocess.py   → tính RSI, ATR, Bollinger Bands, gán nhãn xu thế
   │
   ▼
[Bước 2] 2_fuzzy.py        → lấy tin tức (yfinance) + phân tích cảm xúc (Gemini) + suy luận mờ
   │
   ▼
[Bước 3] 3_dataset.py      → tạo chuỗi thời gian (sliding window), chia train/val/test
   │
   ▼
[Bước 4] 4_train.py        → huấn luyện mô hình LSTM, lưu best_model.pt
   │
   ▼
[Bước 5] evaluation/main.py → tính Accuracy/F1/Recall, vẽ biểu đồ, backtesting
```

---

## Yêu cầu cài đặt

```bash
pip install pandas numpy scikit-learn torch yfinance google-genai joblib matplotlib seaborn
```

---

## Cấu hình trước khi chạy

Mở file `pipeline/config.py` và kiểm tra:

```python
# Gemini API key — đã điền sẵn
GEMINI_API_KEY = "AIzaSy..."

# Khung thời gian dự đoán — chọn 1 trong 3:
HORIZON = 5    # Ngắn hạn  (~1 tuần  giao dịch)
# HORIZON = 20  # Trung hạn (~1 tháng giao dịch)
# HORIZON = 60  # Dài hạn   (~3 tháng giao dịch)
```

---

## Cách chạy

### Cách 1 — Chạy toàn bộ pipeline (khuyên dùng)

```bash
cd C:\TUAN\code\python\manh\pipeline

# Ngắn hạn (mặc định, horizon = 5 phiên)
python run_pipeline.py --horizon short

# Trung hạn (horizon = 20 phiên)
python run_pipeline.py --horizon medium

# Dài hạn (horizon = 60 phiên)
python run_pipeline.py --horizon long
```

Pipeline sẽ tự động chạy tuần tự Bước 1 → 5 và in tiến độ ra màn hình.

---

### Cách 2 — Chạy từng bước thủ công

Nếu muốn kiểm tra kết quả từng bước riêng lẻ:

```bash
cd C:\TUAN\code\python\manh\pipeline

# Bước 1: Tiền xử lý dữ liệu giá
python 1_preprocess.py
# → Tạo: pipeline_data/tsla_processed.csv
# → Tạo: pipeline_data/scaler.pkl

# Bước 2: Suy luận mờ + phân tích cảm xúc Gemini
python 2_fuzzy.py
# → Gọi yfinance lấy tin tức TSLA gần đây
# → Gọi Gemini API phân tích sentiment
# → Tạo: pipeline_data/news_sentiment_cache.csv  (cache, lần sau không gọi lại)
# → Tạo: pipeline_data/tsla_with_fuzzy.csv

# Bước 3: Tạo sequences cho LSTM
python 3_dataset.py
# → Tạo: pipeline_data/X_train.npy, y_train.npy
#         pipeline_data/X_val.npy,   y_val.npy
#         pipeline_data/X_test.npy,  y_test.npy

# Bước 4: Huấn luyện LSTM
python 4_train.py
# → Tạo: model_output/best_model.pt
# → Tạo: model_output/y_true.npy, y_pred.npy
# → Tạo: model_output/training_history.csv

# Bước 5: Đánh giá mô hình
cd C:\TUAN\code\python\manh\evaluation
python main.py --true  ../pipeline/model_output/y_true.npy ^
               --pred  ../pipeline/model_output/y_pred.npy ^
               --price ../TSLA.csv
# → Tạo: evaluation_results/1_metrics_bar.png
# → Tạo: evaluation_results/2_confusion_matrix.png
# → Tạo: evaluation_results/3_classification_report.txt
# → Tạo: evaluation_results/4_backtesting.png
# → Tạo: evaluation_results/5_evaluation_summary.csv
```

---

## Kết quả đầu ra

| Vị trí | Mô tả |
|---|---|
| `pipeline/pipeline_data/tsla_processed.csv` | Dữ liệu giá + RSI/ATR/BB đã chuẩn hóa |
| `pipeline/pipeline_data/news_sentiment_cache.csv` | Cache sentiment theo ngày (Gemini) |
| `pipeline/pipeline_data/tsla_with_fuzzy.csv` | Dữ liệu đầy đủ kèm cột `sentiment` và `fuzzy_label` |
| `pipeline/model_output/best_model.pt` | Model LSTM tốt nhất (theo val loss) |
| `pipeline/model_output/training_history.csv` | Lịch sử train/val loss theo epoch |
| `evaluation/evaluation_results/` | Biểu đồ + báo cáo đánh giá |

---

## Nhãn xu thế (5 lớp phân loại)

| Label | Tên | Điều kiện (return N phiên tới) |
|---|---|---|
| 0 | Giảm mạnh | < −2% |
| 1 | Giảm nhẹ | −2% đến −0.5% |
| 2 | Bình thường | −0.5% đến +0.5% |
| 3 | Tăng nhẹ | +0.5% đến +2% |
| 4 | Tăng mạnh | > +2% |

---

## Lưu ý về sentiment (tin tức)

- **Ngày có tin tức** (chủ yếu vài ngày gần nhất): yfinance cung cấp tiêu đề → Gemini phân tích → điểm `[0.0, 1.0]`
- **Ngày không có tin tức** (phần lớn dữ liệu lịch sử 2010–2022): tự động dùng `0.5` (trung lập)
- **Cache**: kết quả sentiment được lưu vào `news_sentiment_cache.csv`, những lần chạy sau sẽ không gọi lại Gemini API cho các ngày đã xử lý

---

## Cấu trúc thư mục

```
manh/
├── TSLA.csv                          ← Dữ liệu giá cổ phiếu (Yahoo Finance)
├── HUONG_DAN.md                      ← File này
├── pipeline/
│   ├── config.py                     ← Cấu hình (API key, HORIZON)
│   ├── run_pipeline.py               ← Điểm vào chạy toàn bộ
│   ├── 1_preprocess.py
│   ├── 2_fuzzy.py
│   ├── 3_dataset.py
│   ├── 4_train.py
│   └── pipeline_data/                ← Sinh ra khi chạy
│       ├── tsla_processed.csv
│       ├── tsla_with_fuzzy.csv
│       ├── news_sentiment_cache.csv
│       ├── scaler.pkl
│       └── X_train/val/test.npy, y_*.npy
│   └── model_output/                 ← Sinh ra khi chạy
│       ├── best_model.pt
│       ├── y_true.npy, y_pred.npy
│       └── training_history.csv
└── evaluation/
    ├── main.py
    ├── config.py
    ├── data_loader.py
    ├── metrics.py
    ├── backtesting.py
    └── evaluation_results/           ← Sinh ra khi chạy
        ├── 1_metrics_bar.png
        ├── 2_confusion_matrix.png
        ├── 3_classification_report.txt
        ├── 4_backtesting.png
        └── 5_evaluation_summary.csv
```
