
GEMINI_API_KEY = "AIzaSyBxKMm7B_j2s1DkD2ZsCYYu5PSMDl07UW0"

# ── Khung thời gian dự đoán ───────────────────────────────────────────────────
# HORIZON = 5   → Ngắn hạn  (~1 tuần giao dịch)
# HORIZON = 20  → Trung hạn (~1 tháng giao dịch)
# HORIZON = 60  → Dài hạn   (~3 tháng giao dịch)
HORIZON = 5
HORIZON_LABEL = {5: "ngắn hạn (5 phiên)", 20: "trung hạn (20 phiên)", 60: "dài hạn (60 phiên)"}

RAW_CSV_PATH = r"C:\TUAN\code\python\manh\TSLA.csv"

# ── Tham số tạo nhãn & chuỗi ─────────────────────────────────────────────────
SEQ_LEN   = 30      # số phiên mỗi chuỗi đầu vào LSTM
THRESHOLD = 0.015   # ±1.5% ngưỡng gán nhãn Giảm/Giữ/Tăng

# ── Cột đặc trưng ─────────────────────────────────────────────────────────────
PRICE_COLS     = ["Open", "High", "Low", "Close", "Volume"]               # 5 cột giá OHLCV
INDICATOR_COLS = ["RSI", "ATR", "BB_upper", "BB_middle", "BB_lower",
                  "MACD_hist"]                                             # 6 chỉ báo (+MACD)
SENTIMENT_COL  = ["sentiment"]   # momentum 5 ngày chuẩn hóa [0,1] — proxy cảm xúc thị trường
FUZZY_COLS     = ["fuzzy_0", "fuzzy_1", "fuzzy_2", "fuzzy_3", "fuzzy_4"]  # 5 chiều mờ

# Cột đưa vào MinMaxScaler (price + indicator, không scale sentiment/fuzzy vì đã [0,1])
SCALE_COLS = PRICE_COLS + INDICATOR_COLS   # 11 cột

# Tổng đặc trưng đầu vào LSTM: 5 + 6 + 1 + 5 = 17 chiều
FEATURE_COLS = PRICE_COLS + INDICATOR_COLS + SENTIMENT_COL + FUZZY_COLS

# ── Tham số huấn luyện ────────────────────────────────────────────────────────
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
BATCH_SIZE   = 64
EPOCHS       = 100
LR           = 5e-4
PATIENCE     = 15
NUM_CLASSES  = 3
