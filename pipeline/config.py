
GEMINI_API_KEY = "AIzaSyBxKMm7B_j2s1DkD2ZsCYYu5PSMDl07UW0"

# ── Khung thời gian dự đoán (Horizon) ────────────────────────────────────────
# Chọn một trong ba giá trị:
#   HORIZON = 5   → Ngắn hạn  (~1 tuần giao dịch)
#   HORIZON = 20  → Trung hạn (~1 tháng giao dịch)
#   HORIZON = 60  → Dài hạn   (~3 tháng giao dịch)
#
# Giá trị này quyết định: nhãn xu thế được tính dựa trên return
# trong HORIZON phiên tiếp theo.
HORIZON = 5

HORIZON_LABEL = {5: "ngắn hạn (5 phiên)", 20: "trung hạn (20 phiên)", 60: "dài hạn (60 phiên)"}

# ── Đường dẫn dữ liệu ────────────────────────────────────────────────────────
RAW_CSV_PATH = r"C:\TUAN\code\python\manh\TSLA.csv"
