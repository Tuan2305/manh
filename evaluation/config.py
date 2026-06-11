# config.py — Cấu hình chung cho toàn bộ pipeline đánh giá

import os

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES  = ["Giảm", "Giữ", "Tăng"]
LABEL_COLORS = ["#d32f2f", "#90a4ae", "#388e3c"]
