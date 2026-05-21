# config.py — Cấu hình chung cho toàn bộ pipeline đánh giá

import os

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES  = ["Giảm mạnh", "Giảm nhẹ", "Bình thường", "Tăng nhẹ", "Tăng mạnh"]
LABEL_COLORS = ["#d32f2f",   "#ef9a9a",   "#90a4ae",      "#a5d6a7",  "#388e3c"]
