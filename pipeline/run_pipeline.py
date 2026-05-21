"""
run_pipeline.py — Chạy toàn bộ pipeline theo thứ tự

Thứ tự:
    Bước 1: Tiền xử lý dữ liệu giá (RSI, ATR, Bollinger Bands)
    Bước 2: Hệ thống suy luận mờ (Fuzzy Inference + Gemini sentiment)
    Bước 3: Tạo sequences cho LSTM
    Bước 4: Huấn luyện mô hình LSTM
    Bước 5: Đánh giá toàn diện

Cách chạy:
    python run_pipeline.py                    # ngắn hạn (mặc định, horizon=5)
    python run_pipeline.py --horizon short    # ngắn hạn  (5  phiên)
    python run_pipeline.py --horizon medium   # trung hạn (20 phiên)
    python run_pipeline.py --horizon long     # dài hạn   (60 phiên)
"""

import argparse
import os
import subprocess
import sys
import time

# ── Parse tham số ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fuzzy-LSTM Pipeline")
parser.add_argument(
    "--horizon",
    choices=["short", "medium", "long"],
    default="short",
    help="Khung thời gian dự đoán: short=5 phiên, medium=20, long=60 (mặc định: short)",
)
args = parser.parse_args()

HORIZON_MAP   = {"short": 5, "medium": 20, "long": 60}
HORIZON_VALUE = HORIZON_MAP[args.horizon]

# Truyền HORIZON cho các script con qua biến môi trường
env = os.environ.copy()
env["HORIZON"] = str(HORIZON_VALUE)

STEPS = [
    ("Tien xu ly du lieu",    "1_preprocess.py"),
    ("He thong suy luan mo",  "2_fuzzy.py"),
    ("Tao sequences LSTM",    "3_dataset.py"),
    ("Huan luyen mo hinh",    "4_train.py"),
]

EVAL_SCRIPT = os.path.join("..", "evaluation", "main.py")
Y_TRUE_PATH = os.path.join("model_output", "y_true.npy")
Y_PRED_PATH = os.path.join("model_output", "y_pred.npy")
PRICE_PATH  = r"C:\TUAN\code\python\manh\TSLA.csv"


def run_step(label: str, script: str) -> float:
    print(f"\n{'='*55}")
    print(f"   {label}")
    print(f"{'='*55}")
    t0 = time.time()
    subprocess.run([sys.executable, script], check=True, env=env)
    elapsed = time.time() - t0
    print(f"  Hoan thanh trong {elapsed:.1f}s")
    return elapsed


def main():
    horizon_desc = {"short": "ngan han (5 phien)",
                    "medium": "trung han (20 phien)",
                    "long": "dai han (60 phien)"}[args.horizon]

    print("\n" + "=" * 55)
    print("   PIPELINE FUZZY-LSTM: BAT DAU")
    print(f"   Khung thoi gian: {horizon_desc}")
    print("=" * 55)

    total_start = time.time()

    for label, script in STEPS:
        run_step(label, script)

    # Bước 5: Đánh giá
    print(f"\n{'='*55}")
    print(f"   Danh gia mo hinh (Evaluation)")
    print(f"{'='*55}")
    t0 = time.time()
    subprocess.run([
        sys.executable, EVAL_SCRIPT,
        "--true",  Y_TRUE_PATH,
        "--pred",  Y_PRED_PATH,
        "--price", PRICE_PATH,
    ], check=True, env=env)
    print(f"   Hoan thanh trong {time.time()-t0:.1f}s")

    total = time.time() - total_start
    print(f"\n{'='*55}")
    print(f"   TOAN BO PIPELINE HOAN THANH  ({total:.1f}s)")
    print(f"{'='*55}")
    print(f"\n  Ket qua dau ra:")
    print(f"     pipeline/pipeline_data/        <- du lieu da xu ly")
    print(f"     pipeline/model_output/         <- model + y_true/y_pred")
    print(f"     evaluation/evaluation_results/ <- bieu do + bao cao\n")


if __name__ == "__main__":
    main()
