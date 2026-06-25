"""
benchmark_gpu.py — So sanh toc do CPU vs GPU cho mo hinh Fuzzy-LSTM
Chay tu thu muc pipeline/:
    python benchmark_gpu.py
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR  = "pipeline_data"
MODEL_DIR = "model_output"

# ── Kiem tra thiet bi ─────────────────────────────────────────────────────────
CUDA_OK = torch.cuda.is_available()
DEVICES = {"CPU": torch.device("cpu")}
if CUDA_OK:
    DEVICES["GPU"] = torch.device("cuda")
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1024**3
else:
    GPU_NAME = "Khong co GPU"
    GPU_MEM  = 0

N_EPOCHS   = 10          # so epoch do benchmark
BATCH_SIZE = 64
N_WARMUP   = 2           # epoch khoi dong truoc khi do

# ── Mo hinh LSTM ──────────────────────────────────────────────────────────────
class FuzzyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ── Load du lieu ──────────────────────────────────────────────────────────────
def load_data():
    X_train = torch.tensor(np.load(os.path.join(DATA_DIR, "X_train.npy")))
    y_train = torch.tensor(np.load(os.path.join(DATA_DIR, "y_train.npy")), dtype=torch.long)
    X_test  = torch.tensor(np.load(os.path.join(DATA_DIR, "X_test.npy")))
    y_test  = torch.tensor(np.load(os.path.join(DATA_DIR, "y_test.npy")), dtype=torch.long)
    return X_train, y_train, X_test, y_test


# ── Benchmark training ────────────────────────────────────────────────────────
def benchmark_training(device, X_train, y_train, num_classes):
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True)

    model = FuzzyLSTM(
        input_size  = X_train.shape[2],
        num_classes = num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Warmup
    for _ in range(N_WARMUP):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Do thoi gian chinh thuc
    epoch_times = []
    for _ in range(N_EPOCHS):
        t0 = time.perf_counter()
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)

    return {
        "epoch_mean_s":  np.mean(epoch_times),
        "epoch_std_s":   np.std(epoch_times),
        "total_10ep_s":  np.sum(epoch_times),
    }


# ── Benchmark inference ───────────────────────────────────────────────────────
def benchmark_inference(device, X_test, num_classes, n_runs=100):
    model = FuzzyLSTM(
        input_size  = X_test.shape[2],
        num_classes = num_classes,
    ).to(device)
    model.eval()

    # Load best model neu co
    best_path = os.path.join(MODEL_DIR, "best_model.pt")
    if os.path.exists(best_path):
        try:
            model.load_state_dict(torch.load(best_path, map_location=device))
        except Exception:
            pass  # model co the khac so luong lop, dung model ngau nhien

    X_all = X_test.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(X_all)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Inference toan bo tap test
    times_batch = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(X_all)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_batch.append(time.perf_counter() - t0)

    # Inference 1 mau don le
    single = X_all[:1]
    times_single = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(single)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_single.append(time.perf_counter() - t0)

    n_samples = len(X_test)
    batch_mean = np.mean(times_batch)

    return {
        "batch_ms":         batch_mean * 1000,
        "per_sample_us":    batch_mean / n_samples * 1e6,
        "single_ms":        np.mean(times_single) * 1000,
        "throughput_sps":   n_samples / batch_mean,
    }


# ── In bang ket qua ───────────────────────────────────────────────────────────
def print_table(results_train, results_infer):
    devices = list(results_train.keys())

    print("\n" + "=" * 60)
    print("  BANG SO SANH TOC DO: CPU  vs  GPU")
    print("=" * 60)

    # Thong tin thiet bi
    print(f"\n  CPU : {get_cpu_name()}")
    print(f"  GPU : {GPU_NAME}  ({GPU_MEM:.1f} GB VRAM)" if CUDA_OK
          else f"  GPU : Khong co")

    # Training
    print(f"\n  --- TRAINING ({N_EPOCHS} epoch, batch={BATCH_SIZE}) ---")
    header = f"  {'Chi so':<30}"
    for d in devices:
        header += f"  {d:>10}"
    print(header)
    print("  " + "-" * 55)

    rows = [
        ("Thoi gian / epoch (s)",   "epoch_mean_s",  ".3f"),
        ("Do lech chuan (s)",        "epoch_std_s",   ".3f"),
        (f"Tong {N_EPOCHS} epoch (s)", "total_10ep_s", ".2f"),
    ]
    for label, key, fmt in rows:
        row = f"  {label:<30}"
        for d in devices:
            val = results_train[d][key]
            row += f"  {val:>10{fmt}}"
        print(row)

    if len(devices) == 2:
        speedup = results_train["CPU"]["epoch_mean_s"] / results_train["GPU"]["epoch_mean_s"]
        print(f"\n  => Toc do tang (GPU / CPU)  :  {speedup:.1f}x nhanh hon")

    # Inference
    print(f"\n  --- INFERENCE (tap test {results_infer[devices[0]]['throughput_sps']:.0f} mau) ---")
    header2 = f"  {'Chi so':<35}"
    for d in devices:
        header2 += f"  {d:>10}"
    print(header2)
    print("  " + "-" * 58)

    rows2 = [
        ("Toan bo tap test (ms)",     "batch_ms",       ".2f"),
        ("Moi mau (microsecond)",     "per_sample_us",  ".1f"),
        ("1 du bao don le (ms)",      "single_ms",      ".3f"),
        ("Throughput (mau/giay)",     "throughput_sps", ".0f"),
    ]
    for label, key, fmt in rows2:
        row = f"  {label:<35}"
        for d in devices:
            val = results_infer[d][key]
            row += f"  {val:>10{fmt}}"
        print(row)

    if len(devices) == 2:
        sp_inf = results_infer["CPU"]["batch_ms"] / results_infer["GPU"]["batch_ms"]
        sp_thr = results_infer["GPU"]["throughput_sps"] / results_infer["CPU"]["throughput_sps"]
        print(f"\n  => Toc do inference tang     :  {sp_inf:.1f}x nhanh hon")
        print(f"  => Throughput tang           :  {sp_thr:.1f}x")

    print("=" * 60 + "\n")


def get_cpu_name():
    try:
        import subprocess
        r = subprocess.check_output(
            "wmic cpu get name", shell=True, text=True
        ).strip().split("\n")
        return r[-1].strip() if r else "CPU"
    except Exception:
        return "CPU"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  BENCHMARK GPU ACCELERATION - Fuzzy-LSTM")
    print("=" * 60)

    if not CUDA_OK:
        print("\n  [!] CHUA CO GPU / CUDA. Benchmark chi chay tren CPU.")
        print("  Cai PyTorch CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print()

    X_train, y_train, X_test, y_test = load_data()
    num_classes = len(np.unique(y_train.numpy()))
    print(f"\n  Du lieu: train={len(X_train)}  test={len(X_test)}"
          f"  features={X_train.shape[2]}  seq_len={X_train.shape[1]}"
          f"  classes={num_classes}")

    results_train = {}
    results_infer = {}

    for dev_name, device in DEVICES.items():
        print(f"\n  [{dev_name}] Do benchmark training ({N_EPOCHS} epoch)...")
        results_train[dev_name] = benchmark_training(device, X_train, y_train, num_classes)
        print(f"  [{dev_name}] Trung binh: {results_train[dev_name]['epoch_mean_s']:.3f}s/epoch")

        print(f"  [{dev_name}] Do benchmark inference ({len(X_test)} mau)...")
        results_infer[dev_name] = benchmark_inference(device, X_test, num_classes)
        print(f"  [{dev_name}] Throughput: {results_infer[dev_name]['throughput_sps']:.0f} mau/giay")

    print_table(results_train, results_infer)

    # Luu CSV
    import pandas as pd
    rows = []
    for d in DEVICES:
        rows.append({
            "Thiet bi": d,
            "Epoch_mean_s": results_train[d]["epoch_mean_s"],
            "Epoch_std_s":  results_train[d]["epoch_std_s"],
            "Total_10ep_s": results_train[d]["total_10ep_s"],
            "Inference_batch_ms":  results_infer[d]["batch_ms"],
            "Inference_per_sample_us": results_infer[d]["per_sample_us"],
            "Throughput_sps": results_infer[d]["throughput_sps"],
        })
    out = os.path.join("..", "evaluation", "evaluation_results", "7_benchmark_gpu.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  Da luu: {out}")


if __name__ == "__main__":
    main()
