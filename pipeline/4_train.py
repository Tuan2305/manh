# 4_train.py — Định nghĩa và huấn luyện mô hình Fuzzy-LSTM
#
# Input:  pipeline_data/X_train/val/test.npy, y_train/val/test.npy  (từ bước 3)
# Output: model_output/best_model.pt, y_true.npy, y_pred.npy, training_history.csv
#
# Tối ưu GPU:
#   - torch.cuda.amp (Mixed Precision) → giảm memory, tăng tốc 1.5-2x
#   - cudnn.benchmark = True           → tự chọn kernel tốt nhất
#   - pin_memory = True                → transfer CPU→GPU nhanh hơn
#   - torch.compile (PyTorch ≥ 2.0)   → JIT compile thêm ~15-20%

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, EPOCHS, LR, PATIENCE, NUM_CLASSES,
)

DATA_DIR  = "pipeline_data"
MODEL_DIR = "model_output"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Thiết lập device & tối ưu GPU ────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True   # tự tune CUDA kernel


# ── Kiến trúc mô hình Fuzzy-LSTM ─────────────────────────────────────────────

class FuzzyLSTM(nn.Module):
    """
    Mạng LSTM phân loại xu thế cổ phiếu thành 3 lớp (Giảm/Giữ/Tăng).
    Input:  (batch, seq_len, input_size)   → seq_len=30, input_size=16
    Output: (batch, num_classes)           → num_classes=3
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, seq_len, hidden_size)
        out     = out[:, -1, :]      # lấy hidden state tại bước cuối
        out     = self.dropout(out)
        return self.fc(out)          # (batch, num_classes)


# ── Load data ────────────────────────────────────────────────────────────────

def load_data():
    def _load(name):
        return np.load(os.path.join(DATA_DIR, f"{name}.npy"))

    X_train = torch.tensor(_load("X_train"))
    y_train = torch.tensor(_load("y_train"), dtype=torch.long)
    X_val   = torch.tensor(_load("X_val"))
    y_val   = torch.tensor(_load("y_val"),   dtype=torch.long)
    X_test  = torch.tensor(_load("X_test"))
    y_test  = torch.tensor(_load("y_test"),  dtype=torch.long)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Vòng lặp huấn luyện ──────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 55)
    print("  [4/4] HUẤN LUYỆN MÔ HÌNH FUZZY-LSTM")
    print("=" * 55)
    print(f"  Device : {DEVICE}"
          + (" — " + torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else ""))
    amp_enabled = DEVICE.type == "cuda"
    print(f"  AMP    : {'Bật (Mixed Precision)' if amp_enabled else 'Tắt (CPU mode)'}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    input_size = X_train.shape[2]
    print(f"  Input  : seq_len={X_train.shape[1]}, features={input_size}")
    print(f"  Tập    : train={len(X_train):,} / val={len(X_val):,} / test={len(X_test):,}")

    # DataLoader — pin_memory tăng tốc transfer CPU→GPU
    _pin = amp_enabled
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=_pin, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=_pin, num_workers=0,
    )

    # Model
    model = FuzzyLSTM(
        input_size  = input_size,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
    ).to(DEVICE)

    # torch.compile (PyTorch ≥ 2.0) — JIT compile, tăng thêm ~15-20%
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  Compile: torch.compile() — OK")
        except Exception:
            pass   # fallback nếu không hỗ trợ

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params : {total_params:,}")

    # Class weights để xử lý mất cân bằng lớp (Giữ chỉ 18%)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=y_train.numpy(),
    )
    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: Giảm={raw_weights[0]:.3f}  Giữ={raw_weights[1]:.3f}  Tăng={raw_weights[2]:.3f}")

    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    # GradScaler cho AMP
    scaler_amp = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_path    = os.path.join(MODEL_DIR, "best_model.pt")
    best_val_loss = float("inf")
    patience_cnt  = 0
    history       = []

    print(f"\n  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val Acc':>8}  {'Time':>6}")
    print(f"  {'-'*52}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(xb)
            loss = criterion(logits.float(), yb)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(xb).float()
                val_loss += criterion(logits, yb).item() * len(xb)
                correct  += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(X_val)
        val_acc   = correct / len(X_val)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"  {epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  "
              f"{val_acc:>7.2%}  {elapsed:>5.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        # Early stopping + lưu model tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            # Lưu state_dict gốc (không phải compiled model)
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw_model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stopping tại epoch {epoch} (patience={PATIENCE})")
                break

    # ── Đánh giá trên tập TEST ────────────────────────────────────────────────
    print(f"\n  Tải model tốt nhất từ {best_path} ...")
    best_model = FuzzyLSTM(
        input_size  = input_size,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    best_model.eval()

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=_pin, num_workers=0,
    )
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = best_model(xb).float()
            all_preds.append(logits.argmax(1).cpu().numpy())

    y_pred_arr = np.concatenate(all_preds)
    y_true_arr = y_test.numpy()
    test_acc   = (y_pred_arr == y_true_arr).mean()

    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")

    # ── Lưu kết quả ──────────────────────────────────────────────────────────
    np.save(os.path.join(MODEL_DIR, "y_true.npy"), y_true_arr)
    np.save(os.path.join(MODEL_DIR, "y_pred.npy"), y_pred_arr)
    pd.DataFrame(history).to_csv(
        os.path.join(MODEL_DIR, "training_history.csv"), index=False
    )
    print(f"  Đã lưu: {MODEL_DIR}/y_true.npy, y_pred.npy, best_model.pt, training_history.csv")
    print("=" * 55)


if __name__ == "__main__":
    train()
