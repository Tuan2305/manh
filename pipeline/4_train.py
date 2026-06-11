# 4_train.py — Định nghĩa và huấn luyện mô hình LSTM
#
# Input:  X_train/val/test.npy, y_train/val/test.npy (từ bước 3)
# Output: best_model.pt, y_true.npy, y_pred.npy, training_history.csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, time

DATA_DIR  = "pipeline_data"
MODEL_DIR = "model_output"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Cấu hình huấn luyện ───────────────────────────────────────────────────────
CFG = {
    "hidden_size":   128,
    "num_layers":    2,
    "dropout":       0.3,
    "batch_size":    64,
    "epochs":        100,
    "lr":            5e-4,
    "patience":      15,      # early stopping
    "num_classes":   3,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Kiến trúc LSTM ───────────────────────────────────────────────────────────

class FuzzyLSTM(nn.Module):
    """
    LSTM phân loại xu thế giá cổ phiếu thành 5 nhãn.
    Input shape: (batch, seq_len, num_features)
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
        out, _ = self.lstm(x)        # (batch, seq_len, hidden)
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


# ── Huấn luyện ───────────────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 55)
    print("  [4/5] HUẤN LUYỆN MÔ HÌNH LSTM")
    print("=" * 55)
    print(f"   Device: {DEVICE}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    input_size = X_train.shape[2]
    print(f"  Input size   : {input_size} features")
    print(f"  Sequence len : {X_train.shape[1]}")
    print(f"  Train / Val / Test: {len(X_train)} / {len(X_val)} / {len(X_test)}")

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=CFG["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=CFG["batch_size"], shuffle=False,
    )

    # Model
    model = FuzzyLSTM(
        input_size  = input_size,
        hidden_size = CFG["hidden_size"],
        num_layers  = CFG["num_layers"],
        num_classes = CFG["num_classes"],
        dropout     = CFG["dropout"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Tổng tham số : {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # ── Vòng lặp huấn luyện ──────────────────────────────────────────────────
    history = []
    best_val_loss = float("inf")
    patience_cnt  = 0
    best_path     = os.path.join(MODEL_DIR, "best_model.pt")

    print(f"\n  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val Acc':>8}  {'Time':>6}")
    print(f"  {'-'*50}")

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        # Validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits  = model(xb)
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
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"\n   Early stopping tại epoch {epoch}")
                break

    # ── Đánh giá trên tập test ────────────────────────────────────────────────
    print(f"\n  Đánh giá trên tập TEST với model tốt nhất...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=CFG["batch_size"], shuffle=False,
    )
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            all_preds.append(model(xb).argmax(1).cpu().numpy())

    y_pred_arr = np.concatenate(all_preds)
    y_true_arr = y_test.numpy()

    test_acc = (y_pred_arr == y_true_arr).mean()
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")

    # Lưu kết quả
    np.save(os.path.join(MODEL_DIR, "y_true.npy"), y_true_arr)
    np.save(os.path.join(MODEL_DIR, "y_pred.npy"), y_pred_arr)
    pd.DataFrame(history).to_csv(
        os.path.join(MODEL_DIR, "training_history.csv"), index=False
    )

    print(f"  Đã lưu: {MODEL_DIR}/y_true.npy")
    print(f"   Đã lưu: {MODEL_DIR}/y_pred.npy")
    print(f"   Đã lưu: {MODEL_DIR}/best_model.pt")
    print(f"   Đã lưu: {MODEL_DIR}/training_history.csv")
    print("=" * 55)


if __name__ == "__main__":
    train()
