"""
train_improved.py
=================
MYZ307E — Music Genre Recognition
Author  : Mesut Anlak
Script  : Training pipeline for ImprovedCRNN

Keeps the EXACT same train/val split, random seed, Dataset class,
and evaluation protocol as the baseline so results are directly comparable.

Usage (Google Colab)
--------------------
    # Mount Drive first, then:
    !python train_improved.py \
        --data_dir  /content/drive/MyDrive/MYZ\ GRUP/preprocessed_data \
        --save_path /content/drive/MyDrive/MYZ\ GRUP/best_improved_crnn.pth \
        --epochs 30 \
        --lr 3e-4 \
        --batch_size 64 \
        --dropout 0.3
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from model_improved import build_model

# ---------------------------------------------------------------------------
# Reproducibility — same seed as baseline
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Dataset  (identical to baseline implementation)
# ---------------------------------------------------------------------------
GENRES = ["blues", "classical", "country", "disco",
          "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

class GTZANDataset(Dataset):
    """
    Loads pre-computed .npy mel-spectrogram files produced by Ahmet Selim's
    preprocessing module.  Each file has shape (num_segments, 128, 130).
    Segments are cached in memory at init to avoid repeated disk I/O.
    """

    def __init__(self, file_paths: list, labels: list):
        super().__init__()
        segments_list, targets_list = [], []

        for path, label in zip(file_paths, labels):
            arr = np.load(path)                     # (num_segs, 128, 130)
            segments_list.append(arr)
            targets_list.extend([label] * len(arr))

        all_segs = np.concatenate(segments_list, axis=0)   # (N, 128, 130)
        self.data   = torch.tensor(all_segs,    dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(targets_list, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_data(data_dir: str):
    """Collect file paths and integer labels from the preprocessed_data folder."""
    data_dir = Path(data_dir)
    file_paths, labels = [], []

    for genre_idx, genre in enumerate(GENRES):
        genre_dir = data_dir / genre
        if not genre_dir.exists():
            print(f"  [WARNING] Folder not found: {genre_dir}")
            continue
        npy_files = sorted(genre_dir.glob("*.npy"))
        if not npy_files:
            # Support flat layout: data_dir/<genre>_*.npy
            npy_files = sorted(data_dir.glob(f"{genre}*.npy"))
        for f in npy_files:
            file_paths.append(str(f))
            labels.append(genre_idx)

    print(f"  Found {len(file_paths)} files across {len(set(labels))} genres")
    return file_paths, labels


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            preds  = logits.argmax(dim=1)

            total_loss += loss.item() * len(y)
            correct    += (preds == y).sum().item()
            total      += len(y)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_targets


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    # ── Data ────────────────────────────────────────────────────────────
    print("\n[Data] Loading file list …")
    file_paths, labels = load_data(args.data_dir)

    # Song-level stratified split (same 80/20 ratio and seed as baseline)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )
    print(f"  Train songs: {len(train_paths)} | Val songs: {len(val_paths)}")

    print("[Data] Building in-memory datasets …")
    train_ds = GTZANDataset(train_paths, train_labels)
    val_ds   = GTZANDataset(val_paths,   val_labels)
    print(f"  Train segments: {len(train_ds):,} | Val segments: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    print("\n[Model] Building ImprovedCRNN …")
    model = build_model(num_classes=len(GENRES),
                        dropout=args.dropout,
                        lstm_hidden=args.lstm_hidden)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── Optimiser & scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    # Cosine annealing LR scheduler (improves convergence vs. fixed LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training ─────────────────────────────────────────────────────────
    history       = []
    best_val_acc  = 0.0
    patience_cnt  = 0
    patience      = args.patience

    print(f"\n[Train] Starting — {args.epochs} epochs, patience={patience}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            preds            = logits.argmax(dim=1)
            running_loss    += loss.item() * len(y)
            running_correct += (preds == y).sum().item()
            running_total   += len(y)

        scheduler.step()

        train_loss = running_loss    / running_total
        train_acc  = running_correct / running_total
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        })

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            patience_cnt = 0
            flag = "  ← best"
        else:
            patience_cnt += 1

        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f}  acc {val_acc:.4f} | "
              f"lr {lr_now:.2e}{flag}")

        if patience_cnt >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    # ── Final evaluation ─────────────────────────────────────────────────
    print(f"\n[Eval] Best validation accuracy: {best_val_acc:.4f}")
    print("[Eval] Loading best checkpoint for final report …")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    _, _, all_preds, all_targets = evaluate(model, val_loader, criterion, device)

    report = classification_report(all_targets, all_preds,
                                   target_names=GENRES, digits=3)
    cm     = confusion_matrix(all_targets, all_preds)

    print("\nClassification Report:\n" + report)
    print("Confusion Matrix:\n", cm)

    # ── Save artefacts ────────────────────────────────────────────────────
    history_path = str(Path(args.save_path).with_name("improved_training_history.json"))
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Saved] Training history → {history_path}")

    report_path = str(Path(args.save_path).with_name("improved_baseline_report.txt"))
    with open(report_path, "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n\n")
        f.write("Classification Report:\n" + report + "\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")
    print(f"[Saved] Evaluation report  → {report_path}")
    print(f"[Saved] Best model weights → {args.save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train ImprovedCRNN on GTZAN")
    p.add_argument("--data_dir",    type=str, required=True,
                   help="Path to preprocessed_data folder (contains per-genre .npy files)")
    p.add_argument("--save_path",   type=str,
                   default="best_improved_crnn.pth",
                   help="Where to save the best model checkpoint")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--lstm_hidden", type=int,   default=256)
    p.add_argument("--patience",    type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
