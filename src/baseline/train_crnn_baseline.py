import os
import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from crnn_baseline_pytorch import CRNNBaseline


GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}


class GTZANSegmentDatasetCached(Dataset):
    """
    Cached version of the dataset.

    The original dataset reloaded the entire .npy file (40 segments)
    every time a single segment was requested, which created huge
    redundant disk I/O. This version loads everything into RAM once,
    then __getitem__ becomes a simple tensor indexing operation.

    Expected file shape: [40, 128, 130]
    """

    def __init__(self, file_label_pairs, name="dataset"):
        all_segments = []
        all_labels = []

        print(f"[{name}] Loading {len(file_label_pairs)} files into memory...")
        for file_path, label in file_label_pairs:
            data = np.load(file_path)  # [40, 128, 130]

            if data.ndim != 3:
                raise ValueError(f"Expected 3D array, got {data.shape} in {file_path}")

            for segment_idx in range(data.shape[0]):
                all_segments.append(data[segment_idx])
                all_labels.append(label)

        # Stack into a single tensor for fast indexing
        segments = np.stack(all_segments).astype(np.float32)   # [N, 128, 130]
        segments = np.expand_dims(segments, axis=1)             # [N, 1, 128, 130]

        self.segments = torch.from_numpy(segments)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

        size_gb = self.segments.element_size() * self.segments.nelement() / 1e9
        print(f"[{name}] Loaded {len(self.segments)} segments | RAM usage: {size_gb:.2f} GB")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]


def collect_files(data_root):
    file_label_pairs = []

    for genre in GENRES:
        genre_dir = os.path.join(data_root, genre)

        if not os.path.isdir(genre_dir):
            print(f"Warning: folder not found: {genre_dir}")
            continue

        files = glob.glob(os.path.join(genre_dir, "*.npy"))

        for file_path in files:
            file_label_pairs.append((file_path, GENRE_TO_IDX[genre]))

    if len(file_label_pairs) == 0:
        raise ValueError(f"No .npy files found under {data_root}")

    return file_label_pairs


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, all_targets, all_preds


def main():
    # --- Reproducibility ---
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Hyperparameters ---
    data_root = "preprocessed_data"
    batch_size = 64          # was 32; 64 gives better CPU/GPU vectorization
    num_epochs = 12          # upper bound; early stopping will likely cut earlier
    learning_rate = 1e-3
    patience = 3             # stop if val_acc doesn't improve for `patience` epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Split ---
    all_files = collect_files(data_root)
    labels = [label for _, label in all_files]

    train_files, val_files = train_test_split(
        all_files,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Number of song files: {len(all_files)}")
    print(f"Train files: {len(train_files)} | Validation files: {len(val_files)}")

    # --- Datasets ---
    train_dataset = GTZANSegmentDatasetCached(train_files, name="train")
    val_dataset = GTZANSegmentDatasetCached(val_files, name="val")

    print(f"Train segments: {len(train_dataset)} | Validation segments: {len(val_dataset)}")

    # In-memory dataset -> num_workers=0 is fastest (no IPC overhead)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Model / loss / optimizer ---
    model = CRNNBaseline(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training loop with early stopping ---
    best_val_acc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_crnn_baseline.pth")
            print("  -> Best model saved.")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break

    # --- Final evaluation using the BEST checkpoint ---
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load("best_crnn_baseline.pth"))
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print("\nClassification Report:")
    report_text = classification_report(y_true, y_pred, target_names=GENRES)
    print(report_text)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # --- Save artifacts that teammates will need ---
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open("baseline_report.txt", "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))

    print("\nSaved files:")
    print("  - best_crnn_baseline.pth     (model weights, for Mesut)")
    print("  - training_history.json      (per-epoch curves, for Ismail)")
    print("  - baseline_report.txt        (final metrics, for Ismail)")


if __name__ == "__main__":
    main()