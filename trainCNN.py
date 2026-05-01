
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from CNNClassifier import SmallBoardCNN, normalize_piece_crop_gray


class PieceCropDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.samples: List[Tuple[Path, int]] = []
        labels = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.label_to_index: Dict[str, int] = {label: i for i, label in enumerate(labels)}

        for label in labels:
            label_dir = self.root / label
            for path in sorted(label_dir.glob("*")):
                if path.is_file():
                    self.samples.append((path, self.label_to_index[label]))

        if not self.samples:
            raise ValueError(f"No training images found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")

        crop = normalize_piece_crop_gray(img)
        if crop is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        x = torch.from_numpy(crop).float().unsqueeze(0) / 255.0
        return x, torch.tensor(y, dtype=torch.long)


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    ce = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad()

        logits = model(x)
        loss = ce(logits, y)

        if training:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        batch = x.shape[0]
        total_loss += loss.item() * batch
        total_acc += (preds == y).float().mean().item() * batch
        total += batch

    return total_loss / total, total_acc / total


def main():
    parser = argparse.ArgumentParser(description="Train a small CNN board-piece classifier.")
    parser.add_argument("--data", default="templates/board")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="board_cnn.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = PieceCropDataset(args.data)
    train_len = max(1, int(0.9 * len(ds)))
    val_len = max(1, len(ds) - train_len)
    if train_len + val_len > len(ds):
        train_len = len(ds) - val_len

    train_ds, val_ds = random_split(ds, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallBoardCNN(num_classes=len(ds.label_to_index)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, device)
        va_loss, va_acc = run_epoch(model, val_loader, None, device)
        print(f"Epoch {epoch:03d} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "label_to_index": ds.label_to_index,
            }

    if best_state is None:
        raise RuntimeError("No model state saved.")

    torch.save(best_state, args.out)
    print(f"Saved classifier to {args.out}")


if __name__ == "__main__":
    main()
