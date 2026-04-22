from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


class ShogiPolicyValueDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.examples: List[Tuple[torch.Tensor, int, float]] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                x = torch.tensor(obj["input_planes"], dtype=torch.float32)
                if x.ndim != 3 or x.shape[-2:] != (9, 9):
                    raise ValueError(
                        f"Line {line_no}: input_planes must have shape [C, 9, 9], got {tuple(x.shape)}"
                    )

                policy_target = int(obj["policy_target"])
                value_target = float(obj.get("value_target", 0.0))
                self.examples.append((x, policy_target, value_target))

        if not self.examples:
            raise ValueError("Dataset is empty.")

        self.channels = self.examples[0][0].shape[0]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        x, p, v = self.examples[idx]
        return x, torch.tensor(p, dtype=torch.long), torch.tensor(v, dtype=torch.float32)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ShogiPolicyValueNet(nn.Module):
    def __init__(self, in_channels: int, num_policy_classes: int, width: int = 128, blocks: int = 6):
        super().__init__()

        trunk = [
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            trunk.append(ResidualBlock(width))
        self.trunk = nn.Sequential(*trunk)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(width, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, num_policy_classes),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(width, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        policy_logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return policy_logits, value


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    value_loss_weight: float,
):
    training = optimizer is not None
    model.train(training)

    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for x, policy_target, value_target in loader:
        x = x.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)

        if training:
            optimizer.zero_grad()

        policy_logits, value_pred = model(x)
        policy_loss = ce_loss_fn(policy_logits, policy_target)
        value_loss = mse_loss_fn(value_pred, value_target)
        loss = policy_loss + value_loss_weight * value_loss

        if training:
            loss.backward()
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_policy_loss += policy_loss.item() * batch_size
        total_value_loss += value_loss.item() * batch_size
        total_acc += accuracy_from_logits(policy_logits, policy_target) * batch_size
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "policy_loss": total_policy_loss / total_count,
        "value_loss": total_value_loss / total_count,
        "policy_acc": total_acc / total_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a shogi policy-value network.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--num-policy-classes", required=True, type=int, help="Size of move vocabulary.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--value-loss-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="shogi_policy_value.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = ShogiPolicyValueDataset(args.dataset)
    train_len = max(1, int(len(dataset) * args.train_split))
    val_len = max(1, len(dataset) - train_len)
    if train_len + val_len > len(dataset):
        train_len = len(dataset) - val_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShogiPolicyValueNet(
        in_channels=dataset.channels,
        num_policy_classes=args.num_policy_classes,
        width=args.width,
        blocks=args.blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -math.inf
    best_state = None

    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"Channels: {dataset.channels}")
    print(f"Policy classes: {args.num_policy_classes}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            value_loss_weight=args.value_loss_weight,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            value_loss_weight=args.value_loss_weight,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['policy_acc']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['policy_acc']:.4f}"
        )

        if val_metrics["policy_acc"] > best_val_acc:
            best_val_acc = val_metrics["policy_acc"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "channels": dataset.channels,
                "num_policy_classes": args.num_policy_classes,
                "width": args.width,
                "blocks": args.blocks,
            }

    if best_state is None:
        raise RuntimeError("Training ended without a saved model.")

    torch.save(best_state, args.out)
    print(f"Saved best model to {args.out}")


if __name__ == "__main__":
    main()
