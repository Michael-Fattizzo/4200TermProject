from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from moveEncoding import TOTAL_MOVE_CLASSES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class ShogiPolicyValueNet(nn.Module):
    """
    Compact AlphaZero-style network for 9x9 shogi positions.

    Input shape:  [batch, 44, 9, 9]
    Policy shape: [batch, TOTAL_MOVE_CLASSES]
    Value shape:  [batch], in [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 44,
        num_policy_classes: int = TOTAL_MOVE_CLASSES,
        width: int = 128,
        blocks: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_policy_classes = num_policy_classes
        self.width = width
        self.blocks = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(width, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, num_policy_classes),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(width, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 9 * 9, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(1)
        return policy_logits, value


class ShardDataset(IterableDataset):
    """
    Streams converter.py .pt shards.

    Each shard must contain:
      input_planes:  [N, 44, 9, 9]
      policy_target: [N]
      value_target:  [N]
    """

    def __init__(self, shard_paths: Sequence[Path], shuffle_shards: bool = True, shuffle_in_shard: bool = True):
        super().__init__()
        self.shard_paths = [Path(p) for p in shard_paths]
        self.shuffle_shards = shuffle_shards
        self.shuffle_in_shard = shuffle_in_shard

    def _worker_shards(self) -> List[Path]:
        info = torch.utils.data.get_worker_info()
        paths = list(self.shard_paths)
        if info is None:
            return paths
        return paths[info.id::info.num_workers]

    def __iter__(self):
        paths = self._worker_shards()
        if self.shuffle_shards:
            random.shuffle(paths)

        for shard_path in paths:
            shard = torch.load(shard_path, map_location="cpu")
            x = shard["input_planes"]
            policy = shard["policy_target"]
            value = shard["value_target"]

            n = int(x.shape[0])
            order = torch.randperm(n) if self.shuffle_in_shard else torch.arange(n)
            for idx in order.tolist():
                yield x[idx], policy[idx], value[idx]


def load_manifest_or_glob(data_dir: Path) -> Tuple[List[Path], Dict]:
    manifest_path = data_dir / "manifest.json"
    manifest: Dict = {}

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_paths = [Path(p) for p in manifest.get("shards", [])]
        shard_paths = [p if p.is_absolute() else data_dir / p for p in shard_paths]
    else:
        shard_paths = sorted(data_dir.glob("*.pt"))

    shard_paths = [p for p in shard_paths if p.exists() and p.name != "checkpoint.pt"]
    if not shard_paths:
        raise ValueError(f"No .pt training shards found in {data_dir}")
    return shard_paths, manifest


def split_shards(shards: Sequence[Path], val_fraction: float, seed: int) -> Tuple[List[Path], List[Path]]:
    shards = list(shards)
    rng = random.Random(seed)
    rng.shuffle(shards)
    val_count = max(1, int(round(len(shards) * val_fraction))) if len(shards) > 1 else 0
    return shards[val_count:], shards[:val_count]


def estimate_examples(shards: Sequence[Path], manifest: Dict) -> Optional[int]:
    if manifest.get("total_examples") and manifest.get("shards"):
        frac = len(shards) / max(1, len(manifest["shards"]))
        return int(manifest["total_examples"] * frac)
    return None


def make_loader(
    shards: Sequence[Path],
    batch_size: int,
    workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    ds = ShardDataset(shards, shuffle_shards=shuffle, shuffle_in_shard=shuffle)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    value_loss_weight: float,
    grad_clip: float,
    log_every: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = total_policy = total_value = total_acc = 0.0
    total_seen = 0
    start = time.time()

    for batch_idx, (x, policy_target, value_target) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True).float()
        policy_target = policy_target.to(device, non_blocking=True).long()
        value_target = value_target.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            policy_logits, value_pred = model(x)
            policy_loss = F.cross_entropy(policy_logits, policy_target)
            value_loss = F.mse_loss(value_pred, value_target)
            loss = policy_loss + value_loss_weight * value_loss

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch = x.shape[0]
            acc = (policy_logits.argmax(dim=1) == policy_target).float().mean().item()
            total_loss += loss.item() * batch
            total_policy += policy_loss.item() * batch
            total_value += value_loss.item() * batch
            total_acc += acc * batch
            total_seen += batch

        if log_every > 0 and batch_idx % log_every == 0:
            elapsed = max(1e-6, time.time() - start)
            print(
                f"  batch={batch_idx} examples={total_seen} "
                f"loss={total_loss / total_seen:.4f} "
                f"policy={total_policy / total_seen:.4f} "
                f"value={total_value / total_seen:.4f} "
                f"acc={total_acc / total_seen:.4f} "
                f"ex/s={total_seen / elapsed:.1f}"
            )

        if max_batches is not None and batch_idx >= max_batches:
            break

    return {
        "loss": total_loss / max(1, total_seen),
        "policy_loss": total_policy / max(1, total_seen),
        "value_loss": total_value / max(1, total_seen),
        "policy_acc": total_acc / max(1, total_seen),
        "examples": float(total_seen),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    value_loss_weight: float,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = total_policy = total_value = total_acc = 0.0
    total_seen = 0

    for batch_idx, (x, policy_target, value_target) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True).float()
        policy_target = policy_target.to(device, non_blocking=True).long()
        value_target = value_target.to(device, non_blocking=True).float()

        policy_logits, value_pred = model(x)
        policy_loss = F.cross_entropy(policy_logits, policy_target)
        value_loss = F.mse_loss(value_pred, value_target)
        loss = policy_loss + value_loss_weight * value_loss

        batch = x.shape[0]
        total_loss += loss.item() * batch
        total_policy += policy_loss.item() * batch
        total_value += value_loss.item() * batch
        total_acc += (policy_logits.argmax(dim=1) == policy_target).float().mean().item() * batch
        total_seen += batch

        if max_batches is not None and batch_idx >= max_batches:
            break

    return {
        "loss": total_loss / max(1, total_seen),
        "policy_loss": total_policy / max(1, total_seen),
        "value_loss": total_value / max(1, total_seen),
        "policy_acc": total_acc / max(1, total_seen),
        "examples": float(total_seen),
    }


def save_checkpoint(
    path: Path,
    model: ShogiPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    best_val_loss: float,
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "channels": model.in_channels,
            "num_policy_classes": model.num_policy_classes,
            "width": model.width,
            "blocks": model.blocks,
            "args": vars(args),
        },
        tmp,
    )
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a policy-value shogi AI from converter.py PyTorch shards.")
    parser.add_argument("--data", required=True, help="Directory containing train_*.pt shards and optional manifest.json")
    parser.add_argument("--out", default="shogi_policy_value.pt")
    parser.add_argument("--resume", default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=44)

    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data)
    shard_paths, manifest = load_manifest_or_glob(data_dir)
    train_shards, val_shards = split_shards(shard_paths, args.val_fraction, args.seed)

    if not train_shards:
        train_shards, val_shards = shard_paths, []

    model = ShogiPolicyValueNet(
        in_channels=args.channels,
        num_policy_classes=TOTAL_MOVE_CLASSES,
        width=args.width,
        blocks=args.blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_shards, args.batch_size, args.workers, shuffle=True, pin_memory=pin_memory)
    val_loader = make_loader(val_shards, args.batch_size, args.workers, shuffle=False, pin_memory=pin_memory) if val_shards else None

    print(f"Device: {device}")
    print(f"Train shards: {len(train_shards)} | Val shards: {len(val_shards)}")
    est_train = estimate_examples(train_shards, manifest)
    if est_train is not None:
        print(f"Approx. train examples: {est_train:,}")
    print(f"Model: channels={args.channels}, width={args.width}, blocks={args.blocks}, policy_classes={TOTAL_MOVE_CLASSES}")

    out_path = Path(args.out)
    best_path = out_path.with_name(out_path.stem + "_best" + out_path.suffix)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            value_loss_weight=args.value_loss_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            max_batches=args.max_train_batches,
        )

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                value_loss_weight=args.value_loss_weight,
                max_batches=args.max_val_batches,
            )
        else:
            val_metrics = {"loss": train_metrics["loss"], "policy_loss": 0.0, "value_loss": 0.0, "policy_acc": 0.0, "examples": 0.0}

        print(
            f"Epoch {epoch:03d} complete | "
            f"train loss={train_metrics['loss']:.4f} policy={train_metrics['policy_loss']:.4f} "
            f"value={train_metrics['value_loss']:.4f} acc={train_metrics['policy_acc']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} policy={val_metrics['policy_loss']:.4f} "
            f"value={val_metrics['value_loss']:.4f} acc={val_metrics['policy_acc']:.4f}"
        )

        save_checkpoint(out_path, model, optimizer, epoch, args, best_val_loss)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(best_path, model, optimizer, epoch, args, best_val_loss)
            print(f"Saved new best checkpoint: {best_path}")

    print(f"Saved final checkpoint: {out_path}")


if __name__ == "__main__":
    main()
