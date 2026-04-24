from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from converter import encode_position, initial_position
from moveEncoding import TOTAL_MOVE_CLASSES, encode_move_obj
from shogiEngine import (
    Move,
    Position,
    apply_move,
    generate_legal_moves,
    in_check,
    opponent,
)
from train_shogi import ShogiPolicyValueNet


@dataclass
class SelfPlayExample:
    x: torch.Tensor
    policy_indices: torch.Tensor
    policy_probs: torch.Tensor
    value_target: float


class SelfPlayDataset(Dataset):
    def __init__(self, examples: List[SelfPlayExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        policy = torch.zeros(TOTAL_MOVE_CLASSES, dtype=torch.float32)
        policy[ex.policy_indices] = ex.policy_probs

        return (
            ex.x,
            policy,
            torch.tensor(ex.value_target, dtype=torch.float32),
        )


class MCTSEdge:
    def __init__(self, move: Move, prior: float):
        self.move = move
        self.prior = prior
        self.child: Optional[MCTSNode] = None
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSNode:
    def __init__(self, position: Position, ply: int):
        self.position = position
        self.ply = ply
        self.edges: Dict[int, MCTSEdge] = {}
        self.expanded = False
        self.terminal_value: Optional[float] = None

    @property
    def visit_count(self) -> int:
        return sum(edge.visit_count for edge in self.edges.values())


def terminal_value(position: Position) -> Optional[float]:
    legal = generate_legal_moves(position, position.side_to_move)

    if legal:
        return None

    if in_check(position, position.side_to_move):
        return -1.0

    return 0.0


def evaluate_and_expand(
    node: MCTSNode,
    model: nn.Module,
    device: torch.device,
) -> float:
    tv = terminal_value(node.position)
    if tv is not None:
        node.terminal_value = tv
        node.expanded = True
        return tv

    legal_moves = generate_legal_moves(node.position, node.position.side_to_move)
    move_ids = [encode_move_obj(mv) for mv in legal_moves]

    x = torch.tensor(
        encode_position(node.position, ply_index=node.ply),
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, value = model(x)

    logits = logits[0].cpu()
    value_scalar = float(value.item())

    legal_logits = logits[move_ids]
    priors = torch.softmax(legal_logits, dim=0).tolist()

    for mv, move_id, prior in zip(legal_moves, move_ids, priors):
        node.edges[move_id] = MCTSEdge(move=mv, prior=float(prior))

    node.expanded = True
    return value_scalar


def select_edge(node: MCTSNode, c_puct: float) -> Tuple[int, MCTSEdge]:
    total_visits = max(1, node.visit_count)
    sqrt_total = math.sqrt(total_visits)

    best_score = -float("inf")
    best_item = None

    for move_id, edge in node.edges.items():
        u_value = c_puct * edge.prior * sqrt_total / (1 + edge.visit_count)
        score = edge.q_value + u_value

        if score > best_score:
            best_score = score
            best_item = (move_id, edge)

    if best_item is None:
        raise RuntimeError("MCTS selection failed: node has no edges.")

    return best_item


def add_root_dirichlet_noise(
    root: MCTSNode,
    alpha: float,
    epsilon: float,
) -> None:
    if not root.edges:
        return

    move_ids = list(root.edges.keys())
    noise = torch.distributions.Dirichlet(
        torch.full((len(move_ids),), alpha)
    ).sample().tolist()

    for move_id, n in zip(move_ids, noise):
        edge = root.edges[move_id]
        edge.prior = (1.0 - epsilon) * edge.prior + epsilon * float(n)


def run_simulation(
    root: MCTSNode,
    model: nn.Module,
    device: torch.device,
    c_puct: float,
) -> float:
    if not root.expanded:
        return evaluate_and_expand(root, model, device)

    if root.terminal_value is not None:
        return root.terminal_value

    _, edge = select_edge(root, c_puct)

    if edge.child is None:
        next_pos = apply_move(root.position, edge.move)
        edge.child = MCTSNode(next_pos, ply=root.ply + 1)

    child_value = run_simulation(
        root=edge.child,
        model=model,
        device=device,
        c_puct=c_puct,
    )

    value_for_parent = -child_value

    edge.visit_count += 1
    edge.value_sum += value_for_parent

    return value_for_parent


def run_mcts(
    model: nn.Module,
    position: Position,
    device: torch.device,
    ply: int,
    simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> MCTSNode:
    root = MCTSNode(position=position, ply=ply)
    evaluate_and_expand(root, model, device)

    if root.terminal_value is not None:
        return root

    add_root_dirichlet_noise(
        root=root,
        alpha=dirichlet_alpha,
        epsilon=dirichlet_epsilon,
    )

    for _ in range(simulations):
        run_simulation(
            root=root,
            model=model,
            device=device,
            c_puct=c_puct,
        )

    return root


def select_move_from_visits(
    root: MCTSNode,
    temperature: float,
) -> Tuple[Move, torch.Tensor, torch.Tensor]:
    if not root.edges:
        raise RuntimeError("Cannot select move from terminal root.")

    move_ids = list(root.edges.keys())
    visits = torch.tensor(
        [root.edges[mid].visit_count for mid in move_ids],
        dtype=torch.float32,
    )

    if visits.sum().item() <= 0:
        visits = torch.ones_like(visits)

    if temperature <= 0:
        best = int(torch.argmax(visits).item())
        probs = torch.zeros_like(visits)
        probs[best] = 1.0
        chosen_idx = best
    else:
        adjusted = visits.pow(1.0 / temperature)
        probs = adjusted / adjusted.sum()
        chosen_idx = int(torch.multinomial(probs, 1).item())

    chosen_move_id = move_ids[chosen_idx]
    chosen_move = root.edges[chosen_move_id].move

    return (
        chosen_move,
        torch.tensor(move_ids, dtype=torch.long),
        probs.detach().cpu(),
    )


def value_for_side(winner: str, side: str) -> float:
    if winner == "draw":
        return 0.0
    return 1.0 if winner == side else -1.0


def game_result(position: Position) -> Optional[str]:
    legal = generate_legal_moves(position, position.side_to_move)

    if legal:
        return None

    if in_check(position, position.side_to_move):
        return opponent(position.side_to_move)

    return "draw"


def play_self_game(
    model: nn.Module,
    device: torch.device,
    simulations: int,
    c_puct: float,
    max_plies: int,
    temperature: float,
    temperature_drop_ply: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> List[SelfPlayExample]:
    position = initial_position()

    history: List[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]
    ] = []

    for ply in range(1, max_plies + 1):
        result = game_result(position)

        if result is not None:
            return [
                SelfPlayExample(
                    x=x,
                    policy_indices=policy_indices,
                    policy_probs=policy_probs,
                    value_target=value_for_side(result, side_to_move),
                )
                for x, policy_indices, policy_probs, side_to_move in history
            ]

        root = run_mcts(
            model=model,
            position=position,
            device=device,
            ply=ply,
            simulations=simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

        move_temperature = temperature if ply <= temperature_drop_ply else 0.0

        move, policy_indices, policy_probs = select_move_from_visits(
            root=root,
            temperature=move_temperature,
        )

        x = torch.tensor(
            encode_position(position, ply_index=ply),
            dtype=torch.float32,
        )

        history.append(
            (
                x,
                policy_indices,
                policy_probs,
                position.side_to_move,
            )
        )

        position = apply_move(position, move)

    return [
        SelfPlayExample(
            x=x,
            policy_indices=policy_indices,
            policy_probs=policy_probs,
            value_target=0.0,
        )
        for x, policy_indices, policy_probs, _ in history
    ]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_loss_weight: float,
    grad_clip: float,
):
    model.train()

    mse = nn.MSELoss()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total = 0

    for x, policy_target, value_target in loader:
        x = x.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)

        optimizer.zero_grad(set_to_none=True)

        policy_logits, value_pred = model(x)

        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

        value_loss = mse(value_pred, value_target)
        loss = policy_loss + value_loss_weight * value_loss

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch = x.shape[0]
        total_loss += loss.item() * batch
        total_policy_loss += policy_loss.item() * batch
        total_value_loss += value_loss.item() * batch
        total += batch

    return {
        "loss": total_loss / total,
        "policy_loss": total_policy_loss / total,
        "value_loss": total_value_loss / total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero-style MCTS self-play training for shogi."
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="shogi_alphazero_selfplay.pt")

    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--max-plies", type=int, default=300)

    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-ply", type=int, default=30)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-epochs-per-iteration", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device)

    model = ShogiPolicyValueNet(
        in_channels=ckpt["channels"],
        num_policy_classes=ckpt.get("num_policy_classes", TOTAL_MOVE_CLASSES),
        width=ckpt["width"],
        blocks=ckpt["blocks"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    replay_buffer: List[SelfPlayExample] = []

    print(f"Device: {device}")
    print(f"Loaded model: {args.model}")
    print(f"MCTS simulations per move: {args.simulations}")

    for iteration in range(1, args.iterations + 1):
        new_examples: List[SelfPlayExample] = []

        for game_idx in range(1, args.games_per_iteration + 1):
            examples = play_self_game(
                model=model,
                device=device,
                simulations=args.simulations,
                c_puct=args.c_puct,
                max_plies=args.max_plies,
                temperature=args.temperature,
                temperature_drop_ply=args.temperature_drop_ply,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_epsilon=args.dirichlet_epsilon,
            )

            new_examples.extend(examples)

            print(
                f"Iteration {iteration} | "
                f"game {game_idx}/{args.games_per_iteration} | "
                f"positions={len(examples)}"
            )

        replay_buffer.extend(new_examples)

        if len(replay_buffer) > args.replay_size:
            replay_buffer = replay_buffer[-args.replay_size:]

        dataset = SelfPlayDataset(replay_buffer)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        for train_epoch_idx in range(1, args.train_epochs_per_iteration + 1):
            metrics = train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=device,
                value_loss_weight=args.value_loss_weight,
                grad_clip=args.grad_clip,
            )

            print(
                f"Iteration {iteration:03d} | "
                f"train epoch {train_epoch_idx}/{args.train_epochs_per_iteration} | "
                f"examples={len(replay_buffer)} | "
                f"loss={metrics['loss']:.4f} | "
                f"policy={metrics['policy_loss']:.4f} | "
                f"value={metrics['value_loss']:.4f}"
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "channels": ckpt["channels"],
                "num_policy_classes": ckpt.get("num_policy_classes", TOTAL_MOVE_CLASSES),
                "width": ckpt["width"],
                "blocks": ckpt["blocks"],
                "selfplay_iterations": iteration,
                "mcts_simulations": args.simulations,
            },
            args.out,
        )

        print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()