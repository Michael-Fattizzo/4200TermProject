from __future__ import annotations

import argparse
import copy
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
    BLACK,
    WHITE,
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


def evaluate_and_expand_many(
    nodes: List[MCTSNode],
    model: nn.Module,
    device: torch.device,
) -> List[float]:
    values: List[Optional[float]] = [None for _ in nodes]
    batch_nodes: List[Tuple[int, MCTSNode]] = []
    batch_inputs = []

    for i, node in enumerate(nodes):
        if node.expanded:
            if node.terminal_value is not None:
                values[i] = node.terminal_value
            else:
                values[i] = 0.0
            continue

        tv = terminal_value(node.position)
        if tv is not None:
            node.terminal_value = tv
            node.expanded = True
            values[i] = tv
            continue

        batch_nodes.append((i, node))
        batch_inputs.append(encode_position(node.position, ply_index=node.ply))

    if batch_nodes:
        x = torch.tensor(batch_inputs, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits_batch, value_batch = model(x)

        for batch_i, (original_i, node) in enumerate(batch_nodes):
            legal_moves = generate_legal_moves(node.position, node.position.side_to_move)
            move_ids = [encode_move_obj(mv) for mv in legal_moves]

            legal_ids = torch.tensor(move_ids, dtype=torch.long, device=device)
            legal_logits = logits_batch[batch_i].index_select(0, legal_ids)
            priors = torch.softmax(legal_logits, dim=0).detach().cpu().tolist()

            for mv, move_id, prior in zip(legal_moves, move_ids, priors):
                node.edges[move_id] = MCTSEdge(move=mv, prior=float(prior))

            node.expanded = True
            values[original_i] = float(value_batch[batch_i].detach().item())

    return [float(v) for v in values]


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


def select_leaf_for_batch(
    root: MCTSNode,
    c_puct: float,
) -> Tuple[MCTSNode, List[MCTSEdge]]:
    node = root
    path: List[MCTSEdge] = []

    while node.expanded and node.terminal_value is None:
        _, edge = select_edge(node, c_puct)

        edge.visit_count += 1
        path.append(edge)

        if edge.child is None:
            next_pos = apply_move(node.position, edge.move)
            edge.child = MCTSNode(next_pos, ply=node.ply + 1)
            return edge.child, path

        node = edge.child

    return node, path


def backup_path(
    path: List[MCTSEdge],
    leaf_value: float,
) -> None:
    value = leaf_value

    for edge in reversed(path):
        value_for_parent = -value
        edge.value_sum += value_for_parent
        value = value_for_parent


def run_mcts(
    model: nn.Module,
    position: Position,
    device: torch.device,
    ply: int,
    simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    batch_size: int = 32,
) -> MCTSNode:
    root = MCTSNode(position=position, ply=ply)
    evaluate_and_expand_many([root], model, device)

    if root.terminal_value is not None:
        return root

    add_root_dirichlet_noise(
        root=root,
        alpha=dirichlet_alpha,
        epsilon=dirichlet_epsilon,
    )

    completed = 0

    while completed < simulations:
        current_batch = min(batch_size, simulations - completed)

        leaves: List[MCTSNode] = []
        paths: List[List[MCTSEdge]] = []

        for _ in range(current_batch):
            leaf, path = select_leaf_for_batch(root, c_puct)
            leaves.append(leaf)
            paths.append(path)

        leaf_values = evaluate_and_expand_many(leaves, model, device)

        for path, leaf_value in zip(paths, leaf_values):
            backup_path(path, leaf_value)

        completed += current_batch

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


def clone_model(model: ShogiPolicyValueNet, device: torch.device) -> ShogiPolicyValueNet:
    cloned = ShogiPolicyValueNet(
        in_channels=model.in_channels,
        num_policy_classes=model.num_policy_classes,
        width=model.width,
        blocks=model.blocks,
    ).to(device)

    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    cloned.eval()
    return cloned


def play_eval_game(
    candidate_model: nn.Module,
    best_model: nn.Module,
    device: torch.device,
    simulations: int,
    c_puct: float,
    max_plies: int,
    candidate_side: str,
    mcts_batch_size: int,
) -> str:
    position = initial_position()

    for ply in range(1, max_plies + 1):
        result = game_result(position)
        if result is not None:
            return result

        active_model = candidate_model if position.side_to_move == candidate_side else best_model

        root = run_mcts(
            model=active_model,
            position=position,
            device=device,
            ply=ply,
            simulations=simulations,
            c_puct=c_puct,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.0,
            batch_size=mcts_batch_size,
        )

        if root.terminal_value is not None:
            result = game_result(position)
            return result if result is not None else "draw"

        move, _, _ = select_move_from_visits(root=root, temperature=0.0)
        position = apply_move(position, move)
        #print(f"  eval ply {ply} | side={position.side_to_move}")

    return "draw"


def evaluate_candidate_vs_best(
    candidate_model: nn.Module,
    best_model: nn.Module,
    device: torch.device,
    games: int,
    simulations: int,
    c_puct: float,
    max_plies: int,
    mcts_batch_size: int,
) -> Dict[str, float]:
    if games <= 0:
        return {
            "score": 1.0,
            "wins": 0.0,
            "draws": 0.0,
            "losses": 0.0,
            "games": 0.0,
        }

    wins = 0
    draws = 0
    losses = 0

    for game_idx in range(1, games + 1):
        candidate_side = BLACK if game_idx % 2 == 1 else WHITE

        result = play_eval_game(
            candidate_model=candidate_model,
            best_model=best_model,
            device=device,
            simulations=simulations,
            c_puct=c_puct,
            max_plies=max_plies,
            candidate_side=candidate_side,
            mcts_batch_size=mcts_batch_size,
        )

        if result == "draw":
            draws += 1
        elif result == candidate_side:
            wins += 1
        else:
            losses += 1

        score = (wins + 0.5 * draws) / game_idx

        print(
            f"Eval game {game_idx}/{games} | "
            f"candidate_side={candidate_side} | "
            f"result={result} | "
            f"score={score:.3f}"
        )

    score = (wins + 0.5 * draws) / games

    return {
        "score": float(score),
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "games": float(games),
    }


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
    mcts_batch_size: int,
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
            batch_size=mcts_batch_size,
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


def save_selfplay_checkpoint(
    path: str,
    model: nn.Module,
    ckpt: Dict,
    iteration: int,
    accepted_iterations: int,
    simulations: int,
    extra: Optional[Dict] = None,
) -> None:
    data = {
        "model_state_dict": model.state_dict(),
        "channels": ckpt["channels"],
        "num_policy_classes": ckpt.get("num_policy_classes", TOTAL_MOVE_CLASSES),
        "width": ckpt["width"],
        "blocks": ckpt["blocks"],
        "selfplay_iterations": iteration,
        "accepted_selfplay_iterations": accepted_iterations,
        "mcts_simulations": simulations,
    }

    if extra:
        data.update(extra)

    torch.save(data, path)


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero-style MCTS self-play training for shogi."
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="shogi_alphazero_selfplay.pt")

    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--mcts-batch-size", type=int, default=32)
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

    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument("--eval-simulations", type=int, default=None)
    parser.add_argument("--eval-max-plies", type=int, default=None)
    parser.add_argument("--min-eval-score", type=float, default=0.45)
    parser.add_argument("--stop-on-regression", action="store_true")

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

    accepted_best_state = copy.deepcopy(model.state_dict())
    accepted_iterations = 0

    print(f"Device: {device}")
    print(f"Loaded model: {args.model}")
    print(f"MCTS simulations per move: {args.simulations}")
    print(f"MCTS batch size: {args.mcts_batch_size}")

    for iteration in range(1, args.iterations + 1):
        previous_best_model = clone_model(model, device)
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
                mcts_batch_size=args.mcts_batch_size,
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

        save_selfplay_checkpoint(
            path=args.out,
            model=model,
            ckpt=ckpt,
            iteration=iteration,
            accepted_iterations=accepted_iterations,
            simulations=args.simulations,
        )

        print(f"Saved candidate model to {args.out}")

        if args.eval_games > 0:
            eval_simulations = args.eval_simulations or args.simulations
            eval_max_plies = args.eval_max_plies or args.max_plies

            print(
                f"Evaluating candidate vs previous accepted model | "
                f"games={args.eval_games} | simulations={eval_simulations}"
            )

            eval_metrics = evaluate_candidate_vs_best(
                candidate_model=model,
                best_model=previous_best_model,
                device=device,
                games=args.eval_games,
                simulations=eval_simulations,
                c_puct=args.c_puct,
                max_plies=eval_max_plies,
                mcts_batch_size=args.mcts_batch_size,
            )

            print(
                f"Evaluation complete | score={eval_metrics['score']:.3f} | "
                f"wins={int(eval_metrics['wins'])} | "
                f"draws={int(eval_metrics['draws'])} | "
                f"losses={int(eval_metrics['losses'])}"
            )

            if eval_metrics["score"] < args.min_eval_score:
                print(
                    f"Candidate rejected: score {eval_metrics['score']:.3f} "
                    f"< min_eval_score {args.min_eval_score:.3f}"
                )

                model.load_state_dict(accepted_best_state)

                save_selfplay_checkpoint(
                    path=args.out,
                    model=model,
                    ckpt=ckpt,
                    iteration=iteration,
                    accepted_iterations=accepted_iterations,
                    simulations=args.simulations,
                    extra={
                        "rejected_iteration": iteration,
                        "rejection_score": eval_metrics["score"],
                    },
                )

                print(f"Restored previous accepted model and saved it to {args.out}")

                if args.stop_on_regression:
                    print("Stopping self-play because candidate performance regressed.")
                    break

            else:
                accepted_best_state = copy.deepcopy(model.state_dict())
                accepted_iterations += 1

                best_path = args.out.replace(".pt", "_best.pt")

                save_selfplay_checkpoint(
                    path=best_path,
                    model=model,
                    ckpt=ckpt,
                    iteration=iteration,
                    accepted_iterations=accepted_iterations,
                    simulations=args.simulations,
                    extra={"eval_score": eval_metrics["score"]},
                )

                print(f"Candidate accepted and saved to {best_path}")


if __name__ == "__main__":
    main()