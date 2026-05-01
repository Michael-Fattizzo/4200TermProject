from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from converter import (
    ParsedMoveSpec,
    collect_input_files,
    encode_position,
    infer_result_from_moves,
    initial_position,
    parse_game,
    read_kif_file,
    resolve_move,
    value_target_for_side,
)

from moveEncoding import TOTAL_MOVE_CLASSES, encode_move_obj

from shogiEngine import (
    BLACK,
    WHITE,
    Move,
    Piece,
    Position,
    apply_move,
)


SFEN_HAND_ORDER = ["R", "B", "G", "S", "N", "L", "P"]


def square_to_usi(square: Tuple[int, int]) -> str:
    row, col = square
    file_ = 9 - col
    rank = chr(ord("a") + row)
    return f"{file_}{rank}"


def usi_to_square(s: str) -> Tuple[int, int]:
    if len(s) != 2:
        raise ValueError(f"Invalid USI square: {s}")

    file_char = s[0]
    rank_char = s[1]

    if not file_char.isdigit():
        raise ValueError(f"Invalid USI file: {s}")

    file_ = int(file_char)
    row = ord(rank_char) - ord("a")
    col = 9 - file_

    if not (0 <= row < 9 and 0 <= col < 9):
        raise ValueError(f"USI square out of range: {s}")

    return row, col


def piece_to_sfen(piece: Piece) -> str:
    letter = piece.kind.upper()

    if piece.owner == WHITE:
        letter = letter.lower()

    if piece.promoted:
        return "+" + letter

    return letter


def board_to_sfen(position: Position) -> str:
    rows: List[str] = []

    for r in range(9):
        row_parts: List[str] = []
        empty_count = 0

        for c in range(9):
            piece = position.board[r][c]

            if piece is None:
                empty_count += 1
                continue

            if empty_count > 0:
                row_parts.append(str(empty_count))
                empty_count = 0

            row_parts.append(piece_to_sfen(piece))

        if empty_count > 0:
            row_parts.append(str(empty_count))

        rows.append("".join(row_parts))

    return "/".join(rows)


def hands_to_sfen(position: Position) -> str:
    parts: List[str] = []

    for side in (BLACK, WHITE):
        for kind in SFEN_HAND_ORDER:
            count = int(position.hands[side].get(kind, 0))

            if count <= 0:
                continue

            letter = kind if side == BLACK else kind.lower()

            if count == 1:
                parts.append(letter)
            else:
                parts.append(f"{count}{letter}")

    return "".join(parts) if parts else "-"


def position_to_sfen(position: Position, move_number: int) -> str:
    board = board_to_sfen(position)
    side = "b" if position.side_to_move == BLACK else "w"
    hands = hands_to_sfen(position)
    return f"{board} {side} {hands} {move_number}"


def usi_to_move(position: Position, usi: str) -> Move:
    side = position.side_to_move

    if usi in {"resign", "win", "draw", "none"}:
        raise ValueError(f"YaneuraOu returned non-move bestmove: {usi}")

    if "*" in usi:
        piece_kind = usi[0].upper()
        to_sq = usi_to_square(usi[2:4])

        return Move(
            from_sq=None,
            to_sq=to_sq,
            piece=piece_kind,
            side=side,
            promote=False,
            drop=True,
            captured=None,
        )

    promote = usi.endswith("+")
    raw = usi[:-1] if promote else usi

    if len(raw) != 4:
        raise ValueError(f"Invalid USI move: {usi}")

    from_sq = usi_to_square(raw[:2])
    to_sq = usi_to_square(raw[2:4])

    moving_piece = position.piece_at(from_sq)
    if moving_piece is None:
        raise ValueError(f"No piece on source square for USI move {usi}")

    if moving_piece.owner != side:
        raise ValueError(f"USI move {usi} tries to move opponent piece")

    target = position.piece_at(to_sq)

    return Move(
        from_sq=from_sq,
        to_sq=to_sq,
        piece=moving_piece.kind,
        side=side,
        promote=promote,
        drop=False,
        captured=target.code() if target else None,
    )


class YaneuraOuTeacher:
    def __init__(
        self,
        engine_path: str,
        byoyomi_ms: int = 50,
        nodes: Optional[int] = None,
        threads: Optional[int] = None,
        hash_mb: Optional[int] = None,
        multipv: int = 1,
    ):
        self.engine_path = engine_path
        self.byoyomi_ms = byoyomi_ms
        self.nodes = nodes
        self.threads = threads
        self.hash_mb = hash_mb
        self.multipv = multipv

        engine_file = Path(engine_path)

        self.process = subprocess.Popen(
            [str(engine_file)],
            cwd=str(engine_file.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._initialize_engine()

    def _write(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Engine stdin is closed.")

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _readline(self) -> str:
        if self.process.stdout is None:
            raise RuntimeError("Engine stdout is closed.")

        line = self.process.stdout.readline()
        if line == "":
            raise RuntimeError("YaneuraOu process terminated unexpectedly.")

        return line.strip()

    def _wait_for(self, token: str) -> None:
        while True:
            line = self._readline()
            if token in line:
                return

    def _initialize_engine(self) -> None:
        self._write("usi")
        self._wait_for("usiok")

        if self.threads is not None:
            self._write(f"setoption name Threads value {self.threads}")

        if self.hash_mb is not None:
            self._write(f"setoption name Hash value {self.hash_mb}")

        if self.multipv != 1:
            self._write(f"setoption name MultiPV value {self.multipv}")

        self._write("isready")
        self._wait_for("readyok")

        self._write("usinewgame")

    def bestmove(self, sfen: str) -> str:
        self._write(f"position sfen {sfen}")

        if self.nodes is not None:
            self._write(f"go nodes {self.nodes}")
        else:
            self._write(f"go byoyomi {self.byoyomi_ms}")

        best = None

        while True:
            line = self._readline()

            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best = parts[1]
                break

        if best is None:
            raise RuntimeError("YaneuraOu did not return bestmove.")

        return best

    def close(self) -> None:
        try:
            self._write("quit")
        except Exception:
            pass

        try:
            self.process.terminate()
        except Exception:
            pass


def write_shard(
    output_dir: Path,
    task_id: int,
    local_shard_id: int,
    shard_inputs: List,
    shard_policies: List[int],
    shard_values: List[float],
) -> Path:
    shard_path = output_dir / f"teacher_t{task_id:06d}_{local_shard_id:04d}.pt"
    tmp_path = shard_path.with_suffix(".tmp")

    data = {
        "input_planes": torch.tensor(shard_inputs, dtype=torch.float32),
        "policy_target": torch.tensor(shard_policies, dtype=torch.long),
        "value_target": torch.tensor(shard_values, dtype=torch.float32),
    }

    torch.save(data, tmp_path)
    os.replace(tmp_path, shard_path)
    return shard_path


@dataclass
class WorkerArgs:
    task_id: int
    path_strings: List[str]
    output_dir: str
    engine_path: str
    shard_examples: int
    min_free_gb: float
    byoyomi_ms: int
    nodes: Optional[int]
    engine_threads: Optional[int]
    hash_mb: Optional[int]
    max_positions_per_game: Optional[int]
    value_mode: str


def iter_teacher_examples_from_game(
    parsed_moves: List[ParsedMoveSpec],
    teacher: YaneuraOuTeacher,
    max_positions_per_game: Optional[int],
    value_mode: str,
) -> Iterable[Dict]:
    winner = infer_result_from_moves(parsed_moves)
    position = initial_position()

    emitted = 0

    for ply_index, spec in enumerate(parsed_moves, start=1):
        if spec.special is not None:
            break

        if max_positions_per_game is not None and emitted >= max_positions_per_game:
            break

        sfen = position_to_sfen(position, move_number=ply_index)
        teacher_usi = teacher.bestmove(sfen)
        teacher_move = usi_to_move(position, teacher_usi)

        if value_mode == "game":
            value_target = value_target_for_side(winner, position.side_to_move)
        elif value_mode == "zero":
            value_target = 0.0
        else:
            raise ValueError(f"Unsupported value_mode: {value_mode}")

        yield {
            "input_planes": encode_position(position, ply_index=ply_index),
            "policy_target": encode_move_obj(teacher_move),
            "value_target": value_target,
        }

        human_move = resolve_move(position, spec)
        position = apply_move(position, human_move)
        emitted += 1


def convert_file_batch_to_teacher_torch(args: WorkerArgs) -> Dict:
    output_dir = Path(args.output_dir)
    skipped = 0
    examples_written = 0
    files_done = 0
    shard_paths: List[str] = []
    errors: List[str] = []

    teacher = YaneuraOuTeacher(
        engine_path=args.engine_path,
        byoyomi_ms=args.byoyomi_ms,
        nodes=args.nodes,
        threads=args.engine_threads,
        hash_mb=args.hash_mb,
    )

    shard_inputs = []
    shard_policies = []
    shard_values = []
    local_shard_id = 0

    def flush() -> None:
        nonlocal shard_inputs, shard_policies, shard_values, local_shard_id

        if not shard_inputs:
            return

        free_gb = shutil.disk_usage(output_dir).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            raise RuntimeError(
                f"Low disk space before writing shard: {free_gb:.2f} GB free"
            )

        path = write_shard(
            output_dir=output_dir,
            task_id=args.task_id,
            local_shard_id=local_shard_id,
            shard_inputs=shard_inputs,
            shard_policies=shard_policies,
            shard_values=shard_values,
        )

        shard_paths.append(str(path))
        local_shard_id += 1

        shard_inputs = []
        shard_policies = []
        shard_values = []

    try:
        for path_str in args.path_strings:
            path = Path(path_str)

            try:
                text = read_kif_file(path)
                parsed_moves = parse_game(text)

                for ex in iter_teacher_examples_from_game(
                    parsed_moves=parsed_moves,
                    teacher=teacher,
                    max_positions_per_game=args.max_positions_per_game,
                    value_mode=args.value_mode,
                ):
                    shard_inputs.append(ex["input_planes"])
                    shard_policies.append(ex["policy_target"])
                    shard_values.append(ex["value_target"])
                    examples_written += 1

                    if len(shard_inputs) >= args.shard_examples:
                        flush()

                files_done += 1

            except Exception as exc:
                skipped += 1
                errors.append(f"[SKIP] {path.name}: {exc}")

        flush()

    finally:
        teacher.close()

    return {
        "task_id": args.task_id,
        "files_done": files_done,
        "skipped": skipped,
        "examples": examples_written,
        "shards": shard_paths,
        "errors": errors[:20],
    }


def chunk_paths(paths: Sequence[Path], chunk_size: int) -> List[List[str]]:
    return [
        [str(p) for p in paths[i:i + chunk_size]]
        for i in range(0, len(paths), chunk_size)
    ]


def convert_files_to_teacher_torch(
    input_paths: Sequence[Path],
    output_dir: Path,
    engine_path: str,
    workers: Optional[int],
    shard_examples: int,
    files_per_task: int,
    min_free_gb: float,
    byoyomi_ms: int,
    nodes: Optional[int],
    engine_threads: Optional[int],
    hash_mb: Optional[int],
    max_positions_per_game: Optional[int],
    value_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if workers is None:
        workers = 1

    workers = max(1, workers)

    batches = chunk_paths(input_paths, files_per_task)

    tasks = [
        WorkerArgs(
            task_id=task_id,
            path_strings=batch,
            output_dir=str(output_dir),
            engine_path=engine_path,
            shard_examples=shard_examples,
            min_free_gb=min_free_gb,
            byoyomi_ms=byoyomi_ms,
            nodes=nodes,
            engine_threads=engine_threads,
            hash_mb=hash_mb,
            max_positions_per_game=max_positions_per_game,
            value_mode=value_mode,
        )
        for task_id, batch in enumerate(batches)
    ]

    start_time = time.time()
    total_examples = 0
    total_skipped = 0
    total_files_done = 0
    all_shards: List[str] = []

    print(f"Found {len(input_paths)} input files")
    print(f"Using {workers} worker process(es)")
    print(f"YaneuraOu engine: {engine_path}")
    print(f"Writing teacher shards to {output_dir}")
    print(f"Files per worker task: {files_per_task}")
    print(f"Examples per shard: {shard_examples}")
    print(f"Policy classes: {TOTAL_MOVE_CLASSES}")

    if nodes is not None:
        print(f"Teacher search: nodes={nodes}")
    else:
        print(f"Teacher search: byoyomi={byoyomi_ms} ms")

    if workers == 1:
        results_iter = map(convert_file_batch_to_teacher_torch, tasks)

        for i, result in enumerate(results_iter, start=1):
            total_files_done += result["files_done"]
            total_skipped += result["skipped"]
            total_examples += result["examples"]
            all_shards.extend(result["shards"])

            for err in result["errors"]:
                print(err)

            if i == 1 or i % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[tasks {i}/{len(tasks)}] "
                    f"files_done={total_files_done}/{len(input_paths)} | "
                    f"examples={total_examples} | "
                    f"shards={len(all_shards)} | "
                    f"skipped={total_skipped} | "
                    f"elapsed={elapsed:.1f}s"
                )

    else:
        with Pool(processes=workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(convert_file_batch_to_teacher_torch, tasks),
                start=1,
            ):
                total_files_done += result["files_done"]
                total_skipped += result["skipped"]
                total_examples += result["examples"]
                all_shards.extend(result["shards"])

                for err in result["errors"]:
                    print(err)

                if i == 1 or i % 10 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[tasks {i}/{len(tasks)}] "
                        f"files_done={total_files_done}/{len(input_paths)} | "
                        f"examples={total_examples} | "
                        f"shards={len(all_shards)} | "
                        f"skipped={total_skipped} | "
                        f"elapsed={elapsed:.1f}s"
                    )

    manifest = {
        "kind": "yaneuraou_teacher",
        "total_examples": total_examples,
        "total_files": len(input_paths),
        "files_done": total_files_done,
        "skipped": total_skipped,
        "total_move_classes": TOTAL_MOVE_CLASSES,
        "shard_examples": shard_examples,
        "engine_path": engine_path,
        "byoyomi_ms": byoyomi_ms,
        "nodes": nodes,
        "engine_threads": engine_threads,
        "hash_mb": hash_mb,
        "max_positions_per_game": max_positions_per_game,
        "value_mode": value_mode,
        "shards": sorted(all_shards),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"Wrote {total_examples} teacher examples")
    print(f"Wrote {len(all_shards)} shards")
    print(f"Manifest: {manifest_path}")
    print(f"Skipped {total_skipped} files")
    print(f"Elapsed time: {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate YaneuraOu teacher-labeled PyTorch shards from KIF/KI2 games."
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more files or directories containing .txt/.kif/.ki2 files.",
    )

    parser.add_argument(
        "--engine",
        required=True,
        help="Path to YaneuraOu executable.",
    )

    parser.add_argument(
        "--output-dir",
        default="teacher_shards",
        help="Directory for output .pt teacher shards.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Start with 1. Each worker launches its own YaneuraOu process.",
    )

    parser.add_argument(
        "--engine-threads",
        type=int,
        default=1,
        help="Threads per YaneuraOu process.",
    )

    parser.add_argument(
        "--hash-mb",
        type=int,
        default=128,
        help="Hash size in MB per YaneuraOu process.",
    )

    parser.add_argument(
        "--byoyomi-ms",
        type=int,
        default=50,
        help="YaneuraOu search time per position in milliseconds. Ignored if --nodes is set.",
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Fixed YaneuraOu nodes per position. If set, overrides --byoyomi-ms.",
    )

    parser.add_argument(
        "--shard-examples",
        type=int,
        default=5000,
        help="Examples per .pt shard.",
    )

    parser.add_argument(
        "--files-per-task",
        type=int,
        default=25,
        help="Number of KIF files processed by each worker task.",
    )

    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Stop before writing a shard if free disk space is below this value.",
    )

    parser.add_argument(
        "--max-positions-per-game",
        type=int,
        default=None,
        help="Optional cap on teacher-labeled positions per game.",
    )

    parser.add_argument(
        "--value-mode",
        choices=["zero", "game"],
        default="zero",
        help="Use zero value targets or original game-result value targets.",
    )

    args = parser.parse_args()

    input_files = collect_input_files(args.inputs)

    convert_files_to_teacher_torch(
        input_paths=input_files,
        output_dir=Path(args.output_dir),
        engine_path=args.engine,
        workers=args.workers,
        shard_examples=args.shard_examples,
        files_per_task=args.files_per_task,
        min_free_gb=args.min_free_gb,
        byoyomi_ms=args.byoyomi_ms,
        nodes=args.nodes,
        engine_threads=args.engine_threads,
        hash_mb=args.hash_mb,
        max_positions_per_game=args.max_positions_per_game,
        value_mode=args.value_mode,
    )


if __name__ == "__main__":
    main()