from __future__ import annotations

import argparse
import json
import re
import time
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from multiprocessing import Pool, cpu_count

from shogiEngine import (
    BLACK,
    WHITE,
    Move,
    Piece,
    Position,
    apply_move,
    generate_pseudo_legal_moves,
    opponent,
)

from moveEncoding import TOTAL_MOVE_CLASSES, encode_move_obj


FULLWIDTH_TO_ASCII = str.maketrans("０１２３４５６７８９", "0123456789")

KANJI_RANK = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

PIECE_NAME_TO_STATE = {
    "歩": ("P", False),
    "香": ("L", False),
    "桂": ("N", False),
    "銀": ("S", False),
    "金": ("G", False),
    "角": ("B", False),
    "飛": ("R", False),
    "玉": ("K", False),
    "王": ("K", False),
    "と": ("P", True),
    "成香": ("L", True),
    "杏": ("L", True),
    "成桂": ("N", True),
    "圭": ("N", True),
    "成銀": ("S", True),
    "全": ("S", True),
    "馬": ("B", True),
    "龍": ("R", True),
    "竜": ("R", True),
}

SPECIAL_MOVES = {
    "投了",
    "中断",
    "千日手",
    "持将棋",
    "詰み",
    "切れ負け",
    "反則負け",
    "反則勝ち",
    "入玉宣言",
}

MOVE_LINE_RE = re.compile(r"^\s*(?P<num>\d+)\s+(?P<body>.+?)\s*$")


@dataclass
class ParsedMoveSpec:
    move_no: int
    same_as_previous: bool
    dst: Optional[Tuple[int, int]]
    piece_kind: str
    piece_promoted_state: bool
    src: Optional[Tuple[int, int]]
    promote: Optional[bool]
    drop: bool
    special: Optional[str] = None
    raw: str = ""


def read_kif_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis", "euc_jp"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode file: {path}")


def normalize_text(s: str) -> str:
    s = s.replace("\u3000", "")
    s = s.translate(FULLWIDTH_TO_ASCII)
    s = re.sub(r"\s+", "", s)
    return s.strip()


def strip_elapsed_time(body: str) -> str:
    return re.sub(r"\s+\([^)]*/[^)]*\)\s*$", "", body).strip()


def parse_square_digits(two_digits: str) -> Tuple[int, int]:
    two_digits = two_digits.translate(FULLWIDTH_TO_ASCII)
    if len(two_digits) != 2 or not two_digits.isdigit():
        raise ValueError(f"Invalid square digits: {two_digits}")

    file_ = int(two_digits[0])
    rank_ = int(two_digits[1])
    if not (1 <= file_ <= 9 and 1 <= rank_ <= 9):
        raise ValueError(f"Out-of-range square: {two_digits}")

    return rank_ - 1, 9 - file_


def parse_destination_prefix(
    s: str,
    previous_dst: Optional[Tuple[int, int]],
) -> Tuple[bool, Optional[Tuple[int, int]], str]:
    if s.startswith("同"):
        if previous_dst is None:
            raise ValueError("Encountered '同' without previous destination.")
        return True, previous_dst, s[1:]

    if len(s) < 2:
        raise ValueError(f"Cannot parse destination from move text: {s}")

    file_char = s[0]
    rank_char = s[1]

    if not file_char.isdigit():
        raise ValueError(f"Expected file digit at start of move: {s}")
    if rank_char not in KANJI_RANK:
        raise ValueError(f"Expected kanji rank after file digit in move: {s}")

    file_ = int(file_char)
    rank_ = KANJI_RANK[rank_char]

    dst = (rank_ - 1, 9 - file_)
    return False, dst, s[2:]



def parse_piece_token(rest: str) -> Tuple[str, bool, Optional[bool], bool, str]:
    piece_token = None
    remaining = ""

    for token in sorted(PIECE_NAME_TO_STATE.keys(), key=len, reverse=True):
        if rest.startswith(token):
            piece_token = token
            remaining = rest[len(token):]
            break

    if piece_token is None:
        raise ValueError(f"Unknown piece token in move text: {rest}")

    piece_kind, promoted_state = PIECE_NAME_TO_STATE[piece_token]
    promote_flag: Optional[bool]

    if remaining.startswith("不成"):
        promote_flag = False
        remaining = remaining[2:]
    elif remaining.startswith("成"):
        promote_flag = True
        remaining = remaining[1:]
    elif promoted_state:
        promote_flag = None
    else:
        promote_flag = False

    drop = False
    if remaining.startswith("打"):
        drop = True
        remaining = remaining[1:]

    return piece_kind, promoted_state, promote_flag, drop, remaining


def parse_source_square(remaining: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"\((\d\d)\)", remaining)
    if not m:
        return None
    return parse_square_digits(m.group(1))


def parse_move_line(line: str, previous_dst: Optional[Tuple[int, int]]) -> ParsedMoveSpec:
    raw_line = line.rstrip()
    m = MOVE_LINE_RE.match(raw_line)
    if not m:
        raise ValueError(f"Unrecognized move line format: {raw_line}")

    move_no = int(m.group("num"))
    move_text = normalize_text(strip_elapsed_time(m.group("body")))

    if move_text in SPECIAL_MOVES:
        return ParsedMoveSpec(
            move_no=move_no,
            same_as_previous=False,
            dst=None,
            piece_kind="",
            piece_promoted_state=False,
            src=None,
            promote=None,
            drop=False,
            special=move_text,
            raw=raw_line,
        )

    same_as_previous, dst, rest = parse_destination_prefix(move_text, previous_dst)
    piece_kind, piece_promoted_state, promote, drop, remaining = parse_piece_token(rest)
    src = parse_source_square(remaining)

    return ParsedMoveSpec(
        move_no=move_no,
        same_as_previous=same_as_previous,
        dst=dst,
        piece_kind=piece_kind,
        piece_promoted_state=piece_promoted_state,
        src=src,
        promote=promote,
        drop=drop,
        special=None,
        raw=raw_line,
    )


def initial_position() -> Position:
    pos = Position(side_to_move=BLACK)

    pos.board[0] = [
        Piece("L", WHITE), Piece("N", WHITE), Piece("S", WHITE), Piece("G", WHITE),
        Piece("K", WHITE), Piece("G", WHITE), Piece("S", WHITE), Piece("N", WHITE), Piece("L", WHITE),
    ]
    pos.board[1][1] = Piece("R", WHITE)
    pos.board[1][7] = Piece("B", WHITE)

    for c in range(9):
        pos.board[2][c] = Piece("P", WHITE)

    for c in range(9):
        pos.board[6][c] = Piece("P", BLACK)

    pos.board[7][1] = Piece("B", BLACK)
    pos.board[7][7] = Piece("R", BLACK)

    pos.board[8] = [
        Piece("L", BLACK), Piece("N", BLACK), Piece("S", BLACK), Piece("G", BLACK),
        Piece("K", BLACK), Piece("G", BLACK), Piece("S", BLACK), Piece("N", BLACK), Piece("L", BLACK),
    ]

    return pos


def pseudo_move_matches_spec(position: Position, move: Move, spec: ParsedMoveSpec) -> bool:
    if spec.special is not None:
        return False

    if move.drop != spec.drop:
        return False

    if move.to_sq != spec.dst:
        return False

    if spec.src is not None and move.from_sq != spec.src:
        return False

    if move.drop:
        return move.piece == spec.piece_kind and spec.piece_promoted_state is False

    src_piece = position.piece_at(move.from_sq)
    if src_piece is None:
        return False

    if src_piece.kind != spec.piece_kind:
        return False

    promoted_after = src_piece.promoted or bool(move.promote)

    if spec.promote is True:
        return bool(move.promote)

    if spec.promote is False:
        if bool(move.promote):
            return False
        return promoted_after == spec.piece_promoted_state

    if spec.promote is None:
        return promoted_after == spec.piece_promoted_state

    return True


def resolve_move_direct(position: Position, spec: ParsedMoveSpec) -> Optional[Move]:
    if spec.special is not None or spec.dst is None:
        return None

    side = position.side_to_move

    if spec.drop:
        return Move(
            from_sq=None,
            to_sq=spec.dst,
            piece=spec.piece_kind,
            side=side,
            promote=False,
            drop=True,
            captured=None,
        )

    if spec.src is None:
        return None

    src_piece = position.piece_at(spec.src)
    if src_piece is None:
        return None

    if src_piece.kind != spec.piece_kind:
        return None

    if spec.piece_promoted_state and not src_piece.promoted:
        return None

    target = position.piece_at(spec.dst)
    captured = target.code() if target else None
    promote = bool(spec.promote) if spec.promote is not None else False

    return Move(
        from_sq=spec.src,
        to_sq=spec.dst,
        piece=spec.piece_kind,
        side=side,
        promote=promote,
        drop=False,
        captured=captured,
    )


def resolve_move_fallback(position: Position, spec: ParsedMoveSpec) -> Move:
    pseudo = generate_pseudo_legal_moves(position, position.side_to_move)
    matches = [mv for mv in pseudo if pseudo_move_matches_spec(position, mv, spec)]

    if len(matches) == 1:
        return matches[0]

    if spec.src is not None:
        exact_src = [mv for mv in matches if mv.from_sq == spec.src]
        if len(exact_src) == 1:
            return exact_src[0]

    debug_moves = ", ".join(mv.usi() for mv in pseudo[:80])
    raise ValueError(
        f"Could not resolve move at move {spec.move_no}: {spec.raw}\n"
        f"Side to move: {position.side_to_move}\n"
        f"Candidates: {[mv.usi() for mv in matches]}\n"
        f"First pseudo-legal moves: {debug_moves}"
    )



def resolve_move(position: Position, spec: ParsedMoveSpec) -> Move:
    if spec.special is not None:
        raise ValueError(f"Cannot resolve special move: {spec.raw}")

    if spec.dst is None:
        raise ValueError(f"Move has no destination: {spec.raw}")

    side = position.side_to_move

    # Drops can be constructed directly.
    if spec.drop:
        return Move(
            from_sq=None,
            to_sq=spec.dst,
            piece=spec.piece_kind,
            side=side,
            promote=False,
            drop=True,
            captured=None,
        )

    # If KIF gives source square, construct directly.
    if spec.src is not None:
        src_piece = position.piece_at(spec.src)
        if src_piece is None:
            raise ValueError(
                f"No source piece at {spec.src} for move {spec.move_no}: {spec.raw}"
            )

        if src_piece.owner != side:
            raise ValueError(
                f"Source piece belongs to {src_piece.owner}, but side to move is {side}: {spec.raw}"
            )

        if src_piece.kind != spec.piece_kind:
            raise ValueError(
                f"Source piece kind mismatch at move {spec.move_no}: "
                f"expected {spec.piece_kind}, found {src_piece.kind}; {spec.raw}"
            )

        target = position.piece_at(spec.dst)
        if target is not None and target.owner == side:
            raise ValueError(
                f"Move targets own piece at move {spec.move_no}: {spec.raw}"
            )

        promote = bool(spec.promote) if spec.promote is not None else False

        return Move(
            from_sq=spec.src,
            to_sq=spec.dst,
            piece=spec.piece_kind,
            side=side,
            promote=promote,
            drop=False,
            captured=target.code() if target else None,
        )

    # Rare fallback: source square missing.
    pseudo = generate_pseudo_legal_moves(position, side)

    matches: List[Move] = []
    for mv in pseudo:
        if mv.drop:
            continue

        if mv.to_sq != spec.dst:
            continue

        src_piece = position.piece_at(mv.from_sq)
        if src_piece is None:
            continue

        if src_piece.owner != side:
            continue

        if src_piece.kind != spec.piece_kind:
            continue

        promoted_after = src_piece.promoted or bool(mv.promote)

        if spec.promote is True and not mv.promote:
            continue

        if spec.promote is False:
            if mv.promote:
                continue
            if promoted_after != spec.piece_promoted_state:
                continue

        if spec.promote is None:
            if promoted_after != spec.piece_promoted_state:
                continue

        matches.append(mv)

    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Could not uniquely resolve move {spec.move_no}: {spec.raw}\n"
        f"Side to move: {side}\n"
        f"Matches: {[m.usi() for m in matches]}"
    )


PIECE_PLANE_INDEX = {
    ("P", False): 0,
    ("L", False): 1,
    ("N", False): 2,
    ("S", False): 3,
    ("G", False): 4,
    ("B", False): 5,
    ("R", False): 6,
    ("K", False): 7,
    ("P", True): 8,
    ("L", True): 9,
    ("N", True): 10,
    ("S", True): 11,
    ("B", True): 12,
    ("R", True): 13,
}

HAND_ORDER = ["P", "L", "N", "S", "G", "B", "R"]


def empty_planes(channels: int) -> List[List[List[float]]]:
    return [[[0.0 for _ in range(9)] for _ in range(9)] for _ in range(channels)]


def fill_value_plane(plane: List[List[float]], value: float) -> None:
    for r in range(9):
        for c in range(9):
            plane[r][c] = float(value)


def encode_position(position: Position, ply_index: int) -> List[List[List[float]]]:
    planes = empty_planes(44)
    stm = position.side_to_move
    opp = opponent(stm)

    for r in range(9):
        for c in range(9):
            piece = position.board[r][c]
            if piece is None:
                continue

            idx = PIECE_PLANE_INDEX[(piece.kind, piece.promoted)]
            if piece.owner == stm:
                planes[idx][r][c] = 1.0
            else:
                planes[14 + idx][r][c] = 1.0

    for i, kind in enumerate(HAND_ORDER):
        fill_value_plane(planes[28 + i], float(position.hands[stm].get(kind, 0)) / 18.0)
        fill_value_plane(planes[35 + i], float(position.hands[opp].get(kind, 0)) / 18.0)

    fill_value_plane(planes[42], 1.0 if stm == BLACK else 0.0)
    fill_value_plane(planes[43], min(1.0, ply_index / 200.0))

    return planes


def infer_result_from_moves(parsed_moves: List[ParsedMoveSpec]) -> Optional[str]:
    if not parsed_moves:
        return None

    last = parsed_moves[-1]

    if last.special == "投了":
        resigning_side = BLACK if (last.move_no % 2 == 1) else WHITE
        return opponent(resigning_side)

    if last.special == "反則負け":
        losing_side = BLACK if (last.move_no % 2 == 1) else WHITE
        return opponent(losing_side)

    if last.special == "反則勝ち":
        return BLACK if (last.move_no % 2 == 1) else WHITE

    if last.special == "入玉宣言":
        return BLACK if (last.move_no % 2 == 1) else WHITE

    if last.special in {"千日手", "持将棋", "中断"}:
        return "draw"

    return None


def value_target_for_side(winner: Optional[str], side: str) -> float:
    if winner is None or winner == "draw":
        return 0.0
    return 1.0 if winner == side else -1.0


def iter_game_lines(text: str) -> Iterable[str]:
    for raw in text.splitlines():
        stripped = raw.strip()

        if not stripped:
            continue

        if stripped.startswith((
            "開始日時",
            "終了日時",
            "棋戦",
            "戦型",
            "手合割",
            "先手",
            "後手",
            "場所",
            "持ち時間",
            "消費時間",
            "手数----指手",
        )):
            continue

        if stripped.startswith(("*", "#", "&")):
            continue

        if stripped.startswith(("**", "対局", "時間", "深さ", "ノード数", "評価値", "読み筋")):
            continue

        if not re.match(r"^\s*\d+\s+", stripped):
            continue

        yield raw.rstrip()


def parse_game(text: str) -> List[ParsedMoveSpec]:
    parsed: List[ParsedMoveSpec] = []
    prev_dst: Optional[Tuple[int, int]] = None

    for line in iter_game_lines(text):
        spec = parse_move_line(line, prev_dst)
        parsed.append(spec)

        if spec.dst is not None:
            prev_dst = spec.dst

    return parsed


def iter_examples_from_game(parsed_moves: List[ParsedMoveSpec], source_file: str) -> Iterable[Dict]:
    winner = infer_result_from_moves(parsed_moves)
    pos = initial_position()

    for ply_index, spec in enumerate(parsed_moves, start=1):
        if spec.special is not None:
            break

        move = resolve_move(pos, spec)

        if not isinstance(move.to_sq, tuple):
            raise TypeError(f"Bad move.to_sq: {move.to_sq} from {move}")

        yield {
            "input_planes": encode_position(pos, ply_index=ply_index),
            "policy_target": encode_move_obj(move),
            "value_target": value_target_for_side(winner, pos.side_to_move),
            "source_file": source_file,
        }

        pos = apply_move(pos, move)


def convert_one_file(path: Path) -> Tuple[str, int, Optional[str]]:
    """
    Returns:
        jsonl_text, example_count, error_message
    """
    try:
        text = read_kif_file(path)
        parsed_moves = parse_game(text)

        lines = []
        count = 0

        for ex in iter_examples_from_game(parsed_moves, source_file=path.name):
            lines.append(json.dumps(ex, ensure_ascii=False))
            count += 1

        return "\n".join(lines), count, None

    except Exception as exc:
        return "", 0, f"[SKIP] {path.name}: {exc}"


def convert_files(
    input_paths: Sequence[Path],
    output_jsonl: Path,
    workers: Optional[int] = None,
) -> None:
    start_time = time.time()
    skipped = 0
    total_examples = 0

    if workers is None:
        workers = max(1, cpu_count() - 1)

    print(f"Using {workers} worker processes")

    with output_jsonl.open("w", encoding="utf-8") as out:
        with Pool(processes=workers) as pool:
            for i, (jsonl_text, count, error) in enumerate(
                pool.imap_unordered(convert_one_file, input_paths, chunksize=25),
                start=1,
            ):
                if error:
                    skipped += 1
                    print(error)
                else:
                    if jsonl_text:
                        out.write(jsonl_text)
                        out.write("\n")
                    total_examples += count

                if i == 1 or i % 100 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{i}/{len(input_paths)}] "
                        f"examples={total_examples} | "
                        f"skipped={skipped} | "
                        f"elapsed={elapsed:.1f}s"
                    )

    elapsed = time.time() - start_time
    print(f"Wrote {total_examples} examples to {output_jsonl}")
    print(f"Fixed policy classes: {TOTAL_MOVE_CLASSES}")
    print(f"Skipped {skipped} files")
    print(f"Elapsed time: {elapsed:.1f}s")


def collect_input_files(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []

    for p in paths:
        path = Path(p)

        if path.is_dir():
            for ext in ("*.txt", "*.kif", "*.ki2"):
                out.extend(sorted(path.glob(ext)))
        else:
            out.append(path)

    if not out:
        raise ValueError("No input game files found.")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast streaming converter from Japanese shogi KIF/KI2 logs to fixed-encoding JSONL."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more files or directories containing .txt/.kif/.ki2 files.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="training_examples.jsonl",
        help="Output JSONL dataset path.",
    )
    parser.add_argument(
    "--workers",
    type=int,
    default=None,
    help="Number of worker processes. Default: CPU count minus 1.",
)

    args = parser.parse_args()

    input_files = collect_input_files(args.inputs)
    print(f"Found {len(input_files)} input files")
    print(f"Fixed policy classes: {TOTAL_MOVE_CLASSES}")

    convert_files(
        input_paths=input_files,
        output_jsonl=Path(args.output_jsonl),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()