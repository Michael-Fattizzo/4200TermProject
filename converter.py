from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from shogiEngine import (
    BLACK,
    WHITE,
    Move,
    Piece,
    Position,
    apply_move,
    generate_legal_moves,
    opponent,
)

# Parsing helpers

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

MOVE_LINE_RE = re.compile(
    r"^\s*(?P<num>\d+)\s+(?P<move>.+?)\s+\((?P<time>[^)]*)\)\s*$"
)


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


def normalize_text(s: str) -> str:
    s = s.replace("\u3000", "")  # ideographic spaces
    s = s.translate(FULLWIDTH_TO_ASCII)
    return s.strip()


def parse_square_digits(two_digits: str) -> Tuple[int, int]:
    if len(two_digits) != 2 or not two_digits.isdigit():
        raise ValueError(f"Invalid square digits: {two_digits}")
    file_ = int(two_digits[0])
    rank_ = int(two_digits[1])
    if not (1 <= file_ <= 9 and 1 <= rank_ <= 9):
        raise ValueError(f"Out-of-range square: {two_digits}")

    row = rank_ - 1
    col = 9 - file_
    return (row, col)


def parse_destination_prefix(s: str, previous_dst: Optional[Tuple[int, int]]) -> Tuple[bool, Optional[Tuple[int, int]], str]:
    if s.startswith("同"):
        if previous_dst is None:
            raise ValueError("Encountered 'same square' move without a previous destination.")
        rest = s[1:]
        return True, previous_dst, rest

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
    row = rank_ - 1
    col = 9 - file_
    return False, (row, col), s[2:]


def parse_piece_token(rest: str) -> Tuple[str, bool, Optional[bool], bool, str]:
    """
    Returns:
      piece_kind, piece_promoted_state, promote_flag, drop_flag, remaining_rest

    Cases handled:
      歩
      銀
      角成
      成桂
      馬
      飛打
    """
    drop = False
    promote_flag: Optional[bool] = None

    piece_token = None
    remaining = ""

    # Longest prefix match to distinguish 成桂 from 桂, 成銀 from 銀, etc.
    for token in sorted(PIECE_NAME_TO_STATE.keys(), key=len, reverse=True):
        if rest.startswith(token):
            piece_token = token
            remaining = rest[len(token):]
            break

    if piece_token is None:
        raise ValueError(f"Unknown piece token in move text: {rest}")

    piece_kind, promoted_state = PIECE_NAME_TO_STATE[piece_token]

    if remaining.startswith("成"):
        promote_flag = True
        remaining = remaining[1:]
    elif remaining.startswith("不成"):
        promote_flag = False
        remaining = remaining[2:]
    elif promoted_state:
        # The token itself is a promoted-piece token such as 馬 or 成桂.
        promote_flag = None
    else:
        promote_flag = False

    if remaining.startswith("打"):
        drop = True
        remaining = remaining[1:]

    return piece_kind, promoted_state, promote_flag, drop, remaining


def parse_source_square(remaining: str) -> Optional[Tuple[int, int]]:
    if not remaining:
        return None
    m = re.search(r"\((\d\d)\)", remaining)
    if not m:
        return None
    return parse_square_digits(m.group(1))


def parse_move_line(line: str, previous_dst: Optional[Tuple[int, int]]) -> ParsedMoveSpec:
    line = normalize_text(line)
    m = MOVE_LINE_RE.match(line)
    if not m:
        raise ValueError(f"Unrecognized move line format: {line}")

    move_no = int(m.group("num"))
    move_text = normalize_text(m.group("move"))

    if move_text in {"投了", "中断", "千日手", "持将棋", "詰み", "切れ負け", "反則負け", "反則勝ち"}:
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
            raw=line,
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
        raw=line,
    )

# Position setup

def initial_position() -> Position:
    pos = Position(side_to_move=BLACK)

    # White side (top)
    pos.board[0] = [
        Piece("L", WHITE), Piece("N", WHITE), Piece("S", WHITE), Piece("G", WHITE),
        Piece("K", WHITE), Piece("G", WHITE), Piece("S", WHITE), Piece("N", WHITE), Piece("L", WHITE)
    ]
    pos.board[1][1] = Piece("R", WHITE)
    pos.board[1][7] = Piece("B", WHITE)
    for c in range(9):
        pos.board[2][c] = Piece("P", WHITE)

    # Black side (bottom)
    for c in range(9):
        pos.board[6][c] = Piece("P", BLACK)
    pos.board[7][1] = Piece("B", BLACK)
    pos.board[7][7] = Piece("R", BLACK)
    pos.board[8] = [
        Piece("L", BLACK), Piece("N", BLACK), Piece("S", BLACK), Piece("G", BLACK),
        Piece("K", BLACK), Piece("G", BLACK), Piece("S", BLACK), Piece("N", BLACK), Piece("L", BLACK)
    ]

    return pos

# Move matching against legal moves

def move_piece_state_after(move: Move, position: Position) -> Tuple[str, bool]:
    if move.drop:
        return move.piece, False

    src_piece = position.piece_at(move.from_sq)
    if src_piece is None:
        raise ValueError("Source piece missing while matching move.")
    promoted_after = src_piece.promoted or bool(move.promote)
    return src_piece.kind, promoted_after


def legal_move_matches_spec(position: Position, move: Move, spec: ParsedMoveSpec) -> bool:
    if spec.special is not None:
        return False

    if move.drop != spec.drop:
        return False

    if move.to_sq != spec.dst:
        return False

    if spec.src is not None and move.from_sq != spec.src:
        return False

    piece_kind_after, promoted_after = move_piece_state_after(move, position)
    if piece_kind_after != spec.piece_kind:
        return False

    if promoted_after != spec.piece_promoted_state:
        return False

    if spec.promote is True and not move.promote:
        return False
    if spec.promote is False:
        # For tokens like 馬 or 成桂, spec.promote remains None; for normal pieces it is False.
        if move.promote:
            return False

    return True


def resolve_move(position: Position, spec: ParsedMoveSpec) -> Move:
    legal = generate_legal_moves(position, position.side_to_move)
    matches = [mv for mv in legal if legal_move_matches_spec(position, mv, spec)]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        debug_legal = ", ".join(mv.usi() for mv in legal[:50])
        raise ValueError(
            f"No legal move matched spec at move {spec.move_no}: {spec.raw}\n"
            f"Side to move: {position.side_to_move}\n"
            f"First legal moves: {debug_legal}"
        )

    # If multiple legal moves still match, the notation was ambiguous relative to our
    # compressed move representation. Prefer exact source match if present; otherwise fail loudly.
    if spec.src is not None:
        for mv in matches:
            if mv.from_sq == spec.src:
                return mv

    raise ValueError(
        f"Ambiguous move at move {spec.move_no}: {spec.raw}\n"
        f"Candidates: {[mv.usi() for mv in matches]}"
    )

# Tensor encoding

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


def encode_position(position: Position, ply_index: int) -> List[List[List[float]]]:
    """
    44 channels:
      0-13   side-to-move piece planes
      14-27  opponent piece planes
      28-34  side-to-move hand-count planes
      35-41  opponent hand-count planes
      42     side-to-move plane
      43     phase plane
    """
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
        stm_count = float(position.hands[stm].get(kind, 0))
        opp_count = float(position.hands[opp].get(kind, 0))
        fill_value_plane(planes[28 + i], stm_count / 18.0)
        fill_value_plane(planes[35 + i], opp_count / 18.0)

    fill_value_plane(planes[42], 1.0 if stm == BLACK else 0.0)
    fill_value_plane(planes[43], min(1.0, ply_index / 200.0))
    return planes


def fill_value_plane(plane: List[List[float]], value: float) -> None:
    for r in range(9):
        for c in range(9):
            plane[r][c] = float(value)


# Dataset generation

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
        winning_side = BLACK if (last.move_no % 2 == 1) else WHITE
        return winning_side

    if last.special in {"千日手", "持将棋"}:
        return "draw"

    return None


def value_target_for_side(winner: Optional[str], side: str) -> float:
    if winner is None or winner == "draw":
        return 0.0
    return 1.0 if winner == side else -1.0


def iter_game_lines(text: str) -> Iterable[str]:
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("開始日時") or line.startswith("棋戦") or line.startswith("手合割") or line.startswith("先手") or line.startswith("後手"):
            continue
        if line.startswith("手数----指手"):
            continue
        yield line


def parse_game(text: str) -> List[ParsedMoveSpec]:
    parsed: List[ParsedMoveSpec] = []
    prev_dst: Optional[Tuple[int, int]] = None

    for line in iter_game_lines(text):
        spec = parse_move_line(line, prev_dst)
        parsed.append(spec)
        if spec.dst is not None:
            prev_dst = spec.dst

    return parsed


def build_examples_from_game(parsed_moves: List[ParsedMoveSpec]) -> Tuple[List[Dict], Dict[str, int]]:
    winner = infer_result_from_moves(parsed_moves)

    pos = initial_position()
    examples: List[Dict] = []
    vocab: Dict[str, int] = {}

    for ply_index, spec in enumerate(parsed_moves, start=1):
        if spec.special is not None:
            break

        move = resolve_move(pos, spec)
        move_token = move.usi()

        if move_token not in vocab:
            vocab[move_token] = len(vocab)

        example = {
            "input_planes": encode_position(pos, ply_index=ply_index),
            "policy_target_token": move_token,
            "value_target": value_target_for_side(winner, pos.side_to_move),
        }
        examples.append(example)

        pos = apply_move(pos, move)

    return examples, vocab


def merge_vocab(global_vocab: Dict[str, int], local_tokens: Iterable[str]) -> None:
    for token in local_tokens:
        if token not in global_vocab:
            global_vocab[token] = len(global_vocab)


def remap_examples_policy_targets(examples: List[Dict], global_vocab: Dict[str, int]) -> None:
    for ex in examples:
        token = ex.pop("policy_target_token")
        ex["policy_target"] = global_vocab[token]


def convert_files(input_paths: Sequence[Path], output_jsonl: Path, vocab_json: Path) -> None:
    all_examples: List[Dict] = []
    global_vocab: Dict[str, int] = {}

    for path in input_paths:
        text = path.read_text(encoding="utf-8")
        parsed_moves = parse_game(text)
        examples, local_vocab = build_examples_from_game(parsed_moves)

        merge_vocab(global_vocab, local_vocab.keys())
        for ex in examples:
            ex["source_file"] = path.name
        all_examples.extend(examples)

    remap_examples_policy_targets(all_examples, global_vocab)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with vocab_json.open("w", encoding="utf-8") as f:
        json.dump(global_vocab, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_examples)} examples to {output_jsonl}")
    print(f"Wrote {len(global_vocab)} policy tokens to {vocab_json}")


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
        description="Convert Japanese shogi game logs into training_examples.jsonl"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more game files or directories containing .txt/.kif/.ki2 files.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="training_examples.jsonl",
        help="Output JSONL dataset path.",
    )
    parser.add_argument(
        "--output-vocab",
        default="policy_vocab.json",
        help="Output policy vocabulary JSON path.",
    )
    args = parser.parse_args()

    input_files = collect_input_files(args.inputs)
    convert_files(
        input_paths=input_files,
        output_jsonl=Path(args.output_jsonl),
        vocab_json=Path(args.output_vocab),
    )


if __name__ == "__main__":
    main()
