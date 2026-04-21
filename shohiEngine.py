from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

BOARD_SIZE = 9

BLACK = "black"   # sente; moves "up" toward decreasing row indices
WHITE = "white"   # gote; moves "down" toward increasing row indices

PROMOTABLE = {"P", "L", "N", "S", "B", "R"}
GOLD_MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]


@dataclass(frozen=True)
class Piece:
    kind: str               # P, L, N, S, G, B, R, K
    owner: str              # "black" or "white"
    promoted: bool = False

    def base_kind(self) -> str:
        return self.kind

    def code(self) -> str:
        return ("+" if self.promoted else "") + self.kind


@dataclass(frozen=True)
class Move:
    from_sq: Optional[Tuple[int, int]]
    to_sq: Tuple[int, int]
    piece: str
    side: str
    promote: bool = False
    drop: bool = False
    captured: Optional[str] = None

    def usi(self) -> str:
        if self.drop:
            return f"{self.piece}*{square_to_usi(self.to_sq)}"
        base = f"{square_to_usi(self.from_sq)}{square_to_usi(self.to_sq)}"
        return base + ("+" if self.promote else "")


@dataclass
class Position:
    board: List[List[Optional[Piece]]] = field(
        default_factory=lambda: [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    )
    hands: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            BLACK: {k: 0 for k in ("P", "L", "N", "S", "G", "B", "R")},
            WHITE: {k: 0 for k in ("P", "L", "N", "S", "G", "B", "R")},
        }
    )
    side_to_move: str = BLACK

    def clone(self) -> "Position":
        new_board = [[self.board[r][c] for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)]
        new_hands = {
            side: {k: v for k, v in hand.items()}
            for side, hand in self.hands.items()
        }
        return Position(board=new_board, hands=new_hands, side_to_move=self.side_to_move)

    def piece_at(self, sq: Tuple[int, int]) -> Optional[Piece]:
        r, c = sq
        return self.board[r][c]

    def set_piece(self, sq: Tuple[int, int], piece: Optional[Piece]) -> None:
        r, c = sq
        self.board[r][c] = piece

    def king_square(self, side: str) -> Optional[Tuple[int, int]]:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r][c]
                if p and p.owner == side and p.kind == "K":
                    return (r, c)
        return None


def square_to_usi(square: Tuple[int, int] | None) -> str:
    if square is None:
        raise ValueError("Drop moves do not have a source square.")
    row, col = square
    file_ = 9 - col
    rank_ = chr(ord("a") + row)
    return f"{file_}{rank_}"


def inside(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def forward_dir(side: str) -> int:
    return -1 if side == BLACK else 1


def promotion_zone(side: str, row: int) -> bool:
    if side == BLACK:
        return row <= 2
    return row >= 6


def opponent(side: str) -> str:
    return WHITE if side == BLACK else BLACK


def can_promote(piece: Piece, src: Tuple[int, int], dst: Tuple[int, int]) -> bool:
    if piece.promoted or piece.kind not in PROMOTABLE:
        return False
    return promotion_zone(piece.owner, src[0]) or promotion_zone(piece.owner, dst[0])


def must_promote(piece: Piece, dst: Tuple[int, int]) -> bool:
    row, _ = dst
    if piece.owner == BLACK:
        if piece.kind in {"P", "L"} and row == 0:
            return True
        if piece.kind == "N" and row <= 1:
            return True
    else:
        if piece.kind in {"P", "L"} and row == 8:
            return True
        if piece.kind == "N" and row >= 7:
            return True
    return False


def piece_step_moves(piece: Piece) -> List[Tuple[int, int]]:
    f = forward_dir(piece.owner)
    if piece.kind == "K":
        return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if piece.kind == "G" or (piece.promoted and piece.kind in {"P", "L", "N", "S"}):
        return [(f, -1), (f, 0), (f, 1), (0, -1), (0, 1), (-f, 0)]
    if piece.kind == "S":
        return [(f, -1), (f, 0), (f, 1), (-f, -1), (-f, 1)]
    if piece.kind == "P":
        return [(f, 0)]
    if piece.kind == "N":
        return [(2 * f, -1), (2 * f, 1)]
    if piece.kind == "B" and piece.promoted:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if piece.kind == "R" and piece.promoted:
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return []


def piece_slide_dirs(piece: Piece) -> List[Tuple[int, int]]:
    if piece.kind == "L":
        return [(forward_dir(piece.owner), 0)]
    if piece.kind == "B":
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    if piece.kind == "R":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return []

