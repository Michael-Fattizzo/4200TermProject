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


def generate_pseudo_legal_moves(position: Position, side: str) -> List[Move]:
    moves: List[Move] = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = position.board[r][c]
            if piece is None or piece.owner != side:
                continue

            src = (r, c)

            for dr, dc in piece_step_moves(piece):
                nr, nc = r + dr, c + dc
                if not inside(nr, nc):
                    continue
                target = position.board[nr][nc]
                if target and target.owner == side:
                    continue

                base_move = Move(
                    from_sq=src,
                    to_sq=(nr, nc),
                    piece=piece.kind,
                    side=side,
                    captured=target.code() if target else None,
                )

                if can_promote(piece, src, (nr, nc)):
                    if must_promote(piece, (nr, nc)):
                        moves.append(Move(**{**base_move.__dict__, "promote": True}))
                    else:
                        moves.append(base_move)
                        moves.append(Move(**{**base_move.__dict__, "promote": True}))
                else:
                    moves.append(base_move)

            for dr, dc in piece_slide_dirs(piece):
                nr, nc = r + dr, c + dc
                while inside(nr, nc):
                    target = position.board[nr][nc]
                    if target and target.owner == side:
                        break

                    base_move = Move(
                        from_sq=src,
                        to_sq=(nr, nc),
                        piece=piece.kind,
                        side=side,
                        captured=target.code() if target else None,
                    )

                    if can_promote(piece, src, (nr, nc)):
                        if must_promote(piece, (nr, nc)):
                            moves.append(Move(**{**base_move.__dict__, "promote": True}))
                        else:
                            moves.append(base_move)
                            moves.append(Move(**{**base_move.__dict__, "promote": True}))
                    else:
                        moves.append(base_move)

                    if target is not None:
                        break
                    nr += dr
                    nc += dc

    moves.extend(generate_drop_moves(position, side))
    return moves


def generate_drop_moves(position: Position, side: str) -> List[Move]:
    moves: List[Move] = []
    hand = position.hands[side]

    for piece_kind, count in hand.items():
        if count <= 0:
            continue

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if position.board[r][c] is not None:
                    continue

                if piece_kind == "P":
                    if pawn_drop_illegal(position, side, (r, c)):
                        continue
                elif piece_kind == "L":
                    if (side == BLACK and r == 0) or (side == WHITE and r == 8):
                        continue
                elif piece_kind == "N":
                    if (side == BLACK and r <= 1) or (side == WHITE and r >= 7):
                        continue

                moves.append(
                    Move(
                        from_sq=None,
                        to_sq=(r, c),
                        piece=piece_kind,
                        side=side,
                        drop=True,
                    )
                )
    return moves


def pawn_drop_illegal(position: Position, side: str, dst: Tuple[int, int]) -> bool:
    r, c = dst
    if (side == BLACK and r == 0) or (side == WHITE and r == 8):
        return True

    for row in range(BOARD_SIZE):
        p = position.board[row][c]
        if p and p.owner == side and p.kind == "P" and not p.promoted:
            return True

    trial = apply_move(position, Move(from_sq=None, to_sq=dst, piece="P", side=side, drop=True))
    if is_checkmate(trial, opponent(side)):
        return True

    return False


def apply_move(position: Position, move: Move) -> Position:
    new_pos = position.clone()
    side = move.side

    if move.drop:
        if new_pos.hands[side][move.piece] <= 0:
            raise ValueError(f"No {move.piece} in hand for {side}.")
        new_pos.hands[side][move.piece] -= 1
        new_pos.set_piece(move.to_sq, Piece(kind=move.piece, owner=side, promoted=False))
    else:
        moving_piece = new_pos.piece_at(move.from_sq)
        if moving_piece is None:
            raise ValueError("No piece on source square.")

        target = new_pos.piece_at(move.to_sq)
        if target is not None:
            captured_base = target.kind
            new_pos.hands[side][captured_base] += 1

        new_pos.set_piece(move.from_sq, None)
        new_piece = Piece(
            kind=moving_piece.kind,
            owner=side,
            promoted=moving_piece.promoted or move.promote,
        )
        new_pos.set_piece(move.to_sq, new_piece)

    new_pos.side_to_move = opponent(side)
    return new_pos


def attacked_by(position: Position, attacker_side: str, square: Tuple[int, int]) -> bool:
    for move in generate_pseudo_legal_moves_no_drops(position, attacker_side):
        if move.to_sq == square:
            return True
    return False


def generate_pseudo_legal_moves_no_drops(position: Position, side: str) -> List[Move]:
    moves: List[Move] = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = position.board[r][c]
            if piece is None or piece.owner != side:
                continue

            src = (r, c)

            for dr, dc in piece_step_moves(piece):
                nr, nc = r + dr, c + dc
                if not inside(nr, nc):
                    continue
                target = position.board[nr][nc]
                if target and target.owner == side:
                    continue
                moves.append(Move(from_sq=src, to_sq=(nr, nc), piece=piece.kind, side=side))

            for dr, dc in piece_slide_dirs(piece):
                nr, nc = r + dr, c + dc
                while inside(nr, nc):
                    target = position.board[nr][nc]
                    if target and target.owner == side:
                        break
                    moves.append(Move(from_sq=src, to_sq=(nr, nc), piece=piece.kind, side=side))
                    if target is not None:
                        break
                    nr += dr
                    nc += dc

    return moves


def in_check(position: Position, side: str) -> bool:
    king_sq = position.king_square(side)
    if king_sq is None:
        return True
    return attacked_by(position, opponent(side), king_sq)


def generate_legal_moves(position: Position, side: str) -> List[Move]:
    legal: List[Move] = []
    for move in generate_pseudo_legal_moves(position, side):
        nxt = apply_move(position, move)
        if not in_check(nxt, side):
            legal.append(move)
    return legal


def is_checkmate(position: Position, side: str) -> bool:
    if not in_check(position, side):
        return False
    return len(generate_legal_moves(position, side)) == 0

