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

        if moving_piece.owner != side:
            raise ValueError(f"Moving opponent piece: {move}")

        target = new_pos.piece_at(move.to_sq)
        if target is not None:
            if target.owner == side:
                raise ValueError("Cannot capture own piece.")

            captured_base = target.kind

            # Kings are never added to hand.
            if captured_base != "K":
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


PIECE_VALUES = {
    "P": 100,
    "L": 300,
    "N": 300,
    "S": 500,
    "G": 600,
    "B": 800,
    "R": 1000,
    "K": 10000,
}

PROMOTION_BONUS = {
    "P": 400,
    "L": 300,
    "N": 300,
    "S": 200,
    "B": 500,
    "R": 500,
}


def evaluate(position: Position, side: str) -> int:
    score = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = position.board[r][c]
            if p is None:
                continue
            value = PIECE_VALUES[p.kind] + (PROMOTION_BONUS.get(p.kind, 0) if p.promoted else 0)

            centrality = 4 - abs(4 - r) + 4 - abs(4 - c)
            value += 5 * centrality

            if p.owner == side:
                score += value
            else:
                score -= value

    for s in (BLACK, WHITE):
        hand_score = 0
        for kind, count in position.hands[s].items():
            hand_score += PIECE_VALUES[kind] * count
        if s == side:
            score += hand_score
        else:
            score -= hand_score

    my_king = position.king_square(side)
    opp_king = position.king_square(opponent(side))
    if my_king:
        score -= king_exposure_penalty(position, my_king, side)
    if opp_king:
        score += king_exposure_penalty(position, opp_king, opponent(side))

    if in_check(position, opponent(side)):
        score += 250
    if in_check(position, side):
        score -= 250

    return score


def king_exposure_penalty(position: Position, king_sq: Tuple[int, int], side: str) -> int:
    penalty = 0
    kr, kc = king_sq
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = kr + dr, kc + dc
            if not inside(nr, nc):
                penalty += 20
                continue
            p = position.board[nr][nc]
            if p is None:
                penalty += 15
            elif p.owner != side:
                penalty += 30
    return penalty


def choose_best_move(position: Position, side: str, depth: int = 2) -> Optional[Move]:
    legal = generate_legal_moves(position, side)
    if not legal:
        return None

    best_move = None
    best_score = -10**18

    for move in legal:
        nxt = apply_move(position, move)
        score = -negamax(nxt, opponent(side), depth - 1, -10**18, 10**18)
        score += move_ordering_bonus(position, move, side)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def negamax(position: Position, side: str, depth: int, alpha: int, beta: int) -> int:
    legal = generate_legal_moves(position, side)

    if depth == 0 or not legal:
        if not legal and in_check(position, side):
            return -1000000
        return evaluate(position, side)

    best = -10**18
    for move in legal:
        nxt = apply_move(position, move)
        val = -negamax(nxt, opponent(side), depth - 1, -beta, -alpha)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def move_ordering_bonus(position: Position, move: Move, side: str) -> int:
    bonus = 0
    target = position.piece_at(move.to_sq)
    if target is not None and target.owner != side:
        bonus += PIECE_VALUES[target.kind] * 2
    if move.promote:
        bonus += 150

    nxt = apply_move(position, move)
    if in_check(nxt, opponent(side)):
        bonus += 200
    if is_checkmate(nxt, opponent(side)):
        bonus += 100000
    return bonus

#Capture-state integration 

def parse_capture_label(label: str) -> Optional[Piece]:
    """
    Expected labels from the screen reader:
      black_P, white_P, black_+P, white_+P, ...
      or bP / wP / b+P / w+P
    Empty / unknown labels return None.
    """
    if label in {"empty", ".", "blank", "unknown", ""}:
        return None

    promoted = "+" in label
    cleaned = label.replace("+", "")

    if "_" in cleaned:
        side_str, kind = cleaned.split("_", 1)
        owner = BLACK if side_str.lower().startswith(("b", "black", "sente")) else WHITE
        return Piece(kind=kind.upper(), owner=owner, promoted=promoted)

    if len(cleaned) >= 2 and cleaned[0].lower() in {"b", "w"}:
        owner = BLACK if cleaned[0].lower() == "b" else WHITE
        kind = cleaned[1:].upper()
        return Piece(kind=kind, owner=owner, promoted=promoted)

    raise ValueError(f"Unsupported capture label format: {label}")


def position_from_capture_state(
    capture_state: Dict[str, object],
    side_to_move: str,
    left_hand_owner: str = WHITE,
    right_hand_owner: str = BLACK,
) -> Position:
    """
    Converts the existing screen-capture output into an engine position.
    Assumes:
      - capture_state['board'] is a 9x9 matrix of dicts with a 'label' key
      - capture_state['left_hand'] and ['right_hand'] are slot dicts with 'piece' and 'count'
    """
    pos = Position(side_to_move=side_to_move)

    board_rows = capture_state["board"]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = board_rows[r][c]
            piece = parse_capture_label(cell["label"])
            pos.board[r][c] = piece

    for slot in capture_state.get("left_hand", []):
        piece = slot.get("piece", ".")
        count = int(slot.get("count", 0))
        if piece not in {".", "empty", "blank", "unknown"} and count > 0:
            pos.hands[left_hand_owner][piece.replace("+", "").upper()] += count

    for slot in capture_state.get("right_hand", []):
        piece = slot.get("piece", ".")
        count = int(slot.get("count", 0))
        if piece not in {".", "empty", "blank", "unknown"} and count > 0:
            pos.hands[right_hand_owner][piece.replace("+", "").upper()] += count

    return pos


def suggest_move_from_capture_state(
    capture_state: Dict[str, object],
    side_to_move: str,
    depth: int = 2,
    left_hand_owner: str = WHITE,
    right_hand_owner: str = BLACK,
) -> Optional[Dict[str, object]]:
    pos = position_from_capture_state(
        capture_state=capture_state,
        side_to_move=side_to_move,
        left_hand_owner=left_hand_owner,
        right_hand_owner=right_hand_owner,
    )
    move = choose_best_move(pos, side_to_move, depth=depth)
    if move is None:
        return None

    explanation = []
    nxt = apply_move(pos, move)

    if move.drop:
        explanation.append(f"Drop {move.piece} on {square_to_usi(move.to_sq)}.")
    else:
        explanation.append(
            f"Move {move.piece} from {square_to_usi(move.from_sq)} to {square_to_usi(move.to_sq)}"
            + (" and promote." if move.promote else ".")
        )

    if move.captured:
        explanation.append(f"It captures {move.captured}.")
    if in_check(nxt, opponent(side_to_move)):
        explanation.append("The move gives check.")
    if is_checkmate(nxt, opponent(side_to_move)):
        explanation.append("The move is checkmate.")

    return {
        "move": move.usi(),
        "drop": move.drop,
        "promote": move.promote,
        "from": move.from_sq,
        "to": move.to_sq,
        "piece": move.piece,
        "explanation": " ".join(explanation),
    }
