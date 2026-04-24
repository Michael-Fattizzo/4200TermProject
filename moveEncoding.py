from typing import Optional, Tuple
from shogiEngine import Move

BOARD_SIZE = 9

# Maximum theoretical moves:
# 81 from-squares × 81 to-squares × 2 (promotion)
# + drops (7 piece types × 81 squares)
TOTAL_MOVE_CLASSES = 81 * 81 * 2 + 7 * 81  # = 13122


DROP_PIECE_ORDER = ["P", "L", "N", "S", "G", "B", "R"]


def square_to_index(sq: Tuple[int, int]) -> int:
    """Convert (row, col) → 0–80"""
    r, c = sq
    return r * 9 + c


def encode_move_obj(move: Move) -> int:
    """
    Convert Move → fixed global index
    """

    # --- DROP ---
    if move.drop:
        piece_index = DROP_PIECE_ORDER.index(move.piece)
        to_index = square_to_index(move.to_sq)

        return 81 * 81 * 2 + piece_index * 81 + to_index

    # --- NORMAL MOVE ---
    from_index = square_to_index(move.from_sq)
    to_index = square_to_index(move.to_sq)

    base = from_index * 81 + to_index

    if move.promote:
        base += 81 * 81

    return base