from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np

from shogiEngine import (
    BLACK,
    WHITE,
    Move,
    Position,
    apply_move,
    generate_legal_moves,
    opponent,
    parse_capture_label,
    suggest_move_from_capture_state,
)

# Configuration

DEBUG_WINDOW_SCALE = 1.0
BOARD_MIN_AREA = 40_000
BOARD_ASPECT_MIN = 0.80
BOARD_ASPECT_MAX = 1.20

HAND_GAP_RATIO = 0.025
HAND_WIDTH_RATIO = 0.12
BOARD_PAD_RATIO = 0.02

HAND_SLOT_COUNT = 7
SAVE_DIR = "capture_debug"
TEMPLATE_ROOT = "templates"

BOARD_MATCH_THRESHOLD = 0.48
HAND_MATCH_THRESHOLD = 0.48
DIGIT_MATCH_THRESHOLD = 0.45
PROMOTION_RED_RATIO_THRESHOLD = 0.03

SAVE_DETECTION_DEBUG = True
DETECTION_DEBUG_PREFIX = "detection_debug_monitor_"

STARTUP_DELAY_SECONDS = 5

# Agent / tracker configuration

INITIAL_SIDE_TO_MOVE = BLACK
LEFT_HAND_OWNER = WHITE
RIGHT_HAND_OWNER = BLACK

# A position must remain unchanged for this many consecutive frames
# before it is treated as "stable" and eligible for move tracking.
STABLE_FRAMES_REQUIRED = 3

ENGINE_DEPTH = 2

# Data classes

@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_dict(self) -> Dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Regions:
    board: Region
    left_hand: Region
    right_hand: Region


@dataclass
class TrackedMove:
    ply: int
    side: str
    usi: str
    explanation: str


@dataclass
class TrackerState:
    previous_stable_capture_state: Optional[Dict[str, object]] = None
    previous_stable_position: Optional[Position] = None
    side_to_move: Optional[str] = INITIAL_SIDE_TO_MOVE

    pending_signature: Optional[Tuple] = None
    pending_count: int = 0
    current_stable_signature: Optional[Tuple] = None

    move_history: List[TrackedMove] = field(default_factory=list)
    last_detected_move: Optional[TrackedMove] = None


# Screen capture

def capture_monitor(monitor_index: int) -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        img = np.array(sct.grab(monitor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def crop_region(image: np.ndarray, region: Region) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, region.left)
    y1 = max(0, region.top)
    x2 = min(w, region.right)
    y2 = min(h, region.bottom)
    return image[y1:y2, x1:x2].copy()

# Debug / saving helpers

def ensure_save_dir() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)


def save_detection_debug(image: np.ndarray, name: str) -> None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    cv2.imwrite(name, edges)


def save_calibration_images(screen: np.ndarray, regions: Regions) -> None:
    ensure_save_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    cv2.imwrite(os.path.join(SAVE_DIR, f"{timestamp}_full_screen.png"), screen)
    cv2.imwrite(os.path.join(SAVE_DIR, f"{timestamp}_board.png"), crop_region(screen, regions.board))
    cv2.imwrite(os.path.join(SAVE_DIR, f"{timestamp}_left_hand.png"), crop_region(screen, regions.left_hand))
    cv2.imwrite(os.path.join(SAVE_DIR, f"{timestamp}_right_hand.png"), crop_region(screen, regions.right_hand))

# Detection

def detect_shogi_board(image: np.ndarray) -> Optional[Region]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_rect: Optional[Tuple[int, int, int, int]] = None
    best_score = float("-inf")

    img_h, img_w = image.shape[:2]
    center_x = img_w / 2.0
    center_y = img_h / 2.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        if area < BOARD_MIN_AREA:
            continue

        aspect = w / float(h)
        if not (BOARD_ASPECT_MIN <= aspect <= BOARD_ASPECT_MAX):
            continue

        rect_center_x = x + w / 2.0
        rect_center_y = y + h / 2.0
        dist_to_center = ((rect_center_x - center_x) ** 2 + (rect_center_y - center_y) ** 2) ** 0.5

        score = area - dist_to_center * 120
        if score > best_score:
            best_score = score
            best_rect = (x, y, w, h)

    if best_rect is None:
        return None

    x, y, w, h = best_rect
    pad = int(min(w, h) * BOARD_PAD_RATIO)

    return Region(
        left=max(0, x - pad),
        top=max(0, y - pad),
        width=w + 2 * pad,
        height=h + 2 * pad,
    )


def manually_select_board(screen: np.ndarray) -> Optional[Region]:
    roi = cv2.selectROI("Select Board", screen, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Board")

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None

    side = int(min(w, h))
    return Region(left=int(x), top=int(y), width=side, height=side)


def derive_regions(image: np.ndarray, board: Region) -> Regions:
    _, img_w = image.shape[:2]
    side_gap = int(board.width * HAND_GAP_RATIO)
    hand_width = int(board.width * HAND_WIDTH_RATIO)

    left_hand = Region(
        left=max(0, board.left - hand_width - side_gap),
        top=max(0, board.top),
        width=hand_width,
        height=board.height,
    )

    right_hand = Region(
        left=min(img_w - hand_width, board.right + side_gap),
        top=max(0, board.top),
        width=hand_width,
        height=board.height,
    )

    return Regions(board=board, left_hand=left_hand, right_hand=right_hand)


def detect_all_regions() -> Tuple[np.ndarray, Regions, int]:
    with mss.mss() as sct:
        monitor_count = len(sct.monitors) - 1

    for monitor_index in range(1, monitor_count + 1):
        screen = capture_monitor(monitor_index)

        if SAVE_DETECTION_DEBUG:
            save_detection_debug(screen, f"{DETECTION_DEBUG_PREFIX}{monitor_index}.png")

        board = detect_shogi_board(screen)
        if board is not None:
            regions = derive_regions(screen, board)
            print(f"Detected board on monitor {monitor_index}")
            return screen, regions, monitor_index

    screen = capture_monitor(1)
    print("Auto-detection failed. Please drag a box around the board.")
    board = manually_select_board(screen)
    if board is not None:
        regions = derive_regions(screen, board)
        return screen, regions, 1

    raise RuntimeError("Could not detect shogi board.")

# Splitting helpers

def split_board_into_cells(board_img: np.ndarray) -> List[List[np.ndarray]]:
    h, w = board_img.shape[:2]
    cell_h = h / 9.0
    cell_w = w / 9.0

    cells: List[List[np.ndarray]] = []
    for row in range(9):
        row_cells: List[np.ndarray] = []
        for col in range(9):
            y1 = int(round(row * cell_h))
            y2 = int(round((row + 1) * cell_h))
            x1 = int(round(col * cell_w))
            x2 = int(round((col + 1) * cell_w))
            row_cells.append(board_img[y1:y2, x1:x2].copy())
        cells.append(row_cells)
    return cells


def split_hand_into_slots(hand_img: np.ndarray, num_slots: int = HAND_SLOT_COUNT) -> List[np.ndarray]:
    h, _ = hand_img.shape[:2]
    slot_h = h / float(num_slots)

    slots: List[np.ndarray] = []
    for i in range(num_slots):
        y1 = int(round(i * slot_h))
        y2 = int(round((i + 1) * slot_h))
        slots.append(hand_img[y1:y2, :].copy())
    return slots

# Template loading

def load_templates(template_root: str = TEMPLATE_ROOT) -> Dict[str, Dict[str, List[np.ndarray]]]:
    out: Dict[str, Dict[str, List[np.ndarray]]] = {"board": {}, "hand": {}}

    for group in ("board", "hand"):
        group_dir = os.path.join(template_root, group)
        if not os.path.isdir(group_dir):
            continue

        for label in os.listdir(group_dir):
            label_dir = os.path.join(group_dir, label)
            if not os.path.isdir(label_dir):
                continue

            imgs: List[np.ndarray] = []
            for path in glob.glob(os.path.join(label_dir, "*")):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    imgs.append(img)

            if imgs:
                out[group][label] = imgs

    return out


def load_digit_templates(template_root: str = TEMPLATE_ROOT) -> Dict[str, List[np.ndarray]]:
    digit_templates: Dict[str, List[np.ndarray]] = {}
    digits_dir = os.path.join(template_root, "digits")

    if not os.path.isdir(digits_dir):
        return digit_templates

    for label in os.listdir(digits_dir):
        label_dir = os.path.join(digits_dir, label)
        if not os.path.isdir(label_dir):
            continue

        imgs: List[np.ndarray] = []
        for path in glob.glob(os.path.join(label_dir, "*")):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img)

        if imgs:
            digit_templates[label] = imgs

    return digit_templates

# Preprocessing

def normalize_img(img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(img)


def preprocess_board_cell(cell_img: np.ndarray, out_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    mx = int(w * 0.08)
    my = int(h * 0.08)

    x1 = min(mx, w - 1)
    y1 = min(my, h - 1)
    x2 = max(x1 + 1, w - mx)
    y2 = max(y1 + 1, h - my)

    cropped = gray[y1:y2, x1:x2]
    return cv2.resize(cropped, out_size, interpolation=cv2.INTER_AREA)


def preprocess_hand_slot(slot_img: np.ndarray, out_size: Tuple[int, int] = (64, 96)) -> np.ndarray:
    gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    mx = int(w * 0.08)
    my = int(h * 0.05)

    x1 = min(mx, w - 1)
    y1 = min(my, h - 1)
    x2 = max(x1 + 1, w - mx)
    y2 = max(y1 + 1, h - my)

    cropped = gray[y1:y2, x1:x2]
    return cv2.resize(cropped, out_size, interpolation=cv2.INTER_AREA)


def preprocess_digit_img(img: np.ndarray, out_size: Tuple[int, int] = (24, 32)) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.resize(thresh, out_size, interpolation=cv2.INTER_AREA)

# Template scoring

def score_template(query: np.ndarray, tmpl: np.ndarray) -> float:
    if query.shape != tmpl.shape:
        tmpl = cv2.resize(tmpl, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_AREA)

    query_n = normalize_img(query)
    tmpl_n = normalize_img(tmpl)

    res = cv2.matchTemplate(query_n, tmpl_n, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)


def best_template_match(query: np.ndarray, template_bank: Dict[str, List[np.ndarray]]) -> Tuple[str, float]:
    best_label = "unknown"
    best_score = -1.0

    for label, tmpls in template_bank.items():
        for tmpl in tmpls:
            score = score_template(query, tmpl)
            if score > best_score:
                best_score = score
                best_label = label

    return best_label, best_score


def match_digit_templates(digit_img: np.ndarray, digit_templates: Dict[str, List[np.ndarray]]) -> Tuple[str, float]:
    query = preprocess_digit_img(digit_img)

    best_label = "?"
    best_score = -1.0

    for label, tmpls in digit_templates.items():
        for tmpl in tmpls:
            score = score_template(query, preprocess_digit_img(tmpl))
            if score > best_score:
                best_score = score
                best_label = label

    return best_label, best_score

# Promoted piece helpers

def red_ratio(cell_img: np.ndarray) -> float:
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, 70, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    return float(np.count_nonzero(red_mask) / red_mask.size)


def is_red_promoted(cell_img: np.ndarray, red_ratio_threshold: float = PROMOTION_RED_RATIO_THRESHOLD) -> bool:
    return red_ratio(cell_img) >= red_ratio_threshold


def normalize_promoted_label(label: str, promoted_detected: bool) -> str:
    if label in ("empty", ".", "blank", "unknown"):
        return label

    if promoted_detected and "+" not in label:
        if label[-1] in ("P", "L", "N", "S", "B", "R"):
            return label + "+"

    if not promoted_detected and "+" in label:
        return label.replace("+", "")

    return label

# Hand count badge reading

def extract_count_badge(slot_img: np.ndarray) -> Optional[np.ndarray]:
    h, w = slot_img.shape[:2]

    x1 = int(w * 0.45)
    y1 = int(h * 0.55)
    x2 = int(w * 0.98)
    y2 = int(h * 0.98)

    if x2 <= x1 or y2 <= y1:
        return None

    badge = slot_img[y1:y2, x1:x2].copy()
    if badge.size == 0:
        return None

    return badge


def segment_badge_digits(badge_img: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(badge_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = np.count_nonzero(thresh) / thresh.size
    if white_ratio > 0.7:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_boxes: List[Tuple[int, int, int, int]] = []
    h, w = thresh.shape[:2]

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < 20:
            continue
        if ch < h * 0.25:
            continue
        digit_boxes.append((x, y, cw, ch))

    if not digit_boxes:
        return []

    digit_boxes.sort(key=lambda b: b[0])

    digit_imgs: List[np.ndarray] = []
    for x, y, cw, ch in digit_boxes:
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        digit_imgs.append(thresh[y1:y2, x1:x2].copy())

    return digit_imgs


def fallback_hand_count(slot_img: np.ndarray) -> int:
    gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = np.count_nonzero(edges) / edges.size
    return 1 if edge_density > 0.05 else 0


def read_badge_count(
    slot_img: np.ndarray,
    digit_templates: Dict[str, List[np.ndarray]],
    min_score: float = DIGIT_MATCH_THRESHOLD,
) -> int:
    badge = extract_count_badge(slot_img)
    if badge is None:
        return fallback_hand_count(slot_img)

    digit_imgs = segment_badge_digits(badge)
    if not digit_imgs:
        return fallback_hand_count(slot_img)

    digits: List[str] = []
    for digit_img in digit_imgs:
        label, score = match_digit_templates(digit_img, digit_templates)
        if score < min_score or not label.isdigit():
            continue
        digits.append(label)

    if not digits:
        return fallback_hand_count(slot_img)

    try:
        value = int("".join(digits))
        return max(1, value)
    except ValueError:
        return fallback_hand_count(slot_img)

# Recognition

def classify_board_cell(
    cell_img: np.ndarray,
    templates: Dict[str, Dict[str, List[np.ndarray]]],
    threshold: float = BOARD_MATCH_THRESHOLD,
) -> Dict[str, Any]:
    query = preprocess_board_cell(cell_img)
    board_templates = templates.get("board", {})

    if not board_templates:
        return {
            "label": "unknown",
            "score": 0.0,
            "occupied": False,
            "promoted": False,
            "red_ratio": 0.0,
        }

    promoted_detected = is_red_promoted(cell_img)
    rr = red_ratio(cell_img)

    if promoted_detected:
        filtered_templates = {
            label: imgs
            for label, imgs in board_templates.items()
            if "+" in label or label in ("empty", ".", "blank")
        }
    else:
        filtered_templates = {
            label: imgs
            for label, imgs in board_templates.items()
            if "+" not in label or label in ("empty", ".", "blank")
        }

    if not filtered_templates:
        filtered_templates = board_templates

    label, score = best_template_match(query, filtered_templates)
    label = normalize_promoted_label(label, promoted_detected)

    if score < threshold:
        return {
            "label": "unknown",
            "score": round(score, 4),
            "occupied": False,
            "promoted": promoted_detected,
            "red_ratio": round(rr, 4),
        }

    occupied = label not in ("empty", ".", "blank")
    return {
        "label": label,
        "score": round(score, 4),
        "occupied": occupied,
        "promoted": "+" in label,
        "red_ratio": round(rr, 4),
    }


def classify_hand_slot(
    slot_img: np.ndarray,
    templates: Dict[str, Dict[str, List[np.ndarray]]],
    digit_templates: Dict[str, List[np.ndarray]],
    threshold: float = HAND_MATCH_THRESHOLD,
) -> Dict[str, Any]:
    query = preprocess_hand_slot(slot_img)

    hand_templates = templates.get("hand", {})
    if not hand_templates:
        return {"occupied": False, "piece": "unknown", "count": 0, "score": 0.0}

    label, score = best_template_match(query, hand_templates)

    if score < threshold or label in ("empty", ".", "blank"):
        return {"occupied": False, "piece": ".", "count": 0, "score": round(score, 4)}

    count = read_badge_count(slot_img, digit_templates)
    return {
        "occupied": True,
        "piece": label,
        "count": count,
        "score": round(score, 4),
    }

# State extraction

def extract_state(
    screen: np.ndarray,
    regions: Regions,
    templates: Dict[str, Dict[str, List[np.ndarray]]],
    digit_templates: Dict[str, List[np.ndarray]],
) -> Dict[str, object]:
    board_img = crop_region(screen, regions.board)
    left_hand_img = crop_region(screen, regions.left_hand)
    right_hand_img = crop_region(screen, regions.right_hand)

    board_cells = split_board_into_cells(board_img)
    board_state = [
        [classify_board_cell(cell, templates) for cell in row]
        for row in board_cells
    ]

    left_slots = split_hand_into_slots(left_hand_img)
    right_slots = split_hand_into_slots(right_hand_img)

    left_hand_state = [classify_hand_slot(slot, templates, digit_templates) for slot in left_slots]
    right_hand_state = [classify_hand_slot(slot, templates, digit_templates) for slot in right_slots]

    return {
        "board": board_state,
        "left_hand": left_hand_state,
        "right_hand": right_hand_state,
    }

# Visualization

def draw_regions(image: np.ndarray, regions: Regions) -> np.ndarray:
    vis = image.copy()

    region_specs = [
        ("board", regions.board, (0, 255, 0)),
        ("left_hand", regions.left_hand, (255, 200, 0)),
        ("right_hand", regions.right_hand, (0, 200, 255)),
    ]

    for name, r, color in region_specs:
        cv2.rectangle(vis, (r.left, r.top), (r.right, r.bottom), color, 2)
        cv2.putText(
            vis,
            name,
            (r.left, max(20, r.top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis


def draw_board_grid(board_img: np.ndarray) -> np.ndarray:
    vis = board_img.copy()
    h, w = vis.shape[:2]

    for i in range(1, 9):
        x = int(round(i * w / 9.0))
        y = int(round(i * h / 9.0))
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)

    return vis


def draw_hand_slots(hand_img: np.ndarray, num_slots: int = HAND_SLOT_COUNT) -> np.ndarray:
    vis = hand_img.copy()
    h, w = vis.shape[:2]

    for i in range(1, num_slots):
        y = int(round(i * h / float(num_slots)))
        cv2.line(vis, (0, y), (w, y), (255, 0, 255), 1)

    return vis

# Printing helpers

def print_state(state: Dict[str, object]) -> None:
    print("\n=== BOARD ===")
    for row in state["board"]:
        print(" ".join(cell["label"] for cell in row))

    print("\n=== LEFT HAND ===")
    for i, slot in enumerate(state["left_hand"], start=1):
        print(f"slot {i}: {slot}")

    print("\n=== RIGHT HAND ===")
    for i, slot in enumerate(state["right_hand"], start=1):
        print(f"slot {i}: {slot}")


def print_move_history(tracker: TrackerState, max_items: int = 12) -> None:
    print("\n=== MOVE HISTORY ===")
    if not tracker.move_history:
        print("(none)")
        return

    for item in tracker.move_history[-max_items:]:
        move_no = (item.ply + 1) // 2
        mover = "B" if item.side == BLACK else "W"
        print(f"{move_no:>3}.{mover} {item.usi}  {item.explanation}")

# Turn detection + move tracking helpers

def normalize_hand_piece_label(label: str) -> Optional[str]:
    if label in {"", ".", "empty", "blank", "unknown"}:
        return None
    return label.replace("+", "").upper()


def capture_state_to_position(
    capture_state: Dict[str, object],
    side_to_move: Optional[str],
    left_hand_owner: str = LEFT_HAND_OWNER,
    right_hand_owner: str = RIGHT_HAND_OWNER,
) -> Position:
    pos = Position(side_to_move=side_to_move or INITIAL_SIDE_TO_MOVE)

    for r in range(9):
        for c in range(9):
            label = capture_state["board"][r][c]["label"]
            try:
                pos.board[r][c] = parse_capture_label(label)
            except Exception:
                pos.board[r][c] = None

    for slot in capture_state.get("left_hand", []):
        piece = normalize_hand_piece_label(str(slot.get("piece", ".")))
        count = int(slot.get("count", 0))
        if piece and count > 0:
            pos.hands[left_hand_owner][piece] += count

    for slot in capture_state.get("right_hand", []):
        piece = normalize_hand_piece_label(str(slot.get("piece", ".")))
        count = int(slot.get("count", 0))
        if piece and count > 0:
            pos.hands[right_hand_owner][piece] += count

    return pos


def piece_signature(piece: Optional[Any]) -> Optional[Tuple[str, str, bool]]:
    if piece is None:
        return None
    return (piece.kind, piece.owner, piece.promoted)


def position_signature(position: Position) -> Tuple:
    board_sig = tuple(
        tuple(piece_signature(position.board[r][c]) for c in range(9))
        for r in range(9)
    )
    hand_sig = (
        tuple(sorted(position.hands[BLACK].items())),
        tuple(sorted(position.hands[WHITE].items())),
    )
    return board_sig + hand_sig


def positions_equivalent(a: Position, b: Position) -> bool:
    return position_signature(a) == position_signature(b)


def move_to_explanation(move: Move) -> str:
    if move.drop:
        return f"drop {move.piece} to {move.usi().split('*', 1)[1]}"
    if move.promote:
        return f"{move.usi()} (promotion)"
    return move.usi()


def infer_transition_move(
    prev_position: Position,
    curr_position: Position,
    expected_side_to_move: Optional[str],
) -> Tuple[Optional[str], Optional[Move]]:
    candidate_sides = [expected_side_to_move] if expected_side_to_move else [BLACK, WHITE]

    for side in candidate_sides:
        if side is None:
            continue
        trial_prev = prev_position.clone()
        trial_prev.side_to_move = side

        for move in generate_legal_moves(trial_prev, side):
            try:
                nxt = apply_move(trial_prev, move)
            except Exception:
                continue
            if positions_equivalent(nxt, curr_position):
                return side, move

    if expected_side_to_move is None:
        return None, None

    other_side = opponent(expected_side_to_move)
    trial_prev = prev_position.clone()
    trial_prev.side_to_move = other_side
    for move in generate_legal_moves(trial_prev, other_side):
        try:
            nxt = apply_move(trial_prev, move)
        except Exception:
            continue
        if positions_equivalent(nxt, curr_position):
            return other_side, move

    return None, None


def update_tracker_from_capture_state(
    tracker: TrackerState,
    capture_state: Dict[str, object],
) -> bool:
    current_position = capture_state_to_position(
        capture_state=capture_state,
        side_to_move=tracker.side_to_move,
        left_hand_owner=LEFT_HAND_OWNER,
        right_hand_owner=RIGHT_HAND_OWNER,
    )
    current_sig = position_signature(current_position)

    if tracker.pending_signature == current_sig:
        tracker.pending_count += 1
    else:
        tracker.pending_signature = current_sig
        tracker.pending_count = 1

    if tracker.pending_count < STABLE_FRAMES_REQUIRED:
        return False

    if tracker.current_stable_signature == current_sig:
        return False

    tracker.current_stable_signature = current_sig

    if tracker.previous_stable_position is None:
        tracker.previous_stable_position = current_position
        tracker.previous_stable_capture_state = capture_state
        tracker.last_detected_move = None
        return True

    moved_side, detected_move = infer_transition_move(
        prev_position=tracker.previous_stable_position,
        curr_position=current_position,
        expected_side_to_move=tracker.side_to_move,
    )

    if detected_move is not None and moved_side is not None:
        tracked = TrackedMove(
            ply=len(tracker.move_history) + 1,
            side=moved_side,
            usi=detected_move.usi(),
            explanation=move_to_explanation(detected_move),
        )
        tracker.move_history.append(tracked)
        tracker.last_detected_move = tracked
        tracker.side_to_move = opponent(moved_side)
    else:
        # Fallback: keep synchronization even when the exact move could not
        # be reconstructed, but do not invent a move history entry.
        tracker.last_detected_move = None
        if tracker.side_to_move is None:
            tracker.side_to_move = INITIAL_SIDE_TO_MOVE

    tracker.previous_stable_position = current_position
    tracker.previous_stable_capture_state = capture_state
    return True


def side_label(side: Optional[str]) -> str:
    if side == BLACK:
        return "BLACK"
    if side == WHITE:
        return "WHITE"
    return "UNKNOWN"

# Main app loop

def main() -> None:
    print("Starting shogi screen reader...")
    print("Controls: q=quit, r=re-detect board, s=save crops")
    print(f"Switch to the shogi board window now... ({STARTUP_DELAY_SECONDS}s)")
    time.sleep(STARTUP_DELAY_SECONDS)

    templates = load_templates(TEMPLATE_ROOT)
    digit_templates = load_digit_templates(TEMPLATE_ROOT)

    print("Loaded templates:")
    for group, classes in templates.items():
        print(f"  {group}: {list(classes.keys())}")
    print(f"  digits: {list(digit_templates.keys())}")

    if not templates.get("board"):
        print("Warning: no board templates loaded from templates/board")
    if not templates.get("hand"):
        print("Warning: no hand templates loaded from templates/hand")
    if not digit_templates:
        print("Warning: no digit templates loaded from templates/digits")

    try:
        screen, regions, active_monitor = detect_all_regions()
    except RuntimeError as exc:
        print(f"Startup detection failed: {exc}")
        return

    print("Detected regions:")
    print("board     =", regions.board.as_dict())
    print("left_hand =", regions.left_hand.as_dict())
    print("right_hand=", regions.right_hand.as_dict())
    print("monitor   =", active_monitor)

    print("\nTracker configuration:")
    print("  INITIAL_SIDE_TO_MOVE =", side_label(INITIAL_SIDE_TO_MOVE))
    print("  LEFT_HAND_OWNER      =", side_label(LEFT_HAND_OWNER))
    print("  RIGHT_HAND_OWNER     =", side_label(RIGHT_HAND_OWNER))
    print("  STABLE_FRAMES_REQUIRED =", STABLE_FRAMES_REQUIRED)

    tracker = TrackerState(side_to_move=INITIAL_SIDE_TO_MOVE)

    last_print = 0.0
    print_interval = 1.0

    while True:
        screen = capture_monitor(active_monitor)
        state = extract_state(screen, regions, templates, digit_templates)
        became_stable = update_tracker_from_capture_state(tracker, state)

        suggestion = None
        try:
            suggestion = suggest_move_from_capture_state(
                capture_state=state,
                side_to_move=tracker.side_to_move or INITIAL_SIDE_TO_MOVE,
                depth=ENGINE_DEPTH,
                left_hand_owner=LEFT_HAND_OWNER,
                right_hand_owner=RIGHT_HAND_OWNER,
            )
        except Exception as exc:
            suggestion = None
            print(f"Engine error: {exc}")

        overlay = draw_regions(screen, regions)

        board_img = crop_region(screen, regions.board)
        left_hand_img = crop_region(screen, regions.left_hand)
        right_hand_img = crop_region(screen, regions.right_hand)

        board_debug = draw_board_grid(board_img)
        left_hand_debug = draw_hand_slots(left_hand_img)
        right_hand_debug = draw_hand_slots(right_hand_img)

        overlay_lines = [
            f"Turn: {side_label(tracker.side_to_move)}",
            f"Stable frames: {tracker.pending_count}/{STABLE_FRAMES_REQUIRED}",
        ]
        if tracker.last_detected_move is not None:
            overlay_lines.append(f"Last move: {tracker.last_detected_move.usi}")
        if suggestion:
            overlay_lines.append(f"Best: {suggestion['move']}")

        for idx, line in enumerate(overlay_lines):
            cv2.putText(
                overlay,
                line,
                (20, 35 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if DEBUG_WINDOW_SCALE != 1.0:
            overlay = cv2.resize(
                overlay,
                None,
                fx=DEBUG_WINDOW_SCALE,
                fy=DEBUG_WINDOW_SCALE,
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow("Shogi Capture - Overlay", overlay)
        cv2.imshow("Shogi Capture - Board", board_debug)
        cv2.imshow("Shogi Capture - Left Hand", left_hand_debug)
        cv2.imshow("Shogi Capture - Right Hand", right_hand_debug)

        now = time.time()
        if now - last_print >= print_interval:
            print_state(state)

            print("\n=== TURN TRACKER ===")
            print("side_to_move:", side_label(tracker.side_to_move))
            if tracker.last_detected_move is not None:
                print("last_detected_move:", tracker.last_detected_move.usi)
            else:
                print("last_detected_move: none")

            if became_stable:
                print("position_status: new stable position accepted")
            else:
                print("position_status: waiting / unchanged")

            print_move_history(tracker)

            print("\n=== ENGINE SUGGESTION ===")
            if suggestion:
                print("Move:", suggestion["move"])
                print("Explanation:", suggestion["explanation"])
            else:
                print("No legal move found or engine failed.")

            last_print = now

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            try:
                screen, regions, active_monitor = detect_all_regions()
                print("\nRe-detected regions:")
                print("board     =", regions.board.as_dict())
                print("left_hand =", regions.left_hand.as_dict())
                print("right_hand=", regions.right_hand.as_dict())
                print("monitor   =", active_monitor)

                tracker = TrackerState(side_to_move=INITIAL_SIDE_TO_MOVE)
                print("Tracker reset after region re-detection.")
            except RuntimeError as exc:
                print(f"Re-detection failed: {exc}")
        elif key == ord("s"):
            save_calibration_images(screen, regions)
            print(f"Saved debug images to ./{SAVE_DIR}/")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
