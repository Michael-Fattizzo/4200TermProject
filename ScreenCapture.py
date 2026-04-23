
from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np

from CNNClassifier import BoardCNNClassifier
from occupancyCNNClassifier import OccupancyCNNClassifier

DEBUG_WINDOW_SCALE = 1.0
BOARD_MIN_AREA = 40_000
BOARD_ASPECT_MIN = 0.80
BOARD_ASPECT_MAX = 1.20

HAND_GAP_RATIO = 0.025
HAND_WIDTH_RATIO = 0.12
BOARD_PAD_RATIO = 0.02
BOARD_SHRINK_FRAC = 0.00

HAND_SLOT_COUNT = 7
SAVE_DIR = "capture_debug"
CELL_SAVE_DIR = "debug_cells"
TEMPLATE_ROOT = "templates"
CNN_CHECKPOINT = "board_cnn.pt"
OCCUPANCY_CHECKPOINT = "occupancy_cnn.pt"

CNN_CONFIDENCE_THRESHOLD = 0.55
CNN_AMBIGUITY_MARGIN = 0.08
OCCUPANCY_EMPTY_THRESHOLD = 0.70

HAND_MATCH_THRESHOLD = 0.48
DIGIT_MATCH_THRESHOLD = 0.45

SAVE_DETECTION_DEBUG = True
DETECTION_DEBUG_PREFIX = "detection_debug_monitor_"
STARTUP_DELAY_SECONDS = 5


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
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass
class Regions:
    board: Region
    left_hand: Region
    right_hand: Region


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


def save_board_cells(board_img: np.ndarray, out_dir: str = CELL_SAVE_DIR) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    full_dir = os.path.join(out_dir, timestamp)
    os.makedirs(full_dir, exist_ok=True)
    cells = split_board_into_cells(board_img)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            cv2.imwrite(os.path.join(full_dir, f"cell_r{r}_c{c}.png"), cell)
    return full_dir


def show_all_cells(board_img: np.ndarray) -> None:
    cells = split_board_into_cells(board_img)
    rows = [np.hstack(row) for row in cells]
    grid = np.vstack(rows)
    cv2.imshow("All Cells", grid)


def shrink_region(region: Region, frac: float = BOARD_SHRINK_FRAC) -> Region:
    if frac <= 0:
        return region
    dx = int(region.width * frac)
    dy = int(region.height * frac)
    return Region(left=region.left + dx, top=region.top + dy, width=max(1, region.width - 2 * dx), height=max(1, region.height - 2 * dy))


def refine_board_region_from_grid(image: np.ndarray, rough: Region) -> Optional[Region]:
    roi = crop_region(image, rough)
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    min_len = int(min(rough.width, rough.height) * 0.5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=max(20, min_len), maxLineGap=10)
    if lines is None:
        return None

    vertical_x: List[int] = []
    horizontal_y: List[int] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < 8 and dy > rough.height * 0.4:
            vertical_x.append((x1 + x2) // 2)
        elif dy < 8 and dx > rough.width * 0.4:
            horizontal_y.append((y1 + y2) // 2)

    if len(vertical_x) < 6 or len(horizontal_y) < 6:
        return None

    left = min(vertical_x)
    right = max(vertical_x)
    top = min(horizontal_y)
    bottom = max(horizontal_y)
    if right <= left or bottom <= top:
        return None

    pad_x = int((right - left) * 0.02)
    pad_y = int((bottom - top) * 0.02)
    left = max(0, left - pad_x)
    right = min(roi.shape[1] - 1, right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(roi.shape[0] - 1, bottom + pad_y)

    side = max(right - left, bottom - top)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    half = side // 2

    new_left = max(0, cx - half)
    new_top = max(0, cy - half)
    new_right = min(roi.shape[1], new_left + side)
    new_bottom = min(roi.shape[0], new_top + side)

    return Region(left=rough.left + new_left, top=rough.top + new_top, width=new_right - new_left, height=new_bottom - new_top)


def detect_shogi_board(image: np.ndarray) -> Optional[Region]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
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
        dist = ((rect_center_x - center_x) ** 2 + (rect_center_y - center_y) ** 2) ** 0.5
        score = area - dist * 120
        if score > best_score:
            best_score = score
            best_rect = (x, y, w, h)

    if best_rect is None:
        return None

    x, y, w, h = best_rect
    pad = int(min(w, h) * BOARD_PAD_RATIO)
    rough = Region(left=max(0, x - pad), top=max(0, y - pad), width=w + 2 * pad, height=h + 2 * pad)
    refined = refine_board_region_from_grid(image, rough)
    board = refined if refined is not None else rough
    return shrink_region(board, BOARD_SHRINK_FRAC)


def manually_select_board(screen: np.ndarray) -> Optional[Region]:
    roi = cv2.selectROI("Select Board", screen, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Board")
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None
    side = int(min(w, h))
    return shrink_region(Region(left=int(x), top=int(y), width=side, height=side), BOARD_SHRINK_FRAC)


def derive_regions(image: np.ndarray, board: Region) -> Regions:
    _, img_w = image.shape[:2]
    side_gap = int(board.width * HAND_GAP_RATIO)
    hand_width = int(board.width * HAND_WIDTH_RATIO)

    left_hand = Region(left=max(0, board.left - hand_width - side_gap), top=max(0, board.top), width=hand_width, height=board.height)
    right_hand = Region(left=min(img_w - hand_width, board.right + side_gap), top=max(0, board.top), width=hand_width, height=board.height)
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
            return screen, derive_regions(screen, board), monitor_index
    screen = capture_monitor(1)
    print("Auto-detection failed. Please drag a box around the board.")
    board = manually_select_board(screen)
    if board is not None:
        return screen, derive_regions(screen, board), 1
    raise RuntimeError("Could not detect shogi board.")


def detect_internal_grid_bounds(board_img: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=int(min(board_img.shape[:2]) * 0.5), maxLineGap=8)
    if lines is None:
        return None

    vertical: List[int] = []
    horizontal: List[int] = []
    h, w = board_img.shape[:2]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < 6 and dy > h * 0.5:
            vertical.append((x1 + x2) // 2)
        elif dy < 6 and dx > w * 0.5:
            horizontal.append((y1 + y2) // 2)

    if len(vertical) < 6 or len(horizontal) < 6:
        return None

    def cluster(vals: List[int], gap: int = 10) -> np.ndarray:
        vals = sorted(vals)
        groups = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - groups[-1][-1]) <= gap:
                groups[-1].append(v)
            else:
                groups.append([v])
        return np.array([int(round(sum(g) / len(g))) for g in groups], dtype=int)

    vx = cluster(vertical)
    hy = cluster(horizontal)
    if len(vx) < 2 or len(hy) < 2:
        return None
    return np.linspace(int(vx[0]), int(vx[-1]), 10).astype(int), np.linspace(int(hy[0]), int(hy[-1]), 10).astype(int)


def get_board_grid_boundaries(board_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    detected = detect_internal_grid_bounds(board_img)
    if detected is not None:
        return detected
    h, w = board_img.shape[:2]
    return np.linspace(0, w, 10).astype(int), np.linspace(0, h, 10).astype(int)


def split_board_into_cells(board_img: np.ndarray) -> List[List[np.ndarray]]:
    xs, ys = get_board_grid_boundaries(board_img)
    out = []
    for row in range(9):
        row_cells = []
        for col in range(9):
            x1, x2 = xs[col], xs[col + 1]
            y1, y2 = ys[row], ys[row + 1]
            row_cells.append(board_img[y1:y2, x1:x2].copy())
        out.append(row_cells)
    return out


def normalize_img(img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(img)


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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.resize(thresh, out_size, interpolation=cv2.INTER_AREA)


def score_template(query: np.ndarray, tmpl: np.ndarray) -> float:
    if len(tmpl.shape) == 3:
        tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    if query.shape != tmpl.shape:
        tmpl = cv2.resize(tmpl, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(normalize_img(query), normalize_img(tmpl), cv2.TM_CCOEFF_NORMED)
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


def classify_board_cell(cell_img: np.ndarray, occupancy_classifier: OccupancyCNNClassifier, piece_classifier: BoardCNNClassifier) -> Dict[str, Any]:
    occ_label, occ_conf, occ_prob = occupancy_classifier.predict(cell_img)
    if occ_label == "empty" and occ_conf >= OCCUPANCY_EMPTY_THRESHOLD:
        return {"label": "empty", "score": round(occ_conf, 4), "occupied": False, "occ_prob": round(occ_prob, 4)}

    preds = piece_classifier.predict_topk(cell_img, k=3)
    if not preds:
        return {"label": "unknown", "score": 0.0, "occupied": True, "occ_prob": round(occ_prob, 4)}

    best_label, best_score = preds[0]
    second_score = preds[1][1] if len(preds) > 1 else 0.0

    if best_score < CNN_CONFIDENCE_THRESHOLD or (best_score - second_score) < CNN_AMBIGUITY_MARGIN:
        return {"label": "unknown", "score": round(best_score, 4), "occupied": True, "occ_prob": round(occ_prob, 4)}

    return {"label": best_label, "score": round(best_score, 4), "occupied": True, "occ_prob": round(occ_prob, 4)}


def extract_count_badge(slot_img: np.ndarray) -> Optional[np.ndarray]:
    h, w = slot_img.shape[:2]
    x1 = int(w * 0.45)
    y1 = int(h * 0.55)
    x2 = int(w * 0.98)
    y2 = int(h * 0.98)
    if x2 <= x1 or y2 <= y1:
        return None
    badge = slot_img[y1:y2, x1:x2].copy()
    return None if badge.size == 0 else badge


def segment_badge_digits(badge_img: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(badge_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(thresh) / thresh.size > 0.7:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_boxes: List[Tuple[int, int, int, int]] = []
    h, w = thresh.shape[:2]
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 20 or ch < h * 0.25:
            continue
        digit_boxes.append((x, y, cw, ch))
    digit_boxes.sort(key=lambda b: b[0])

    out = []
    for x, y, cw, ch in digit_boxes:
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        out.append(thresh[y1:y2, x1:x2].copy())
    return out


def fallback_hand_count(slot_img: np.ndarray) -> int:
    gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    return 1 if np.count_nonzero(edges) / edges.size > 0.05 else 0


def read_badge_count(slot_img: np.ndarray, digit_templates: Dict[str, List[np.ndarray]], min_score: float = DIGIT_MATCH_THRESHOLD) -> int:
    badge = extract_count_badge(slot_img)
    if badge is None:
        return fallback_hand_count(slot_img)

    digit_imgs = segment_badge_digits(badge)
    if not digit_imgs:
        return fallback_hand_count(slot_img)

    digits: List[str] = []
    for digit_img in digit_imgs:
        label, score = match_digit_templates(digit_img, digit_templates)
        if score >= min_score and label.isdigit():
            digits.append(label)

    if not digits:
        return fallback_hand_count(slot_img)

    try:
        return max(1, int("".join(digits)))
    except ValueError:
        return fallback_hand_count(slot_img)


def classify_hand_slot(slot_img: np.ndarray, templates: Dict[str, Dict[str, List[np.ndarray]]], digit_templates: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
    query = preprocess_hand_slot(slot_img)
    hand_templates = templates.get("hand", {})
    if not hand_templates:
        return {"occupied": False, "piece": "unknown", "count": 0, "score": 0.0}

    label, score = best_template_match(query, hand_templates)
    if score < HAND_MATCH_THRESHOLD or label in ("empty", ".", "blank"):
        return {"occupied": False, "piece": ".", "count": 0, "score": round(score, 4)}

    return {"occupied": True, "piece": label, "count": read_badge_count(slot_img, digit_templates), "score": round(score, 4)}


def split_hand_into_slots(hand_img: np.ndarray, num_slots: int = HAND_SLOT_COUNT) -> List[np.ndarray]:
    h, _ = hand_img.shape[:2]
    slot_h = h / float(num_slots)
    return [hand_img[int(round(i * slot_h)):int(round((i + 1) * slot_h)), :].copy() for i in range(num_slots)]


def extract_state(screen: np.ndarray, regions: Regions, occupancy_classifier: OccupancyCNNClassifier, piece_classifier: BoardCNNClassifier, templates: Dict[str, Dict[str, List[np.ndarray]]], digit_templates: Dict[str, List[np.ndarray]]) -> Dict[str, object]:
    board_img = crop_region(screen, regions.board)
    left_hand_img = crop_region(screen, regions.left_hand)
    right_hand_img = crop_region(screen, regions.right_hand)

    board_cells = split_board_into_cells(board_img)
    board_state = [[classify_board_cell(cell, occupancy_classifier, piece_classifier) for cell in row] for row in board_cells]

    left_hand_state = [classify_hand_slot(slot, templates, digit_templates) for slot in split_hand_into_slots(left_hand_img)]
    right_hand_state = [classify_hand_slot(slot, templates, digit_templates) for slot in split_hand_into_slots(right_hand_img)]

    return {"board": board_state, "left_hand": left_hand_state, "right_hand": right_hand_state}


def draw_regions(image: np.ndarray, regions: Regions) -> np.ndarray:
    vis = image.copy()
    for name, r, color in [
        ("board", regions.board, (0, 255, 0)),
        ("left_hand", regions.left_hand, (255, 200, 0)),
        ("right_hand", regions.right_hand, (0, 200, 255)),
    ]:
        cv2.rectangle(vis, (r.left, r.top), (r.right, r.bottom), color, 2)
        cv2.putText(vis, name, (r.left, max(20, r.top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return vis


def draw_board_grid(board_img: np.ndarray) -> np.ndarray:
    vis = board_img.copy()
    h, w = vis.shape[:2]
    xs, ys = get_board_grid_boundaries(board_img)
    for x in xs:
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
    for y in ys:
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
    return vis


def draw_hand_slots(hand_img: np.ndarray, num_slots: int = HAND_SLOT_COUNT) -> np.ndarray:
    vis = hand_img.copy()
    h, w = vis.shape[:2]
    for i in range(1, num_slots):
        y = int(round(i * h / float(num_slots)))
        cv2.line(vis, (0, y), (w, y), (255, 0, 255), 1)
    return vis


def print_state(state: Dict[str, object]) -> None:
    print("\\n=== BOARD ===")
    for row in state["board"]:
        print(" | ".join(f'{cell["label"]}:{cell["score"]:.2f}' for cell in row))

    print("\\n=== LEFT HAND ===")
    for i, slot in enumerate(state["left_hand"], start=1):
        print(f"slot {i}: {slot}")

    print("\\n=== RIGHT HAND ===")
    for i, slot in enumerate(state["right_hand"], start=1):
        print(f"slot {i}: {slot}")


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
            imgs = []
            for path in glob.glob(os.path.join(label_dir, "*")):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
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
        imgs = []
        for path in glob.glob(os.path.join(label_dir, "*")):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)
        if imgs:
            digit_templates[label] = imgs
    return digit_templates


def main() -> None:
    print("Starting shogi screen reader with occupancy CNN + piece CNN...")
    print("Controls: q=quit, r=re-detect board, s=save crops, d=save 81 cells, a=show 81-cell grid")
    print(f"Switch to the shogi board window now... ({STARTUP_DELAY_SECONDS}s)")
    time.sleep(STARTUP_DELAY_SECONDS)

    occupancy_classifier = OccupancyCNNClassifier.load(OCCUPANCY_CHECKPOINT)
    piece_classifier = BoardCNNClassifier.load(CNN_CHECKPOINT)
    templates = load_templates(TEMPLATE_ROOT)
    digit_templates = load_digit_templates(TEMPLATE_ROOT)

    print("Loaded occupancy CNN:", OCCUPANCY_CHECKPOINT)
    print("Loaded piece CNN:", CNN_CHECKPOINT)
    print("Piece labels:", sorted(piece_classifier.label_to_index.keys()))

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

    last_print = 0.0
    print_interval = 1.0

    while True:
        screen = capture_monitor(active_monitor)
        state = extract_state(screen, regions, occupancy_classifier, piece_classifier, templates, digit_templates)

        overlay = draw_regions(screen, regions)
        board_img = crop_region(screen, regions.board)
        left_hand_img = crop_region(screen, regions.left_hand)
        right_hand_img = crop_region(screen, regions.right_hand)

        board_debug = draw_board_grid(board_img)
        left_hand_debug = draw_hand_slots(left_hand_img)
        right_hand_debug = draw_hand_slots(right_hand_img)

        if DEBUG_WINDOW_SCALE != 1.0:
            overlay = cv2.resize(overlay, None, fx=DEBUG_WINDOW_SCALE, fy=DEBUG_WINDOW_SCALE, interpolation=cv2.INTER_AREA)

        cv2.imshow("Shogi Capture - Overlay", overlay)
        cv2.imshow("Shogi Capture - Board", board_debug)
        cv2.imshow("Shogi Capture - Left Hand", left_hand_debug)
        cv2.imshow("Shogi Capture - Right Hand", right_hand_debug)

        now = time.time()
        if now - last_print >= print_interval:
            print_state(state)
            last_print = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            try:
                screen, regions, active_monitor = detect_all_regions()
                print("\\nRe-detected regions:")
                print("board     =", regions.board.as_dict())
                print("left_hand =", regions.left_hand.as_dict())
                print("right_hand=", regions.right_hand.as_dict())
                print("monitor   =", active_monitor)
            except RuntimeError as exc:
                print(f"Re-detection failed: {exc}")
        elif key == ord("s"):
            save_calibration_images(screen, regions)
            print(f"Saved debug images to ./{SAVE_DIR}/")
        elif key == ord("d"):
            saved_dir = save_board_cells(board_img)
            print(f"Saved 81 board cells to ./{saved_dir}/")
        elif key == ord("a"):
            show_all_cells(board_img)
            print("Showing 81-cell grid preview.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
