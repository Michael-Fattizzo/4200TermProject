
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

PIECE_CANONICAL_SIZE = (64, 64)
CELL_INNER_MARGIN_X = 0.14
CELL_INNER_MARGIN_Y = 0.12
MIN_CONTOUR_AREA_RATIO = 0.015
PIECE_BOX_PAD_FRAC = 0.10


def crop_inner_cell(cell_img: np.ndarray, mx_frac: float = CELL_INNER_MARGIN_X, my_frac: float = CELL_INNER_MARGIN_Y) -> np.ndarray:
    h, w = cell_img.shape[:2]
    mx = int(round(w * mx_frac))
    my = int(round(h * my_frac))
    x1 = max(0, min(w - 1, mx))
    y1 = max(0, min(h - 1, my))
    x2 = max(x1 + 1, w - mx)
    y2 = max(y1 + 1, h - my)
    return cell_img[y1:y2, x1:x2].copy()


def piece_binary_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = thresh_inv if np.count_nonzero(thresh_inv) < np.count_nonzero(thresh) else thresh

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def largest_piece_component(mask: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = mask.shape[:2]
    min_area = h * w * MIN_CONTOUR_AREA_RATIO
    best_cnt = None
    best_area = -1

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None

    x, y, cw, ch = cv2.boundingRect(best_cnt)
    comp = np.zeros_like(mask)
    cv2.drawContours(comp, [best_cnt], -1, 255, thickness=cv2.FILLED)
    return comp, (x, y, cw, ch)


def normalize_piece_crop_gray(cell_img: np.ndarray, out_size: Tuple[int, int] = PIECE_CANONICAL_SIZE) -> Optional[np.ndarray]:
    inner = crop_inner_cell(cell_img)
    if inner.size == 0:
        return None

    mask = piece_binary_mask(inner)
    found = largest_piece_component(mask)
    if found is None:
        return None

    mask_full, (x, y, w, h) = found
    pad_x = int(round(w * PIECE_BOX_PAD_FRAC))
    pad_y = int(round(h * PIECE_BOX_PAD_FRAC))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(inner.shape[1], x + w + pad_x)
    y2 = min(inner.shape[0], y + h + pad_y)

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    piece = gray[y1:y2, x1:x2]
    piece_mask = mask_full[y1:y2, x1:x2]
    if piece.size == 0:
        return None

    piece = piece.copy()
    piece[piece_mask == 0] = 255

    h2, w2 = piece.shape
    side = max(h2, w2)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    ox = (side - w2) // 2
    oy = (side - h2) // 2
    canvas[oy:oy+h2, ox:ox+w2] = piece
    canvas = cv2.equalizeHist(canvas)
    canvas = cv2.resize(canvas, out_size, interpolation=cv2.INTER_AREA)
    return canvas


class SmallBoardCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class BoardCNNClassifier:
    model: SmallBoardCNN
    label_to_index: Dict[str, int]
    device: str = "cpu"

    @property
    def index_to_label(self) -> Dict[int, str]:
        return {v: k for k, v in self.label_to_index.items()}

    @classmethod
    def load(cls, checkpoint_path: str, device: Optional[str] = None) -> "BoardCNNClassifier":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        label_to_index = ckpt["label_to_index"]
        model = SmallBoardCNN(num_classes=len(label_to_index))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return cls(model=model, label_to_index=label_to_index, device=device)

    def predict_topk(self, cell_img: np.ndarray, k: int = 3) -> Optional[List[Tuple[str, float]]]:
        crop = normalize_piece_crop_gray(cell_img)
        if crop is None:
            return None
        x = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0) / 255.0
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0]
            probs = torch.softmax(logits, dim=0)
            vals, idxs = torch.topk(probs, k=min(k, probs.numel()))
        idx_to_label = self.index_to_label
        return [(idx_to_label[int(i)], float(v)) for v, i in zip(vals.cpu(), idxs.cpu())]
