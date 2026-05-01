
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

OCCUPANCY_SIZE = (64, 64)
CELL_INNER_MARGIN_X = 0.14
CELL_INNER_MARGIN_Y = 0.12


def crop_inner_cell(cell_img: np.ndarray, mx_frac: float = CELL_INNER_MARGIN_X, my_frac: float = CELL_INNER_MARGIN_Y) -> np.ndarray:
    h, w = cell_img.shape[:2]
    mx = int(round(w * mx_frac))
    my = int(round(h * my_frac))
    x1 = max(0, min(w - 1, mx))
    y1 = max(0, min(h - 1, my))
    x2 = max(x1 + 1, w - mx)
    y2 = max(y1 + 1, h - my)
    return cell_img[y1:y2, x1:x2].copy()


def preprocess_occupancy_cell(cell_img: np.ndarray, out_size: Tuple[int, int] = OCCUPANCY_SIZE) -> np.ndarray:
    inner = crop_inner_cell(cell_img)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    return cv2.resize(gray, out_size, interpolation=cv2.INTER_AREA)


class SmallOccupancyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class OccupancyCNNClassifier:
    model: SmallOccupancyCNN
    device: str = "cpu"

    @classmethod
    def load(cls, checkpoint_path: str, device: Optional[str] = None) -> "OccupancyCNNClassifier":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = SmallOccupancyCNN()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return cls(model=model, device=device)

    def predict(self, cell_img: np.ndarray) -> Tuple[str, float, float]:
        crop = preprocess_occupancy_cell(cell_img)
        x = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0) / 255.0
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        empty_prob = float(probs[0])
        occupied_prob = float(probs[1])
        label = "empty" if empty_prob >= occupied_prob else "occupied"
        confidence = max(empty_prob, occupied_prob)
        return label, confidence, occupied_prob
