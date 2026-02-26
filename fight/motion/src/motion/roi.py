from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np
from ..core.config import ROIBox

def build_ignore_mask(shape_hw: Tuple[int, int], ignore_zones: List[ROIBox]) -> np.ndarray:
    h, w = shape_hw
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if not ignore_zones:
        return mask

    for z in ignore_zones:
        x1 = int(round(z.x * w))
        y1 = int(round(z.y * h))
        x2 = int(round((z.x + z.w) * w))
        y2 = int(round((z.y + z.h) * h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0

    return mask

def apply_mask(gray: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return gray
    return cv2.bitwise_and(gray, gray, mask=mask)