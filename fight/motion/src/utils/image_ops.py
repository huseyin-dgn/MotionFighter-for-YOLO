from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    nh = int(round(h * scale))
    return cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def blur(frame: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 0:
        return frame
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)