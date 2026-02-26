from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

@dataclass
class FrameDiffResult:
    score: float
    diff: np.ndarray  

class FrameDiffer:
    def __init__(self) -> None:
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def compute(self, gray: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> FrameDiffResult:
        if self._prev is None:
            self._prev = gray.copy()
            return FrameDiffResult(score=0.0, diff=np.zeros_like(gray))

        diff = cv2.absdiff(gray, self._prev)

        if roi_mask is not None:
            diff_masked = cv2.bitwise_and(diff, diff, mask=roi_mask)
            allowed = roi_mask > 0
            score = float(np.mean(diff_masked[allowed])) if np.any(allowed) else 0.0
            diff_out = diff_masked
        else:
            score = float(np.mean(diff))
            diff_out = diff

        self._prev = gray.copy()
        return FrameDiffResult(score=score, diff=diff_out)