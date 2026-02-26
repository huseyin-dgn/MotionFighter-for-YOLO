from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(slots=True)
class BGSubResult:
    score: float
    fgmask: np.ndarray
    largest_area: int
    num_components: int
    centroids: list[tuple[int, int]]


class BGSubtractor:
    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 55,
        detect_shadows: bool = False,
        morph_ksize: int = 5,
        min_contour_area: int = 100,
    ) -> None:
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.morph_ksize = morph_ksize
        self.min_contour_area = min_contour_area

    def compute(
        self,
        gray: np.ndarray,
        ignore_mask: Optional[np.ndarray] = None,
        learning_rate: float = -1.0,
    ) -> BGSubResult:
        fg = self.sub.apply(gray, learningRate=float(learning_rate))

        fg = np.where(fg == 127, 0, fg).astype(np.uint8)

        k = int(self.morph_ksize or 0)
        if k > 0:
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

        found = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]

        clean = np.zeros_like(fg)

        largest_area = 0
        centroids: list[tuple[int, int]] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area:
                continue

            cv2.drawContours(clean, [c], -1, 255, -1)

            if area > largest_area:
                largest_area = int(area)

            M = cv2.moments(c)
            if M["m00"] != 0.0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        fg = clean
        num_components = len(centroids)

        allowed: Optional[np.ndarray]
        if ignore_mask is not None:
            ignore_u8 = (ignore_mask > 0).astype(np.uint8) * 255
            fg[ignore_u8 > 0] = 0
            allowed = ignore_u8 == 0
            denom = int(np.sum(allowed))
        else:
            allowed = None
            denom = int(fg.size)

        if denom <= 0:
            score = 0.0
        else:
            if allowed is None:
                moving = int(np.sum(fg > 0))
            else:
                moving = int(np.sum((fg > 0)[allowed]))
            score = float(moving) / float(denom)

        return BGSubResult(
            score=score,
            fgmask=fg,
            largest_area=int(largest_area),
            num_components=int(num_components),
            centroids=centroids,
        )