from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class PoseGateDecision:
    pose_ok: bool
    hist_positive: int
    current_positive: bool
    score: float


class PoseGate:
    def __init__(self, window_size: int = 6, need_positive: int = 2):
        self.window_size = int(max(1, window_size))
        self.need_positive = int(max(1, need_positive))
        self.hist: Deque[bool] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self.hist.clear()

    def update(self, score: float, positive: bool) -> PoseGateDecision:
        self.hist.append(bool(positive))
        hist_positive = int(sum(self.hist))
        pose_ok = hist_positive >= self.need_positive
        return PoseGateDecision(
            pose_ok=bool(pose_ok),
            hist_positive=hist_positive,
            current_positive=bool(positive),
            score=float(score),
        )