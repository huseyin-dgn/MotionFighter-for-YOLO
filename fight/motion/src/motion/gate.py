from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class GateDecision:
    pass_frame: bool
    score: float
    current: bool
    motion_on_before: bool
    motion_on_after: bool
    thr_used: float
    hist_sum: int
    reason: str


class MotionGate:
    def __init__(
        self,
        open_threshold: float,
        close_threshold: float,
        window_size: int = 5,
        min_pass: int = 3,
        adaptive_thr: bool = False,
        adapt_frames: int = 60,
        k_on: float = 8.0,
        k_off: float = 4.0,
        thr_min: float = 1e-6,
        min_on_frames: int = 0,
        off_run: int = 0,
    ) -> None:
        self.open_threshold = float(open_threshold)
        self.close_threshold = float(close_threshold)
        self.window_size = int(window_size)
        self.min_pass = int(min_pass)

        self.adaptive_thr = bool(adaptive_thr)
        self.adapt_frames = int(max(1, adapt_frames))
        self.k_on = float(k_on)
        self.k_off = float(k_off)
        self.thr_min = float(thr_min)

        self.min_on_frames = int(max(0, min_on_frames))
        self.off_run = int(max(0, off_run))

        self.history: Deque[bool] = deque(maxlen=self.window_size)
        self.motion_on = False

        self._score_hist: Deque[float] = deque(maxlen=self.adapt_frames)
        self._thr_on = float(self.open_threshold)
        self._thr_off = float(self.close_threshold)

        self._on_age = 0
        self._off_streak = 0

    def reset(self) -> None:
        self.history.clear()
        self.motion_on = False
        self._score_hist.clear()
        self._thr_on = float(self.open_threshold)
        self._thr_off = float(self.close_threshold)
        self._on_age = 0
        self._off_streak = 0

    def _update_adaptive_thresholds(self, score: float) -> None:
        s = float(score)
        self._score_hist.append(s)

        if not self._score_hist:
            base = 0.0
        else:
            vals = sorted(self._score_hist)
            base = float(vals[len(vals) // 2])

        thr_on = max(self.thr_min, base * self.k_on)
        thr_off = max(self.thr_min, base * self.k_off)

        self._thr_on = float(thr_on)
        self._thr_off = float(thr_off)

    def decide(self, score: float) -> GateDecision:
        motion_on_before = self.motion_on

        if self.adaptive_thr:
            self._update_adaptive_thresholds(score)
            open_thr = self._thr_on
            close_thr = self._thr_off
        else:
            open_thr = self.open_threshold
            close_thr = self.close_threshold

        if self.motion_on:
            thr_used = close_thr
            current = score >= close_thr
        else:
            thr_used = open_thr
            current = score >= open_thr

        self.history.append(bool(current))
        hist_sum = int(sum(self.history))

        reason = "HOLD"

        if not self.motion_on:
            if hist_sum >= self.min_pass:
                self.motion_on = True
                self._on_age = 0
                self._off_streak = 0
                reason = "OPEN(hist_sum>=min_pass)"
        else:
            self._on_age += 1

            if current:
                self._off_streak = 0
            else:
                self._off_streak += 1

            can_close = True
            if self.min_on_frames > 0 and self._on_age < self.min_on_frames:
                can_close = False

            if self.off_run > 0:
                if can_close and self._off_streak >= self.off_run:
                    self.motion_on = False
                    self._on_age = 0
                    self._off_streak = 0
                    reason = "CLOSE(off_streak>=off_run)"
            else:
                if can_close and hist_sum == 0:
                    self.motion_on = False
                    self._on_age = 0
                    self._off_streak = 0
                    reason = "CLOSE(hist_sum==0)"

        return GateDecision(
            pass_frame=bool(self.motion_on),
            score=float(score),
            current=bool(current),
            motion_on_before=bool(motion_on_before),
            motion_on_after=bool(self.motion_on),
            thr_used=float(thr_used),
            hist_sum=hist_sum,
            reason=reason,
        )