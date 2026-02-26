from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Segment:
    start_f: int
    end_f: int
    peak: float

    @property
    def length(self) -> int:
        return self.end_f - self.start_f + 1

def ema_smooth(values: List[float], alpha: float = 0.2) -> List[float]:
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out

def moving_average(values: List[float], win: int = 5) -> List[float]:
    if win <= 1 or len(values) == 0:
        return values[:]
    out = []
    s = 0.0
    q = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > win:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def detect_segments(
    scores: List[float],
    fps: float,
    thr_on: float,
    thr_off: float,
    min_len_sec: float = 0.5,
    merge_gap_sec: float = 0.15,
    smooth: str = "ema",      
    ema_alpha: float = 0.2,
    ma_win: int = 5,
) -> List[Segment]:
    if not scores:
        return []

    if thr_off > thr_on:
        raise ValueError("thr_off must be <= thr_on (hysteresis).")

    if smooth == "ema":
        s = ema_smooth(scores, alpha=ema_alpha)
    elif smooth == "ma":
        s = moving_average(scores, win=ma_win)
    else:
        s = scores[:]

    min_len = max(1, int(round(min_len_sec * fps)))
    merge_gap = max(0, int(round(merge_gap_sec * fps)))

    segs: List[Segment] = []
    in_seg = False
    start = 0
    peak = 0.0

    for i, val in enumerate(s):
        if not in_seg:
            if val >= thr_on:
                in_seg = True
                start = i
                peak = val
        else:
            peak = max(peak, val)
            if val < thr_off:
                end = i
                segs.append(Segment(start_f=start, end_f=end, peak=peak))
                in_seg = False

    if in_seg:
        segs.append(Segment(start_f=start, end_f=len(s) - 1, peak=peak))
    segs = [x for x in segs if x.length >= min_len]

    if not segs:
        return []

    merged = [segs[0]]
    for cur in segs[1:]:
        prev = merged[-1]
        gap = cur.start_f - prev.end_f - 1
        if gap <= merge_gap:
            prev.end_f = cur.end_f
            prev.peak = max(prev.peak, cur.peak)
        else:
            merged.append(cur)

    return merged

def frames_to_time(seg: Segment, fps: float) -> tuple[float, float]:
    return seg.start_f / fps, seg.end_f / fps