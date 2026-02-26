from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionCfg:
    resize_w: int = 640              
    resize_h: int = 640             
    gray: bool = True
    blur_ksize: int = 5              
    bg_history: int = 300
    bg_var_threshold: int = 35
    bg_detect_shadows: bool = True

    morph_ksize: int = 3             
    morph_iters: int = 1

    thr_area_ratio: float = 0.0015   
    win: int = 16
    cand_min_hits: int = 6           

    min_event_len: int = 12        
    end_debounce: int = 10          
    cooldown: int = 6               

    save_debug_video: bool = False  
    debug_fps: float = 30.0


@dataclass
class Segment:
    start: int
    end: int
    peak: float


def resize_frame(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    if w <= 0 and h <= 0:
        return frame
    if w > 0 and h > 0:
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    H, W = frame.shape[:2]
    if w > 0:
        nh = int(round(H * (w / W)))
        return cv2.resize(frame, (w, nh), interpolation=cv2.INTER_AREA)
    else:
        nw = int(round(W * (h / H)))
        return cv2.resize(frame, (nw, h), interpolation=cv2.INTER_AREA)


def preprocess(frame: np.ndarray, cfg: MotionCfg) -> np.ndarray:
    frame = resize_frame(frame, cfg.resize_w, cfg.resize_h)
    if cfg.gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cfg.blur_ksize and cfg.blur_ksize > 0:
        k = cfg.blur_ksize
        if k % 2 == 0:
            k += 1
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    return frame


def clean_mask(mask: np.ndarray, cfg: MotionCfg) -> np.ndarray:
    if cfg.morph_ksize and cfg.morph_ksize > 0:
        k = cfg.morph_ksize
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=cfg.morph_iters)
    return mask


def fg_area_ratio(mask: np.ndarray) -> float:
    fg = float(np.count_nonzero(mask))
    total = float(mask.size)
    return fg / max(total, 1.0)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    x = int(stats[i, cv2.CC_STAT_LEFT])
    y = int(stats[i, cv2.CC_STAT_TOP])
    w = int(stats[i, cv2.CC_STAT_WIDTH])
    h = int(stats[i, cv2.CC_STAT_HEIGHT])
    return (x, y, w, h)

def run(video_path: Path, out_dir: Path, cfg: MotionCfg) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    bgs = cv2.createBackgroundSubtractorMOG2(
        history=cfg.bg_history,
        varThreshold=cfg.bg_var_threshold,
        detectShadows=cfg.bg_detect_shadows,
    )

    csv_path = out_dir / "motion_log.csv"
    seg_path = out_dir / "segments.json"

    debug_writer = None

    window_hits: List[int] = []
    in_event = False
    event_start = -1
    event_peak = 0.0
    end_zero_streak = 0
    cooldown_left = 0

    segments: List[Segment] = []

    frame_idx = -1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame",
            "area_ratio",
            "thr",
            "motion",
            "win_hits",
            "cand_ok",
            "in_event",
            "zero_streak",
            "cooldown_left",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        ])

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1

            proc = preprocess(frame_bgr, cfg)

            fg = bgs.apply(proc)
            if cfg.bg_detect_shadows:
                fg = np.where(fg == 127, 0, fg).astype(np.uint8)

            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            fg = clean_mask(fg, cfg)

            ar = fg_area_ratio(fg)
            motion = 1 if ar >= cfg.thr_area_ratio else 0

            window_hits.append(motion)
            if len(window_hits) > cfg.win:
                window_hits.pop(0)
            win_sum = int(sum(window_hits))
            cand_ok = 1 if (len(window_hits) == cfg.win and win_sum >= cfg.cand_min_hits) else 0

            bbox = mask_bbox(fg)
            bx = by = bw = bh = -1
            if bbox is not None:
                bx, by, bw, bh = bbox

            if cooldown_left > 0:
                cooldown_left -= 1

            if not in_event:
                if cand_ok and cooldown_left == 0:
                    in_event = True
                    event_start = frame_idx
                    event_peak = ar
                    end_zero_streak = 0
            else:
                if ar > event_peak:
                    event_peak = ar

                if motion == 0:
                    end_zero_streak += 1
                else:
                    end_zero_streak = 0

                if end_zero_streak >= cfg.end_debounce:
                    event_end = frame_idx - cfg.end_debounce  
                    length = event_end - event_start + 1

                    if length >= cfg.min_event_len:
                        segments.append(Segment(start=event_start, end=event_end, peak=float(event_peak)))

                    in_event = False
                    event_start = -1
                    event_peak = 0.0
                    end_zero_streak = 0
                    cooldown_left = cfg.cooldown

                    window_hits.clear()

            w.writerow([
                frame_idx,
                f"{ar:.8f}",
                f"{cfg.thr_area_ratio:.8f}",
                motion,
                win_sum,
                cand_ok,
                int(in_event),
                end_zero_streak,
                cooldown_left,
                bx, by, bw, bh
            ])

            if cfg.save_debug_video:
                if debug_writer is None:
                    H, W = resize_frame(frame_bgr, cfg.resize_w, cfg.resize_h).shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    debug_writer = cv2.VideoWriter(
                        str(out_dir / "debug_overlay.mp4"),
                        fourcc,
                        cfg.debug_fps,
                        (W, H),
                    )

                vis = resize_frame(frame_bgr, cfg.resize_w, cfg.resize_h).copy()
                mask3 = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
                mask3[:, :, 1] = 0
                mask3[:, :, 2] = mask3[:, :, 2]  # red
                vis = cv2.addWeighted(vis, 1.0, mask3, 0.35, 0)

                if bbox is not None:
                    x, y, ww, hh = bbox
                    cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)

                txt = f"f={frame_idx} ar={ar:.4f} m={motion} win={win_sum}/{cfg.win} cand={cand_ok} ev={int(in_event)} z={end_zero_streak} cd={cooldown_left}"
                cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                debug_writer.write(vis)

    cap.release()
    if debug_writer is not None:
        debug_writer.release()

    if in_event and event_start >= 0:
        event_end = frame_idx
        length = event_end - event_start + 1
        if length >= cfg.min_event_len:
            segments.append(Segment(start=event_start, end=event_end, peak=float(event_peak)))

    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in segments], f, indent=2)

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {seg_path}")
    if cfg.save_debug_video:
        print(f"[OK] wrote: {out_dir / 'debug_overlay.mp4'}")
    print(f"[OK] segments: {len(segments)}")
    for i, s in enumerate(segments, 1):
        print(f"  seg#{i}: frames={s.start}-{s.end} len={s.end - s.start + 1} peak={s.peak:.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=str)
    p.add_argument("--out", type=str, default="motion_fixed_out")
    p.add_argument("--thr", type=float, default=None, help="override thr_area_ratio")
    p.add_argument("--debug-video", action="store_true", help="save debug_overlay.mp4")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = MotionCfg()
    if args.thr is not None:
        cfg.thr_area_ratio = float(args.thr)
    if args.debug_video:
        cfg.save_debug_video = True

    run(Path(args.video), Path(args.out), cfg)