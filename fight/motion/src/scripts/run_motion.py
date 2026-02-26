from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

@dataclass
class MotionConfig:
    resize_w: int = 320                
    gray: bool = True
    blur_ksize: int = 5                 
    min_interval_ms: int = 33           

    bg_history: int = 300
    bg_var_threshold: float = 35.0
    bg_detect_shadows: bool = True

    morph_ksize: int = 3                
    morph_iters: int = 1

    thr_area_ratio: float = 0.0012     
    ema_alpha: float = 0.20            
    score_scale: float = 1.0            

    start_thr: float = 0.0012           
    end_thr: float = 0.0007             
    start_debounce: int = 3            
    end_debounce: int = 6               
    min_event_len: int = 18            
    merge_gap: int = 8                  

    save_every_n: int = 1               
    debug_overlay: bool = True          


def resize_keep_aspect(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def ensure_odd(k: int) -> int:
    if k <= 0:
        return 0
    return k if (k % 2 == 1) else (k + 1)


def find_video(video_path: Path) -> Path:
    if video_path.is_file():
        return video_path

    candidates = []

    candidates.append(Path.cwd() / video_path)

    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir / video_path)
    candidates.append(Path.cwd().parent / video_path.name)
    candidates.append(script_dir.parent / video_path.name)
    candidates.append(Path.cwd() / video_path.name)

    for c in candidates:
        if c.is_file():
            return c.resolve()

    tried = "\n".join(str(c.resolve() if c.exists() else c) for c in candidates)
    raise RuntimeError(f"Cannot open video: {video_path}\nTried:\n{tried}")

def build_segments(scores: List[float], cfg: MotionConfig) -> List[Dict]:
    segments: List[Tuple[int, int, float]] = []

    in_event = False
    start_idx = -1
    peak = 0.0

    above_cnt = 0
    below_cnt = 0

    for i, s in enumerate(scores):
        if not in_event:
            if s >= cfg.start_thr:
                above_cnt += 1
            else:
                above_cnt = 0

            if above_cnt >= cfg.start_debounce:
                in_event = True
                start_idx = i - cfg.start_debounce + 1
                peak = max(scores[start_idx:i+1])
                below_cnt = 0
        else:
            peak = max(peak, s)

            if s < cfg.end_thr:
                below_cnt += 1
            else:
                below_cnt = 0

            if below_cnt >= cfg.end_debounce:
                end_idx = i - cfg.end_debounce + 1
                segments.append((start_idx, end_idx, float(peak)))
                in_event = False
                start_idx = -1
                peak = 0.0
                above_cnt = 0
                below_cnt = 0

    if in_event and start_idx >= 0:
        segments.append((start_idx, len(scores) - 1, float(peak)))

    segments = [s for s in segments if (s[1] - s[0] + 1) >= cfg.min_event_len]

    if not segments:
        return []

    merged: List[Tuple[int, int, float]] = [segments[0]]
    for s in segments[1:]:
        ps = merged[-1]
        gap = s[0] - ps[1] - 1
        if gap <= cfg.merge_gap:
            merged[-1] = (ps[0], s[1], max(ps[2], s[2]))
        else:
            merged.append(s)

    out = [{"start": a, "end": b, "peak": float(p)} for (a, b, p) in merged]
    return out

def run(video_path: Path, out_dir: Path, cfg: MotionConfig, debug_video: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = find_video(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 1e-3:
        fps = 30.0

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=int(cfg.bg_history),
        varThreshold=float(cfg.bg_var_threshold),
        detectShadows=bool(cfg.bg_detect_shadows),
    )

    k = int(cfg.morph_ksize)
    kernel = None
    if k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    writer = None
    debug_path = out_dir / "debug_motion.mp4"

    csv_path = out_dir / "motion_log.csv"
    segments_path = out_dir / "segments.json"
    cfg_path = out_dir / "motion_config.json"

    cfg_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    rows: List[Dict] = []
    scores_raw: List[float] = []
    scores_ema: List[float] = []

    ema = 0.0
    frame_idx = -1
    last_ts_ms = -1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ts_ms is None or math.isnan(ts_ms):
            ts_ms = frame_idx * (1000.0 / fps)

        if last_ts_ms >= 0 and (ts_ms - last_ts_ms) < cfg.min_interval_ms:
            continue
        last_ts_ms = ts_ms

        orig_h, orig_w = frame.shape[:2]

        img = resize_keep_aspect(frame, cfg.resize_w)
        if cfg.gray:
            img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_proc = img

        bk = ensure_odd(int(cfg.blur_ksize))
        if bk > 0:
            img_proc = cv2.GaussianBlur(img_proc, (bk, bk), 0)

        fg = mog2.apply(img_proc)

        if kernel is not None and cfg.morph_iters > 0:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=int(cfg.morph_iters))
            fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=int(cfg.morph_iters))

        fg_bin = (fg >= 200).astype(np.uint8)

        fg_area = int(fg_bin.sum())
        h, w = fg_bin.shape[:2]
        denom = float(h * w) if (h > 0 and w > 0) else 1.0
        area_ratio = (fg_area / denom) * float(cfg.score_scale)

        ema = (1.0 - cfg.ema_alpha) * ema + cfg.ema_alpha * area_ratio

        scores_raw.append(float(area_ratio))
        scores_ema.append(float(ema))

        if (frame_idx % max(1, cfg.save_every_n)) == 0:
            rows.append({
                "frame": frame_idx,
                "t_ms": float(ts_ms),
                "score_raw": float(area_ratio),
                "score_ema": float(ema),
                "fg_area": int(fg_area),
                "w": int(w),
                "h": int(h),
            })

        if debug_video:
            dbg = frame.copy()
            if cfg.debug_overlay:
                cv2.putText(dbg, f"frame={frame_idx}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(dbg, f"raw={area_ratio:.6f}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(dbg, f"ema={ema:.6f}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(dbg, f"fg_area={fg_area}", (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (dbg.shape[1], dbg.shape[0]))
            writer.write(dbg)

    cap.release()
    if writer is not None:
        writer.release()

    segments = build_segments(scores_ema, cfg)

    segments_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame","t_ms","score_raw","score_ema","fg_area","w","h"])
        wri.writeheader()
        for r in rows:
            wri.writerow(r)

    print(f"[OK] video: {video_path}")
    print(f"[OK] out_dir: {out_dir}")
    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {segments_path}")
    if debug_video:
        print(f"[OK] wrote: {debug_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=str, help="video path (relative or absolute)")
    p.add_argument("--out", type=str, default="out_motion", help="output directory")
    p.add_argument("--debug-video", action="store_true", help="export debug overlay video")
    p.add_argument("--resize-w", type=int, default=320)
    p.add_argument("--thr-area", type=float, default=0.0012)
    p.add_argument("--start-thr", type=float, default=0.0012)
    p.add_argument("--end-thr", type=float, default=0.0007)
    p.add_argument("--min-len", type=int, default=18)
    p.add_argument("--merge-gap", type=int, default=8)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cfg = MotionConfig(
        resize_w=int(args.resize_w),
        thr_area_ratio=float(args.thr_area),
        start_thr=float(args.start_thr),
        end_thr=float(args.end_thr),
        min_event_len=int(args.min_len),
        merge_gap=int(args.merge_gap),
    )

    run(Path(args.video), Path(args.out), cfg, debug_video=bool(args.debug_video))