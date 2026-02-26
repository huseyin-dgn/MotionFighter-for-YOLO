from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.core.config import MotionConfig
from src.ingest.cam_reader import frame_generator
from src.motion.bg_subtractor import BGSubtractor
from src.motion.frame_diff import FrameDiffer
from src.motion.gate import MotionGate
from src.motion.roi import apply_mask, build_ignore_mask
from src.utils.image_ops import blur, resize_keep_aspect, to_gray
from src.utils.logger import setup_logger


@dataclass
class MotionStepOut:
    ts: float
    now_ms: int
    score: float
    pass_frame: bool
    frame_resized: np.ndarray
    vis_bgr: np.ndarray
    gray_used: bool


class _CSVTrace:
    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / "motion_trace.csv"
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(
            [
                "frame_idx",
                "ts_ms",
                "method",
                "resize_w",
                "raw_score",
                "gate_pass",
                "gate_reason",
                "gate_thr_used",
                "gate_hist_sum",
                "current_over_thr",
                "diff_mean",
                "diff_max",
                "moving_px",
                "cc",
                "largest_area",
                "learning_rate",
            ]
        )

    def write_row(self, row: list) -> None:
        self._w.writerow(row)

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()


class MotionRunner:
    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self.log = setup_logger("motion")
        self.out_dir = Path(cfg.debug_out_dir)

        if cfg.debug_enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        self.gate = MotionGate(
            open_threshold=cfg.threshold_open,
            close_threshold=cfg.threshold_close,
            window_size=cfg.window_size,
            min_pass=cfg.min_pass,
        )

        self.differ = FrameDiffer()
        self.bgsub = BGSubtractor(
            history=cfg.bg_history,
            var_threshold=cfg.bg_var_threshold,
            detect_shadows=cfg.bg_detect_shadows,
            morph_ksize=cfg.bg_morph_ksize,
            min_contour_area=cfg.min_contour_area,
        )

        self.ignore_mask: Optional[np.ndarray] = None
        self.mask_ready = False
        self.last_ts_ms = 0
        self.pass_count = 0
        self.total_count = 0
        self.frame_idx = -1
        self.method = cfg.method.lower().strip()

        self.trace: Optional[_CSVTrace] = None
        if cfg.debug_enabled:
            self.trace = _CSVTrace(self.out_dir)

        self.log.info(f"CWD={os.getcwd()}")
        self.log.info(f"DEBUG_OUT={self.out_dir.resolve()}")
        self.log.info(
            "CFG "
            f"method={cfg.method} "
            f"resize_width={cfg.resize_width} gray={cfg.grayscale} blur={cfg.blur_ksize} "
            f"open_thr={cfg.threshold_open} close_thr={cfg.threshold_close} "
            f"window={cfg.window_size} min_pass={cfg.min_pass} "
            f"min_interval_ms={cfg.min_interval_ms} "
            f"min_contour_area={cfg.min_contour_area} "
            f"warmup_frames={cfg.warmup_frames}"
        )

    def close(self) -> None:
        if self.trace is not None:
            self.trace.close()
            self.trace = None

    def step(self, ts: float, frame_bgr: np.ndarray) -> Optional[MotionStepOut]:
        now_ms = int(ts * 1000)
        self.frame_idx += 1

        if self.cfg.min_interval_ms > 0 and (now_ms - self.last_ts_ms) < self.cfg.min_interval_ms:
            return None

        self.last_ts_ms = now_ms
        self.total_count += 1

        f = resize_keep_aspect(frame_bgr, self.cfg.resize_width)
        g = to_gray(f) if self.cfg.grayscale else f
        g = blur(g, self.cfg.blur_ksize)

        if self.cfg.roi_enabled and (not self.mask_ready):
            h, w = g.shape[:2]
            self.ignore_mask = build_ignore_mask((h, w), self.cfg.roi_ignore_zones)
            self.mask_ready = True

        g_m = apply_mask(g, self.ignore_mask) if self.ignore_mask is not None else g

        diff_mean = ""
        diff_max = ""
        moving_px = ""
        cc = ""
        largest_area = ""
        lr_used = ""

        if self.method == "frame_diff":
            res = self.differ.compute(g_m, roi_mask=None)
            raw_score = float(res.score)
            if res.diff.size:
                diff_mean = float(np.mean(res.diff))
                diff_max = float(np.max(res.diff))
            else:
                diff_mean = 0.0
                diff_max = 0.0
            vis = cv2.cvtColor(res.diff, cv2.COLOR_GRAY2BGR)

        elif self.method == "bg_subtractor":
            if self.frame_idx < int(self.cfg.warmup_frames):
                lr = -1.0
            else:
                lr = 0.0
            res = self.bgsub.compute(g_m, ignore_mask=self.ignore_mask, learning_rate=lr)
            raw_score = float(res.score)
            moving_px = int(np.count_nonzero(res.fgmask))
            cc = int(res.num_components)
            largest_area = int(res.largest_area)
            lr_used = float(lr)
            vis = cv2.cvtColor(res.fgmask, cv2.COLOR_GRAY2BGR)

        else:
            raise ValueError(f"Unknown method: {self.cfg.method}")

        dec = self.gate.decide(raw_score)

        if dec.pass_frame:
            self.pass_count += 1

        if self.trace is not None:
            self.trace.write_row(
                [
                    self.frame_idx,
                    now_ms,
                    self.method,
                    self.cfg.resize_width,
                    f"{raw_score:.8f}",
                    int(dec.pass_frame),
                    dec.reason,
                    f"{dec.thr_used:.8f}",
                    dec.hist_sum,
                    int(dec.current),
                    diff_mean,
                    diff_max,
                    moving_px,
                    cc,
                    largest_area,
                    lr_used,
                ]
            )

        return MotionStepOut(
            ts=ts,
            now_ms=now_ms,
            score=dec.score,
            pass_frame=dec.pass_frame,
            frame_resized=f,
            vis_bgr=vis,
            gray_used=self.cfg.grayscale,
        )


def run_motion(source: str, cfg: MotionConfig) -> None:
    log = setup_logger("motion")
    runner = MotionRunner(cfg)

    try:
        for ts, frame in frame_generator(source):
            out = runner.step(ts, frame)
            if out is None:
                continue
            print(
                f"t={out.ts:7.3f}s  score={out.score:.8f}  "
                f"{'PASS' if out.pass_frame else 'DROP'}"
            )
    finally:
        runner.close()
        log.info(
            f"SUMMARY total={runner.total_count} pass={runner.pass_count} "
            f"drop={runner.total_count - runner.pass_count}"
        )