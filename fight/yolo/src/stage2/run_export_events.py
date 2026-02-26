from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

THIS = Path(__file__).resolve()
FIGHT_ROOT = THIS.parents[3]
MOTION_ROOT = FIGHT_ROOT / "motion"
sys.path.insert(0, str(MOTION_ROOT))
sys.path.insert(0, str(FIGHT_ROOT))

from src.core.config import MotionConfig, load_config
from src.ingest.cam_reader import frame_generator
from src.utils.image_ops import blur, resize_keep_aspect, to_gray
from src.utils.logger import setup_logger

from yolo.src.stage2.stage2_core import (
    RoiStabilizer,
    apply_min_area_ratio,
    build_segment_mask,
    compute_motion_scores,
    crop,
    detect_motion_segments,
    extract_boxes,
    load_yaml,
    make_writer,
    merge_segments_by_gap,
    parse_cfg,
    select_fight_roi,
    topk_by_area,
    track_ttl_update,
    uniform_probe_indices,
    write_json,
    yolo_infer,
    yolo_model,
)


@dataclass
class SegInfo:
    start_f: int
    end_f: int
    peak: float
    accepted: int
    probe_seen: int
    probe_2p: int


def _expand_xyxy(x1: int, y1: int, x2: int, y2: int, W: int, H: int, margin: float) -> Tuple[int, int, int, int]:
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad = int(round(float(margin) * max(w, h)))
    nx1 = max(0, x1 - pad)
    ny1 = max(0, y1 - pad)
    nx2 = min(W, x2 + pad)
    ny2 = min(H, y2 + pad)
    if nx2 <= nx1:
        nx2 = min(W, nx1 + 2)
    if ny2 <= ny1:
        ny2 = min(H, ny1 + 2)
    return nx1, ny1, nx2, ny2


class MotionRoi:
    def __init__(self, resize_w: int, blur_ksize: int, diff_thr: int, morph_ksize: int, min_area_ratio: float):
        self.resize_w = int(resize_w)
        self.blur_ksize = int(blur_ksize)
        self.diff_thr = int(diff_thr)
        self.morph_ksize = int(morph_ksize)
        self.min_area_ratio = float(min_area_ratio)
        self.prev_gray: Optional[np.ndarray] = None

    def reset(self):
        self.prev_gray = None

    def get_roi(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        H0, W0 = frame_bgr.shape[:2]
        if W0 <= 0 or H0 <= 0:
            return None

        if self.resize_w > 0 and W0 != self.resize_w:
            small = resize_keep_aspect(frame_bgr, self.resize_w)
        else:
            small = frame_bgr

        gray = to_gray(small)
        if self.blur_ksize > 0:
            gray = blur(gray, self.blur_ksize)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        _, bw = cv2.threshold(diff, self.diff_thr, 255, cv2.THRESH_BINARY)

        if self.morph_ksize > 0:
            k = np.ones((self.morph_ksize, self.morph_ksize), dtype=np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
            bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, k, iterations=1)

        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        x1, y1, x2, y2 = 10**9, 10**9, -1, -1
        area_sum = 0.0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x + w)
            y2 = max(y2, y + h)
            area_sum += float(w * h)

        if x2 <= x1 or y2 <= y1:
            return None

        Hs, Ws = bw.shape[:2]
        frame_area = float(max(Ws * Hs, 1))
        if (area_sum / frame_area) < self.min_area_ratio:
            return None

        scale_x = float(W0) / float(Ws)
        scale_y = float(H0) / float(Hs)

        fx1 = int(round(float(x1) * scale_x))
        fy1 = int(round(float(y1) * scale_y))
        fx2 = int(round(float(x2) * scale_x))
        fy2 = int(round(float(y2) * scale_y))

        fx1 = max(0, min(W0 - 1, fx1))
        fy1 = max(0, min(H0 - 1, fy1))
        fx2 = max(0, min(W0, fx2))
        fy2 = max(0, min(H0, fy2))

        if fx2 <= fx1 or fy2 <= fy1:
            return None

        return fx1, fy1, fx2, fy2


def _yolo_probe_on_crop(
    yolo,
    frame_bgr: np.ndarray,
    y,
    t,
    f,
    roi: Optional[Tuple[int, int, int, int]],
    crop_imgsz: int,
    crop_conf: float,
    crop_iou: float,
    topk: int,
    min_area_ratio: float,
) -> int:
    if roi is None:
        return -1

    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return -1

    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return -1

    y2cfg = type(y)(
        weights=y.weights,
        imgsz=int(crop_imgsz),
        conf=float(crop_conf),
        iou=float(crop_iou),
        classes=list(y.classes),
        device=y.device,
        half=bool(y.half),
    )

    r = yolo_infer(yolo, crop_bgr, y2cfg, t)
    boxes_xyxy, confs, _ = extract_boxes(r, False)

    hC, wC = crop_bgr.shape[:2]
    boxes_xyxy, confs, _ = apply_min_area_ratio(boxes_xyxy, confs, wC, hC, float(min_area_ratio))
    if topk and topk > 0 and len(boxes_xyxy) > 0:
        boxes_xyxy, confs = topk_by_area(boxes_xyxy, confs, int(topk))

    return int(len(boxes_xyxy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str)
    ap.add_argument("-c", "--motion-config", type=str, required=True)
    ap.add_argument("--yolo-config", type=str, required=True)
    args = ap.parse_args()

    log = setup_logger("stage2_export")

    mcfg: MotionConfig = load_config(args.motion_config)
    ycfg = load_yaml(args.yolo_config)
    y, t, f, s, e, inter, stab, p, m = parse_cfg(mcfg, ycfg)

    out_root = Path(e.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ms = compute_motion_scores(args.video, mcfg, resize_keep_aspect, to_gray, blur)
    scores = ms["scores"]
    fps_est = ms["fps"] if ms["fps"] > 0 else 30.0
    fps_out = fps_est if e.fps_out <= 0 else e.fps_out

    skip = int(s.skip_first_frames)
    scores_for_seg = scores
    if skip > 0 and len(scores) > skip:
        scores_for_seg = [0.0] * skip + scores[skip:]

    segs = detect_motion_segments(
        scores_for_seg,
        fps_est,
        s.thr_on,
        s.thr_off,
        s.min_len_sec,
        s.merge_gap_sec,
        s.smooth,
        s.ema_alpha,
        s.ma_win,
    )

    log.info(f"Motion pass finished: frames={len(scores_for_seg)} approx_fps={fps_est:.2f}")
    log.info(f"Segments found: {len(segs)}")
    for i, seg in enumerate(segs, 1):
        t0 = float(seg.start_f) / float(fps_est)
        t1 = float(seg.end_f) / float(fps_est)
        log.info(f"  seg#{i}: frames={seg.start_f}-{seg.end_f} len={seg.length} peak={seg.peak:.6f} time={t0:.2f}-{t1:.2f}s")

    if not segs:
        log.info("No segments. Exiting.")
        return

    video_path = Path(args.video)
    vid_out = out_root / video_path.stem
    vid_out.mkdir(parents=True, exist_ok=True)

    seg_mask = build_segment_mask(len(scores_for_seg), segs)
    seg_id_map = np.full((len(seg_mask),), -1, dtype=np.int32)
    for si, seg in enumerate(segs):
        seg_id_map[int(seg.start_f) : int(seg.end_f) + 1] = si

    yolo = yolo_model(y.weights)

    seg_infos: List[SegInfo] = []
    accepted_spans: List[Tuple[int, int]] = []

    seg_probe_plan: Dict[int, List[int]] = {}
    for si, seg in enumerate(segs):
        if not p.enabled:
            seg_probe_plan[si] = []
            continue
        idxs = uniform_probe_indices(int(seg.start_f), int(seg.end_f), int(p.frames))
        if int(p.stride) > 1:
            idxs = [x for k, x in enumerate(idxs) if (k % int(p.stride)) == 0]
        seg_probe_plan[si] = idxs

    seg_probe_seen: Dict[int, int] = {si: 0 for si in range(len(segs))}
    seg_probe_2p: Dict[int, int] = {si: 0 for si in range(len(segs))}
    seg_probe_done: Dict[int, bool] = {si: (not p.enabled) for si in range(len(segs))}
    seg_accepted: Dict[int, int] = {si: 1 if not p.enabled else 0 for si in range(len(segs))}

    probe_targets_inv: Dict[int, List[int]] = {}
    for si, idxs in seg_probe_plan.items():
        for fi in idxs:
            probe_targets_inv.setdefault(int(fi), []).append(int(si))

    probe_cfg = ycfg.get("probe", {}) if isinstance(ycfg.get("probe", {}), dict) else {}
    crop_enabled = bool(probe_cfg.get("crop_enabled", True))
    crop_imgsz = int(probe_cfg.get("crop_imgsz", 512))
    crop_conf = float(probe_cfg.get("crop_conf", float(y.conf)))
    crop_iou = float(probe_cfg.get("crop_iou", float(y.iou)))
    crop_margin = float(probe_cfg.get("crop_margin", 0.30))
    fallback_fullframe = bool(probe_cfg.get("fallback_fullframe", True))

    motion_roi_cfg = probe_cfg.get("motion_roi", {}) if isinstance(probe_cfg.get("motion_roi", {}), dict) else {}
    mroi_enabled = bool(motion_roi_cfg.get("enabled", True))
    mroi = MotionRoi(
        resize_w=int(motion_roi_cfg.get("resize_w", 320)),
        blur_ksize=int(motion_roi_cfg.get("blur_ksize", 5)),
        diff_thr=int(motion_roi_cfg.get("diff_thr", 18)),
        morph_ksize=int(motion_roi_cfg.get("morph_ksize", 5)),
        min_area_ratio=float(motion_roi_cfg.get("min_area_ratio", 0.004)),
    )

    last_ts_ms = 0
    proc_i = -1

    for ts, frame in frame_generator(args.video):
        now_ms = int(ts * 1000)
        if int(mcfg.min_interval_ms) > 0 and (now_ms - last_ts_ms) < int(mcfg.min_interval_ms):
            continue
        last_ts_ms = now_ms

        proc_i += 1
        if proc_i >= len(seg_mask):
            break

        if proc_i not in probe_targets_inv:
            if mroi_enabled:
                _ = mroi.get_roi(frame)
            continue

        roi_motion = mroi.get_roi(frame) if mroi_enabled else None
        H, W = frame.shape[:2]
        if roi_motion is not None:
            roi_motion = _expand_xyxy(roi_motion[0], roi_motion[1], roi_motion[2], roi_motion[3], W, H, crop_margin)

        det_count_crop = -1
        if crop_enabled and roi_motion is not None:
            det_count_crop = _yolo_probe_on_crop(
                yolo=yolo,
                frame_bgr=frame,
                y=y,
                t=t,
                f=f,
                roi=roi_motion,
                crop_imgsz=crop_imgsz,
                crop_conf=crop_conf,
                crop_iou=crop_iou,
                topk=int(f.topk_by_area),
                min_area_ratio=float(f.min_box_area_ratio),
            )

        det_count_full = -1
        if (det_count_crop < 0) and fallback_fullframe:
            r = yolo_infer(yolo, frame, y, t)
            boxes_xyxy, confs, _ = extract_boxes(r, False)
            det_count_full = int(len(boxes_xyxy))

        det_count_for_gate = det_count_crop if det_count_crop >= 0 else det_count_full

        for si in probe_targets_inv[int(proc_i)]:
            if seg_probe_done.get(si, False):
                continue

            seg_probe_seen[si] += 1

            two_p = 1 if int(det_count_for_gate) >= int(f.min_persons) else 0
            seg_probe_2p[si] += int(two_p)

            if seg_probe_seen[si] >= int(max(1, p.frames)):
                if seg_probe_2p[si] >= int(p.need_2p):
                    seg_accepted[si] = 1
                else:
                    seg_accepted[si] = 0
                seg_probe_done[si] = True

    for si, seg in enumerate(segs):
        acc = int(seg_accepted.get(si, 0))
        seen = int(seg_probe_seen.get(si, 0))
        twop = int(seg_probe_2p.get(si, 0))
        seg_infos.append(
            SegInfo(
                start_f=int(seg.start_f),
                end_f=int(seg.end_f),
                peak=float(seg.peak),
                accepted=acc,
                probe_seen=seen,
                probe_2p=twop,
            )
        )
        if acc == 1:
            accepted_spans.append((int(seg.start_f), int(seg.end_f)))

    if not accepted_spans:
        manifest_path = vid_out / "manifest.csv"
        mf = open(manifest_path, "w", newline="", encoding="utf-8")
        mw = csv.writer(mf)
        mw.writerow(["seg_id", "start_f", "end_f", "t0", "t1", "fps_est", "accepted", "probe_seen", "probe_2p"])
        for i, info in enumerate(seg_infos, 1):
            t0 = float(info.start_f) / float(fps_est)
            t1 = float(info.end_f) / float(fps_est)
            mw.writerow([i, info.start_f, info.end_f, f"{t0:.3f}", f"{t1:.3f}", f"{fps_est:.3f}", info.accepted, info.probe_seen, info.probe_2p])
        mf.close()
        meta = {
            "video": str(args.video),
            "motion_config": str(args.motion_config),
            "yolo_config": str(args.yolo_config),
            "fps_est": float(fps_est),
            "segments": [
                {
                    "seg_id": i + 1,
                    "start_f": info.start_f,
                    "end_f": info.end_f,
                    "peak": info.peak,
                    "accepted": info.accepted,
                    "probe_seen": info.probe_seen,
                    "probe_2p": info.probe_2p,
                }
                for i, info in enumerate(seg_infos)
            ],
            "merged_events": [],
        }
        write_json(vid_out / "meta.json", meta)
        log.info("All segments dropped by probe. Exiting.")
        return

    if m.enabled:
        merged_spans = merge_segments_by_gap(
            segs=accepted_spans,
            fps=fps_est,
            gap_sec=m.gap_sec,
            pad_pre_sec=m.pad_pre_sec,
            pad_post_sec=m.pad_post_sec,
            max_merge_len_sec=m.max_merge_len_sec,
            n_total_frames=len(scores_for_seg),
        )
    else:
        merged_spans = sorted(accepted_spans, key=lambda x: x[0])

    log.info(f"Accepted segments: {len(accepted_spans)} -> merged events: {len(merged_spans)}")

    manifest_path = vid_out / "manifest.csv"
    mf = open(manifest_path, "w", newline="", encoding="utf-8")
    mw = csv.writer(mf)
    mw.writerow(
        [
            "event_id",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "fps_est",
            "frames_written",
            "crop_used_ratio",
            "crop_out_w",
            "crop_out_h",
        ]
    )

    stabilizer = RoiStabilizer(stab)
    active_last_seen: Dict[int, int] = {}
    track_last_box: Dict[int, np.ndarray] = {}
    track_last_seen: Dict[int, int] = {}
    last_roi_i = -10_000

    last_ts_ms = 0
    proc_i = -1

    cur_ev = -1
    ev_start = 0
    ev_end = 0

    ev_dir: Optional[Path] = None
    frames_dir: Optional[Path] = None
    full_writer: Optional[cv2.VideoWriter] = None
    crop_writer: Optional[cv2.VideoWriter] = None
    roi_log = None
    roi_w = None

    frames_written = 0
    crop_used = 0
    ev_frames_seen = 0

    def close_event():
        nonlocal full_writer, crop_writer, roi_log, roi_w, frames_written, crop_used, ev_frames_seen, cur_ev, ev_dir
        if cur_ev < 0:
            return

        if full_writer is not None:
            full_writer.release()
            full_writer = None

        if crop_writer is not None:
            crop_writer.release()
            crop_writer = None

        if roi_log is not None:
            roi_log.close()
            roi_log = None
            roi_w = None

        ratio = (crop_used / frames_written) if frames_written > 0 else 0.0
        t0 = float(ev_start) / float(fps_est)
        t1 = float(ev_end) / float(fps_est)
        mw.writerow(
            [
                cur_ev + 1,
                ev_start,
                ev_end,
                f"{t0:.3f}",
                f"{t1:.3f}",
                f"{fps_est:.3f}",
                frames_written,
                f"{ratio:.3f}",
                int(e.crop_out_w),
                int(e.crop_out_h),
            ]
        )
        log.info(f"EXPORT event#{cur_ev+1} frames={frames_written} crop_used_ratio={ratio:.2f}")

        frames_written = 0
        crop_used = 0
        ev_frames_seen = 0
        ev_dir = None

    def open_event(ei: int):
        nonlocal cur_ev, ev_dir, frames_dir, roi_log, roi_w, full_writer, crop_writer
        nonlocal ev_start, ev_end, active_last_seen, last_roi_i, ev_frames_seen, frames_written, crop_used
        nonlocal track_last_box, track_last_seen

        if cur_ev >= 0:
            close_event()

        cur_ev = ei
        ev_start, ev_end = merged_spans[ei]

        if e.max_event_sec > 0:
            max_len = int(round(float(e.max_event_sec) * float(fps_est)))
            if (ev_end - ev_start + 1) > max_len:
                ev_end = ev_start + max_len - 1

        ev_dir = vid_out / f"event_{cur_ev+1:03d}"
        ev_dir.mkdir(parents=True, exist_ok=True)

        if e.save_frames:
            frames_dir = ev_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
        else:
            frames_dir = None

        roi_log = open(ev_dir / "roi_log.csv", "w", newline="", encoding="utf-8")
        roi_w = csv.writer(roi_log)
        roi_w.writerow(
            [
                "proc_i",
                "ts",
                "det_count",
                "track_count",
                "roi_x1",
                "roi_y1",
                "roi_x2",
                "roi_y2",
                "roi_source",
                "roi_score",
                "roi_iou_prev",
                "pair_idx",
                "jump_accepted",
            ]
        )

        full_writer = None
        crop_writer = None
        stabilizer.reset()
        last_roi_i = -10_000
        active_last_seen.clear()
        track_last_box.clear()
        track_last_seen.clear()
        ev_frames_seen = 0
        frames_written = 0
        crop_used = 0

    merged_id_map = np.full((len(scores_for_seg),), -1, dtype=np.int32)
    for ei, (a, b) in enumerate(merged_spans):
        merged_id_map[int(a) : int(b) + 1] = int(ei)

    for ts, frame in frame_generator(args.video):
        now_ms = int(ts * 1000)
        if int(mcfg.min_interval_ms) > 0 and (now_ms - last_ts_ms) < int(mcfg.min_interval_ms):
            continue
        last_ts_ms = now_ms

        proc_i += 1
        if proc_i >= len(merged_id_map):
            break

        ei = int(merged_id_map[int(proc_i)])
        if ei < 0:
            if cur_ev >= 0:
                close_event()
                cur_ev = -1
            continue

        if ei != cur_ev:
            open_event(ei)

        if proc_i < ev_start or proc_i > ev_end:
            continue

        ev_frames_seen += 1

        if e.sample_stride > 1 and (ev_frames_seen - 1) % int(e.sample_stride) != 0:
            continue

        H, W = frame.shape[:2]
        if ev_dir is None:
            continue

        if e.save_fullframe_mp4 and full_writer is None:
            full_writer = make_writer(ev_dir / "full.avi", fps_out, (W, H), fourcc=e.video_fourcc)

        if e.save_crop_mp4 and crop_writer is None:
            crop_writer = make_writer(
                ev_dir / "crop.avi",
                fps_out,
                (int(e.crop_out_w), int(e.crop_out_h)),
                fourcc=e.video_fourcc,
            )

        r = yolo_infer(yolo, frame, y, t)
        boxes_xyxy, confs, ids_now = extract_boxes(r, t.enabled)

        boxes_xyxy, confs, keep = apply_min_area_ratio(boxes_xyxy, confs, W, H, f.min_box_area_ratio)
        if t.enabled and len(ids_now) > 0:
            keep_idx = np.where(keep)[0].tolist()
            ids_now = [ids_now[i] for i in keep_idx] if len(keep_idx) > 0 else []

        if f.topk_by_area and len(boxes_xyxy) > 0:
            boxes_xyxy, confs = topk_by_area(boxes_xyxy, confs, f.topk_by_area)

        det_count = int(len(boxes_xyxy))

        track_count = det_count
        if t.enabled:
            track_count = track_ttl_update(active_last_seen, ids_now, proc_i, t.max_lost_frames)
            if len(ids_now) == boxes_xyxy.shape[0] and boxes_xyxy.shape[0] > 0:
                for k, tid in enumerate(ids_now):
                    tid = int(tid)
                    track_last_box[tid] = boxes_xyxy[k].copy()
                    track_last_seen[tid] = int(proc_i)

        roi_source_override: Optional[str] = None
        if t.enabled and t.gate_use_tracks and det_count == 0 and track_count >= 2:
            alive_ids = list(active_last_seen.keys())
            alive_ids = sorted(alive_ids, key=lambda x: track_last_seen.get(int(x), -1), reverse=True)

            cand = []
            for tid in alive_ids:
                b = track_last_box.get(int(tid), None)
                if b is not None:
                    cand.append(b)
                if len(cand) >= 2:
                    break

            if len(cand) >= 2:
                boxes_xyxy = np.stack(cand, axis=0).astype(np.float32)
                confs = np.ones((boxes_xyxy.shape[0],), dtype=np.float32) * 0.5
                det_count = int(len(boxes_xyxy))
                roi_source_override = "track_fallback"

        roi_raw: Optional[Tuple[int, int, int, int]] = None
        roi_source = "none"
        roi_score = 0.0
        pair_idx: Optional[Tuple[int, int]] = None

        if det_count >= 1:
            roi_raw, roi_source, roi_score, pair_idx = select_fight_roi(
                boxes_xyxy=boxes_xyxy,
                confs=confs,
                W=W,
                H=H,
                margin=e.crop_margin,
                inter=inter,
            )
            if roi_source_override is not None and roi_source != "none":
                roi_source = roi_source_override

        gate_count = det_count
        if t.enabled and t.gate_use_tracks:
            gate_count = max(det_count, track_count)

        roi_gate_ok = gate_count >= int(f.min_persons)
        if not roi_gate_ok:
            if stabilizer.prev_roi is not None and (proc_i - last_roi_i) <= int(t.max_lost_frames):
                roi_raw = stabilizer.prev_roi
                roi_source = "hold"
            else:
                roi_raw = None
                roi_source = "none"

        roi_out, roi_iou_prev, jump_accepted = stabilizer.update(roi_raw)

        if roi_out is not None:
            last_roi_i = proc_i

        if e.save_fullframe_mp4 and full_writer is not None:
            full_writer.write(frame)

        if e.save_crop_mp4 and roi_out is not None and crop_writer is not None:
            crop_img = crop(frame, roi_out)
            crop_img = cv2.resize(
                crop_img,
                (int(e.crop_out_w), int(e.crop_out_h)),
                interpolation=cv2.INTER_LINEAR,
            )
            crop_writer.write(crop_img)
            crop_used += 1

        if e.save_frames and frames_dir is not None:
            cv2.imwrite(str(frames_dir / f"f_{proc_i:06d}_{int(ts*1000)}ms.jpg"), frame)

        frames_written += 1

        if roi_w is not None:
            if roi_out is None:
                roi_w.writerow(
                    [
                        proc_i,
                        f"{ts:.6f}",
                        det_count,
                        track_count,
                        "",
                        "",
                        "",
                        "",
                        roi_source,
                        f"{roi_score:.4f}",
                        f"{roi_iou_prev:.4f}",
                        str(pair_idx) if pair_idx else "",
                        int(jump_accepted),
                    ]
                )
            else:
                x1, y1, x2, y2 = roi_out
                roi_w.writerow(
                    [
                        proc_i,
                        f"{ts:.6f}",
                        det_count,
                        track_count,
                        x1,
                        y1,
                        x2,
                        y2,
                        roi_source,
                        f"{roi_score:.4f}",
                        f"{roi_iou_prev:.4f}",
                        str(pair_idx) if pair_idx else "",
                        int(jump_accepted),
                    ]
                )

    if cur_ev >= 0:
        close_event()

    mf.close()

    meta = {
        "video": str(args.video),
        "motion_config": str(args.motion_config),
        "yolo_config": str(args.yolo_config),
        "fps_est": float(fps_est),
        "segments": [
            {
                "seg_id": i + 1,
                "start_f": info.start_f,
                "end_f": info.end_f,
                "peak": info.peak,
                "accepted": info.accepted,
                "probe_seen": info.probe_seen,
                "probe_2p": info.probe_2p,
            }
            for i, info in enumerate(seg_infos)
        ],
        "merged_events": [
            {
                "event_id": i + 1,
                "start_f": int(a),
                "end_f": int(b),
                "t0": float(a) / float(fps_est),
                "t1": float(b) / float(fps_est),
            }
            for i, (a, b) in enumerate(merged_spans)
        ],
    }
    write_json(vid_out / "meta.json", meta)
    log.info(f"EXPORT DONE -> {vid_out} (manifest={manifest_path})")


if __name__ == "__main__":
    main()