from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import cv2

THIS = Path(__file__).resolve()
FIGHT_ROOT = THIS.parents[3]
MOTION_ROOT = FIGHT_ROOT / "motion"
sys.path.insert(0, str(MOTION_ROOT))
sys.path.insert(0, str(FIGHT_ROOT))

from src.core.config import load_config, MotionConfig
from src.utils.image_ops import resize_keep_aspect, to_gray, blur
from src.utils.logger import setup_logger

from yolo.src.stage2.stage2_core import (
    load_yaml,
    parse_cfg,
    compute_motion_scores,
    detect_motion_segments,
    build_segment_mask,
    yolo_model,
    yolo_infer,
    extract_boxes,
    apply_min_area_ratio,
    topk_by_area,
    track_ttl_update,
    draw_boxes,
    select_fight_roi,
    crop,
)

from src.ingest.cam_reader import frame_generator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str)
    ap.add_argument("-c", "--motion-config", type=str, required=True)
    ap.add_argument("--yolo-config", type=str, required=True)
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument("--csv-out", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    log = setup_logger("stage2_yolo_debug")

    mcfg: MotionConfig = load_config(args.motion_config)
    ycfg = load_yaml(args.yolo_config)
    y, t, f, s, e, inter, stab = parse_cfg(mcfg, ycfg)

    out_dir = Path(args.out_dir) if args.out_dir else Path(str(e.out_dir)).parent / "yolo_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_out) if args.csv_out else out_dir / "yolo_event_log.csv"

    ms = compute_motion_scores(args.video, mcfg, resize_keep_aspect, to_gray, blur)
    scores = ms["scores"]
    fps = ms["fps"] if ms["fps"] > 0 else 30.0

    if s.skip_first_frames > 0 and len(scores) > s.skip_first_frames:
        scores_for_seg = [0.0] * s.skip_first_frames + scores[s.skip_first_frames:]
    else:
        scores_for_seg = scores

    segs = detect_motion_segments(scores_for_seg, fps, s.thr_on, s.thr_off, s.min_len_sec, s.merge_gap_sec, s.smooth, s.ema_alpha, s.ma_win)
    log.info(f"Motion pass finished: frames={len(scores_for_seg)} approx_fps={fps:.2f}")
    log.info(f"Segments found: {len(segs)}")
    for i, seg in enumerate(segs, 1):
        t0, t1 = seg.start_f / fps, seg.end_f / fps
        log.info(f"  seg#{i}: frames={seg.start_f}-{seg.end_f} len={seg.length} peak={seg.peak:.6f} time={t0:.2f}-{t1:.2f}s")

    if not segs:
        log.info("No segments. Exiting.")
        return

    seg_mask = build_segment_mask(len(scores_for_seg), segs)
    yolo = yolo_model(y.weights)

    last_ts_ms = 0
    proc_i = -1
    active_last_seen = {}
    accepted = 0
    rejected_person = 0
    in_segments = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["proc_i", "ts", "in_segment", "det_count", "track_count", "accepted", "roi_source", "roi_score"])
        for ts, frame in frame_generator(args.video):
            now_ms = int(ts * 1000)
            if mcfg.min_interval_ms > 0 and (now_ms - last_ts_ms) < mcfg.min_interval_ms:
                continue
            last_ts_ms = now_ms
            proc_i += 1
            if proc_i >= len(seg_mask):
                break
            if not bool(seg_mask[proc_i]):
                continue
            in_segments += 1

            r = yolo_infer(yolo, frame, y, t)
            boxes_xyxy, confs, ids_now = extract_boxes(r, t.enabled)

            H, W = frame.shape[:2]
            boxes_xyxy, confs, keep = apply_min_area_ratio(boxes_xyxy, confs, W, H, f.min_box_area_ratio)
            if t.enabled and len(ids_now) > 0:
                keep_idx = np.where(keep)[0].tolist()
                ids_now = [ids_now[i] for i in keep_idx] if len(keep_idx) > 0 else []

            if f.topk_by_area and len(boxes_xyxy) > 0:
                boxes_xyxy, confs = topk_by_area(boxes_xyxy, confs, f.topk_by_area)

            det_count = int(len(boxes_xyxy))
            track_count = track_ttl_update(active_last_seen, ids_now, proc_i, t.max_lost_frames) if t.enabled else det_count

            ok = 1 if det_count >= f.min_persons else 0
            roi_source = "none"
            roi_score = 0.0

            if det_count >= 1:
                roi, roi_source, roi_score, _ = select_fight_roi(
                    boxes_xyxy=boxes_xyxy,
                    confs=confs,
                    W=W,
                    H=H,
                    margin=e.crop_margin,
                    inter=inter,
                )
                if args.save_vis:
                    vis = draw_boxes(frame, boxes_xyxy, confs)
                    if roi is not None:
                        x1, y1, x2, y2 = roi
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(vis, f"{roi_source} score={roi_score:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imwrite(str(out_dir / f"seg_{proc_i:06d}_{int(ts*1000)}ms_det{det_count}_trk{track_count}.jpg"), vis)

            if ok:
                accepted += 1
            else:
                rejected_person += 1

            w.writerow([proc_i, f"{ts:.6f}", 1, det_count, track_count, ok, roi_source, f"{roi_score:.4f}"])

    log.info(f"DONE in_segments={in_segments} accepted={accepted} rejected_person<{f.min_persons}={rejected_person} csv={csv_path}")


if __name__ == "__main__":
    main()