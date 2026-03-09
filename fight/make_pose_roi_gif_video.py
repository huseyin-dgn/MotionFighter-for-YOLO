from __future__ import annotations

from pathlib import Path
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


INPUT_VIDEO = r"C:\Users\hdgn5\OneDrive\Masaüstü\Fight-Detection\fight\sample_2.mp4"
OUTPUT_MP4 = r"C:\Users\hdgn5\OneDrive\Masaüstü\Fight-Detection\fight\clip_debug\sample_2_annotated.mp4"

PERSON_WEIGHTS = r"C:\Users\hdgn5\OneDrive\Masaüstü\Fight-Detection\fight\yolo11n.pt"
POSE_WEIGHTS = r"C:\Users\hdgn5\OneDrive\Masaüstü\Fight-Detection\fight\pose\weights\yolo11n-pose.pt"

PERSON_IMGSZ = 416
PERSON_CONF = 0.25

POSE_IMGSZ = 320
POSE_CONF = 0.25
POSE_MIN_KPT_CONF = 0.30
POSE_MIN_VALID_KPTS = 6

POSE_WINDOW = 6
POSE_NEED_POSITIVE = 2

PAD_RATIO = 0.30
ROI_SMOOTH_ALPHA = 0.30


COCO_KPTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

person_model = YOLO(PERSON_WEIGHTS)
pose_model = YOLO(POSE_WEIGHTS)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smooth_box(prev_box, new_box, alpha=0.30):
    if prev_box is None:
        return new_box
    if new_box is None:
        return prev_box

    px1, py1, px2, py2 = prev_box
    nx1, ny1, nx2, ny2 = new_box

    sx1 = int((1.0 - alpha) * px1 + alpha * nx1)
    sy1 = int((1.0 - alpha) * py1 + alpha * ny1)
    sx2 = int((1.0 - alpha) * px2 + alpha * nx2)
    sy2 = int((1.0 - alpha) * py2 + alpha * ny2)
    return (sx1, sy1, sx2, sy2)


def sanitize_box(box, shape):
    if box is None:
        return None
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1 = clamp(int(x1), 0, w - 1)
    y1 = clamp(int(y1), 0, h - 1)
    x2 = clamp(int(x2), 0, w - 1)
    y2 = clamp(int(y2), 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def build_union_box(persons, frame_shape, pad_ratio=0.30):
    if len(persons) < 2:
        return None

    h, w = frame_shape[:2]
    boxes = [box for _, box in persons[:2]]

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = clamp(x1 - pad_x, 0, w - 1)
    y1 = clamp(y1 - pad_y, 0, h - 1)
    x2 = clamp(x2 + pad_x, 0, w - 1)
    y2 = clamp(y2 + pad_y, 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def crop_from_box(frame, box, out_size=320):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def detect_persons(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = person_model.predict(
        source=rgb,
        imgsz=PERSON_IMGSZ,
        conf=PERSON_CONF,
        verbose=False,
    )[0]

    out = []
    if res.boxes is None:
        return out

    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    confs = res.boxes.conf.detach().cpu().numpy()
    clss = res.boxes.cls.detach().cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
        if cls != 0:
            continue
        out.append((float(c), (int(x1), int(y1), int(x2), int(y2))))

    out.sort(key=lambda x: x[0], reverse=True)
    return out


def get_xy(kpts_xy, kpts_conf, idx, min_conf):
    if idx < 0 or idx >= len(kpts_xy):
        return None
    if float(kpts_conf[idx]) < float(min_conf):
        return None
    return np.asarray(kpts_xy[idx], dtype=np.float32)


def count_valid_kpts(kpts_conf, min_conf):
    return int(np.sum(np.asarray(kpts_conf) >= float(min_conf)))


def person_center_from_shoulders_hips(kpts_xy, kpts_conf, min_conf):
    ids = [5, 6, 11, 12]
    pts = []
    for idx in ids:
        p = get_xy(kpts_xy, kpts_conf, idx, min_conf)
        if p is not None:
            pts.append(p)
    if not pts:
        return None
    return np.stack(pts, axis=0).mean(axis=0)


def safe_norm_dist(p1, p2, norm):
    if p1 is None or p2 is None or norm <= 1e-6:
        return 1e9
    return float(np.linalg.norm(p1 - p2) / norm)


def compute_pose_score(a_xy, a_conf, b_xy, b_conf, roi_w, roi_h):
    norm = float(max(roi_w, roi_h, 1))

    center_a = person_center_from_shoulders_hips(a_xy, a_conf, POSE_MIN_KPT_CONF)
    center_b = person_center_from_shoulders_hips(b_xy, b_conf, POSE_MIN_KPT_CONF)
    center_dist = safe_norm_dist(center_a, center_b, norm)

    lw_a = get_xy(a_xy, a_conf, COCO_KPTS["left_wrist"], POSE_MIN_KPT_CONF)
    rw_a = get_xy(a_xy, a_conf, COCO_KPTS["right_wrist"], POSE_MIN_KPT_CONF)
    lw_b = get_xy(b_xy, b_conf, COCO_KPTS["left_wrist"], POSE_MIN_KPT_CONF)
    rw_b = get_xy(b_xy, b_conf, COCO_KPTS["right_wrist"], POSE_MIN_KPT_CONF)

    wrist_dists = []
    for pa in [lw_a, rw_a]:
        for pb in [lw_b, rw_b]:
            if pa is not None and pb is not None:
                wrist_dists.append(safe_norm_dist(pa, pb, norm))
    wrist_dist = min(wrist_dists) if wrist_dists else 1e9

    shoulders_a, shoulders_b = [], []
    for idx in [COCO_KPTS["left_shoulder"], COCO_KPTS["right_shoulder"]]:
        pa = get_xy(a_xy, a_conf, idx, POSE_MIN_KPT_CONF)
        pb = get_xy(b_xy, b_conf, idx, POSE_MIN_KPT_CONF)
        if pa is not None:
            shoulders_a.append(pa)
        if pb is not None:
            shoulders_b.append(pb)

    upper_body_dist = 1e9
    if shoulders_a and shoulders_b:
        sa = np.stack(shoulders_a, axis=0).mean(axis=0)
        sb = np.stack(shoulders_b, axis=0).mean(axis=0)
        upper_body_dist = safe_norm_dist(sa, sb, norm)

    s_center = 0.0 if center_dist >= 1e9 else max(0.0, 1.0 - center_dist / 0.35)
    s_wrist = 0.0 if wrist_dist >= 1e9 else max(0.0, 1.0 - wrist_dist / 0.22)
    s_upper = 0.0 if upper_body_dist >= 1e9 else max(0.0, 1.0 - upper_body_dist / 0.28)

    score = 0.40 * s_center + 0.35 * s_wrist + 0.25 * s_upper
    return float(max(0.0, min(1.0, score))), center_dist, wrist_dist, upper_body_dist


def draw_pose(roi_vis, kp_xy, kp_conf, color):
    for a, b in SKELETON:
        pa = get_xy(kp_xy, kp_conf, a, POSE_MIN_KPT_CONF)
        pb = get_xy(kp_xy, kp_conf, b, POSE_MIN_KPT_CONF)
        if pa is not None and pb is not None:
            cv2.line(
                roi_vis,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                color,
                2,
                cv2.LINE_AA,
            )

    for i in range(len(kp_xy)):
        p = get_xy(kp_xy, kp_conf, i, POSE_MIN_KPT_CONF)
        if p is not None:
            cv2.circle(roi_vis, (int(p[0]), int(p[1])), 3, color, -1, cv2.LINE_AA)


def run_pose_on_roi(roi_bgr, hist_positive):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    res = pose_model.predict(
        source=roi_rgb,
        imgsz=POSE_IMGSZ,
        conf=POSE_CONF,
        verbose=False,
    )[0]

    roi_vis = roi_bgr.copy()

    if res.boxes is None or len(res.boxes) == 0:
        return roi_vis, 0.0, False, 0, 0, 1e9, 1e9, 1e9

    if res.keypoints is None:
        return roi_vis, 0.0, False, 0, 0, 1e9, 1e9, 1e9

    boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
    if len(boxes_xyxy) < 2:
        return roi_vis, 0.0, False, 0, 0, 1e9, 1e9, 1e9

    areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_xyxy], dtype=np.float32)
    idx2 = np.argsort(-areas)[:2].tolist()

    kp_xy = res.keypoints.xy.detach().cpu().numpy()
    kp_conf = res.keypoints.conf.detach().cpu().numpy()

    i0, i1 = idx2[0], idx2[1]
    a_xy, b_xy = kp_xy[i0], kp_xy[i1]
    a_conf, b_conf = kp_conf[i0], kp_conf[i1]

    valid_a = count_valid_kpts(a_conf, POSE_MIN_KPT_CONF)
    valid_b = count_valid_kpts(b_conf, POSE_MIN_KPT_CONF)

    draw_pose(roi_vis, a_xy, a_conf, (0, 255, 0))
    draw_pose(roi_vis, b_xy, b_conf, (0, 165, 255))

    for idx in idx2:
        x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
        cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

    if valid_a < POSE_MIN_VALID_KPTS or valid_b < POSE_MIN_VALID_KPTS:
        return roi_vis, 0.0, False, valid_a, valid_b, 1e9, 1e9, 1e9

    h, w = roi_vis.shape[:2]
    score, cd, wd, ud = compute_pose_score(a_xy, a_conf, b_xy, b_conf, w, h)

    return roi_vis, score, score >= 0.45, valid_a, valid_b, cd, wd, ud


def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 16.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(OUTPUT_MP4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    last_union_box = None
    pose_hist = deque(maxlen=POSE_WINDOW)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vis = frame.copy()

            persons = detect_persons(frame)

            for c, (x1, y1, x2, y2) in persons:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    vis,
                    f"person {c:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            current_union_box = build_union_box(persons, frame.shape, pad_ratio=PAD_RATIO)
            current_union_box = sanitize_box(current_union_box, frame.shape)

            if current_union_box is not None:
                last_union_box = smooth_box(last_union_box, current_union_box, alpha=ROI_SMOOTH_ALPHA)
                last_union_box = sanitize_box(last_union_box, frame.shape)
            else:
                last_union_box = None
                pose_hist.clear()

            roi_vis_small = None
            pose_score = 0.0
            pose_ok = False
            valid_a = 0
            valid_b = 0
            cd = wd = ud = 1e9

            if last_union_box is not None:
                ux1, uy1, ux2, uy2 = last_union_box
                cv2.rectangle(vis, (ux1, uy1), (ux2, uy2), (0, 255, 255), 2)
                cv2.putText(
                    vis,
                    "ROI",
                    (ux1, max(20, uy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                roi = crop_from_box(frame, last_union_box, out_size=320)
                if roi is not None:
                    roi_vis_small, pose_score, pose_positive, valid_a, valid_b, cd, wd, ud = run_pose_on_roi(
                        roi, int(sum(pose_hist))
                    )
                    pose_hist.append(bool(pose_positive))
                    pose_ok = int(sum(pose_hist)) >= POSE_NEED_POSITIVE

            overlay_lines = [
                f"persons={len(persons)}",
                f"pose_score={pose_score:.3f}",
                f"pose_ok={int(pose_ok)} hist={int(sum(pose_hist))}/{POSE_WINDOW}",
                f"valid_a={valid_a} valid_b={valid_b}",
            ]

            if cd < 1e8:
                overlay_lines.append(f"cd={cd:.2f} wd={wd:.2f} ud={ud:.2f}")

            y = 28
            for line in overlay_lines:
                cv2.putText(
                    vis,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                y += 26

            if roi_vis_small is not None:
                inset_h = 220
                inset_w = 220
                inset = cv2.resize(roi_vis_small, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)

                x0 = w - inset_w - 20
                y0 = 20

                cv2.rectangle(vis, (x0 - 2, y0 - 2), (x0 + inset_w + 2, y0 + inset_h + 2), (255, 255, 255), 2)
                vis[y0:y0 + inset_h, x0:x0 + inset_w] = inset

                cv2.putText(
                    vis,
                    "POSE ROI",
                    (x0, y0 + inset_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(vis)

    finally:
        cap.release()
        writer.release()

    print(f"saved: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()