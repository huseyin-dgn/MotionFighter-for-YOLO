from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


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


def safe_norm_dist(p1: Optional[np.ndarray], p2: Optional[np.ndarray], norm: float) -> float:
    if p1 is None or p2 is None or norm <= 1e-6:
        return 1e9
    return float(np.linalg.norm(p1 - p2) / norm)


def get_xy(kpts_xy: np.ndarray, kpts_conf: np.ndarray, idx: int, min_conf: float) -> Optional[np.ndarray]:
    if idx < 0 or idx >= len(kpts_xy):
        return None
    if float(kpts_conf[idx]) < float(min_conf):
        return None
    return np.asarray(kpts_xy[idx], dtype=np.float32)


def person_center_from_shoulders_hips(
    kpts_xy: np.ndarray, kpts_conf: np.ndarray, min_conf: float
) -> Optional[np.ndarray]:
    ids = [
        COCO_KPTS["left_shoulder"],
        COCO_KPTS["right_shoulder"],
        COCO_KPTS["left_hip"],
        COCO_KPTS["right_hip"],
    ]
    pts = []
    for idx in ids:
        p = get_xy(kpts_xy, kpts_conf, idx, min_conf)
        if p is not None:
            pts.append(p)
    if not pts:
        return None
    arr = np.stack(pts, axis=0)
    return arr.mean(axis=0)


def count_valid_kpts(kpts_conf: np.ndarray, min_conf: float) -> int:
    return int(np.sum(np.asarray(kpts_conf) >= float(min_conf)))


def bbox_area_xyxy(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box[:4]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def top2_person_indices_by_area(boxes_xyxy: np.ndarray) -> List[int]:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []
    areas = np.array([bbox_area_xyxy(b) for b in boxes_xyxy], dtype=np.float32)
    idx = np.argsort(-areas)
    return [int(i) for i in idx[:2]]


def compute_pair_features(
    kpts_a_xy: np.ndarray,
    kpts_a_conf: np.ndarray,
    kpts_b_xy: np.ndarray,
    kpts_b_conf: np.ndarray,
    roi_w: int,
    roi_h: int,
    min_conf: float,
) -> Dict[str, float]:
    norm = float(max(roi_w, roi_h, 1))

    center_a = person_center_from_shoulders_hips(kpts_a_xy, kpts_a_conf, min_conf)
    center_b = person_center_from_shoulders_hips(kpts_b_xy, kpts_b_conf, min_conf)
    center_dist = safe_norm_dist(center_a, center_b, norm)

    lw_a = get_xy(kpts_a_xy, kpts_a_conf, COCO_KPTS["left_wrist"], min_conf)
    rw_a = get_xy(kpts_a_xy, kpts_a_conf, COCO_KPTS["right_wrist"], min_conf)
    lw_b = get_xy(kpts_b_xy, kpts_b_conf, COCO_KPTS["left_wrist"], min_conf)
    rw_b = get_xy(kpts_b_xy, kpts_b_conf, COCO_KPTS["right_wrist"], min_conf)

    wrist_dists = []
    for pa in [lw_a, rw_a]:
        for pb in [lw_b, rw_b]:
            if pa is not None and pb is not None:
                wrist_dists.append(safe_norm_dist(pa, pb, norm))
    wrist_dist = min(wrist_dists) if wrist_dists else 1e9

    shoulders_a = []
    shoulders_b = []
    for idx in [COCO_KPTS["left_shoulder"], COCO_KPTS["right_shoulder"]]:
        p = get_xy(kpts_a_xy, kpts_a_conf, idx, min_conf)
        if p is not None:
            shoulders_a.append(p)
        p = get_xy(kpts_b_xy, kpts_b_conf, idx, min_conf)
        if p is not None:
            shoulders_b.append(p)

    upper_body_dist = 1e9
    if shoulders_a and shoulders_b:
        sa = np.stack(shoulders_a, axis=0).mean(axis=0)
        sb = np.stack(shoulders_b, axis=0).mean(axis=0)
        upper_body_dist = safe_norm_dist(sa, sb, norm)

    return {
        "center_dist_norm": float(center_dist),
        "wrist_dist_norm": float(wrist_dist),
        "upper_body_dist_norm": float(upper_body_dist),
    }


def compute_interaction_score(
    center_dist_norm: float,
    wrist_dist_norm: float,
    upper_body_dist_norm: float,
    max_center_dist_norm: float,
    wrist_thr: float,
    upper_thr: float,
) -> float:
    s_center = 0.0
    if center_dist_norm < 1e9:
        s_center = max(0.0, 1.0 - (center_dist_norm / max(max_center_dist_norm, 1e-6)))

    s_wrist = 0.0
    if wrist_dist_norm < 1e9:
        s_wrist = max(0.0, 1.0 - (wrist_dist_norm / max(wrist_thr, 1e-6)))

    s_upper = 0.0
    if upper_body_dist_norm < 1e9:
        s_upper = max(0.0, 1.0 - (upper_body_dist_norm / max(upper_thr, 1e-6)))

    score = 0.40 * s_center + 0.35 * s_wrist + 0.25 * s_upper
    return float(max(0.0, min(1.0, score)))