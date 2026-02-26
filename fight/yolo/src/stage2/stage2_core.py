from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

THIS = Path(__file__).resolve()
FIGHT_ROOT = THIS.parents[3]
MOTION_ROOT = FIGHT_ROOT / "motion"
sys.path.insert(0, str(MOTION_ROOT))
sys.path.insert(0, str(FIGHT_ROOT))

from src.ingest.cam_reader import frame_generator
from src.motion.bg_subtractor import BGSubtractor
from src.motion.frame_diff import FrameDiffer
from src.motion.gate import MotionGate
from src.motion.roi import apply_mask, build_ignore_mask
from src.service.segmenter import Segment, detect_segments


@dataclass
class YoloCfg:
    weights: str
    imgsz: int
    conf: float
    iou: float
    classes: List[int]
    device: str
    half: bool


@dataclass
class TrackingCfg:
    enabled: bool
    tracker: str
    max_lost_frames: int
    gate_use_tracks: bool
    persist: bool


@dataclass
class FilterCfg:
    min_persons: int
    topk_by_area: int
    min_box_area_ratio: float


@dataclass
class SegmentsCfg:
    thr_on: float
    thr_off: float
    skip_first_frames: int
    min_len_sec: float
    merge_gap_sec: float
    smooth: str
    ema_alpha: float
    ma_win: int


@dataclass
class ExportCfg:
    out_dir: str
    save_fullframe_mp4: bool
    save_crop_mp4: bool
    save_frames: bool
    crop_margin: float
    crop_out_w: int
    crop_out_h: int
    fps_out: float
    sample_stride: int
    max_event_sec: float
    video_fourcc: str


@dataclass
class InteractionCfg:
    enabled: bool
    max_center_dist_norm: float
    min_iou: float
    use_topk: int
    w_proximity: float
    w_iou: float
    w_union: float
    max_union_area_ratio: float
    dyn_margin_gain: float
    dyn_margin_max: float


@dataclass
class RoiStabilizerCfg:
    enabled: bool
    ema_alpha: float
    jump_iou_thr: float
    jump_confirm_frames: int


@dataclass
class ProbeCfg:
    enabled: bool
    frames: int
    need_2p: int
    strategy: str
    gate_source: str
    stride: int


@dataclass
class MergeCfg:
    enabled: bool
    gap_sec: float
    pad_pre_sec: float
    pad_post_sec: float
    max_merge_len_sec: float


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("yolo-config must be a YAML mapping")
    return data


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def _auto_device(dev: str) -> str:
    d = str(dev).lower().strip()
    if d in ("auto", ""):
        return "0" if torch.cuda.is_available() else "cpu"
    return dev


def _auto_half(half: Any, device: str) -> bool:
    if isinstance(half, bool):
        return half
    s = str(half).lower().strip()
    if s == "auto":
        return str(device) != "cpu" and torch.cuda.is_available()
    if s in ("1", "true", "yes", "y"):
        return True
    return False


def parse_cfg(mcfg, ycfg: Dict[str, Any]) -> Tuple[YoloCfg, TrackingCfg, FilterCfg, SegmentsCfg, ExportCfg, InteractionCfg, RoiStabilizerCfg, ProbeCfg, MergeCfg]:
    yolo_d = _get(ycfg, "yolo", {})
    if not isinstance(yolo_d, dict):
        yolo_d = {}

    trk_d = _get(ycfg, "tracking", {})
    if not isinstance(trk_d, dict):
        trk_d = {}

    seg_d = _get(ycfg, "segments", {})
    if not isinstance(seg_d, dict):
        seg_d = {}

    flt_d = _get(ycfg, "filter", {})
    if not isinstance(flt_d, dict):
        flt_d = {}

    exp_d = _get(ycfg, "export", {})
    if not isinstance(exp_d, dict):
        exp_d = {}

    inter_d = _get(ycfg, "interaction", {})
    if not isinstance(inter_d, dict):
        inter_d = {}

    stab_d = _get(ycfg, "roi_stabilizer", {})
    if not isinstance(stab_d, dict):
        stab_d = {}

    probe_d = _get(ycfg, "probe", {})
    if not isinstance(probe_d, dict):
        probe_d = {}

    merge_d = _get(ycfg, "merge", {})
    if not isinstance(merge_d, dict):
        merge_d = {}

    weights = str(_get(yolo_d, "weights", "yolo11n.pt"))
    imgsz = int(_get(yolo_d, "imgsz", 640))
    conf = float(_get(yolo_d, "conf", 0.35))
    iou = float(_get(yolo_d, "iou", 0.65))
    classes = _get(yolo_d, "classes", [0])
    if not isinstance(classes, list):
        classes = [0]
    classes = [int(x) for x in classes]
    device = _auto_device(_get(yolo_d, "device", "auto"))
    half = _auto_half(_get(yolo_d, "half", "auto"), str(device))
    y = YoloCfg(weights=weights, imgsz=imgsz, conf=conf, iou=iou, classes=classes, device=str(device), half=half)

    t = TrackingCfg(
        enabled=bool(_get(trk_d, "enabled", False)),
        tracker=str(_get(trk_d, "tracker", "bytetrack.yaml")),
        max_lost_frames=int(_get(trk_d, "max_lost_frames", 6)),
        gate_use_tracks=bool(_get(trk_d, "gate_use_tracks", True)),
        persist=bool(_get(trk_d, "persist", True)),
    )

    thr_on_raw = _get(seg_d, "thr_on", "motion_open")
    thr_off_raw = _get(seg_d, "thr_off", "motion_close")
    thr_on = float(mcfg.threshold_open) if str(thr_on_raw) == "motion_open" else float(thr_on_raw)
    thr_off = float(mcfg.threshold_close) if str(thr_off_raw) == "motion_close" else float(thr_off_raw)

    s = SegmentsCfg(
        thr_on=thr_on,
        thr_off=thr_off,
        skip_first_frames=int(_get(seg_d, "skip_first_frames", 0)),
        min_len_sec=float(_get(seg_d, "min_len_sec", 0.5)),
        merge_gap_sec=float(_get(seg_d, "merge_gap_sec", 0.25)),
        smooth=str(_get(seg_d, "smooth", "ema")),
        ema_alpha=float(_get(seg_d, "ema_alpha", 0.2)),
        ma_win=int(_get(seg_d, "ma_win", 5)),
    )

    f = FilterCfg(
        min_persons=int(_get(flt_d, "min_persons", 2)),
        topk_by_area=int(_get(flt_d, "topk_by_area", 0)),
        min_box_area_ratio=float(_get(flt_d, "min_box_area_ratio", 0.0)),
    )

    e = ExportCfg(
        out_dir=str(_get(exp_d, "out_dir", "outputs/events")),
        save_fullframe_mp4=bool(_get(exp_d, "save_fullframe_mp4", True)),
        save_crop_mp4=bool(_get(exp_d, "save_crop_mp4", True)),
        save_frames=bool(_get(exp_d, "save_frames", False)),
        crop_margin=float(_get(exp_d, "crop_margin", 0.08)),
        crop_out_w=int(_get(exp_d, "crop_out_w", 640)),
        crop_out_h=int(_get(exp_d, "crop_out_h", 640)),
        fps_out=float(_get(exp_d, "fps_out", 0.0)),
        sample_stride=int(_get(exp_d, "sample_stride", 1)),
        max_event_sec=float(_get(exp_d, "max_event_sec", 0.0)),
        video_fourcc=str(_get(exp_d, "video_fourcc", "XVID")),
    )

    inter = InteractionCfg(
        enabled=bool(_get(inter_d, "enabled", True)),
        max_center_dist_norm=float(_get(inter_d, "max_center_dist_norm", 0.35)),
        min_iou=float(_get(inter_d, "min_iou", 0.0)),
        use_topk=int(_get(inter_d, "use_topk", 6)),
        w_proximity=float(_get(inter_d, "w_proximity", 0.75)),
        w_iou=float(_get(inter_d, "w_iou", 0.25)),
        w_union=float(_get(inter_d, "w_union", 0.55)),
        max_union_area_ratio=float(_get(inter_d, "max_union_area_ratio", 0.55)),
        dyn_margin_gain=float(_get(inter_d, "dyn_margin_gain", 0.35)),
        dyn_margin_max=float(_get(inter_d, "dyn_margin_max", 0.30)),
    )

    stab = RoiStabilizerCfg(
        enabled=bool(_get(stab_d, "enabled", True)),
        ema_alpha=float(_get(stab_d, "ema_alpha", 0.45)),
        jump_iou_thr=float(_get(stab_d, "jump_iou_thr", 0.20)),
        jump_confirm_frames=int(_get(stab_d, "jump_confirm_frames", 2)),
    )

    p = ProbeCfg(
        enabled=bool(_get(probe_d, "enabled", True)),
        frames=int(_get(probe_d, "frames", 18)),
        need_2p=int(_get(probe_d, "need_2p", 2)),
        strategy=str(_get(probe_d, "strategy", "uniform")).lower().strip(),
        gate_source=str(_get(probe_d, "gate_source", "det")).lower().strip(),
        stride=int(_get(probe_d, "stride", 1)),
    )

    m = MergeCfg(
        enabled=bool(_get(merge_d, "enabled", True)),
        gap_sec=float(_get(merge_d, "gap_sec", 1.0)),
        pad_pre_sec=float(_get(merge_d, "pad_pre_sec", 0.25)),
        pad_post_sec=float(_get(merge_d, "pad_post_sec", 0.35)),
        max_merge_len_sec=float(_get(merge_d, "max_merge_len_sec", 0.0)),
    )

    return y, t, f, s, e, inter, stab, p, m


def compute_motion_scores(video: str, mcfg, resize_keep_aspect_fn, to_gray_fn, blur_fn) -> Dict[str, Any]:
    method = str(mcfg.method).lower().strip()

    window_size = 5
    min_pass = 3
    if hasattr(mcfg, "temporal"):
        tcfg = getattr(mcfg, "temporal")
        if hasattr(tcfg, "window_size"):
            try:
                window_size = int(getattr(tcfg, "window_size"))
            except Exception:
                pass
        if hasattr(tcfg, "min_pass"):
            try:
                min_pass = int(getattr(tcfg, "min_pass"))
            except Exception:
                pass

    gate = MotionGate(
        open_threshold=float(mcfg.threshold_open),
        close_threshold=float(mcfg.threshold_close),
        window_size=int(window_size),
        min_pass=int(min_pass),
    )

    differ = FrameDiffer()

    min_contour_area = 300
    if hasattr(mcfg, "postprocess"):
        pcfg = getattr(mcfg, "postprocess")
        if hasattr(pcfg, "min_contour_area"):
            try:
                min_contour_area = int(getattr(pcfg, "min_contour_area"))
            except Exception:
                pass

    bgsub = BGSubtractor(
        history=int(mcfg.bg_history),
        var_threshold=float(mcfg.bg_var_threshold),
        detect_shadows=bool(mcfg.bg_detect_shadows),
        morph_ksize=int(mcfg.bg_morph_ksize),
        min_contour_area=int(min_contour_area),
    )

    ignore_mask: Optional[np.ndarray] = None
    mask_ready = False

    last_ts_ms = 0
    scores: List[float] = []
    ts_list: List[float] = []

    for ts, frame in frame_generator(video):
        now_ms = int(ts * 1000)
        if int(mcfg.min_interval_ms) > 0 and (now_ms - last_ts_ms) < int(mcfg.min_interval_ms):
            continue
        last_ts_ms = now_ms

        f = resize_keep_aspect_fn(frame, int(mcfg.resize_width))
        g = to_gray_fn(f) if bool(mcfg.grayscale) else f
        g = blur_fn(g, int(mcfg.blur_ksize))

        if bool(mcfg.roi_enabled) and not mask_ready:
            h, w = g.shape[:2]
            ignore_mask = build_ignore_mask((h, w), mcfg.roi_ignore_zones)
            mask_ready = True

        g_m = apply_mask(g, ignore_mask) if ignore_mask is not None else g

        if method == "frame_diff":
            res = differ.compute(g_m, roi_mask=None)
            raw_score = float(res.score)
        elif method == "bg_subtractor":
            res = bgsub.compute(g_m, ignore_mask=ignore_mask)
            raw_score = float(res.score)
        else:
            raise ValueError(f"Unknown motion method: {mcfg.method}")

        _ = gate.decide(raw_score)
        scores.append(float(raw_score))
        ts_list.append(float(ts))

    fps = 0.0
    if len(ts_list) >= 2:
        dt = (ts_list[-1] - ts_list[0]) / max(len(ts_list) - 1, 1)
        fps = (1.0 / dt) if dt > 0 else 0.0

    return {"scores": scores, "ts_list": ts_list, "fps": fps}


def detect_motion_segments(
    scores: List[float],
    fps: float,
    thr_on: float,
    thr_off: float,
    min_len_sec: float,
    merge_gap_sec: float,
    smooth: str,
    ema_alpha: float,
    ma_win: int,
) -> List[Segment]:
    return detect_segments(
        scores=scores,
        fps=fps,
        thr_on=float(thr_on),
        thr_off=float(thr_off),
        min_len_sec=float(min_len_sec),
        merge_gap_sec=float(merge_gap_sec),
        smooth=str(smooth),
        ema_alpha=float(ema_alpha),
        ma_win=int(ma_win),
    )


def build_segment_mask(n: int, segs: List[Segment]) -> np.ndarray:
    mask = np.zeros((n,), dtype=np.uint8)
    for s in segs:
        mask[int(s.start_f) : int(s.end_f) + 1] = 1
    return mask


def yolo_model(weights: str) -> YOLO:
    return YOLO(weights)


def _expand_roi(x1: int, y1: int, x2: int, y2: int, W: int, H: int, margin: float) -> Tuple[int, int, int, int]:
    x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
    W = int(W); H = int(H)

    if W <= 0 or H <= 0:
        w = max(2, x2 - x1)
        h = max(2, y2 - y1)
        return (0, 0, w, h)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    if float(margin) <= 1.0:
        pad = int(round(float(margin) * float(max(w, h))))
    else:
        pad = int(round(float(margin)))

    nx1 = max(0, x1 - pad)
    ny1 = max(0, y1 - pad)
    nx2 = min(W, x2 + pad)
    ny2 = min(H, y2 + pad)

    if nx2 <= nx1:
        nx2 = min(W, nx1 + 2)
    if ny2 <= ny1:
        ny2 = min(H, ny1 + 2)

    nx1 = max(0, min(W - 1, nx1))
    ny1 = max(0, min(H - 1, ny1))
    nx2 = max(0, min(W, nx2))
    ny2 = max(0, min(H, ny2))

    return (int(nx1), int(ny1), int(nx2), int(ny2))


def _dyn_margin(base_margin: float, roi: Tuple[int, int, int, int], W: int, H: int, inter: InteractionCfg) -> float:
    if W <= 0 or H <= 0:
        return float(base_margin)
    x1, y1, x2, y2 = roi
    uarea = float(max(0, x2 - x1) * max(0, y2 - y1))
    union_ratio = uarea / float(max(W * H, 1))
    m = float(base_margin) + float(inter.dyn_margin_gain) * float(union_ratio)
    return float(min(m, float(inter.dyn_margin_max)))


def yolo_infer(model: YOLO, frame_bgr: np.ndarray, y: YoloCfg, t: TrackingCfg):
    if t.enabled:
        r = model.track(
            source=frame_bgr,
            imgsz=int(y.imgsz),
            conf=float(y.conf),
            iou=float(y.iou),
            classes=y.classes,
            device=y.device,
            half=bool(y.half),
            tracker=t.tracker,
            persist=bool(t.persist),
            verbose=False,
        )[0]
        return r

    r = model.predict(
        source=frame_bgr,
        imgsz=int(y.imgsz),
        conf=float(y.conf),
        iou=float(y.iou),
        classes=y.classes,
        device=y.device,
        half=bool(y.half),
        verbose=False,
    )[0]
    return r


def extract_boxes(result, tracking_enabled: bool) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

    b = result.boxes
    xyxy = b.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = b.conf.detach().cpu().numpy().astype(np.float32)

    ids: List[int] = []
    if tracking_enabled and getattr(b, "id", None) is not None:
        tid = b.id.detach().cpu().numpy().astype(np.int64).tolist()
        ids = [int(x) for x in tid]
    return xyxy, conf, ids


def apply_min_area_ratio(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    W: int,
    H: int,
    min_area_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boxes_xyxy.shape[0] == 0:
        keep = np.zeros((0,), dtype=bool)
        return boxes_xyxy, confs, keep
    if min_area_ratio <= 0:
        keep = np.ones((boxes_xyxy.shape[0],), dtype=bool)
        return boxes_xyxy, confs, keep

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    total = float(W * H) if (W > 0 and H > 0) else 1.0
    keep = (area / total) >= float(min_area_ratio)
    return boxes_xyxy[keep], confs[keep], keep


def topk_by_area(boxes_xyxy: np.ndarray, confs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if boxes_xyxy.shape[0] == 0 or k <= 0:
        return boxes_xyxy, confs
    if boxes_xyxy.shape[0] <= k:
        return boxes_xyxy, confs
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    idx = np.argsort(-area)[:k]
    return boxes_xyxy[idx], confs[idx]


def union_roi_top2(boxes_xyxy: np.ndarray, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    n = boxes_xyxy.shape[0]
    if n == 0:
        return None

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    idx = np.argsort(-area)

    if n == 1:
        i = int(idx[0])
        rx1, ry1, rx2, ry2 = map(int, boxes_xyxy[i])
        rx1 = max(0, min(W - 1, rx1))
        ry1 = max(0, min(H - 1, ry1))
        rx2 = max(0, min(W, rx2))
        ry2 = max(0, min(H, ry2))
        return (rx1, ry1, rx2, ry2)

    i0 = int(idx[0]); i1 = int(idx[1])
    ax1, ay1, ax2, ay2 = boxes_xyxy[i0]
    bx1, by1, bx2, by2 = boxes_xyxy[i1]

    rx1 = int(min(ax1, bx1))
    ry1 = int(min(ay1, by1))
    rx2 = int(max(ax2, bx2))
    ry2 = int(max(ay2, by2))

    rx1 = max(0, min(W - 1, rx1))
    ry1 = max(0, min(H - 1, ry1))
    rx2 = max(0, min(W, rx2))
    ry2 = max(0, min(H, ry2))
    return (rx1, ry1, rx2, ry2)


def crop(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = max(0, int(x2))
    y2 = max(0, int(y2))
    return frame_bgr[y1:y2, x1:x2].copy()


def make_writer(path: Path, fps: float, size_wh: Tuple[int, int], fourcc: str = "XVID") -> cv2.VideoWriter:
    w, h = int(size_wh[0]), int(size_wh[1])
    p = str(path)
    c = cv2.VideoWriter_fourcc(*str(fourcc)[:4])
    wr = cv2.VideoWriter(p, c, float(fps), (w, h))
    if wr.isOpened():
        return wr
    c2 = cv2.VideoWriter_fourcc(*"MJPG")
    wr2 = cv2.VideoWriter(p, c2, float(fps), (w, h))
    return wr2


def track_ttl_update(active_last_seen: Dict[int, int], ids_now: List[int], frame_i: int, max_lost: int) -> int:
    for tid in ids_now:
        active_last_seen[int(tid)] = int(frame_i)
    kill = [tid for tid, last_i in active_last_seen.items() if (int(frame_i) - int(last_i)) > int(max_lost)]
    for tid in kill:
        active_last_seen.pop(tid, None)
    return len(active_last_seen)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _box_area(b: np.ndarray) -> float:
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def _box_center(b: np.ndarray) -> Tuple[float, float]:
    return (float(b[0] + b[2]) * 0.5, float(b[1] + b[3]) * 0.5)


def iou_xyxy(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def union_pair_roi(boxA: np.ndarray, boxB: np.ndarray, W: int, H: int) -> Tuple[int, int, int, int]:
    rx1 = int(min(boxA[0], boxB[0]))
    ry1 = int(min(boxA[1], boxB[1]))
    rx2 = int(max(boxA[2], boxB[2]))
    ry2 = int(max(boxA[3], boxB[3]))
    rx1 = max(0, min(W - 1, rx1))
    ry1 = max(0, min(H - 1, ry1))
    rx2 = max(0, min(W, rx2))
    ry2 = max(0, min(H, ry2))
    return (rx1, ry1, rx2, ry2)


def pair_score(boxA: np.ndarray, boxB: np.ndarray, W: int, H: int, inter: InteractionCfg) -> float:
    acx, acy = _box_center(boxA)
    bcx, bcy = _box_center(boxB)
    dist = float(np.hypot(acx - bcx, acy - bcy))
    norm_dist = dist / float(max(W, H, 1))
    proximity = 1.0 - norm_dist

    ix1 = max(float(boxA[0]), float(boxB[0]))
    iy1 = max(float(boxA[1]), float(boxB[1]))
    ix2 = min(float(boxA[2]), float(boxB[2]))
    iy2 = min(float(boxA[3]), float(boxB[3]))
    inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union_area = _box_area(boxA) + _box_area(boxB) - inter_area
    iou = (inter_area / union_area) if union_area > 0 else 0.0

    ux1 = min(float(boxA[0]), float(boxB[0]))
    uy1 = min(float(boxA[1]), float(boxB[1]))
    ux2 = max(float(boxA[2]), float(boxB[2]))
    uy2 = max(float(boxA[3]), float(boxB[3]))
    uarea = max(0.0, ux2 - ux1) * max(0.0, uy2 - uy1)
    frame_area = float(max(W * H, 1))
    union_ratio = uarea / frame_area

    return float(inter.w_proximity) * float(proximity) + float(inter.w_iou) * float(iou) - float(inter.w_union) * float(union_ratio)


def select_best_pair(boxes_xyxy: np.ndarray, W: int, H: int, inter: InteractionCfg) -> Tuple[Optional[Tuple[int, int]], float]:
    n = int(boxes_xyxy.shape[0])
    if n < 2:
        return None, -1.0

    best_s = -1.0
    best_pair: Optional[Tuple[int, int]] = None
    frame_area = float(max(W * H, 1))

    for i in range(n):
        for j in range(i + 1, n):
            boxA = boxes_xyxy[i]
            boxB = boxes_xyxy[j]

            acx, acy = _box_center(boxA)
            bcx, bcy = _box_center(boxB)
            dist = float(np.hypot(acx - bcx, acy - bcy))
            norm_dist = dist / float(max(W, H, 1))
            if norm_dist > float(inter.max_center_dist_norm):
                continue

            ux1 = min(float(boxA[0]), float(boxB[0]))
            uy1 = min(float(boxA[1]), float(boxB[1]))
            ux2 = max(float(boxA[2]), float(boxB[2]))
            uy2 = max(float(boxA[3]), float(boxB[3]))
            uarea = max(0.0, ux2 - ux1) * max(0.0, uy2 - uy1)
            union_ratio = uarea / frame_area
            if union_ratio > float(inter.max_union_area_ratio):
                continue

            if float(inter.min_iou) > 0:
                ix1 = max(float(boxA[0]), float(boxB[0]))
                iy1 = max(float(boxA[1]), float(boxB[1]))
                ix2 = min(float(boxA[2]), float(boxB[2]))
                iy2 = min(float(boxA[3]), float(boxB[3]))
                inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union_area = _box_area(boxA) + _box_area(boxB) - inter_area
                iou = (inter_area / union_area) if union_area > 0 else 0.0
                if iou < float(inter.min_iou):
                    continue

            s = pair_score(boxA, boxB, W, H, inter)
            if s > best_s:
                best_s = s
                best_pair = (i, j)

    if best_pair is None:
        best_dist = 1e9
        best_ratio = 1e9
        for i in range(n):
            for j in range(i + 1, n):
                boxA = boxes_xyxy[i]
                boxB = boxes_xyxy[j]

                ux1 = min(float(boxA[0]), float(boxB[0]))
                uy1 = min(float(boxA[1]), float(boxB[1]))
                ux2 = max(float(boxA[2]), float(boxB[2]))
                uy2 = max(float(boxA[3]), float(boxB[3]))
                uarea = max(0.0, ux2 - ux1) * max(0.0, uy2 - uy1)
                union_ratio = uarea / frame_area
                if union_ratio > float(inter.max_union_area_ratio):
                    continue

                acx, acy = _box_center(boxA)
                bcx, bcy = _box_center(boxB)
                dist = float(np.hypot(acx - bcx, acy - bcy))
                if (dist < best_dist) or (abs(dist - best_dist) < 1e-6 and union_ratio < best_ratio):
                    best_dist = dist
                    best_ratio = union_ratio
                    best_pair = (i, j)

        if best_pair is not None:
            best_s = pair_score(boxes_xyxy[best_pair[0]], boxes_xyxy[best_pair[1]], W, H, inter)

    return best_pair, float(best_s)


def select_fight_roi(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    W: int,
    H: int,
    margin: float,
    inter: InteractionCfg,
) -> Tuple[Optional[Tuple[int, int, int, int]], str, float, Optional[Tuple[int, int]]]:
    n = int(boxes_xyxy.shape[0])
    if n <= 0:
        return None, "none", 0.0, None

    if inter.use_topk > 0 and n > int(inter.use_topk):
        boxes_xyxy, confs = topk_by_area(boxes_xyxy, confs, int(inter.use_topk))
        n = int(boxes_xyxy.shape[0])

    if inter.enabled and n >= 2:
        pair, score = select_best_pair(boxes_xyxy, W, H, inter)
        if pair is not None:
            i, j = pair
            roi0 = union_pair_roi(boxes_xyxy[i], boxes_xyxy[j], W, H)
            m2 = _dyn_margin(margin, roi0, W, H, inter)
            roi = _expand_roi(roi0[0], roi0[1], roi0[2], roi0[3], W, H, m2)
            return roi, "pair", float(score), (int(i), int(j))

    if n >= 2:
        roi0 = union_roi_top2(boxes_xyxy, W, H)
        if roi0 is not None:
            m2 = _dyn_margin(margin, roi0, W, H, inter)
            roi = _expand_roi(roi0[0], roi0[1], roi0[2], roi0[3], W, H, m2)
            return roi, "top2", 0.0, None

    rx1, ry1, rx2, ry2 = map(int, boxes_xyxy[0])
    roi0 = (rx1, ry1, rx2, ry2)
    m2 = _dyn_margin(margin, roi0, W, H, inter)
    roi = _expand_roi(roi0[0], roi0[1], roi0[2], roi0[3], W, H, m2)
    return roi, "single", 0.0, None


class RoiStabilizer:
    def __init__(self, cfg: RoiStabilizerCfg):
        self.cfg = cfg
        self.prev_roi: Optional[Tuple[int, int, int, int]] = None
        self._jump_counter = 0

    def reset(self):
        self.prev_roi = None
        self._jump_counter = 0

    def update(self, roi_new: Optional[Tuple[int, int, int, int]]) -> Tuple[Optional[Tuple[int, int, int, int]], float, bool]:
        if not self.cfg.enabled:
            i = iou_xyxy(self.prev_roi, roi_new)
            self.prev_roi = roi_new
            return roi_new, float(i), False

        if roi_new is None:
            return self.prev_roi, 0.0, False

        if self.prev_roi is None:
            self.prev_roi = roi_new
            return roi_new, 0.0, False

        i = iou_xyxy(self.prev_roi, roi_new)

        jump = i < float(self.cfg.jump_iou_thr)
        jump_accepted = False
        if jump:
            self._jump_counter += 1
            if self._jump_counter < int(self.cfg.jump_confirm_frames):
                return self.prev_roi, float(i), False
            jump_accepted = True
        else:
            self._jump_counter = 0

        base_a = float(self.cfg.ema_alpha)
        if i < 0.35:
            a = min(0.75, max(base_a, 0.65))
        elif i < 0.55:
            a = min(0.65, max(base_a, 0.55))
        else:
            a = base_a

        px1, py1, px2, py2 = self.prev_roi
        nx1, ny1, nx2, ny2 = roi_new
        sx1 = int(round((1 - a) * px1 + a * nx1))
        sy1 = int(round((1 - a) * py1 + a * ny1))
        sx2 = int(round((1 - a) * px2 + a * nx2))
        sy2 = int(round((1 - a) * py2 + a * ny2))
        smoothed = (sx1, sy1, sx2, sy2)

        self.prev_roi = smoothed
        return smoothed, float(i), jump_accepted


def uniform_probe_indices(start_f: int, end_f: int, n: int) -> List[int]:
    start_f = int(start_f)
    end_f = int(end_f)
    length = max(1, end_f - start_f + 1)
    n = int(max(1, n))
    if n >= length:
        return [start_f + i for i in range(length)]
    xs = np.linspace(0, length - 1, num=n)
    idx = sorted({start_f + int(round(x)) for x in xs})
    return idx


def merge_segments_by_gap(
    segs: List[Tuple[int, int]],
    fps: float,
    gap_sec: float,
    pad_pre_sec: float,
    pad_post_sec: float,
    max_merge_len_sec: float,
    n_total_frames: int,
) -> List[Tuple[int, int]]:
    if not segs:
        return []

    fps = float(fps) if fps > 0 else 30.0
    gap_f = int(round(float(gap_sec) * fps))
    pre_f = int(round(float(pad_pre_sec) * fps))
    post_f = int(round(float(pad_post_sec) * fps))
    max_len_f = int(round(float(max_merge_len_sec) * fps)) if max_merge_len_sec and max_merge_len_sec > 0 else 0

    segs_sorted = sorted([(int(a), int(b)) for a, b in segs], key=lambda x: x[0])
    out: List[Tuple[int, int]] = []
    cur_s, cur_e = segs_sorted[0]

    for s, e in segs_sorted[1:]:
        if s <= cur_e + gap_f:
            cur_e = max(cur_e, e)
            if max_len_f > 0 and (cur_e - cur_s + 1) > max_len_f:
                out.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    out.append((cur_s, cur_e))

    padded: List[Tuple[int, int]] = []
    for s, e in out:
        ps = max(0, s - pre_f)
        pe = min(int(n_total_frames) - 1, e + post_f)
        padded.append((int(ps), int(pe)))

    return padded