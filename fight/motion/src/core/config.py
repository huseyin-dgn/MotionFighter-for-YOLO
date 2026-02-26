from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _as_int(v, default: int) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v, default: float) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_bool(v, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
    return bool(v)


def _get(data: Dict[str, Any], key: str, default=None):
    try:
        return data.get(key, default)
    except AttributeError:
        return default


@dataclass
class ROIBox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class MotionConfig:
    method: str
    threshold_open: float
    threshold_close: float

    window_size: int
    min_pass: int
    min_interval_ms: int

    resize_width: int
    grayscale: bool
    blur_ksize: int

    roi_enabled: bool
    roi_ignore_zones: List[ROIBox]

    bg_history: int
    bg_var_threshold: int
    bg_detect_shadows: bool
    bg_morph_ksize: int

    min_contour_area: int

    debug_enabled: bool
    debug_save_every_n_pass: int
    debug_out_dir: str

    max_blob_ratio: float
    mean_jump_thr: float

    fight_thr_activity_score: float
    fight_min_activity_area: int
    fight_min_moving_pixels: int

    fight_E_ref: float
    fight_V_ref: float
    fight_F_ref: float
    fight_D_ref: float
    fight_C_ref: float

    fight_wE: float
    fight_wV: float
    fight_wF: float
    fight_wD: float
    fight_wC: float
    fight_wJR: float

    fight_thr_score: float

    vote_window: int
    vote_need: int

    jitter_score_small: float
    jitter_delta_big: int
    jitter_cc_max: int

    walk_filter_enabled: bool
    walk_max_speed: float

    warmup_frames: int
    pre_padding: int
    prebuf_len_base: int
    cooldown_frames: int
    post_padding: int

    motion_gate_win: int
    motion_prebuf: int

    motion_adaptive_thr: bool
    motion_adapt_frames: int
    motion_k_on: float
    motion_k_off: float
    motion_thr_min: float

    motion_min_on_frames: int
    motion_off_run: int
    motion_max_event_len_frames: int


def load_config(path: str | Path) -> MotionConfig:
    p = Path(path)
    data: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    preprocess = _get(data, "preprocess", {}) or {}
    roi = _get(data, "roi", {}) or {}
    bg = _get(data, "bg_subtractor", {}) or {}
    post = _get(data, "postprocess", {}) or {}
    temporal = _get(data, "temporal", {}) or {}
    debug = _get(data, "debug", {}) or {}
    fight = _get(data, "fight", {}) or {}
    event = _get(data, "event", {}) or {}
    motion_filters = _get(data, "motion_filters", {}) or {}
    motion = _get(data, "motion", {}) or {}

    fight_activity = _get(fight, "activity", {}) or {}
    fight_norm = _get(fight, "normalize", {}) or {}
    fight_weights = _get(fight, "weights", {}) or {}
    fight_cand = _get(fight, "candidate", {}) or {}
    fight_voting = _get(fight, "voting", {}) or {}
    fight_jitter = _get(fight, "jitter_reject", {}) or {}
    fight_walk = _get(fight, "walk_filter", {}) or {}

    zones: List[ROIBox] = []
    for z in (_get(roi, "ignore_zones", []) or []):
        try:
            zones.append(
                ROIBox(
                    x=float(_get(z, "x", 0.0)),
                    y=float(_get(z, "y", 0.0)),
                    w=float(_get(z, "w", 0.0)),
                    h=float(_get(z, "h", 0.0)),
                )
            )
        except Exception:
            continue

    return MotionConfig(
        method=str(_get(data, "method", "bg_subtractor")),
        threshold_open=_as_float(_get(data, "threshold_open"), 0.004),
        threshold_close=_as_float(_get(data, "threshold_close"), 0.002),

        window_size=_as_int(_get(temporal, "window_size"), 5),
        min_pass=_as_int(_get(temporal, "min_pass"), 3),
        min_interval_ms=_as_int(_get(data, "min_interval_ms"), 0),

        resize_width=_as_int(_get(preprocess, "resize_width"), 480),
        grayscale=_as_bool(_get(preprocess, "grayscale"), True),
        blur_ksize=_as_int(_get(preprocess, "blur_ksize"), 5),

        roi_enabled=_as_bool(_get(roi, "enabled"), False),
        roi_ignore_zones=zones,

        bg_history=_as_int(_get(bg, "history"), 500),
        bg_var_threshold=_as_int(_get(bg, "var_threshold"), 55),
        bg_detect_shadows=_as_bool(_get(bg, "detect_shadows"), False),
        bg_morph_ksize=_as_int(_get(bg, "morph_ksize"), 5),

        min_contour_area=_as_int(_get(post, "min_contour_area"), 120),

        debug_enabled=_as_bool(_get(debug, "enabled"), True),
        debug_save_every_n_pass=_as_int(_get(debug, "save_every_n_pass"), 1),
        debug_out_dir=str(_get(debug, "out_dir") or "outputs/motion_debug"),

        max_blob_ratio=_as_float(_get(motion_filters, "max_blob_ratio"), 0.10),
        mean_jump_thr=_as_float(_get(motion_filters, "mean_jump_thr"), 10.0),

        fight_thr_activity_score=_as_float(_get(fight_activity, "thr_score"), 0.0018),
        fight_min_activity_area=_as_int(_get(fight_activity, "min_area"), 450),
        fight_min_moving_pixels=_as_int(_get(fight_activity, "min_moving_pixels"), 0),

        fight_E_ref=_as_float(_get(fight_norm, "E_ref"), 0.0060),
        fight_V_ref=_as_float(_get(fight_norm, "V_ref"), 250.0),
        fight_F_ref=_as_float(_get(fight_norm, "F_ref"), 3.0),
        fight_D_ref=_as_float(_get(fight_norm, "D_ref"), 25.0),
        fight_C_ref=_as_float(_get(fight_norm, "C_ref"), 0.55),

        fight_wE=_as_float(_get(fight_weights, "wE"), 0.25),
        fight_wV=_as_float(_get(fight_weights, "wV"), 0.25),
        fight_wF=_as_float(_get(fight_weights, "wF"), 0.15),
        fight_wD=_as_float(_get(fight_weights, "wD"), 0.20),
        fight_wC=_as_float(_get(fight_weights, "wC"), 0.15),
        fight_wJR=_as_float(_get(fight_weights, "wJR"), 1.0),

        fight_thr_score=_as_float(_get(fight_cand, "thr_fight_score"), 0.60),

        vote_window=_as_int(_get(fight_voting, "window"), 16),
        vote_need=_as_int(_get(fight_voting, "need"), 7),

        jitter_score_small=_as_float(_get(fight_jitter, "score_small"), 0.0012),
        jitter_delta_big=_as_int(_get(fight_jitter, "delta_big"), 150),
        jitter_cc_max=_as_int(_get(fight_jitter, "cc_max"), 2),

        walk_filter_enabled=_as_bool(_get(fight_walk, "enabled"), True),
        walk_max_speed=_as_float(_get(fight_walk, "max_speed"), 3.2),

        warmup_frames=_as_int(_get(event, "warmup_frames"), 30),
        pre_padding=_as_int(_get(event, "pre_padding"), 2),
        prebuf_len_base=_as_int(_get(event, "prebuf_len_base"), 30),
        cooldown_frames=_as_int(_get(event, "cooldown_frames"), 60),
        post_padding=_as_int(_get(event, "post_padding"), 30),

        motion_gate_win=_as_int(_get(motion, "gate_win"), 0),
        motion_prebuf=_as_int(_get(motion, "prebuf"), 0),

        motion_adaptive_thr=_as_bool(_get(motion, "adaptive_thr"), False),
        motion_adapt_frames=_as_int(_get(motion, "adapt_frames"), 60),
        motion_k_on=_as_float(_get(motion, "k_on"), 8.0),
        motion_k_off=_as_float(_get(motion, "k_off"), 4.0),
        motion_thr_min=_as_float(_get(motion, "thr_min"), 1e-6),

        motion_min_on_frames=_as_int(_get(motion, "min_on_frames"), 10),
        motion_off_run=_as_int(_get(motion, "off_run"), 20),
        motion_max_event_len_frames=_as_int(_get(motion, "max_event_len_frames"), 300),
    )