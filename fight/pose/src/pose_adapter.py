from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from fight.pose.src.pose_utils import (
    compute_interaction_score,
    compute_pair_features,
    count_valid_kpts,
    top2_person_indices_by_area,
)


@dataclass
class PoseResult:
    ok: bool
    score: float
    num_persons: int
    hist_positive: int
    valid_a: int
    valid_b: int
    kp_conf_a: float
    kp_conf_b: float
    center_dist_norm: float
    wrist_dist_norm: float
    upper_body_dist_norm: float
    debug_frame: Optional[np.ndarray]


class PoseAdapter:
    def __init__(self, cfg_path: str):
        cfgp = Path(cfg_path)
        with open(cfgp, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        mcfg = self.cfg.get("model", {})
        icfg = self.cfg.get("input", {})
        fcfg = self.cfg.get("filter", {})
        xcfg = self.cfg.get("interaction", {})
        dcfg = self.cfg.get("debug", {})

        self.weights = str(mcfg.get("weights", "fight/yolo11n-pose.pt"))
        self.device = mcfg.get("device", 0)
        self.imgsz = int(mcfg.get("imgsz", 320))
        self.conf = float(mcfg.get("conf", 0.25))
        self.verbose = bool(mcfg.get("verbose", False))

        self.roi_size = int(icfg.get("roi_size", 320))

        self.min_persons = int(fcfg.get("min_persons", 2))
        self.min_kpt_conf = float(fcfg.get("min_kpt_conf", 0.30))
        self.min_valid_kpts = int(fcfg.get("min_valid_kpts", 6))

        self.max_center_dist_norm = float(xcfg.get("max_center_dist_norm", 0.35))
        self.wrist_dist_norm_thr = float(xcfg.get("wrist_dist_norm", 0.22))
        self.upper_body_dist_norm_thr = float(xcfg.get("upper_body_dist_norm", 0.28))
        self.min_interaction_score = float(xcfg.get("min_interaction_score", 0.45))

        self.draw_debug = bool(dcfg.get("draw", True))

        self.model = YOLO(self.weights)

    def _empty(self, debug_frame: Optional[np.ndarray] = None) -> PoseResult:
        return PoseResult(
            ok=False,
            score=0.0,
            num_persons=0,
            hist_positive=0,
            valid_a=0,
            valid_b=0,
            kp_conf_a=0.0,
            kp_conf_b=0.0,
            center_dist_norm=1e9,
            wrist_dist_norm=1e9,
            upper_body_dist_norm=1e9,
            debug_frame=debug_frame,
        )

    def infer_roi(self, roi_bgr: np.ndarray) -> Dict[str, Any]:
        roi_resized = cv2.resize(
            roi_bgr,
            (self.roi_size, self.roi_size),
            interpolation=cv2.INTER_LINEAR,
        )
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

        res = self.model.predict(
            source=roi_rgb,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=self.verbose,
        )[0]

        return {
            "resized_bgr": roi_resized,
            "result": res,
        }

    def evaluate(self, roi_bgr: np.ndarray, hist_positive: int = 0) -> PoseResult:
        out = self.infer_roi(roi_bgr)
        roi_vis = out["resized_bgr"].copy()
        res = out["result"]

        if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
            return self._empty(debug_frame=roi_vis)

        if getattr(res, "keypoints", None) is None:
            return self._empty(debug_frame=roi_vis)

        boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
        idx2 = top2_person_indices_by_area(boxes_xyxy)

        if len(idx2) < self.min_persons:
            return self._empty(debug_frame=roi_vis)

        kp_xy = res.keypoints.xy.detach().cpu().numpy()
        kp_conf = res.keypoints.conf.detach().cpu().numpy()

        i0, i1 = idx2[0], idx2[1]
        a_xy = kp_xy[i0]
        b_xy = kp_xy[i1]
        a_conf = kp_conf[i0]
        b_conf = kp_conf[i1]

        valid_a = count_valid_kpts(a_conf, self.min_kpt_conf)
        valid_b = count_valid_kpts(b_conf, self.min_kpt_conf)
        kp_conf_a = float(np.mean(a_conf)) if len(a_conf) else 0.0
        kp_conf_b = float(np.mean(b_conf)) if len(b_conf) else 0.0

        if valid_a < self.min_valid_kpts or valid_b < self.min_valid_kpts:
            return PoseResult(
                ok=False,
                score=0.0,
                num_persons=len(idx2),
                hist_positive=hist_positive,
                valid_a=valid_a,
                valid_b=valid_b,
                kp_conf_a=kp_conf_a,
                kp_conf_b=kp_conf_b,
                center_dist_norm=1e9,
                wrist_dist_norm=1e9,
                upper_body_dist_norm=1e9,
                debug_frame=roi_vis,
            )

        h, w = roi_vis.shape[:2]
        feats = compute_pair_features(
            kpts_a_xy=a_xy,
            kpts_a_conf=a_conf,
            kpts_b_xy=b_xy,
            kpts_b_conf=b_conf,
            roi_w=w,
            roi_h=h,
            min_conf=self.min_kpt_conf,
        )

        score = compute_interaction_score(
            center_dist_norm=feats["center_dist_norm"],
            wrist_dist_norm=feats["wrist_dist_norm"],
            upper_body_dist_norm=feats["upper_body_dist_norm"],
            max_center_dist_norm=self.max_center_dist_norm,
            wrist_thr=self.wrist_dist_norm_thr,
            upper_thr=self.upper_body_dist_norm_thr,
        )
        ok = score >= self.min_interaction_score

        if self.draw_debug:
            txt1 = f"pose_score={score:.3f} ok={int(ok)} hist={hist_positive}"
            txt2 = (
                f"va={valid_a} vb={valid_b} "
                f"cd={feats['center_dist_norm']:.2f} "
                f"wd={feats['wrist_dist_norm']:.2f} "
                f"ud={feats['upper_body_dist_norm']:.2f}"
            )
            cv2.putText(roi_vis, txt1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(roi_vis, txt2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        return PoseResult(
            ok=bool(ok),
            score=float(score),
            num_persons=len(idx2),
            hist_positive=hist_positive,
            valid_a=valid_a,
            valid_b=valid_b,
            kp_conf_a=kp_conf_a,
            kp_conf_b=kp_conf_b,
            center_dist_norm=float(feats["center_dist_norm"]),
            wrist_dist_norm=float(feats["wrist_dist_norm"]),
            upper_body_dist_norm=float(feats["upper_body_dist_norm"]),
            debug_frame=roi_vis,
        )