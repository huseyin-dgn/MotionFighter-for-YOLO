from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


class MotionAdapter:
    def __init__(self, cfg_path: str):
        from fight.motion.src.core.config import load_config
        from fight.motion.src.service.motion_service import MotionRunner

        self.cfg = load_config(Path(cfg_path))
        self.runner = MotionRunner(self.cfg)

    def step(self, ts: float, frame_bgr: np.ndarray):
        out = self.runner.step(ts, frame_bgr)
        if out is None:
            return 0.0, False, None, None
        return float(out.score), bool(out.pass_frame), out.frame_resized, out.vis_bgr

    def close(self):
        try:
            self.runner.close()
        except Exception:
            pass


class YoloAdapter:
    def __init__(self, cfg_path: str, weights_path: str):
        from ultralytics import YOLO
        import yaml

        self.model = YOLO(weights_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        self.imgsz = int(self.cfg.get("imgsz", 416))
        self.conf = float(self.cfg.get("conf", 0.25))
        self.iou = float(self.cfg.get("iou", 0.45))
        self.device = self.cfg.get("device", 0)
        self.verbose = bool(self.cfg.get("verbose", False))

    def detect_persons(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.model.predict(
            source=frame_rgb,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
        )[0]

        out = []
        if res.boxes is None:
            return out

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
            if cls != 0:
                continue
            out.append((float(c), (int(x1), int(y1), int(x2), int(y2))))

        out.sort(key=lambda x: x[0], reverse=True)
        return out


class PoseLiveAdapter:
    def __init__(self, cfg_path: str):
        from fight.pose.src.pose_adapter import PoseAdapter
        from fight.pose.src.pose_gate import PoseGate
        import yaml

        self.adapter = PoseAdapter(cfg_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        tcfg = cfg.get("temporal", {})
        self.gate = PoseGate(
            window_size=int(tcfg.get("window_size", 6)),
            need_positive=int(tcfg.get("need_positive", 2)),
        )

    def check(self, roi_bgr: np.ndarray):
        raw = self.adapter.evaluate(roi_bgr, hist_positive=int(sum(self.gate.hist)))
        dec = self.gate.update(raw.score, raw.ok)

        raw.ok = dec.pose_ok
        raw.hist_positive = dec.hist_positive
        return raw

    def reset(self):
        self.gate.reset()


class Stage3Adapter:
    def __init__(self, cfg_path: str):
        import importlib.util
        import yaml
        import torch

        self.torch = torch

        cfgp = Path(cfg_path).resolve()
        with open(cfgp, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        stage3_root = cfgp.parent.parent
        self.stage3_root = stage3_root

        mcfg = self.cfg.get("model", {})
        icfg = self.cfg.get("input", {})

        self.num_classes = int(mcfg.get("num_classes", 2))
        self.device = str(mcfg.get("device", "cuda"))
        self.amp = bool(mcfg.get("amp", False))

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        self.clip_len = int(icfg.get("clip_len", 32))
        self.size = int(icfg.get("size", 224))

        ckpt_rel = str(mcfg.get("ckpt_path", "")).replace("\\", "/")
        if not ckpt_rel:
            raise RuntimeError("stage3.yaml içinde model.ckpt_path yok")

        p = Path(ckpt_rel)
        candidates = [
            (Path.cwd() / p).resolve(),
            (stage3_root / p).resolve(),
            (stage3_root / "weights" / p.name).resolve(),
        ]
        ckpt_path = None
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            raise FileNotFoundError(f"ckpt bulunamadı: {ckpt_rel}")

        ml_path = stage3_root / "src" / "model_loader.py"
        spec = importlib.util.spec_from_file_location("stage3_model_loader", str(ml_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)

        self.model = mod.load_model(str(ckpt_path), device=self.device, num_classes=self.num_classes)
        self.model.eval()

        mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)

    def _preprocess(self, clip_bgr_list):
        torch = self.torch

        frames = clip_bgr_list
        T = self.clip_len

        if len(frames) < T:
            if len(frames) == 0:
                raise RuntimeError("clip boş")
            last = frames[-1]
            frames = frames + [last] * (T - len(frames))
        elif len(frames) > T:
            frames = frames[-T:]

        out = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            out.append(rgb)

        arr = np.stack(out, axis=0)
        x = torch.from_numpy(arr).to(torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.to(self.device)
        x = (x - self.mean) / self.std
        return x

    def infer(self, clip_bgr_list):
        torch = self.torch
        x = self._preprocess(clip_bgr_list)

        use_amp = self.amp and str(self.device).startswith("cuda")
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self.model(x)

            if hasattr(logits, "ndim") and logits.ndim == 2 and logits.shape[1] >= 2:
                prob = torch.softmax(logits, dim=1)[:, 1].item()
            else:
                prob = torch.sigmoid(logits).item()

        return float(prob)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_union_box(persons, frame_shape, pad_ratio=0.30):
    h, w = frame_shape[:2]

    if len(persons) < 2:
        return None

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


def sanitize_box(box, frame_shape):
    if box is None:
        return None

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box

    x1 = clamp(int(x1), 0, w - 1)
    y1 = clamp(int(y1), 0, h - 1)
    x2 = clamp(int(x2), 0, w - 1)
    y2 = clamp(int(y2), 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


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


def crop_from_box(frame_bgr, box, out_size=320):
    if box is None:
        return None

    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def save_clip_mp4(frames_bgr, out_path: str, fps: float = 16.0):
    if not frames_bgr:
        return

    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    try:
        for f in frames_bgr:
            writer.write(f)
    finally:
        writer.release()


def _open_source(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--motion-config", type=str, default="fight/motion/configs/motion.yaml")
    ap.add_argument("--yolo-config", type=str, default="fight/yolo/configs/yolo.yaml")
    ap.add_argument("--yolo-weights", type=str, default="fight/yolo11n.pt")
    ap.add_argument("--yolo-stride", type=int, default=6)
    ap.add_argument("--person-conf", type=float, default=0.25)

    ap.add_argument("--use-pose", action="store_true")
    ap.add_argument("--pose-config", type=str, default="fight/pose/configs/pose.yaml")
    ap.add_argument("--pose-stride", type=int, default=2)

    ap.add_argument("--use-stage3", action="store_true")
    ap.add_argument("--stage3-config", type=str, default="fight/3D_CNN/configs/stage3.yaml")
    ap.add_argument("--stage3-stride", type=int, default=10)
    ap.add_argument("--min-2p-frames", type=int, default=6)
    ap.add_argument("--fight-thr", type=float, default=0.6)
    ap.add_argument("--show", action="store_true", default=True)
    ap.add_argument("--reconnect-sec", type=float, default=1.0)
    args = ap.parse_args()

    cap = _open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    motion = MotionAdapter(args.motion_config)
    yolo = YoloAdapter(args.yolo_config, args.yolo_weights)
    pose = PoseLiveAdapter(args.pose_config) if args.use_pose else None
    stage3 = Stage3Adapter(args.stage3_config) if args.use_stage3 else None

    if stage3 is not None:
        print(
            f"[INIT] stage3_clip_len={stage3.clip_len} "
            f"stage3_stride={args.stage3_stride} "
            f"use_pose={int(pose is not None)}"
        )

    clip_debug_dir = Path("fight/clip_debug")
    clip_debug_dir.mkdir(parents=True, exist_ok=True)
    clip_save_idx = 0

    clip = []
    yolo_ctr = 0
    pose_ctr = 0
    stage3_ctr = 0
    stage3_cooldown = 0

    two_p_ctr = 0
    two_p_miss_ctr = 0
    two_p_grace_frames = 6

    last_persons = []

    last_union_box = None
    last_union_seen_ts = 0.0
    roi_missing_since = None
    roi_hold_sec = 1.0
    clip_reset_sec = 1.0
    roi_smooth_alpha = 0.30

    last_valid_roi_frame = None

    last_fight_prob = 0.0
    last_pose_score = 0.0
    last_pose_ok = False
    last_pose_vis = None

    # pose grace state
    last_pose_ok_ts = 0.0
    pose_missing_since = None
    pose_hold_sec = 1.0
    pose_reset_sec = 1.0

    fight_state = False
    fight_on_need = 3
    fight_off_need = 4
    fight_on_ctr = 0
    fight_off_ctr = 0

    last_t = time.time()
    fps = 0.0
    debug_ctr = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if Path(args.source).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                    print("video ended.")
                    break

                print("cap.read() failed, reconnecting...")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(args.reconnect_sec)
                cap = _open_source(args.source)
                continue

            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            last_t = now

            score, active, frame_resized, motion_vis = motion.step(now, frame)
            view = frame if frame_resized is None else frame_resized.copy()

            yolo_ctr += 1
            yolo_ran = False
            persons = []

            if active and (yolo_ctr % max(1, args.yolo_stride) == 0):
                dets = yolo.detect_persons(view)
                persons = [(c, box) for (c, box) in dets if c >= args.person_conf]
                persons.sort(key=lambda x: x[0], reverse=True)
                last_persons = persons
                yolo_ran = True
            elif active:
                persons = last_persons
            else:
                last_persons = []
                persons = []

            conf_list = [round(c, 2) for c, _ in persons]

            for c, (x1, y1, x2, y2) in persons:
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    view,
                    f"{c:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

            if active and len(persons) >= 2:
                two_p_ctr += 1
                two_p_miss_ctr = 0
            else:
                if two_p_ctr > 0:
                    two_p_miss_ctr += 1
                    if two_p_miss_ctr >= two_p_grace_frames:
                        two_p_ctr = 0
                        two_p_miss_ctr = 0
                else:
                    two_p_ctr = 0
                    two_p_miss_ctr = 0

            fight_active = False
            can_classify = False

            if stage3 is not None:
                if stage3_cooldown > 0:
                    stage3_cooldown -= 1

                base_frame = frame if frame_resized is None else frame_resized

                current_union_box = None
                if len(persons) >= 2:
                    current_union_box = build_union_box(persons, base_frame.shape, pad_ratio=0.30)

                if current_union_box is not None:
                    current_union_box = sanitize_box(current_union_box, base_frame.shape)
                    last_union_box = smooth_box(last_union_box, current_union_box, alpha=roi_smooth_alpha)
                    last_union_box = sanitize_box(last_union_box, base_frame.shape)
                    last_union_seen_ts = now
                    roi_missing_since = None
                else:
                    if last_union_box is not None:
                        if roi_missing_since is None:
                            roi_missing_since = now

                        if (now - last_union_seen_ts) > roi_hold_sec:
                            last_union_box = None
                            roi_missing_since = None

                view_s3 = None
                if last_union_box is not None:
                    view_s3 = crop_from_box(base_frame, last_union_box, out_size=320)
                    if view_s3 is not None:
                        last_valid_roi_frame = view_s3.copy()

                if view_s3 is None and last_valid_roi_frame is not None:
                    if roi_missing_since is not None and (now - roi_missing_since) <= roi_hold_sec:
                        view_s3 = last_valid_roi_frame.copy()

                # ---------------- POSE (soft gate with grace) ----------------
                raw_pose_ready = True

                if pose is not None:
                    if active and (last_union_box is not None) and (view_s3 is not None):
                        pose_ctr += 1
                        if pose_ctr % max(1, args.pose_stride) == 0:
                            pres = pose.check(view_s3)
                            last_pose_score = pres.score
                            last_pose_ok = pres.ok
                            last_pose_vis = pres.debug_frame
                        raw_pose_ready = bool(last_pose_ok)
                    else:
                        # pose extractor çalışamayacak durum
                        last_pose_score = 0.0
                        last_pose_ok = False
                        last_pose_vis = None
                        raw_pose_ready = False

                    if raw_pose_ready:
                        last_pose_ok_ts = now
                        pose_missing_since = None
                    else:
                        if pose_missing_since is None:
                            pose_missing_since = now

                    effective_pose_ready = raw_pose_ready
                    if not effective_pose_ready:
                        if last_pose_ok_ts > 0 and (now - last_pose_ok_ts) <= pose_hold_sec:
                            effective_pose_ready = True
                else:
                    effective_pose_ready = True
                    raw_pose_ready = True

                roi_available = (last_union_box is not None) and (view_s3 is not None)

                hard_reset = False

                if not active:
                    hard_reset = True

                if roi_missing_since is not None and (now - roi_missing_since) > clip_reset_sec:
                    hard_reset = True

                if pose is not None and pose_missing_since is not None:
                    if (now - pose_missing_since) > pose_reset_sec:
                        hard_reset = True

                if hard_reset:
                    clip = []
                    fight_on_ctr = 0
                    fight_off_ctr += 1
                    last_fight_prob = 0.0

                    if pose is not None and not active:
                        pose.reset()
                        pose_missing_since = None
                        last_pose_ok_ts = 0.0
                        last_pose_ok = False
                        last_pose_score = 0.0
                        last_pose_vis = None

                    if fight_state and fight_off_ctr >= fight_off_need:
                        fight_state = False
                else:
                    # clip başladıktan sonra kısa pose düşüşlerinde akış devam etsin
                    if roi_available and effective_pose_ready:
                        clip.append(view_s3)
                        if len(clip) > stage3.clip_len:
                            clip = clip[-stage3.clip_len:]

                stage3_ctr += 1

                can_classify = (
                    active
                    and roi_available
                    and effective_pose_ready
                    and (two_p_ctr >= args.min_2p_frames)
                    and (len(clip) >= stage3.clip_len)
                    and (stage3_cooldown == 0)
                )

                if can_classify:
                    clip_path = clip_debug_dir / f"clip_{clip_save_idx:04d}.mp4"
                    save_clip_mp4(clip, str(clip_path), fps=16.0)

                    last_fight_prob = stage3.infer(clip)

                    print(
                        f"[STAGE3] clip={clip_path.name} "
                        f"fight_prob={last_fight_prob:.4f} "
                        f"thr={args.fight_thr:.2f} "
                        f"two_p_ctr={two_p_ctr} "
                        f"clip_len={len(clip)} "
                        f"pose_raw={int(raw_pose_ready) if pose is not None else -1} "
                        f"pose_eff={int(effective_pose_ready) if pose is not None else -1} "
                        f"pose_score={last_pose_score:.3f}"
                    )

                    clip_save_idx += 1
                    stage3_cooldown = max(1, args.stage3_stride)

                    if last_fight_prob >= args.fight_thr:
                        fight_on_ctr += 1
                        fight_off_ctr = 0
                    else:
                        fight_off_ctr += 1
                        fight_on_ctr = 0

                    if (not fight_state) and fight_on_ctr >= fight_on_need:
                        fight_state = True

                    if fight_state and fight_off_ctr >= fight_off_need:
                        fight_state = False

                fight_active = fight_state

            if last_union_box is not None:
                ux1, uy1, ux2, uy2 = last_union_box
                cv2.rectangle(view, (ux1, uy1), (ux2, uy2), (0, 255, 255), 2)

            roi_miss_sec = 0.0 if roi_missing_since is None else (now - roi_missing_since)
            pose_miss_sec = 0.0 if pose_missing_since is None else (now - pose_missing_since)
            pose_eff = last_pose_ok
            if pose is not None and not last_pose_ok and last_pose_ok_ts > 0 and (now - last_pose_ok_ts) <= pose_hold_sec:
                pose_eff = True

            debug_ctr += 1
            if debug_ctr % 15 == 0:
                print(
                    f"[PIPE] motion_active={int(active)} "
                    f"yolo_ran={int(yolo_ran)} "
                    f"persons={len(persons)} "
                    f"conf={conf_list} "
                    f"two_p_ctr={two_p_ctr} "
                    f"roi_ok={int(last_union_box is not None)} "
                    f"roi_miss_sec={roi_miss_sec:.2f} "
                    f"pose_ok={int(last_pose_ok) if pose is not None else -1} "
                    f"pose_eff={int(pose_eff) if pose is not None else -1} "
                    f"pose_score={last_pose_score:.3f} "
                    f"pose_miss_sec={pose_miss_sec:.2f} "
                    f"clip_len={len(clip)} "
                    f"cooldown={stage3_cooldown} "
                    f"stage3_ready={int(can_classify)} "
                    f"fight_prob={last_fight_prob:.4f} "
                    f"fight_state={int(fight_state)}"
                )

            draw_text(view, f"fps={fps:.1f}", 10, 25)
            draw_text(view, f"motion={score:.4f} active={1 if active else 0}", 10, 50)
            draw_text(view, f"persons={len(persons)}", 10, 75)
            draw_text(view, f"yolo_ran={1 if yolo_ran else 0}", 10, 100)
            draw_text(view, f"two_p_ctr={two_p_ctr}", 10, 125)

            y_text = 150
            if pose is not None:
                draw_text(
                    view,
                    f"pose_ok={1 if last_pose_ok else 0} pose_eff={1 if pose_eff else 0} score={last_pose_score:.3f}",
                    10,
                    y_text,
                )
                y_text += 25

            if stage3 is not None:
                draw_text(view, f"fight_prob={last_fight_prob:.3f} thr={args.fight_thr:.2f}", 10, y_text)
                draw_text(view, f"clip_len={len(clip)} roi_ok={1 if last_union_box is not None else 0}", 10, y_text + 25)
                draw_text(view, f"roi_miss_sec={roi_miss_sec:.2f}", 10, y_text + 50)
                if pose is not None:
                    draw_text(view, f"pose_miss_sec={pose_miss_sec:.2f}", 10, y_text + 75)
                    draw_text(view, f"cd={stage3_cooldown}", 10, y_text + 100)
                    draw_text(view, "FIGHT" if fight_active else "NO_FIGHT", 10, y_text + 125)
                else:
                    draw_text(view, f"cd={stage3_cooldown}", 10, y_text + 75)
                    draw_text(view, "FIGHT" if fight_active else "NO_FIGHT", 10, y_text + 100)

            if args.show:
                cv2.imshow("fight_live", view)
                if pose is not None and last_pose_vis is not None:
                    cv2.imshow("pose_roi", last_pose_vis)

                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord("q"):
                    break
            else:
                time.sleep(0.001)

    finally:
        motion.close()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()