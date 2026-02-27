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
        return out


class Stage3Adapter:
    def __init__(self, cfg_path: str):
        import importlib.util
        import yaml
        import torch

        self.torch = torch

        cfgp = Path(cfg_path).resolve()
        self.cfg = yaml.safe_load(open(cfgp, "r", encoding="utf-8")) or {}

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--motion-config", type=str, default="fight/motion/configs/motion.yaml")
    ap.add_argument("--yolo-config", type=str, default="fight/yolo/configs/yolo.yaml")
    ap.add_argument("--yolo-weights", type=str, default="fight/yolo11n.pt")
    ap.add_argument("--yolo-stride", type=int, default=6)
    ap.add_argument("--person-conf", type=float, default=0.25)
    ap.add_argument("--use-stage3", action="store_true")
    ap.add_argument("--stage3-config", type=str, default="fight/3D_CNN/configs/stage3.yaml")
    ap.add_argument("--stage3-stride", type=int, default=10)
    ap.add_argument("--min-2p-frames", type=int, default=6)
    ap.add_argument("--fight-thr", type=float, default=0.6)
    ap.add_argument("--show", action="store_true", default=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.cam}")

    motion = MotionAdapter(args.motion_config)
    yolo = YoloAdapter(args.yolo_config, args.yolo_weights)
    stage3 = Stage3Adapter(args.stage3_config) if args.use_stage3 else None

    clip = []
    yolo_ctr = 0
    stage3_ctr = 0
    two_p_ctr = 0
    last_persons = []
    last_fight_prob = 0.0

    fight_state = False
    fight_on_need = 3
    fight_off_need = 6
    fight_on_ctr = 0
    fight_off_ctr = 0

    last_t = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("cap.read() failed")
                break

            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            last_t = now

            score, active, frame_resized, motion_vis = motion.step(now, frame)

            view = frame if frame_resized is None else frame_resized.copy()

            yolo_ctr += 1
            persons = last_persons
            if active and (yolo_ctr % max(1, args.yolo_stride) == 0):
                dets = yolo.detect_persons(view)
                persons = [(c, box) for (c, box) in dets if c >= args.person_conf]
                last_persons = persons

            for c, (x1, y1, x2, y2) in persons:
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(view, f"{c:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            fight_active = False
            if stage3 is not None:
                view_s3 = cv2.resize(view, (320, 320), interpolation=cv2.INTER_LINEAR)
                clip.append(view_s3)
                if len(clip) > stage3.clip_len:
                    clip = clip[-stage3.clip_len :]

                stage3_ctr += 1
                if len(persons) >= 2:
                    two_p_ctr += 1
                else:
                    two_p_ctr = 0

                can_classify = (
                    active
                    and (two_p_ctr >= args.min_2p_frames)
                    and (len(clip) >= stage3.clip_len)
                    and (stage3_ctr % max(1, args.stage3_stride) == 0)
                )

                if can_classify:
                    last_fight_prob = stage3.infer(clip)
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

            draw_text(view, f"fps={fps:.1f}", 10, 25)
            draw_text(view, f"motion={score:.4f} active={1 if active else 0}", 10, 50)
            draw_text(view, f"persons={len(persons)}", 10, 75)
            if stage3 is not None:
                draw_text(view, f"fight_prob={last_fight_prob:.3f} thr={args.fight_thr:.2f}", 10, 100)
                draw_text(view, "FIGHT" if fight_active else "NO_FIGHT", 10, 130)

            if args.show:
                cv2.imshow("fight_live", view)
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