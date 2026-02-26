import os
import json
import glob
import csv
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml

from model_loader import load_model
from clip_sampler import load_event_clips
from transforms import preprocess_clip
from aggregate import aggregate_scores


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError(f"Invalid stage3 yaml: {cfg_path}")
    return cfg


@torch.no_grad()
def infer_events(events_dir: str, out_dir: str, cfg_path: str, weights_path: str | None = None) -> Dict[str, float]:
    cfg = _load_cfg(cfg_path)

    device = cfg.get("model", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.get("model", {}).get("amp", True))
    num_classes = int(cfg.get("model", {}).get("num_classes", 2))

    clip_len = int(cfg.get("input", {}).get("clip_len", 32))
    size = int(cfg.get("input", {}).get("size", 224))
    fps_sample = int(cfg.get("input", {}).get("fps_sample", 16))

    batch_size = int(cfg.get("inference", {}).get("batch", 4))
    agg_mode = str(cfg.get("inference", {}).get("agg", "mean"))

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        amp = False

    if weights_path is None:
        weights_path = cfg.get("model", {}).get("ckpt_path", "")

    _ensure_dir(out_dir)

    model = load_model(weights_path, device=device, num_classes=num_classes)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    event_dirs = sorted([p for p in glob.glob(os.path.join(events_dir, "event_*")) if os.path.isdir(p)])
    if not event_dirs:
        raise RuntimeError(f"No event_* dirs found under: {events_dir}")

    clip_csv_path = os.path.join(out_dir, "clip_scores.csv")
    event_json_path = os.path.join(out_dir, "event_scores.json")

    clip_rows = []
    event_scores: Dict[str, float] = {}

    for ev_dir in event_dirs:
        ev_id = os.path.basename(ev_dir)
        crop_path = os.path.join(ev_dir, "crop.mp4")
        if not os.path.isfile(crop_path):
            alt = os.path.join(ev_dir, "crop.avi")
            if os.path.isfile(alt):
                crop_path = alt
            else:
                continue

        _, clips = load_event_clips(crop_path, clip_len=clip_len, fps_sample=fps_sample)

        clip_tensors: List[torch.Tensor] = []
        for clip_frames in clips:
            x = preprocess_clip(clip_frames, size=size)
            clip_tensors.append(x)

        scores = []
        for i in range(0, len(clip_tensors), batch_size):
            batch = torch.stack(clip_tensors[i:i + batch_size], dim=0).to(device)

            if device.startswith("cuda"):
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(batch)
            else:
                logits = model(batch)

            probs = F.softmax(logits, dim=1)[:, 1]
            probs = probs.detach().float().cpu().tolist()

            for j, p in enumerate(probs):
                clip_index = i + j
                fp = float(p)
                scores.append(fp)
                clip_rows.append([ev_id, os.path.basename(crop_path), clip_index, fp])

        event_score = aggregate_scores(scores, mode=agg_mode)
        event_scores[ev_id] = float(event_score)

    with open(clip_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "video", "clip_index", "fight_prob"])
        w.writerows(clip_rows)

    with open(event_json_path, "w", encoding="utf-8") as f:
        json.dump(event_scores, f, indent=2)

    return event_scores