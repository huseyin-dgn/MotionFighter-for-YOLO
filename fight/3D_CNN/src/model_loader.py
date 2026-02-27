import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torchvision


def build_model(num_classes: int = 2) -> torch.nn.Module:
    model = torchvision.models.video.r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resolve_ckpt_path(ckpt_path: str) -> str:
    p = Path(ckpt_path)

    if p.is_absolute() and p.exists():
        return str(p)

    cwd_cand = (Path.cwd() / p).resolve()
    if cwd_cand.exists():
        return str(cwd_cand)

    stage3_root = Path(__file__).resolve().parents[1]

    cand1 = (stage3_root / p).resolve()
    if cand1.exists():
        return str(cand1)

    cand2 = (stage3_root / "weights" / p.name).resolve()
    if cand2.exists():
        return str(cand2)

    raise FileNotFoundError(f"ckpt not found: {ckpt_path}")


def _is_lfs_pointer(path: str) -> bool:
    head = Path(path).read_bytes()[:64]
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _is_zip(path: str) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def load_ckpt(model: torch.nn.Module, ckpt_path: str):
    ckpt_path = _resolve_ckpt_path(ckpt_path)

    if _is_lfs_pointer(ckpt_path):
        raise RuntimeError(f"Checkpoint is a Git LFS pointer, not real weights: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    sd = None
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd = ckpt["model_state_dict"]

    if sd is None:
        sd = ckpt

    model.load_state_dict(sd, strict=False)
    return ckpt


def load_model(ckpt_path: str, device: str = "cuda", num_classes: int = 2) -> torch.nn.Module:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = build_model(num_classes=num_classes)
    load_ckpt(model, ckpt_path)
    model.eval()
    model.to(device)
    return model