import os
import torch
import torch.nn as nn
import torchvision


def build_model(num_classes: int = 2) -> torch.nn.Module:
    model = torchvision.models.video.r3d_18(
        weights=torchvision.models.video.R3D_18_Weights.DEFAULT
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_ckpt(model: torch.nn.Module, ckpt_path: str) -> dict:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return ckpt


def load_model(ckpt_path: str, device: str = "cuda", num_classes: int = 2) -> torch.nn.Module:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = build_model(num_classes=num_classes)
    load_ckpt(model, ckpt_path)
    model.eval()
    model.to(device)
    return model