import cv2
import numpy as np
import torch

KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

def preprocess_clip(frames_bgr: list[np.ndarray], size: int = 224) -> torch.Tensor:
    processed = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - KINETICS_MEAN) / KINETICS_STD
        processed.append(x)

    clip = np.stack(processed, axis=0)
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()
    return clip