from __future__ import annotations
from typing import Generator, Tuple
import cv2

def frame_generator(source: str) -> Generator[Tuple[float, "cv2.Mat"], None, None]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 
            yield (ts, frame)
    finally:
        cap.release()