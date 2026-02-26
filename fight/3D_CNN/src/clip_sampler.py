import cv2
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class VideoInfo:
    fps: float
    total_frames: int
    w: int
    h: int


def read_video_info(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoInfo(fps=float(fps), total_frames=total, w=w, h=h)


def sample_indices(total_frames: int, step: int) -> List[int]:
    if total_frames <= 0:
        return []
    step = max(1, int(step))
    return list(range(0, total_frames, step))


def extract_frames_by_indices(video_path: str, indices: List[int]) -> List:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    frames = []
    idx_set = set(indices)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in idx_set:
            frames.append(frame)
        i += 1
    cap.release()
    return frames


def make_clips(frames: List, clip_len: int = 32, stride: int | None = None) -> List[List]:
    if stride is None:
        stride = clip_len

    clips = []
    n = len(frames)
    if n == 0:
        return clips

    if n < clip_len:
        last = frames[-1]
        frames = frames + [last] * (clip_len - n)
        clips.append(frames[:clip_len])
        return clips

    for start in range(0, n - clip_len + 1, stride):
        clips.append(frames[start:start + clip_len])

    if (n - clip_len) % stride != 0:
        clips.append(frames[-clip_len:])

    return clips


def load_event_clips(video_path: str, clip_len: int = 32, fps_sample: int = 16) -> Tuple[VideoInfo, List[List]]:
    info = read_video_info(video_path)
    fps = info.fps if info.fps and info.fps > 1e-3 else 30.0
    step = max(1, int(round(fps / float(fps_sample))))
    idx = sample_indices(info.total_frames, step=step)
    frames = extract_frames_by_indices(video_path, idx)
    clips = make_clips(frames, clip_len=clip_len, stride=clip_len)
    return info, clips