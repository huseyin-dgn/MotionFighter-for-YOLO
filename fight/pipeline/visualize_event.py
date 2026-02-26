import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def _read_clip_scores(csv_path: Path) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = row.get("event_id", "")
            p = row.get("fight_prob", "")
            if not eid:
                continue
            try:
                fp = float(p)
            except Exception:
                continue
            out.setdefault(eid, []).append(fp)
    return out


def _find_crop(events_root: Path, event_id: str) -> Path:
    p1 = events_root / event_id / "crop.mp4"
    p2 = events_root / event_id / "crop.avi"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"crop not found for {event_id} under {events_root}")


def annotate_event_video(
    events_root: str,
    event_id: str,
    out_path: str,
    label: str,
    reason: str,
    event_score: float,
    clip_scores: List[float],
    fps_out: float = 30.0,
) -> str:
    events_root_p = Path(events_root)
    crop = _find_crop(events_root_p, event_id)

    cap = cv2.VideoCapture(str(crop))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {crop}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps_in > 1e-3:
        fps_out = fps_in

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_p), fourcc, fps_out, (w, h))

    clip_len = 32
    n_clips = max(1, len(clip_scores))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_per_clip = max(1, total_frames // n_clips)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        clip_idx = min(n_clips - 1, i // frames_per_clip)
        clip_p = clip_scores[clip_idx] if clip_scores else 0.0

        header1 = f"{event_id}  label={label}  score={event_score:.3f}"
        header2 = f"reason={reason}"
        header3 = f"clip={clip_idx+1}/{n_clips}  clip_p={clip_p:.3f}"

        cv2.putText(frame, header1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, header2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, header3, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        bar_w = int((w - 20) * max(0.0, min(1.0, clip_p)))
        cv2.rectangle(frame, (10, h - 30), (w - 10, h - 10), (255, 255, 255), 2)
        cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 10), (255, 255, 255), -1)

        vw.write(frame)
        i += 1

    cap.release()
    vw.release()
    return str(out_p)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-root", type=str, required=True)
    ap.add_argument("--event-id", type=str, required=True)
    ap.add_argument("--clip-csv", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--label", type=str, default="fight")
    ap.add_argument("--reason", type=str, default="")
    ap.add_argument("--event-score", type=float, default=0.0)
    args = ap.parse_args()

    clip_scores_by_event = _read_clip_scores(Path(args.clip_csv))
    clip_scores = clip_scores_by_event.get(args.event_id, [])

    p = annotate_event_video(
        events_root=args.events_root,
        event_id=args.event_id,
        out_path=args.out,
        label=args.label,
        reason=args.reason,
        event_score=float(args.event_score),
        clip_scores=clip_scores,
    )
    print(p)


if __name__ == "__main__":
    main()