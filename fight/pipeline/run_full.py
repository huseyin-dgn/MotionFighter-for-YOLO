from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _now_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copytree_overwrite(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    _mkdir(dst)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        out = dst / rel
        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, out)


def _run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"YAML parsed as None (empty/invalid): {p}")
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML must be a mapping/dict, got {type(cfg)}: {p}")
    return cfg


def _save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _exists_or_raise(p: Path, msg: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{msg}: {p}")


def _read_clip_scores(csv_path: Path) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = (row.get("event_id", "") or "").strip()
            p = (row.get("fight_prob", "") or "").strip()
            if not eid:
                continue
            try:
                fp = float(p)
            except Exception:
                continue
            out.setdefault(eid, []).append(fp)
    return out


def _read_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not manifest_path.exists():
        return out
    with open(manifest_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = (row.get("event_id") or row.get("event") or row.get("id") or "").strip()
            if not eid:
                continue

            def _to_int(x: str, default: int = -1) -> int:
                try:
                    return int(float(x))
                except Exception:
                    return default

            def _to_float(x: str, default: float = 0.0) -> float:
                try:
                    return float(x)
                except Exception:
                    return default

            out[eid] = {
                "start_frame": _to_int((row.get("start_frame", row.get("start", "")) or "").strip(), -1),
                "end_frame": _to_int((row.get("end_frame", row.get("end", "")) or "").strip(), -1),
                "start_s": _to_float((row.get("start_s", row.get("t0", "")) or "").strip(), 0.0),
                "end_s": _to_float((row.get("end_s", row.get("t1", "")) or "").strip(), 0.0),
                "frames": _to_int((row.get("frames", row.get("len", "")) or "").strip(), -1),
            }
    return out


def _find_crop(events_root: Path, event_id: str) -> Path:
    p1 = events_root / event_id / "crop.mp4"
    p2 = events_root / event_id / "crop.avi"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"crop not found: {events_root}/{event_id}/crop.(mp4|avi)")


def annotate_event_video(
    events_root: str,
    event_id: str,
    out_path: str,
    label: str,
    reason: str,
    why: str,
    event_score: float,
    clip_scores: List[float],
) -> str:
    import cv2

    events_root_p = Path(events_root)
    crop = _find_crop(events_root_p, event_id)

    cap = cv2.VideoCapture(str(crop))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {crop}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h))

    n_clips = max(1, len(clip_scores))
    frames_per_clip = max(1, total_frames // n_clips)

    def _wrap(text: str, max_len: int) -> List[str]:
        if not text:
            return []
        words = text.split()
        lines: List[str] = []
        cur = ""
        for wd in words:
            nxt = wd if not cur else (cur + " " + wd)
            if len(nxt) > max_len:
                if cur:
                    lines.append(cur)
                cur = wd
            else:
                cur = nxt
        if cur:
            lines.append(cur)
        return lines

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

        y = 112
        for ln in _wrap(why, 55)[:3]:
            cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y += 24

        bar_w = int((w - 20) * max(0.0, min(1.0, clip_p)))
        cv2.rectangle(frame, (10, h - 30), (w - 10, h - 10), (255, 255, 255), 2)
        cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 10), (255, 255, 255), -1)

        vw.write(frame)
        i += 1

    cap.release()
    vw.release()
    return str(out_p)


@dataclass
class PipelinePaths:
    root: Path
    video: Path
    motion_cfg: Path
    yolo_cfg: Path
    stage3_cfg: Path
    stage3_ckpt: Path
    motion_out: Path
    events_root: Path


def _resolve_paths(repo_root: Path, cfg: dict) -> PipelinePaths:
    video = repo_root / cfg["video"]

    motion_cfg = repo_root / cfg.get("motion", {}).get("config", "motion/configs/motion.yaml")
    yolo_cfg = repo_root / cfg["yolo"]["config"]

    stage3_cfg = repo_root / cfg["stage3"]["config"]
    stage3_ckpt = repo_root / cfg["stage3"]["weights"]

    motion_out = repo_root / cfg.get("motion", {}).get("out_dir", "motion/out_motion")

    video_stem = Path(cfg["video"]).stem
    events_root = repo_root / cfg.get("yolo", {}).get("events_root", f"outputs/events/{video_stem}")

    return PipelinePaths(
        root=repo_root,
        video=video,
        motion_cfg=motion_cfg,
        yolo_cfg=yolo_cfg,
        stage3_cfg=stage3_cfg,
        stage3_ckpt=stage3_ckpt,
        motion_out=motion_out,
        events_root=events_root,
    )


def run_motion(paths: PipelinePaths, do_run: bool, cfg: dict) -> None:
    if not do_run:
        print("[SKIP] Motion run disabled.")
        return

    motion_cfg = cfg.get("motion", {}) if isinstance(cfg, dict) else {}
    out_dir = motion_cfg.get("out_dir", "motion/out_motion")
    thr = motion_cfg.get("thr", None)
    debug_video = bool(motion_cfg.get("debug_video", False))

    cmd = ["python", "motion/run_motion_fixed.py", str(paths.video)]
    if out_dir:
        cmd += ["--out", str(paths.root / out_dir)]
    if thr is not None:
        cmd += ["--thr", str(thr)]
    if debug_video:
        cmd += ["--debug-video"]

    _run_cmd(cmd, cwd=paths.root)

    seg = (paths.root / out_dir) / "segments.json"
    _exists_or_raise(seg, "Motion segments.json not produced")


def run_yolo(paths: PipelinePaths, do_run: bool) -> None:
    if not do_run:
        print("[SKIP] YOLO run disabled.")
        return

    _exists_or_raise(paths.motion_cfg, "Motion config not found (required by yolo exporter)")
    cmd = [
        "python",
        "-m",
        "yolo.src.stage2.run_export_events",
        str(paths.video),
        "-c",
        str(paths.motion_cfg),
        "--yolo-config",
        str(paths.yolo_cfg),
    ]
    _run_cmd(cmd, cwd=paths.root)

    _exists_or_raise(paths.events_root, "YOLO events_root not found after export")
    ev_dirs = sorted([p for p in paths.events_root.glob("event_*") if p.is_dir()])
    if not ev_dirs:
        raise RuntimeError(f"No event_* folders found under {paths.events_root}")

    ok_crop = any((ev / "crop.mp4").exists() or (ev / "crop.avi").exists() for ev in ev_dirs)
    if not ok_crop:
        raise RuntimeError(f"event_* exists but no crop.(mp4|avi) found under {paths.events_root}")


def run_stage3(paths: PipelinePaths, out_stage3_dir: Path) -> Dict[str, float]:
    stage3_src = paths.root / "3D_CNN" / "src"
    _exists_or_raise(stage3_src, "Stage3 src not found")

    sys.path.insert(0, str(stage3_src))

    try:
        from infer import infer_events
    except Exception as e:
        raise RuntimeError(f"Failed to import Stage3 infer.py from {stage3_src}: {e}") from e

    _mkdir(out_stage3_dir)
    event_scores = infer_events(
        events_dir=str(paths.events_root),
        out_dir=str(out_stage3_dir),
        cfg_path=str(paths.stage3_cfg),
        weights_path=str(paths.stage3_ckpt),
    )
    if not isinstance(event_scores, dict) or not event_scores:
        raise RuntimeError("Stage3 produced empty event_scores")
    return {k: float(v) for k, v in event_scores.items()}


def fuse_final(event_scores: Dict[str, float], stage3_cfg: dict, clip_scores_by_event: Dict[str, List[float]]) -> Dict[str, Any]:
    inf = stage3_cfg.get("inference", {}) if isinstance(stage3_cfg, dict) else {}

    thr_conf = float(inf.get("thr_confident", 0.60))
    thr_border = float(inf.get("thr_borderline", 0.45))

    ev_max_thr = float(inf.get("evidence_max_clip", 0.70))
    ev_clip_thr = float(inf.get("evidence_clip_thr", 0.55))
    ev_ratio_thr = float(inf.get("evidence_ratio", 0.25))

    out: Dict[str, Any] = {}
    for eid, s0 in event_scores.items():
        s = float(s0)
        clips = clip_scores_by_event.get(eid, [])
        max_clip = max(clips) if clips else 0.0
        ratio = 0.0
        if clips:
            ratio = sum(1 for x in clips if x >= ev_clip_thr) / float(len(clips))

        label = "non_fight"
        fight = False
        reason = "score_low"
        why = f"score({s:.3f}) < thr_borderline({thr_border:.2f})"

        if s >= thr_conf:
            label = "fight"
            fight = True
            reason = "confident_score"
            why = f"score({s:.3f}) >= thr_confident({thr_conf:.2f})"
        elif s >= thr_border:
            evidence = (max_clip >= ev_max_thr) or (ratio >= ev_ratio_thr)
            if evidence:
                label = "fight"
                fight = True
                reason = "borderline_with_evidence"
                why = (
                    f"borderline score({s:.3f}) >= thr_borderline({thr_border:.2f}) AND "
                    f"(max_clip({max_clip:.3f}) >= {ev_max_thr:.2f} OR ratio({ratio:.2f}) >= {ev_ratio_thr:.2f})"
                )
            else:
                label = "non_fight"
                fight = False
                reason = "borderline_no_evidence"
                why = (
                    f"borderline score({s:.3f}) >= thr_borderline({thr_border:.2f}) BUT "
                    f"max_clip({max_clip:.3f}) < {ev_max_thr:.2f} AND ratio({ratio:.2f}) < {ev_ratio_thr:.2f}"
                )

        out[eid] = {
            "score": float(s),
            "label": label,
            "fight": bool(fight),
            "reason": reason,
            "why": why,
            "thresholds": {
                "thr_confident": thr_conf,
                "thr_borderline": thr_border,
                "evidence_max_clip": ev_max_thr,
                "evidence_clip_thr": ev_clip_thr,
                "evidence_ratio": ev_ratio_thr,
            },
            "evidence": {
                "num_clips": int(len(clips)),
                "max_clip": float(max_clip),
                "ratio_ge_clip_thr": float(ratio),
            },
        }
    return out


def write_and_print_report(
    out_final: Path,
    events_root: Path,
    event_scores: Dict[str, float],
    decisions: Dict[str, Any],
    clip_scores_by_event: Dict[str, List[float]],
    manifest: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for eid in sorted(event_scores.keys()):
        s = float(event_scores[eid])
        d = decisions.get(eid, {})
        clips = clip_scores_by_event.get(eid, [])
        max_clip = max(clips) if clips else 0.0
        ratio = 0.0
        ev_thr = float(d.get("thresholds", {}).get("evidence_clip_thr", 0.55))
        if clips:
            ratio = sum(1 for x in clips if x >= ev_thr) / float(len(clips))

        m = manifest.get(eid, {})
        t0 = float(m.get("start_s", 0.0))
        t1 = float(m.get("end_s", 0.0))
        dur = max(0.0, t1 - t0)

        crop_path = ""
        try:
            crop_path = str(_find_crop(events_root, eid))
        except Exception:
            crop_path = ""

        top_clips: List[Dict[str, Any]] = []
        if clips:
            idx_sorted = sorted(range(len(clips)), key=lambda i: clips[i], reverse=True)[:3]
            top_clips = [{"clip_index": int(i), "fight_prob": float(clips[i])} for i in idx_sorted]

        rows.append(
            {
                "event_id": eid,
                "crop_path": crop_path,
                "start_s": round(t0, 3),
                "end_s": round(t1, 3),
                "dur_s": round(dur, 3),
                "event_score": round(s, 6),
                "label": d.get("label", "unknown"),
                "reason": d.get("reason", ""),
                "why": d.get("why", ""),
                "max_clip": round(float(max_clip), 6),
                "ratio_ge_clip_thr": round(float(ratio), 6),
                "num_clips": int(len(clips)),
                "top_clips": top_clips,
            }
        )

    fight_rows = [r for r in rows if r["label"] == "fight"]
    verdict = "FIGHT DETECTED " if fight_rows else "NO FIGHT "

    out_final.mkdir(parents=True, exist_ok=True)

    report_csv = out_final / "report.csv"
    fieldnames = [
        "event_id",
        "crop_path",
        "start_s",
        "end_s",
        "dur_s",
        "event_score",
        "label",
        "reason",
        "why",
        "max_clip",
        "ratio_ge_clip_thr",
        "num_clips",
        "top_clips",
    ]
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r0 in rows:
            r = dict(r0)
            r["top_clips"] = json.dumps(r.get("top_clips", []), ensure_ascii=False)
            w.writerow(r)

    report_md = out_final / "report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Final Verification Report\n\n")
        f.write(f"**Verdict:** {verdict}\n\n")
        f.write("| event | start(s) | end(s) | dur(s) | score | label | reason | max_clip | ratio | n_clips | crop |\n")
        f.write("|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|\n")
        for r in rows:
            crop = str(r["crop_path"]).replace("\\", "/")
            f.write(
                f"| {r['event_id']} | {r['start_s']} | {r['end_s']} | {r['dur_s']} | {r['event_score']} | "
                f"{r['label']} | {r['reason']} | {r['max_clip']} | {r['ratio_ge_clip_thr']} | {r['num_clips']} | {crop} |\n"
            )
        f.write("\n## Why (Evidence)\n\n")
        for r in rows:
            f.write(f"- **{r['event_id']}**: {r['label']} | {r['why']}\n")
            if r["top_clips"]:
                tc = ", ".join([f"#{x['clip_index']}:{x['fight_prob']:.3f}" for x in r["top_clips"]])
                f.write(f"  - top_clips: {tc}\n")

    verify_txt = out_final / "verify.txt"
    with open(verify_txt, "w", encoding="utf-8") as f:
        f.write(verdict + "\n")
        if fight_rows:
            f.write("Fight events:\n")
            for r in fight_rows:
                f.write(
                    f"- {r['event_id']}  t={r['start_s']}-{r['end_s']}s  score={r['event_score']}  crop={r['crop_path']}\n"
                )
                f.write(f"  why: {r['why']}\n")

    print("\n========== FINAL VERIFICATION ==========")
    print(verdict)
    print("event_id | start-end(s) | dur(s) | score | label | reason | max_clip | ratio | n_clips")
    for r in rows:
        print(
            f"{r['event_id']:>8} | {r['start_s']:>6}-{r['end_s']:<6} | {r['dur_s']:<5} | {r['event_score']:<8} | "
            f"{r['label']:<9} | {r['reason']:<24} | {r['max_clip']:<8} | {r['ratio_ge_clip_thr']:<6} | {r['num_clips']}"
        )
        if r["label"] == "fight":
            print(f"         WHY: {r['why']}")
            if r["top_clips"]:
                tc = ", ".join([f"#{x['clip_index']}:{x['fight_prob']:.3f}" for x in r["top_clips"]])
                print(f"         TOP_CLIPS: {tc}")
            if r["crop_path"]:
                print(f"         CROP: {r['crop_path']}")
    print("Saved:", report_md)
    print("Saved:", report_csv)
    print("Saved:", verify_txt)
    print("======================================\n")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="pipeline/configs/pipeline.yaml")
    ap.add_argument("--skip-motion", action="store_true")
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--run-name", type=str, default="auto")
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / args.config
    cfg = _load_yaml(cfg_path)
    paths = _resolve_paths(repo_root, cfg)

    _exists_or_raise(paths.video, "Video not found")
    _exists_or_raise(paths.yolo_cfg, "YOLO config not found")
    _exists_or_raise(paths.stage3_cfg, "Stage3 config not found")
    _exists_or_raise(paths.stage3_ckpt, "Stage3 checkpoint not found")

    run_name = _now_run_name() if args.run_name == "auto" else args.run_name
    run_dir = _mkdir(repo_root / "pipeline" / "outputs" / run_name)

    out_motion = _mkdir(run_dir / "motion")
    out_yolo = _mkdir(run_dir / "yolo")
    out_stage3 = _mkdir(run_dir / "stage3")
    out_final = _mkdir(run_dir / "final")

    do_motion = not args.skip_motion
    do_yolo = not args.skip_yolo

    run_motion(paths, do_run=do_motion, cfg=cfg)
    _copytree_overwrite(paths.motion_out, out_motion)

    run_yolo(paths, do_run=do_yolo)
    if paths.events_root.exists():
        _copytree_overwrite(paths.events_root, out_yolo / paths.events_root.name)

    event_scores = run_stage3(paths, out_stage3)

    stage3_cfg = _load_yaml(paths.stage3_cfg)
    clip_scores_by_event = _read_clip_scores(out_stage3 / "clip_scores.csv")

    decisions = fuse_final(event_scores, stage3_cfg, clip_scores_by_event)

    _save_json(out_stage3 / "event_scores.json", event_scores)
    _save_json(out_final / "decisions.json", decisions)

    manifest = _read_manifest(paths.events_root / "manifest.csv")
    rows = write_and_print_report(
        out_final=out_final,
        events_root=paths.events_root,
        event_scores=event_scores,
        decisions=decisions,
        clip_scores_by_event=clip_scores_by_event,
        manifest=manifest,
    )

    if args.visualize:
        for r in rows:
            if r.get("label") == "fight" and r.get("crop_path"):
                eid = str(r["event_id"])
                out_vid = out_final / f"{eid}_annotated.mp4"
                annotate_event_video(
                    events_root=str(paths.events_root),
                    event_id=eid,
                    out_path=str(out_vid),
                    label=str(r.get("label", "")),
                    reason=str(r.get("reason", "")),
                    why=str(r.get("why", "")),
                    event_score=float(r.get("event_score", 0.0)),
                    clip_scores=clip_scores_by_event.get(eid, []),
                )

    summary = {
        "video": str(paths.video),
        "events_root": str(paths.events_root),
        "num_events": len(event_scores),
        "run_dir": str(run_dir),
        "visualize": bool(args.visualize),
    }
    _save_json(out_final / "summary.json", summary)

    print("\n DONE")
    print("Run dir:", run_dir)
    print("Final decisions:", out_final / "decisions.json")
    if args.visualize:
        print("Annotated videos (fight only):", out_final)


if __name__ == "__main__":
    main()