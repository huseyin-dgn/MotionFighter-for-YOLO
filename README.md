# MotionFighter-for-YOLO

Multi-stage fight detection pipeline combining:

-   Motion-based event segmentation
-   ROI-driven YOLO person detection
-   Temporal analysis & final decision layer
-   Structured reporting (CSV / JSON outputs)

Designed for efficiency: YOLO does **not** run on full-frame
continuously.\
Motion stage filters irrelevant frames before spatial detection.

------------------------------------------------------------------------

# 🧠 System Architecture

Pipeline Stages:

1.  **Motion Detection (BG Subtractor)**
    -   Frame differencing / background subtraction
    -   Motion score computation
    -   Event segmentation (start--end frames)
2.  **YOLO Stage (ROI-Based)**
    -   Person detection only inside motion regions
    -   Interaction-based ROI selection
    -   ROI stabilization
3.  **Final Stage**
    -   Event-level aggregation
    -   Structured reporting
    -   Optional visualization export

------------------------------------------------------------------------

# 🎞 Motion Debug Overlay (6s--10s)

Below GIF shows motion mask + ROI behavior between seconds 6--10:

![Motion Debug Overlay
6-10s](fight/pipeline/outputs/run_20260226_045804/motion/debug_overlay_6s_10s.gif)

File:
fight/pipeline/outputs/run_20260226_045804/motion/debug_overlay_6s_10s.gif

------------------------------------------------------------------------

# 📊 Final Report (Summarized CSV)

File: fight/pipeline/outputs/run_20260226_045804/final/report.csv

Example (trimmed for readability):

  --------------------------------------------------------------------------
  event_id   start_sec   end_sec   motion_score    person_count   decision
  ---------- ----------- --------- --------------- -------------- ----------
  003        7.92        12.48     0.81            2              Fight

  --------------------------------------------------------------------------

------------------------------------------------------------------------

# 📄 Verification Output

File: fight/pipeline/outputs/run_20260226_045804/final/verify.txt

Example:

    Run ID: run_20260226_045804
    Events detected: 4
    Final decision: Fight detected in event_003
    Confidence: High

------------------------------------------------------------------------

# 🚀 How To Run

## Stage-2 (Motion + YOLO export)

``` powershell
python -m yolo.src.stage2.run_export_events `
  "sample_2.mp4" `
  -c "motion/configs/motion.yaml" `
  --yolo-config "yolo/configs/yolo.yaml"
```

## Full Pipeline (Skip motion & YOLO if already computed)

``` powershell
python -m pipeline.run_full --config pipeline/configs/pipeline.yaml --skip-motion --skip-yolo --visualize
```

------------------------------------------------------------------------

# 📁 Output Structure

fight/pipeline/outputs/run\_`<timestamp>`{=html}/

-   motion/
-   yolo/
-   stage3/
-   final/
    -   report.csv
    -   verify.txt
    -   summary.json
    -   annotated videos

------------------------------------------------------------------------

# 🧩 Design Goals

-   Avoid full-frame YOLO inference
-   Reduce computational overhead
-   Maintain temporal consistency
-   Export structured, analyzable logs
-   Research-oriented modular design

------------------------------------------------------------------------

# Notes

-   Motion stage eliminates irrelevant frames.
-   YOLO runs only on motion-triggered segments.
-   Outputs are reproducible via YAML configs.
-   Suitable for research and prototyping.
