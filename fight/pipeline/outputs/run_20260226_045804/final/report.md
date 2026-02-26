# Final Verification Report

**Verdict:** FIGHT DETECTED 

| event | start(s) | end(s) | dur(s) | score | label | reason | max_clip | ratio | n_clips | crop |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| event_001 | 0.0 | 0.0 | 0.0 | 0.002617 | non_fight | score_low | 0.002617 | 0.0 | 1 | C:/Users/hdgn5/OneDrive/Masaüstü/figth/outputs/events/sample_2/event_001/crop.avi |
| event_002 | 0.0 | 0.0 | 0.0 | 0.383005 | non_fight | score_low | 0.813965 | 0.4 | 5 | C:/Users/hdgn5/OneDrive/Masaüstü/figth/outputs/events/sample_2/event_002/crop.avi |
| event_003 | 0.0 | 0.0 | 0.0 | 0.537231 | fight | borderline_with_evidence | 0.714844 | 0.5 | 2 | C:/Users/hdgn5/OneDrive/Masaüstü/figth/outputs/events/sample_2/event_003/crop.avi |
| event_004 | 0.0 | 0.0 | 0.0 | 0.156738 | non_fight | score_low | 0.16333 | 0.0 | 2 | C:/Users/hdgn5/OneDrive/Masaüstü/figth/outputs/events/sample_2/event_004/crop.avi |

## Why (Evidence)

- **event_001**: non_fight | score(0.003) < thr_borderline(0.45)
  - top_clips: #0:0.003
- **event_002**: non_fight | score(0.383) < thr_borderline(0.45)
  - top_clips: #3:0.814, #4:0.596, #2:0.437
- **event_003**: fight | borderline score(0.537) >= thr_borderline(0.45) AND (max_clip(0.715) >= 0.70 OR ratio(0.50) >= 0.25)
  - top_clips: #1:0.715, #0:0.360
- **event_004**: non_fight | score(0.157) < thr_borderline(0.45)
  - top_clips: #1:0.163, #0:0.150
