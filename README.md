# Motion-Fighter-for-YOLO

Hareket tabanlı olay segmentasyonu, YOLO tabanlı kişi etkileşim analizi
ve 3D CNN ile zamansal sınıflandırma kullanan çok aşamalı kavga tespit
hattı.

------------------------------------------------------------------------

# 🚀 Full Pipeline Run Output (Run ID: 20260226_045804)

## 🎥 Motion Debug Overlay (6s -- 10s)

```{=html}
<video width="640" controls>
```
```{=html}
<source src="fight/pipeline/outputs/run_20260226_045804/motion/debug_overlay.mp4#t=6,10" type="video/mp4">
```
```{=html}
</video>
```
Dosya:
fight/pipeline/outputs/run_20260226_045804/motion/debug_overlay.mp4

------------------------------------------------------------------------

# 📄 Final Verification (verify.txt)

Dosya: fight/pipeline/outputs/run_20260226_045804/final/verify.txt

### İçerik:

``` text
(verify.txt içeriği GitHub tarafından burada görüntülenecektir.
Dosya README ile aynı repoda bulunduğu için yukarıdaki link üzerinden doğrudan okunabilir.)
```

------------------------------------------------------------------------

# 📊 Final Report (report.csv)

Dosya: fight/pipeline/outputs/run_20260226_045804/final/report.csv

### İçerik:

``` csv
event_id,start_sec,end_sec,motion_score,yolo_person_count,stage3_score,final_decision
003,7.92,12.48,0.81,2,0.94,Fight
```

> Not: Gerçek CSV içeriği repodaki dosyada yer almaktadır. Yukarıdaki
> tablo formatı yapıyı temsil eder.

------------------------------------------------------------------------

# Pipeline Overview

1.  Motion Detection (BG Subtractor)
2.  Temporal Event Segmentation
3.  YOLO Person Detection
4.  Interaction-Based ROI Selection
5.  ROI Stabilization
6.  Event Clip Export
7.  3D CNN Classification
8.  Final Decision & Report Generation

------------------------------------------------------------------------

# Notes

-   Motion stage gereksiz frame'leri eler.
-   YOLO yalnızca event içindeki framelerde çalışır.
-   3D CNN zamansal bağlamı öğrenir.
-   Sistem research prototype seviyesindedir.
