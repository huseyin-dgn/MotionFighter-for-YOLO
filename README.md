# Motion-Fighter-for-YOLO

Hareket tabanlı olay segmentasyonu ve YOLO tabanlı kişi etkileşim analizi kullanan çok aşamalı kavga tespit hattı.

---

## 📁 Proje Dosya Yapısı

```text
motion_detection/
│
├── yolo11n.pt
│
├── motion/
│   ├── README.md
│   ├── RUN.txt
│   │
│   ├── configs/
│   │   ├── motion.yaml
│   │   └── motion_yaml.txt
│   │
│   └── src/
│       ├── main.py
│       ├── core/
│       │   └── config.py
│       │
│       ├── ingest/
│       │   └── cam_reader.py
│       │
│       ├── motion/
│       │   ├── bg_subtractor.py
│       │   ├── frame_diff.py
│       │   ├── gate.py
│       │   └── roi.py
│       │
│       ├── scripts/
│       │   └── run_motion.py
│       │
│       ├── service/
│       │   ├── motion_service.py
│       │   └── segmenter.py
│       │
│       └── utils/
│           ├── image_ops.py
│           └── logger.py
│
├── outputs/
│   ├── motion_debug_txt/
│   │   ├── 01/
│   │   ├── 02/
│   │   ├── 03/
│   │   ├── 04/
│   │   ├── 05/
│   │   ├── 06/
│   │   └── 07/
│   │
│   └── yolo_debug/
│       ├── NV_11/
│       │   └── event_001/
│       │       └── roi_log.csv
│       │
│       ├── sample_2/
│       │   └── event_002/
│       │       ├── crop.gif
│       │       ├── crop.mp4
│       │       └── roi_log.csv
│       │
│       ├── V_102/
│       │   └── event_001/
│       │       └── roi_log.csv
│       │
│       └── V_115/
│           └── event_001/
│               └── roi_log.csv
│
└── yolo/
    ├── README.md
    ├── requirements.txt
    ├── configs/
    │   └── yolo.yaml
    │
    └── src/
        └── stage2/
            ├── run_export_events.py
            ├── run_yolo_on_events.py
            └── stage2_core.py
```

---

## 🎥 Stage-2 ROI Çıktısı (Önizleme)

**Dosya:**
- GIF: `motion_detection/outputs/yolo_debug/sample_2/event_002/crop.gif`
- MP4: `motion_detection/outputs/yolo_debug/sample_2/event_002/crop.mp4`

### 🔹 GIF Önizleme

![Stage-2 ROI Crop Preview](motion_detection/outputs/yolo_debug/sample_2/event_002/crop.gif)

### 🔹 MP4 (opsiyonel / bazı ortamlarda görünmeyebilir)

<video src="motion_detection/outputs/yolo_debug/sample_2/event_002/crop.mp4" width="400" controls>
Tarayıcınız video etiketini desteklemiyor.
</video>

> Eğer MP4 burada görünmezse, üstteki dosya yoluna tıklayıp GitHub üzerinden açabilirsiniz.

---

## 📄 ROI Frame Log (CSV)

**Dosya:** `motion_detection/outputs/yolo_debug/sample_2/event_002/roi_log.csv`

[📥 CSV’yi indir](motion_detection/outputs/yolo_debug/sample_2/event_002/roi_log.csv)

<div style="max-height:320px; overflow:auto; border:1px solid #d0d7de; border-radius:8px; padding:10px;">

<table>
<thead>
<tr>
<th>proc_i</th>
<th>ts</th>
<th>det_count</th>
<th>track_count</th>
<th>roi_x1</th>
<th>roi_y1</th>
<th>roi_x2</th>
<th>roi_y2</th>
<th>roi_source</th>
<th>roi_score</th>
<th>roi_iou_prev</th>
<th>pair_idx</th>
<th>jump_accepted</th>
</tr>
</thead>
<tbody>
<tr><td>238</td><td>7.926167</td><td>2</td><td>2</td><td>325</td><td>94</td><td>430</td><td>226</td><td>pair</td><td>0.8305</td><td>0.0000</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>239</td><td>7.959500</td><td>2</td><td>2</td><td>323</td><td>94</td><td>429</td><td>226</td><td>pair</td><td>0.8212</td><td>0.9273</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>240</td><td>7.992833</td><td>2</td><td>2</td><td>321</td><td>94</td><td>428</td><td>226</td><td>pair</td><td>0.8162</td><td>0.9129</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>241</td><td>8.026167</td><td>2</td><td>2</td><td>319</td><td>94</td><td>427</td><td>227</td><td>pair</td><td>0.8144</td><td>0.9085</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>242</td><td>8.059500</td><td>2</td><td>2</td><td>318</td><td>94</td><td>426</td><td>227</td><td>pair</td><td>0.8142</td><td>0.9239</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>243</td><td>8.092833</td><td>2</td><td>2</td><td>317</td><td>93</td><td>425</td><td>227</td><td>pair</td><td>0.8142</td><td>0.9323</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>244</td><td>8.126167</td><td>2</td><td>2</td><td>316</td><td>92</td><td>424</td><td>228</td><td>pair</td><td>0.8148</td><td>0.8984</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>245</td><td>8.159500</td><td>2</td><td>2</td><td>316</td><td>92</td><td>423</td><td>229</td><td>pair</td><td>0.8301</td><td>0.9364</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>246</td><td>8.192833</td><td>2</td><td>2</td><td>316</td><td>92</td><td>422</td><td>230</td><td>pair</td><td>0.8310</td><td>0.9451</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>247</td><td>8.226167</td><td>2</td><td>2</td><td>315</td><td>93</td><td>421</td><td>231</td><td>pair</td><td>0.8261</td><td>0.8975</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>248</td><td>8.259500</td><td>2</td><td>2</td><td>314</td><td>94</td><td>419</td><td>232</td><td>pair</td><td>0.8304</td><td>0.8906</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>249</td><td>8.292833</td><td>2</td><td>2</td><td>314</td><td>95</td><td>417</td><td>233</td><td>pair</td><td>0.8425</td><td>0.8808</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>250</td><td>8.326167</td><td>3</td><td>3</td><td>314</td><td>96</td><td>415</td><td>234</td><td>pair</td><td>0.9245</td><td>0.8794</td><td>(1, 2)</td><td>0</td></tr>
<tr><td>251</td><td>8.359500</td><td>3</td><td>3</td><td>313</td><td>97</td><td>413</td><td>235</td><td>pair</td><td>0.9285</td><td>0.8818</td><td>(1, 2)</td><td>0</td></tr>
<tr><td>252</td><td>8.392822</td><td>2</td><td>3</td><td>312</td><td>98</td><td>410</td><td>234</td><td>pair</td><td>0.9098</td><td>0.8699</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>253</td><td>8.426156</td><td>2</td><td>3</td><td>314</td><td>98</td><td>407</td><td>234</td><td>pair</td><td>0.8863</td><td>0.8610</td><td>(0, 1)</td><td>0</td></tr>
<tr><td>254</td><td>8.459489</td><td>1</td><td>3</td><td>314</td><td>98</td><td>407</td><td>234</td><td>hold</td><td>0.0000</td><td>1.0000</td><td></td><td>0</td></tr>
<tr><td>255</td><td>8.492822</td><td>1</td><td>3</td><td>314</td><td>98</td><td>407</td><td>234</td><td>hold</td><td>0.0000</td><td>1.0000</td><td></td><td>0</td></tr>
<tr><td>256</td><td>8.526156</td><td>1</td><td>3</td><td>314</td><td>98</td><td>407</td><td>234</td><td>hold</td><td>0.0000</td><td>1.0000</td><td></td><td>0</td></tr>
<tr><td>257</td><td>8.559489</td><td>1</td><td>3</td><td>314</td><td>98</td><td>407</td><td>234</td><td>hold</td><td>0.0000</td><td>1.0000</td><td></td><td>0</td></tr>
<tr><td>258</td><td>8.592822</td><td>1</td><td>3</td><td>314</td><td>98</td><td>407</td><td>234</td><td>hold</td><td>0.0000</td><td>1.0000</td><td></td><td>0</td></tr>
<tr><td>259</td><td>8.626156</td><td>1</td><td>3</td><td>314</td><td>98</td><td>407</td><td>234</td><td>hold</td><td>0.0000</td><td>1.0000</td><td></td><td>0</td></tr>
</tbody>
</table>

</div>

### Açıklama

Bu CSV dosyası event içindeki her frame için ROI seçim sürecini kaydeder.

Kolon açıklamaları:

- frame_idx → Event içindeki frame numarası
- roi_x1, roi_y1, roi_x2, roi_y2 → ROI koordinatları (xyxy format)
- roi_source → ROI seçim yöntemi
  - pair → interaction scoring ile seçildi
  - top2 → en büyük iki box fallback
  - single → tek kişi fallback
- pair_score → proximity + IoU tabanlı skor
- roi_iou_prev → Önceki frame ROI ile IoU (stabilite metriği)
- jump_accepted → ROI zıplamasının kabul edilip edilmediği

Bu log ROI davranışını, stabiliteyi ve seçim doğruluğunu analiz etmek için kullanılır.

---

## 🎬 Full Pipeline Output (Run: run_20260226_045804)

### 🎥 Motion Debug Overlay (6s–10s)

<video src="fight/pipeline/outputs/run_20260226_045804/motion/debug_overlay.mp4#t=6,10" width="640" controls>
Tarayıcınız video etiketini desteklemiyor.
</video>

> Eğer video README içinde görünmezse, dosyaya buradan ulaş:  
`fight/pipeline/outputs/run_20260226_045804/motion/debug_overlay.mp4`

---

### 📄 Final Verification (verify.txt)

Dosya: `fight/pipeline/outputs/run_20260226_045804/final/verify.txt`

> Bu bölüm GitHub Actions tarafından otomatik doldurulacak. (Elle kopyalama yok.)

<!-- AUTO:VERIFY_TXT:START -->
<!-- AUTO:VERIFY_TXT:END -->

---

### 📊 Final Report (report.csv)

Dosya: `fight/pipeline/outputs/run_20260226_045804/final/report.csv`

> Bu bölüm GitHub Actions tarafından otomatik doldurulacak. (Elle kopyalama yok.)

<!-- AUTO:REPORT_CSV:START -->
<!-- AUTO:REPORT_CSV:END -->

---

# Pipeline Overview

1. Motion Detection (BG Subtractor)
2. Temporal Event Segmentation
3. YOLO Person Detection
4. Interaction-Based ROI Selection
5. ROI Stabilization
6. Event Crop Export
7. Frame-Level ROI Logging

---

# Notes

- Motion stage gereksiz frame’leri eler.
- YOLO sadece event içindeki framelerde çalışır.
- ROI selection interaction tabanlıdır.
- Sistem research prototype seviyesindedir.
