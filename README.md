# MotionFighter-for-YOLO — Nihai Rapor (TR)

Bu sayfa, **run_20260226_045804** çalıştırmasına ait çıktıları README içinde **gömülü** şekilde gösterir:
- ✅ GIF (6–10 sn) önizleme
- ✅ `verify.txt` içeriği (dosya yolları temizlenmiş)
- ✅ `report.csv` içeriği (tabloya dönüştürülmüş, sade)

---

## 🎞 Motion Debug Overlay (6–10 saniye)

![Motion Debug Overlay 6–10s](fight/pipeline/outputs/run_20260226_045804/motion/debug_overlay_6s_10s.gif)

> Dosya: `fight/pipeline/outputs/run_20260226_045804/motion/debug_overlay_6s_10s.gif`

---

## 📄 Final Verification (verify.txt) — Gömülü

### ✅ Karar: **KAVGA TESPİT EDİLDİ**

| Olay | Başlangıç (sn) | Bitiş (sn) | Süre (sn) | Skor | Etiket | Gerekçe | max_clip | oran | clip_sayısı |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| event_001 | 0.0 | 0.0 | 0.0 | 0.002617 | non_fight | score_low | 0.002617 | 0.0 | 1 |
| event_002 | 0.0 | 0.0 | 0.0 | 0.383005 | non_fight | score_low | 0.813965 | 0.4 | 5 |
| event_003 | 0.0 | 0.0 | 0.0 | 0.537231 | fight | borderline_with_evidence | 0.714844 | 0.5 | 2 |
| event_004 | 0.0 | 0.0 | 0.0 | 0.156738 | non_fight | score_low | 0.163330 | 0.0 | 2 |

### 🔎 Kanıt (Why / Evidence)

- **event_001**: non_fight — score(0.003) < thr_borderline(0.45)  
  - top_clips: #0:0.003
- **event_002**: non_fight — score(0.383) < thr_borderline(0.45)  
  - top_clips: #3:0.814, #4:0.596, #2:0.437
- **event_003**: fight — borderline score(0.537) ≥ thr_borderline(0.45) **ve** (max_clip(0.715) ≥ 0.70 **veya** ratio(0.50) ≥ 0.25)  
  - top_clips: #1:0.715, #0:0.360
- **event_004**: non_fight — score(0.157) < thr_borderline(0.45)  
  - top_clips: #1:0.163, #0:0.150

### ✅ Tespit Edilen Kavga Olayı

- **event_003** — skor=0.537231  
  - neden: borderline score(0.537) ≥ thr_borderline(0.45) **ve** (max_clip(0.715) ≥ 0.70 **veya** ratio(0.50) ≥ 0.25)

---

## 📊 Final Report (report.csv) — Gömülü (Sade)

> Dosya: `fight/pipeline/outputs/run_20260226_045804/final/report.csv`  
> Not: Elinde CSV’nin tam içeriği varsa (satırların hepsi), buraya **tam tablo** olarak da gömerim. Şimdilik `verify.txt` tablosundaki ana metriklerle aynı özet gösteriliyor.

| Olay | Skor | Etiket | Gerekçe | max_clip | oran | clip_sayısı |
|---|---:|---|---|---:|---:|---:|
| event_001 | 0.002617 | non_fight | score_low | 0.002617 | 0.0 | 1 |
| event_002 | 0.383005 | non_fight | score_low | 0.813965 | 0.4 | 5 |
| event_003 | 0.537231 | fight | borderline_with_evidence | 0.714844 | 0.5 | 2 |
| event_004 | 0.156738 | non_fight | score_low | 0.163330 | 0.0 | 2 |

---

## 🧠 Karar Mantığı (Okunaklı)

event_003 için karar koşulu:

```text
score >= thr_borderline
VE
( max_clip >= 0.70  VEYA  ratio >= 0.25 )
```

Bu yüzden event_003 **fight** olarak işaretlenir.

---

## 🏃‍♂️ Çalıştırma Komutları

### Stage-2 Export (Motion + YOLO)

```powershell
python -m yolo.src.stage2.run_export_events `
  "sample_2.mp4" `
  -c "motion/configs/motion.yaml" `
  --yolo-config "yolo/configs/yolo.yaml"
```

### Full Pipeline (önceden hesaplandıysa motion + yolo atla)

```powershell
python -m pipeline.run_full --config pipeline/configs/pipeline.yaml --skip-motion --skip-yolo --visualize
```

---

## 📌 Notlar

- GitHub README içinde MP4 çoğu zaman oynatılmadığı için **GIF** önerilir.
- Dosya yolu görünmesi istenmiyorsa rapor/verify çıktılarında path alanları temizlenmelidir (bu sayfada temizlendi).
