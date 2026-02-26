# Fight Detection Pipeline — Stage2 Yapısı (Motion → YOLO → Export)

Bu repo, **kavga tespiti** için 3 katmanlı bir akış hedefliyor:

- **Stage 1 (motion)**: Videodaki “anlamlı hareket” olan kısımları bulur, gereksiz frame’leri eler.
- **Stage 2 (yolo)**: Motion ile seçilmiş aralıklarda **person detection + tracking** yapar.
  - **Debug mod**: Analiz/ayar/kalite kontrol
  - **Export mod**: Stage 3 için **event clip** üretimi (dataset/zemin)
- **Stage 3 (ileri model)**: Kavga sınıflandırma (CNN/3D-CNN/Transformer vs.) veya eğitilmiş fight detector.

---

## Klasörler ne işe yarıyor?

Aşağıdaki yapı senin mevcut tree’ne göre anlatıldı.

### `motion/`

**Stage 1**. Hareket skorları, segment çıkarımı ve debug çıktıları.

- `motion/configs/motion.yaml`  
  Motion tarafının tüm threshold/parametreleri. (BG subtractor, gate, postprocess, fight thresholds vs.)
- `motion/src/`  
  Motion pipeline’ın kodu.
  - `motion/src/scripts/run_motion.py`  
    Motion’ı tek başına çalıştırmak için script.
  - `motion/src/service/segmenter.py`  
    Skor dizisinden **segment (event)** üretir (thr_on/thr_off + smoothing + merge).
  - `motion/src/motion/*`  
    `bg_subtractor`, `frame_diff`, `gate`, `roi` gibi çekirdek hareket modülleri.
  - `motion/src/utils/*`  
    resize/blur/to_gray ve logger yardımcıları.
- `outputs/motion_debug_txt/*`  
  Motion testlerinin config/log/CSV çıktıları (versiyonlayarak saklıyorsun).

**Amaç:** Motion, YOLO’ya “tüm video” yerine **muhtemel kavga bölgesi** olabilecek aralıkları bırakır.

---

### `yolo/`

**Stage 2**. Motion segmentleri üzerinde person detection, tracking ve event export.

- `yolo/configs/yolo.yaml`  
  YOLO tarafı config (weights, imgsz, conf, iou, tracking, export ayarları).
- `yolo/src/stage2/`  
  Stage2 scriptleri ve core.

#### `yolo/src/stage2/stage2_core.py`

Stage2’nin **ortak motoru**.  
Hem debug hem export scripti buradaki fonksiyonları kullanır.

İçerik (özet):

- Motion skorlarını çıkarma (`compute_motion_scores`)
- Segment çıkarma (`detect_motion_segments`, `build_segment_mask`)
- YOLO inference wrapper (`yolo_infer`, `extract_boxes`)
- Filtreler (`min_box_area_ratio`, `topk_by_area`)
- Tracking TTL mantığı (`track_ttl_update`)
- ROI/crop yardımcıları (`union_roi_top2`, `crop`, `make_writer`)

**Amaç:** Kod tekrarını bitirir. Bug fix/param değişimi tek yerde yapılır.

#### `yolo/src/stage2/run_yolo_on_events.py` (DEBUG)

Stage2’nin **analiz ve kalite kontrol** runner’ı.

Ne yapar?

- Motion → segment bulur
- Segment içindeki frame’lerde YOLO (ve varsa tracking) çalıştırır
- `accepted / rejected` sayar
- CSV log çıkarır
- İstersen `--save-vis` ile görsel basar (bbox çizilmiş)

Çıktı:

- `outputs/yolo_debug/yolo_event_log.csv`
- `outputs/yolo_debug/ok_*.jpg` (opsiyonel)

#### `yolo/src/stage2/run_export_events.py` (EXPORT)

Stage2’nin **üretim/export** runner’ı.

Ne yapar?

- Motion → segment bulur
- Segment içindeki frame’lerde YOLO+tracking ile **ROI** üretir
- Event başına klasör açar, video clip çıkarır:
  - `full.mp4` (full frame event clip)
  - `crop.mp4` (2 kişiye göre union ROI crop event clip)
  - `roi_log.csv` (frame-frame ROI kaydı)
- En üste `manifest.csv` ve `meta.json` yazar

Çıktı dizini:

- `outputs/events/<VIDEO_ADI>/event_001/ ...`
- `outputs/events/<VIDEO_ADI>/manifest.csv`
- `outputs/events/<VIDEO_ADI>/meta.json`

**Amaç:** Stage 3’e gidecek dataset/clip üretimi.

---

### `outputs/`

Pipeline çıktılarının toplandığı yer.

- `outputs/yolo_debug/`  
  Stage2 debug çıktıları (CSV + görsel)
- `outputs/events/`  
  Stage2 export çıktıları (event klasörleri + mp4 + manifest)

---

## Stage2’de “Debug” ve “Export” farkı

### Debug ne zaman?

- “YOLO kaç kişi görüyor?” gibi doğrulama
- `conf/imgsz/min_box_area_ratio` gibi parametre tuning
- Tracking’in katkısını ölçme
- Hangi frameler neden reject oldu görmek

**Debug komutu:**

```powershell
python -m yolo.src.stage2.run_yolo_on_events "V_102.mp4" `
  -c ".\motion\configs\motion.yaml" `
  --yolo-config ".\yolo\configs\yolo.yaml" `
  --save-vis
```

---

## Önerilen çalışma sırası

1. **Motion config** oturt: `motion.yaml` ile segmentler makul mü?
2. **Stage2 Debug**: `run_yolo_on_events` ile person sayımı/kalite kontrol.
3. **Stage2 Export**: `run_export_events` ile event clip üret.
4. **Stage3**: Export edilen clip’lerle fight classifier / action model eğitimi.

---

## Notlar / Pratik ayar mantığı

- `filter.min_box_area_ratio`  
  Küçük/uzak insanları “person” saymasına engel olur ama çok yükselirse insan kaçırır.
- `yolo.imgsz`  
  Büyüdükçe küçük insan kaçırma azalır ama hız düşer.
- `tracking.max_lost_frames`  
  Track kaybolsa bile kısa süre “hold” yapar; ROI stabil olur.
- Export’ta `crop_margin`  
  Union ROI’yı biraz genişletir, yumruk/kol taşması azalır.

---

## Kısa özet

- **motion/** = “Nerede olay var?”
- **yolo stage2 debug** = “Burada kaç kişi var, doğru görüyor mu?”
- **yolo stage2 export** = “Stage3’e event clip üret”

---

## Stage-2 kullanımı için aşağıdaki powershell komutunu çalıştırabilirsiniz.

```text
cd fight_detection

python -m yolo.src.stage2.run_export_events `
 "sample_2.mp4" `
 -c ".\motion\configs\motion.yaml" `
 --yolo-config ".\yolo\configs\yolo.yaml"
```
