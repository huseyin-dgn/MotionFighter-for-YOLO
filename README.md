
# MotionFighter-for-YOLO

MotionFighter-for-YOLO, çok aşamalı (multi-stage) bir kavga tespit mimarisidir. Sistem; düşük seviyeli hareket (motion) analizi ile zaman tabanlı olay segmentasyonu üretir, ardından yalnızca anlamlı bölgelerde ROI tabanlı YOLO kişi tespiti gerçekleştirir ve son aşamada zamansal bağlamı modelleyen karar mekanizması ile nihai sınıflandırmayı yapar. Bu tasarım, full-frame sürekli inference yaklaşımına kıyasla hesaplama maliyetini azaltırken tutarlı ve analiz edilebilir sonuçlar üretmeyi amaçlar.

Projede hafif ve hızlı çıkarım için YOLOv11n modeli kullanılmakta; ön filtreleme aşamasında özel tasarlanmış bir motion segmentasyon mekanizması, karar aşamasında ise olay bazlı değerlendirme yapan 3D CNN tabanlı zamansal sınıflandırma mimarisi yer almaktadır.

Eğer .pt model dosyasına doğrudan erişilemiyorsa, modeli yeniden oluşturmak / paketlemek için:
```text
fight/tools/pack_pt_from_folder_v2.py
```

betiği kullanılabilir.

------------------------------------------------------------------------

# 🧠 Sistem Mimarisi

Pipeline üç ana katmandan oluşur:

## 1️⃣ Motion Stage (Hareket Analizi)

-   Background Subtraction / Frame Differencing
-   Motion score hesaplama
-   Zaman tabanlı event segmentasyonu
-   Gereksiz frame'lerin elenmesi

Amaç: YOLO'nun tüm video boyunca çalışmasını engelleyerek performansı
artırmak.

------------------------------------------------------------------------

## 2️⃣ YOLO Stage (ROI Tabanlı Kişi Tespiti)

-   Motion ile tetiklenen segmentlerde çalışır
-   Full-frame yerine yalnızca ROI üzerinde inference yapılır
-   Interaction-based ROI seçimi uygulanır
-   Frame bazlı ROI log tutulur

Amaç: Hesaplama yükünü azaltmak ve anlamlı bölgeleri analiz etmek.

------------------------------------------------------------------------

## 3️⃣ Final Stage (Olay Bazlı Karar)

-   Event-level skor hesaplama
-   Borderline eşik kontrolü
-   max_clip ve ratio analizi
-   Nihai fight / non_fight kararı
-   CSV / TXT rapor üretimi

------------------------------------------------------------------------

# 🎞 Motion Debug Overlay (6--10 saniye)

Aşağıdaki GIF, motion mask + ROI davranışını 6--10 saniye aralığında
göstermektedir:

![Motion Debug
Overlay](fight/pipeline/outputs/run_20260226_045804/motion/debug_overlay_6s_10s.gif)

------------------------------------------------------------------------

# 📊 Nihai Sonuç Özeti

## ✅ Karar: KAVGA TESPİT EDİLDİ
```text
  Olay        Skor       Etiket      max_clip   oran   clip_sayısı
  ----------- ---------- ----------- ---------- ------ -------------
  event_001   0.002617   non_fight   0.002617   0.0    1
  event_002   0.383005   non_fight   0.813965   0.4    5
  event_003   0.537231   fight       0.714844   0.5    2
  event_004   0.156738   non_fight   0.163330   0.0    2
```

------------------------------------------------------------------------

## 🔎 Karar Mekanizması (event_003)

Her olay (event) için sistem önce olay bazlı ortalama skor (`score`) hesaplar.  
Ardından iki aşamalı bir karar mekanizması uygulanır:


### 1️⃣ Eşik Kontrolü (Borderline Threshold)

Olayın kavga adayı olarak değerlendirilebilmesi için ortalama skorun belirlenen eşik değeri aşması gerekir:

```text
score ≥ thr_borderline
```

### 2️⃣ Güçlü Zamansal Kanıt Kontrolü

Eşik koşulu sağlandıktan sonra, olayın gerçekten kavga olarak etiketlenebilmesi için aşağıdaki koşullardan en az biri sağlanmalıdır:
```text
max_clip ≥ 0.70
VEYA
ratio ≥ 0.25
```

Tanımlar:

- max_clip → Olay içerisindeki en yüksek tekil clip skorudur.

- ratio → Pozitif (yüksek skorlu) clip’lerin toplam clip sayısına oranıdır.

#### ✅ event_003 İçin Sonuç

* Ortalama skor eşik değerinin üzerindedir.

- En az bir clip güçlü kanıt üretmiştir (max_clip ≥ 0.70).

**Bu nedenle event_003, nihai karar mekanizması tarafından fight olarak sınıflandırılmıştır.**

------------------------------------------------------------------------

# 🚀 Çalıştırma

## Stage-2 (Motion + YOLO)

``` powershell
python -m yolo.src.stage2.run_export_events `
  "sample_2.mp4" `
  -c "motion/configs/motion.yaml" `
  --yolo-config "yolo/configs/yolo.yaml"
```

## Full Pipeline

``` powershell
python -m pipeline.run_full --config pipeline/configs/pipeline.yaml --skip-motion --skip-yolo --visualize
```

------------------------------------------------------------------------

# 📁 Çıktı Yapısı

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

# 🎯 Tasarım Hedefleri

-   Full-frame inference'dan kaçınmak
-   Hesaplama maliyetini düşürmek
-   Zamansal tutarlılığı korumak
-   Analiz edilebilir log üretmek
-   Modüler ve genişletilebilir yapı sunmak
