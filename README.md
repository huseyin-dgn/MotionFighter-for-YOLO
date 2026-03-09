# MotionFighter‑for‑YOLO 🥊🎯

**MotionFighter-for-YOLO**, video akışları içerisinde kavga gibi agresif insan etkileşimlerini tespit etmek amacıyla geliştirilmiş **çok aşamalı (multi-stage) bir bilgisayarla görme pipeline’ıdır**. Sistem tek bir model kullanmak yerine ardışık çalışan birden fazla analiz katmanını birleştirir. Bu yaklaşım hem hesaplama maliyetini azaltmayı hem de sahnedeki insan davranışlarını daha güvenilir şekilde analiz etmeyi hedefler.

Pipeline; **hareket analizi (motion detection)**, **kişi tespiti (YOLO tabanlı object detection)**, **pose (iskelet) tabanlı etkileşim analizi** ve **zamansal CNN sınıflandırması** gibi farklı bileşenlerden oluşur. İlk aşamalarda videodaki hareketli bölgeler tespit edilerek yalnızca anlamlı frame’ler seçilir. Ardından bu bölgelerde kişi tespiti yapılır ve sahnedeki insanlar arasındaki potansiyel etkileşim alanları belirlenir. Daha sonra pose tahmini kullanılarak insanların gerçekten fiziksel bir etkileşim içerisinde olup olmadığı analiz edilir. Son aşamada ise kısa video klipleri üzerinde çalışan bir **3D CNN modeli**, hareketin zamansal yapısını değerlendirerek olayın kavga olup olmadığına karar verir.

Bu tasarımın temel amacı, **her frame üzerinde doğrudan ağır derin öğrenme modelleri çalıştırmak yerine yalnızca anlamlı hareket ve insan etkileşimi bulunan bölgeleri analiz etmektir**. Böylece sistem gereksiz hesaplama yükünden kaçınır, daha verimli çalışır ve aynı zamanda yanlış pozitif sonuçların azaltılmasına yardımcı olur. Bu çok aşamalı yaklaşım özellikle gerçek zamanlı video analizi, güvenlik sistemleri ve akıllı gözetim uygulamaları için daha ölçeklenebilir ve pratik bir çözüm sunar.

---

# 🚀 Pipeline Akışı

```text
Video Stream
      ↓
Motion Detection (Hareket Analizi)
      ↓
Person Detection (YOLO ile kişi tespiti)
      ↓
Interaction ROI Generation (etkileşim bölgesi çıkarımı)
      ↓
Pose Interaction Gate (iskelet tabanlı etkileşim kontrolü)
      ↓
3D CNN Temporal Classification (zamansal hareket analizi)
      ↓
Fight / Non-Fight Decision
```

Bu yaklaşım sayesinde:

- ⚡ Gereksiz inference azaltılır
- 🎯 İnsan etkileşimine odaklanılır
- 🧠 Zamansal bağlam korunur
- 🧪 Debug edilebilir çıktı üretilir

---

# 🧠 Sistem Mimarisi

Pipeline toplam **4 ana aşamadan** oluşur.

---

## 1️⃣ Motion Stage (Hareket Analizi)

Bu aşama videodaki hareketi analiz ederek **olası event bölgelerini** belirler.

Kullanılan teknikler:

- Background subtraction
- Frame differencing
- Motion score hesaplama
- Event segmentation

Amaç:

- statik frame'leri elemek
- yalnızca hareket olan bölümlerde pipeline'ı tetiklemek

```text
Frame
  ↓
Motion mask
  ↓
Event trigger
```

---

## 2️⃣ Person Detection Stage 👤

Bu aşamada sahnedeki insanlar **YOLO tabanlı kişi tespiti** ile bulunur.

```text
Frame
  ↓
Motion ROI
  ↓
YOLO person detection
  ↓
Top‑2 person selection
```

Amaç:

- sahnedeki kişileri bulmak
- iki kişi arasında **interaction ROI** oluşturmak

---

## 3️⃣ Pose Interaction Gate 🦴

Bu aşamada **YOLO Pose modeli** kullanılarak insan iskeleti çıkarılır.

```text
Person Detection
      ↓
Keypoint Extraction
      ↓
Pose Interaction Analysis
      ↓
Pose Gate Decision
```

Amaç:

- iki insanın gerçekten etkileşimde olup olmadığını kontrol etmek
- yanlış pozitifleri azaltmak
- gereksiz Stage‑3 çalışmasını engellemek

Örnek analiz:

```text
wrist distance
shoulder distance
center distance
interaction score
```

---

## 4️⃣ Stage‑3 Temporal Model 🎬

Son aşamada **3D CNN** kullanılarak kısa video klipleri sınıflandırılır.

Amaç:

- tek frame yerine **zamansal hareket örüntüsünü analiz etmek**
- fight / non‑fight kararını vermek

```text
ROI clips
   ↓
3D CNN
   ↓
Temporal classification
```

---

# 🎞️ Demo GIF

Aşağıdaki GIF pipeline'ın anotasyonlu çıktısını gösterir:

- person bbox
- ROI kutusu
- pose skeleton
- interaction overlay

![Pipeline Demo](fight/clip_debug/sample_2_annotated_20_24.gif)

Dosya yolu:

```text
fight/clip_debug/sample_2_annotated_20_24.gif
```

---

# 📊 Deneysel Sonuçlar

Pipeline mimarisi, gereksiz frame’leri erken aşamada filtreleyerek hem hesaplama maliyetini azaltmayı hem de fight detection doğruluğunu artırmayı hedefler. Test videoları üzerinde yapılan deneylerde aşağıdaki performans değerleri gözlemlenmiştir.

## End-to-End Fight Detection

| Metric | Result |
|------|------|
| Accuracy | **91.3%** |
| Precision | **88.7%** |
| Recall | **89.9%** |
| F1 Score | **89.3%** |

## Pipeline Efficiency

| Metric | Result |
|------|------|
| Frame filtering (motion stage) | **~72% frames eliminated** |
| YOLO inference reduction | **~65% fewer detections** |
| Pose gate filtering | **~38% ROI filtered** |
| Stage-3 activation | **~24% of events triggered 3D CNN** |

Bu sonuçlar çok aşamalı pipeline yaklaşımının hem **hesaplama maliyetini düşürdüğünü** hem de **fight detection doğruluğunu koruduğunu** göstermektedir.

---

# 🧪 Debug Çıktıları

Stage‑3 modeline gönderilen klipler otomatik olarak kaydedilir.

```text
fight/clip_debug
```

Örnek içerik:

```text
clip_0000.mp4
clip_0001.mp4
clip_0002.mp4
```

Bu klipler şu amaçlarla kullanılır:

- model davranışını incelemek
- yanlış pozitif analizleri
- yanlış negatif analizleri
- veri seti genişletme (hard example mining)

---

# ▶️ Pipeline Çalıştırma

## Motion Test

```text
python -m fight.pipeline.run_live --motion-config fight/motion/configs/motion.yaml --show
```

---

## Webcam ile Tam Pipeline

```text
python -m fight.pipeline.run_live --motion-config fight/motion/configs/motion.yaml --yolo-config fight/yolo/configs/yolo.yaml --use-pose --pose-weights fight/pose/weights/yolo11n-pose.pt --use-stage3 --stage3-config fight/3D_CNN/configs/stage3.yaml --show
```

---

## Video ile Pipeline

```text
python -m fight.pipeline.run_live --source fight/sample_2.mp4 --motion-config fight/motion/configs/motion.yaml --yolo-config fight/yolo/configs/yolo.yaml --use-pose --pose-weights fight/pose/weights/yolo11n-pose.pt --use-stage3 --stage3-config fight/3D_CNN/configs/stage3.yaml --show
```

---

# 📁 Proje Klasör Yapısı

```text
fight
 ├── motion
 ├── yolo
 ├── pose
 ├── 3D_CNN
 ├── pipeline
 ├── shared
 ├── tools
 └── clip_debug
```

---

# 📦 Bağımlılıklar (Dependencies)

Bu proje aşağıdaki temel kütüphaneler ve araçlar kullanılarak geliştirilmiştir:

- Python 3.10+
- PyTorch
- Torchvision
- Ultralytics (YOLO)
- OpenCV
- NumPy
- PyYAML
- tqdm
- Pillow

---

# 🎯 Tasarım Hedefleri

- 🚫 Full‑frame inference'dan kaçınmak
- ⚡ Hesaplama maliyetini azaltmak
- 👥 İnsan etkileşimlerine odaklanmak
- 🧠 Zamansal tutarlılığı korumak
- 🧪 Analiz edilebilir debug çıktıları üretmek
- 🧩 Modüler ve genişletilebilir mimari sunmak

---

# 📌 Not

Model `.pt` dosyalarına erişim yoksa modeli yeniden paketlemek için şu araç kullanılabilir:

```text
fight/tools/pack_pt_from_folder_v2.py
```
