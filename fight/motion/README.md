# Motion Gate Module

## 📌 Amaç

Bu modül, kamera görüntüsünde **anlamlı hareket olup olmadığını** tespit eden ön filtre katmanıdır.  
Amaç, gereksiz yere ağır modelleri çalıştırmamak ve sistem yükünü azaltmaktır.

---

# 🎯 Motion Nedir?

Motion (hareket kontrolü), ardışık iki frame arasındaki piksel değişimini analiz ederek:

> “Bu görüntüde kayda değer bir hareket var mı?”

sorusuna cevap verir.

Bu aşamada:

- İnsan algılama yapılmaz
- Nesne sınıflandırma yapılmaz
- Sadece hareket yoğunluğu ölçülür

Eğer hareket düşükse frame **drop edilir**,  
hareket yüksekse frame bir sonraki aşamaya gönderilir.

---

# ⚙️ Bu Adımda Neler Yapılıyor?

1. Kamera veya video kaynağından frame alınır.
2. Frame grayscale'e çevrilir.
3. Bir önceki frame ile piksel farkı hesaplanır.
4. Ortalama fark değeri (`motion_score`) üretilir.
5. `motion_score` belirlenen eşik değeri ile karşılaştırılır.
6. Eşik üzerindeyse PASS, altındaysa DROP kararı verilir.

---

# 📁 Dosya Yapısı ve Açıklamaları

## `configs/`

### `motion.yaml`

Motion threshold, kullanılacak yöntem ve görüntü işleme parametrelerini tanımlar.

---

## `scripts/`

### `run_motion.py`

Motion modülünü tek kamera veya video üzerinde test etmek için çalıştırma scriptidir.

---

## `src/`

### `main.py`

Modülün ana giriş noktasıdır; ingest ve motion pipeline’ını başlatır.

---

## `src/ingest/`

### `cam_reader.py`

RTSP veya video dosyasından frame okuma işlemini gerçekleştirir.

---

## `src/motion/`

### `frame_diff.py`

Ardışık frame’ler arasındaki piksel farkını hesaplayarak motion_score üretir.

### `bg_subtractor.py`

Arka plan çıkarma (MOG2/KNN) yöntemiyle hareket maskesi üretir.

### `gate.py`

motion_score’u eşik ile karşılaştırarak PASS veya DROP kararını verir.

### `roi.py`

Belirli bölgeleri analiz dışında bırakmak için maskeleme işlemi yapar.

---

## `src/utils/`

### `image_ops.py`

Resize, blur, grayscale gibi temel görüntü ön işleme işlemlerini içerir.

### `logger.py`

Loglama ve debug mesajlarını yönetir.

---

# 🧠 Özet

Motion Gate, sistemin ilk savunma hattıdır.  
Hareket olmayan frame’ler elenir.  
Bu sayede sonraki aşamalarda gereksiz hesaplama yapılmaz ve sistem ölçeklenebilir kalır.

python -m src.scripts.run_motion "C:\Users\hdgn5\OneDrive\Masaüstü\fight_detection\V_102.mp4" -c ".\configs\motion.yaml"
