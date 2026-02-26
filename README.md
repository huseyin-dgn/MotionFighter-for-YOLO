# MotionFighter-for-YOLO

Çok aşamalı kavga tespit sistemi:

-   Hareket tabanlı olay segmentasyonu (Motion Stage)
-   ROI tabanlı YOLO kişi analizi (Spatial Stage)
-   Olay bazlı karar üretimi ve raporlama (Final Stage)

Bu belge, **run_20260226_045804** çıktısına ait nihai doğrulama ve rapor
verilerini doğrudan gömülü şekilde içerir.

------------------------------------------------------------------------

# 🎯 Nihai Doğrulama Sonucu

## ✅ KARAR: **KAVGA TESPİT EDİLDİ**

# 📊 Olay Bazlı Sonuç Tablosu

  -------------------------------------------------------------------------------------------------------------
  Olay        Başlangıç   Bitiş    Süre    Skor       Etiket      Sebep             Maksimum   Oran   Clip
              (sn)        (sn)     (sn)                                             Clip              Sayısı
  ----------- ----------- -------- ------- ---------- ----------- ----------------- ---------- ------ ---------
  event_001   0.0         0.0      0.0     0.002617   non_fight   skor_düşük        0.002617   0.0    1

  event_002   0.0         0.0      0.0     0.383005   non_fight   skor_düşük        0.813965   0.4    5

  event_003   0.0         0.0      0.0     0.537231   fight       sınırda_kanıtlı   0.714844   0.5    2

  event_004   0.0         0.0      0.0     0.156738   non_fight   skor_düşük        0.16333    0.0    2
  -------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 🔎 Kanıt Analizi 

### event_001

-   Etiket: non_fight\
-   Skor: 0.003\
-   Açıklama: skor(0.003) \< eşik(0.45)\
-   En yüksek clip: #0 → 0.003

### event_002

-   Etiket: non_fight\
-   Skor: 0.383\
-   Açıklama: skor(0.383) \< eşik(0.45)\
-   En yüksek clip skorları:
    -   #3 → 0.814\
    -   #4 → 0.596\
    -   #2 → 0.437

### event_003 ← **Kavga Olayı**

-   Etiket: fight\
-   Skor: 0.537\
-   Karar Mantığı:

```{=html}
<!-- -->
```
    skor(0.537) ≥ eşik(0.45)
    VE
    (max_clip(0.715) ≥ 0.70
     VEYA
     oran(0.50) ≥ 0.25)

-   En yüksek clip skorları:
    -   #1 → 0.715\
    -   #0 → 0.360

### event_004

-   Etiket: non_fight\
-   Skor: 0.157\
-   Açıklama: skor(0.157) \< eşik(0.45)\
-   En yüksek clip:
    -   #1 → 0.163

------------------------------------------------------------------------

# 🚨 Tespit Edilen Kavga Olayı

**event_003**\
Skor: 0.537231

Karar nedeni:

Sınırda skor ≥ eşik VE güçlü clip kanıtı (max_clip ≥ 0.70 veya oran ≥
0.25).

------------------------------------------------------------------------

# 🧠 Sistem Karar Mantığı

1.  Ortalama olay skoru hesaplanır\
2.  Borderline eşik kontrol edilir\
3.  Maksimum clip skoru değerlendirilir\
4.  Pozitif clip oranı analiz edilir\
5.  Nihai karar üretilir

------------------------------------------------------------------------

# 📌 Özet

-   Toplam 4 olay analiz edildi\
-   3 olay kavga dışı olarak sınıflandırıldı\
-   1 olay (event_003) kavga olarak işaretlendi\
-   Karar güçlü clip kanıtı ile desteklendi

------------------------------------------------------------------------

Bu çıktı, Motion + YOLO + olay bazlı karar mekanizmasının birleşik
sonucudur.
