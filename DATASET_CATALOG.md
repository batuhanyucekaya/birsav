# Veri kataloğu

## Mapillary birleşik segmentasyon kümesi

`data/mapillary_seg_combined/` yerel eğitim kümesidir. Mevcut Mapillary segmentasyon verisi, ek Mapillary Vistas maskeleriyle birleştirilmiştir.

- Toplam: **831 görüntü**
- Eğitim: **673 görüntü**
- Doğrulama: **158 görüntü**
- Sınıflar: `road`, `sidewalk`, `lane_marking`, `trash_can`, `billboard`, `sign`
- Etiket biçimi: YOLO segmentation

Ek kaynak küme `data/mapillary_seg_expanded/` altında tutulur (300 görüntü: 240 train, 60 val). Büyük veri klasörleri GitHub’a gönderilmez; eğitim yerel ortamda yapılır.
