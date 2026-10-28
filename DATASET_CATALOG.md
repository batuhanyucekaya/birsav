# Veri kataloğu

## Denetlenmiş yerel sürümler — 28 Ekim 2026

- `data/ceymo_clean_v1`: 99 eğitim, 1 doğrulama görüntüsü. Doğrulama görselinin eğitimdeki birebir kopyası çıkarıldı. Aynı rota/video grupları bilinmediğinden bağımsız test değildir; eski ağırlıkların geçmişte gördüğü veri geri alınamaz.
- `data/mapillary_seg_dedup_v1`: 629 eğitim, 158 doğrulama görüntüsü. 56.739 kaynak satırdan 37.058 birebir tekrar çıkarıldı; 19.681 anotasyon kaldı. Hazırlayıcı mümkünse hardlink kullanır, etiketler ayrı dosyalardır. Diskler arası taşıma bağlantıları kopyaya çevirebilir; görüntüleri yerinde değiştirmeyin.
- `data/container_localization_draft_v1`: 10 benzersiz görüntüye ait nesne kutusu taslakları. İnsan incelemesi ve durum etiketleri eksiktir; eğitim YAML'i üretilmez ve aktif modele bağlanmaz.
- Taşan çöp eğitim klasörü boştur; bu modül için tamamlanmış eğitim yoktur.

Özgün veri kümeleri korunur. Sürüm değişiklikleri ilgili `repair_manifest.json` dosyalarında kayıtlıdır. Denetim `reports/project_audit.json`, hızlı düzeltme özeti `reports/data_repair_summary.json` altındadır. Hiçbir kümede doğrulanmış bağımsız rota/gün/kamera bazlı saha test bölümü bulunmamıştır.

## Mapillary birleşik segmentasyon kümesi

`data/mapillary_seg_combined/` yerel eğitim kümesidir. Mevcut Mapillary segmentasyon verisi, ek Mapillary Vistas maskeleriyle birleştirilmiştir.

- Toplam: **831 görüntü**
- Eğitim: **673 görüntü**
- Doğrulama: **158 görüntü**
- Sınıflar: `road`, `sidewalk`, `lane_marking`, `trash_can`, `billboard`, `sign`
- Etiket biçimi: YOLO segmentation

Ek kaynak küme `data/mapillary_seg_expanded/` altında tutulur (300 görüntü: 240 train, 60 val). Büyük veri klasörleri GitHub’a gönderilmez; eğitim yerel ortamda yapılır.
