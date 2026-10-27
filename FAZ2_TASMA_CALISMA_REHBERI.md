# Faz 2 — Taşan çöp ve yol/kaldırım ilişkisi

## Görsel etiketleme

Her görselde görünen nesnelerin etrafına ayrı YOLO kutusu çizilmeli:

- `container`: konteynerin tamamı
- `overflowing_litter`: konteyner dışına taşmış veya yere yayılmış çöp
- `ground_litter`: konteynerden bağımsız yerde biriken çöp

Bir görselde birden fazla konteyner veya çöp grubu varsa her biri ayrı kutu olmalı. Kutu olmayan temiz bölgeler için etiket dosyası boş bırakılabilir.

Dosya yapısı:

```text
data/overflow_detection/
  images/train/  labels/train/
  images/val/    labels/val/
```

Etiket sınıf sırası: `container=0`, `overflowing_litter=1`, `ground_litter=2`.

## Eğitim

Görseller ve `.txt` etiketleri eklendikten sonra:

```powershell
python -m src.train_overflow_model --epochs 40
```

Model çıktısı `runs/overflow/train/weights/best.pt` olur. Sonraki adımda API bu modeli kullanarak konteyner dışındaki çöp kutularını konteyner kutusuyla ilişkilendirir ve raporu şu biçimde üretir:

> Konteyner taşmış — kaldırımda taşan çöp tespit edildi.

Yol/kaldırım ayrımı için ayrıca yüzey/alan etiketleri gerektiğinden, ilk sürümde görüntüdeki konum kuralı ve sonrasında ayrı bir segmentasyon modeli kullanılacaktır.
