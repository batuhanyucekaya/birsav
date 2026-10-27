# Çoklu konteyner tespiti

Mevcut sınıflandırma modeli tek görüntüye tek sınıf veriyordu. Çoklu konteyner için YOLO detection veri seti gerekir.

## Kullanıcının yapacağı işlem

CVAT, Roboflow veya LabelImg ile her görseldeki her konteynerin etrafına ayrı bir kutu çiz ve sınıfı seç:

- `empty`: görünür atık yok
- `full`: atık var, dışarı taşmıyor
- `overflowing`: atık dışarı taşıyor

Bir görüntüde üç konteyner varsa üç kutu çizilmelidir. Görselleri ve YOLO `.txt` etiketlerini şu yapıya koy:

```text
data/container_detection/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Her görüntünün aynı isimli etiketi olmalı:

```text
images/train/sokak_01.jpg
labels/train/sokak_01.txt
```

Etiket satırı:

```text
class_id center_x center_y width height
```

Sınıf kimlikleri `empty=0`, `full=1`, `overflowing=2`.

Etiketleri koyduktan sonra şu kontrolü çalıştır:

```powershell
python -m src.prepare_container_detection
```

Kontrol geçerse detection eğitimini başlatacağız.
