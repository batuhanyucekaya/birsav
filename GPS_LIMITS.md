# GPS sınır kontrolü

Güncel API, GPS kapsamı dışındaki video anlarına konum atamaz ve olaylarda konumu `camera_gps_estimate` olarak işaretler. Referans dosyası boşsa karşılaştırma yapılmaz. Yeni sonuç sözleşmesindeki `pixel_reference_exceeded` ve `within_pixel_reference` yalnızca görüntü pikseli karşılaştırmasıdır; fiziksel boyut veya mevzuata uygunluk kararı değildir. Eski `SINIR ASILDI/UYGUN` metinleri tarihsel demo davranışını anlatır.

Video formuna isteğe bağlı GPS CSV eklenebilir. Desteklenen CSV biçimleri:

```csv
video_time_seconds,latitude,longitude
0,41.0121,28.9762
5,41.0122,28.9763
```

veya:

```csv
timestamp,latitude,longitude
2026-10-27T08:00:00+03:00,41.0121,28.9762
2026-10-27T08:00:05+03:00,41.0122,28.9763
```

`data/reference_limits.json` içine konuma bağlı sınır eklenir:

```json
{
  "locations": [
    {
      "name": "Dükkan X",
      "latitude": 41.0121,
      "longitude": 28.9762,
      "radius_m": 30,
      "max_width_px": 250,
      "max_height_px": 100
    }
  ]
}
```

Sistem %5 tolerans uygular. Ölçü sınırı aşarsa çıktı karesine `SINIR ASILDI`, aşmıyorsa `UYGUN` yazılır ve olay detayına sınır, ölçü, GPS mesafesi ve tolerans eklenir. Şu an değerler piksel cinsindedir; metreye çevirmek için kamera kalibrasyonu gerekir.
