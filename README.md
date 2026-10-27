# Araç Kamerasıyla Kent Sorunları Tespiti

İlk akış, konumlu bir videoyu okuyup kareleri zaman ve GPS bilgisiyle eşleştirir. Ham videolar yerel çalışma makinesinde tutulur; Git'e alınmaz.

## Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Video ve GPS formatı

Video dosyasını `data/raw/` altına koyun. GPS yan dosyası aynı ada sahip `.csv` dosyasıdır:

```csv
timestamp,latitude,longitude
2026-10-27T08:00:00+03:00,41.012345,28.987654
2026-10-27T08:00:01+03:00,41.012400,28.987700
```

`timestamp` ISO-8601 biçiminde olmalıdır. GPS örnekleri arasındaki konum, kare zamanına doğrusal olarak yaklaştırılır.

## İlk akışı çalıştırma

```powershell
python -m src.ingest --video data/raw/route.mp4 --gps data/raw/route.csv --output data/outputs/route.jsonl --sample-every 5
```

Her satır bir örneklenen kareyi temsil eder ve `frame_index`, `video_time_seconds`, `timestamp`, `latitude`, `longitude`, `image_path` alanlarını içerir. Görüntü kırpıntıları `data/processed/frames/` altında üretilir.

## Hazır veri setlerini hazırlama

RDD2022 arşivi `data/external/` altına çıkarıldıktan sonra Pascal VOC anotasyonlarını YOLO formatına dönüştürün:

```powershell
python -m src.prepare_rdd --source data/external/RDD2022_United_States --output data/ready/rdd2022_us
```

Kaynak ve lisans notları `data/external/SOURCES.md` dosyasındadır. Hazır veri eğitim için, gerçek rota verisi ise son test için ayrılmalıdır.

## Tasarım kararları

İlk sürümde kare örnekleme saniye aralığı yerine kare adımıyla yapılır; böylece düşük FPS videolarda davranış öngörülebilir kalır. GPS eşleştirmesi için kare zamanı video başlangıç zamanına eklenir. GPS yan dosyasında başlangıç zamanı yoksa ilk GPS satırı video başlangıcı kabul edilir ve bu karar `--video-start` ile değiştirilebilir.

