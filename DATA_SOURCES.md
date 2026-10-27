# Veri kaynakları

## RDD2022

Resmî kaynak: https://github.com/sekilab/RoadDamageDetector

Bu projede yol hasarı için kullanılacak. İndirilen arşiv `data/external/` altına çıkarılmalı, ardından:

```powershell
python -m src.prepare_rdd --source data/external/RDD2022_United_States --output data/ready/rdd2022_us
```

## TACO

Kaynak: https://huggingface.co/datasets/pictograph/taco

Yerde çöp için kullanılacak. Veri seti CC BY 4.0 olarak belirtilmiştir; atıf bilgisi veri seti kartında korunmalıdır.

Hazır veri eğitim ve geliştirme için kullanılacak; nihai başarı ölçümü ayrı bir test kümesinde yapılmalıdır.
