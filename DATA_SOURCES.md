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

## Demo görüntüleri

`data/demo/road_damaged.jpg` ve `data/demo/road_clean.jpg`, RDD2022’nin Hugging Face aynasından yalnızca yerel demo testi için alınmıştır. İkinci görüntü veri etiketinde boş olsa da fiziksel olarak kusursuz olduğu garanti edilmez; modelin yanlış alarm davranışını göstermek için kullanılır. Kaynak: https://huggingface.co/datasets/dronefreak/RDD2022
