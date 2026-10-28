# Araç Kamerasıyla Kent Sorunları Tespiti

## Güncel durum — 28 Ekim 2026

Deneysel bootstrap konteyner modelinin görüntünün yarısından fazlasını kaplayan kutuları konum kanıtı kabul edilmez; sınıflandırma ve çizimden önce elenir. Bu koruma yalnızca doğrulanmamış konteyner detector'üne uygulanır. `unknown`, konteynerin boş/dolu/taşmış durumunun belirlenemediğini ifade eder; çizimde tespit skoru (`det`) ve durum skoru (`state`) ayrı verilir. Boyut elemesi bir model eğitimi veya doğruluk kalibrasyonu değildir; çok yakın konteynerler için gerçek kutularla doğrulanmış model gerekir.

Kaldırım segmentleri çizim ve video takibinden önce maske tabanlı bölgeler halinde birleştirilir. Örtüşen parçalar tek bölge olur; nesneyle desteklenen kısa örtülme aralıkları yol maskesi kontrolüyle bağlanabilir. Ayrı paralel kaldırımlar ve görünür yol boşlukları birleştirilmez. Kaldırım için parça başına mor dikdörtgen yerine bölgenin dış konturu çizilir. Bu birleştirme orijinal sahne maskelerini veya çöpün zemin ilişkisi hesabını değiştirmez; örtülme bağlantıları sonuçta tahmin olarak işaretlenir.

Değişken kare aralıklarına sahip videolar giriş PTS zaman damgalarıyla sabit FPS çıktısına örneklenir; yalnızca değişken aralık bulunduğu için reddedilmez. GPS ve olay zamanları özgün giriş zamanını kullanır. Çıktıdaki tekrar/atlama sayıları `timing_conversion` içinde döner. Son kare süresi OpenCV tarafından verilmediği için son pozitif zaman aralıklarının medyanından tahmin edilir ve bu bilgi sonuçta belirtilir. Artmayan/geçersiz zaman damgaları hâlâ hata olarak bildirilir.

Uygulama yerel inceleme prototipidir. Yapılan denetimler, geliştirme ölçümleri ve kalan eksikler [teknik ilerleme raporunda](TECHNICAL_PROGRESS_20261028.md) bulunur. Model skoru doğruluk yüzdesi değildir; sahne nesnesi tespiti tek başına sorun/ihlal sayılmaz.

Video analizi artık arka plan kuyruğunda çalışır. `POST /api/analyze-video` HTTP 202 ve iş kimliği döndürür; `GET /api/jobs/{id}` ilerlemeyi bildirir. `POST /api/jobs/{id}/cancel` iptal eder. İşlemden sonra çıktı `review_required` durumundadır; yerel önizleme incelendikten sonra `POST /api/jobs/{id}/review` yerel kullanıcı incelemesini kaydeder. Bu kayıt otomatik gizlilik sertifikası değildir.

Tüm API ve video sonuçları ana sayfanın açtığı yerel oturum çerezine bağlıdır. Oturumlar ve sonuçlar geçicidir; kullanıcı hesabı/yetkilendirme sistemi yerine geçmez. Sunucu yalnızca `127.0.0.1` üzerinde açılır. Sonuçların saklama süresi bir saattir; temizlik en geç sonraki periyodik kontrolle uygulanır. Ham giriş işlem sonrasında ve hata yollarında silinir. Algılanan yüz/plakalar opak maskelenir; tespit başarımı bağımsız veride doğrulanmamıştır.

Harita varsayılan olarak gerçek analiz olaylarını listeler; altı örnek kayıt ayrı kutuyla açılır. GPS bulunmayan olaylara koordinat uydurulmaz. Mevcut modeller `config/models.json` üzerinden yönetilir; zayıf CeyMo ve bootstrap konteyner pilotları kullanıcının talebiyle deneysel olarak aktiftir. Eski ağırlıklar korunmuştur. Mapillary adayı geliştirme kümesinde iyileşmiştir, bağımsız test bulunmadığı için etkinleştirilmemiştir.

Kullanıcının son etkinleştirme talebiyle mevcut modellerin tamamı görsel ve video analizinde açılmıştır. CeyMo ve konteyner bootstrap ağırlıkları deneysel modda çalışır; `localization_validated=false` korunur. Etkinleştirme, eğitim/doğrulama başarımını değiştirmez. Konteyner düşük durum skorunda `unknown` döndürür; sonuçlar inceleme adayıdır. Yeni Mapillary adayı otomatik seçilmemiştir.

Proje başka bir klasöre taşındığında veri yollarını önce `python -m src.relocate_workspace`, ardından çıktıyı kontrol ederek `python -m src.relocate_workspace --apply` ile güncelleyin. Eski eğitim kayıtlarının tarihsel yolları değiştirilmez.

### Yerel doğrulama komutları

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.audit_project
.\.venv\Scripts\python.exe -m src.smoke_models
.\.venv\Scripts\python.exe -m src.demo_server
```

Denetim büyük resimleri okuduğu için birkaç dakika sürebilir. Duman testleri gerçek ağırlıkları çalıştırır; model doğruluğunu ölçen bağımsız testler değildir. Veri ve model değerlendirme araçlarının çıktı dosyaları `reports/` altındadır. `src.evaluate_models` açıkça bir yerel veri kümesi ister; eksik test kümesi yerine otomatik eğitim verisi kullanmaz. Eğitim komutları yeni adaylar üretir ve aktif model manifestini değiştirmez. Eğitim öncesi en az 256–512 MB boş disk alanı denetlenir; büyük deneyler daha fazla alan gerektirir.

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

Güncel ingest akışı tespit edilen yüz/plakaları kare kaydından önce opak maskeler ve `privacy_status=review_required` yazar. Kareler rota adına ait alt klasöre gider; mevcut JSONL veya dolu rota klasörü üzerine yazılmaz. GPS kapsamı dışında konum alanları `null` olur. Saat dilimi ve geçerli video zaman damgaları gereklidir; FPS/zaman bilgisi eksikse uydurulmaz. Dedektörün kaçırdığı yüz/plakalar için insan incelemesi hâlâ gereklidir.

## Hazır veri setlerini hazırlama

RDD2022 arşivi `data/external/` altına çıkarıldıktan sonra Pascal VOC anotasyonlarını YOLO formatına dönüştürün:

```powershell
python -m src.prepare_rdd --source data/external/RDD2022_United_States --output data/ready/rdd2022_us
```

Kaynak ve lisans notları `data/external/SOURCES.md` dosyasındadır. Hazır veri eğitim için, gerçek rota verisi ise son test için ayrılmalıdır.

## Hazır yol hasarı modeliyle çıkarım

RDD2022 üzerinde eğitilmiş model ağırlığı `models/yolo12s_seed0_best.pt` altında tutulur ve Git'e alınmaz. Görsel veya video üzerinde çalıştırmak için:

```powershell
python -m src.road_damage --source data/processed/frames/frame_00000000.jpg
```

Çıktılar `data/outputs/road_damage/annotated/` altında işaretli görüntü, `detections.jsonl` altında makine tarafından okunabilir kayıt olarak oluşur. Modelin benchmark deposu: https://huggingface.co/SreekarAditya/yolo-rdd2022-benchmark

## Geçici demo sitesi

```powershell
python -m src.demo_server
```

Ardından `http://127.0.0.1:8000` adresini açın. Site örnek tespitleri haritada ve listede gösterir; kalıcı veritabanı kullanmaz.

## Tasarım kararları

İlk sürümde kare örnekleme saniye aralığı yerine kare adımıyla yapılır; böylece düşük FPS videolarda davranış öngörülebilir kalır. GPS eşleştirmesi için kare zamanı video başlangıç zamanına eklenir. GPS yan dosyasında başlangıç zamanı yoksa ilk GPS satırı video başlangıcı kabul edilir ve bu karar `--video-start` ile değiştirilebilir.

