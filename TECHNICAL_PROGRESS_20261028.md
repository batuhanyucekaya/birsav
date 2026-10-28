# Kent Sorunları Tespiti — teknik ilerleme ve doğrulama raporu

Tarih: **28 Ekim 2026**. Güncel çalışma klasörü: `E:\birsav`.

> Sonraki kullanıcı talebiyle mevcut CeyMo ve konteyner modelleri deneysel olarak görsel/video analizinde etkinleştirildi; doğrulanmış oldukları iddia edilmez. İlk iterasyonun aşağıdaki kapatma/kabul kararları tarihsel kayıttır. Etkinleştirme sonrası 37 regresyon testi ve gerçek ağırlıklı duman testi geçti; kayıtlar `tmp/activation-tests.log` ve `tmp/activation-model-smoke.log` altındadır.

Bu çalışma mevcut projede kod/veri düzeltmeleri, gerçek ağırlıklarla geliştirme değerlendirmeleri ve kontrollü eğitimler yaptı. **Bağımsız saha doğruluğu veya üretime hazır olma iddiası yoktur.** Özellikle gizlilik recall'u ve fiziksel olay doğruluğu için etiketli, eğitimden bağımsız rota videoları bulunmadı. İstenen kapsamın bu veriye bağlı bölümleri aşağıda tamamlanmamış olarak işaretlendi.

## A. Başlangıç durumu

Başlangıç Git revizyonu `b47545c171b3ad45ea11b56d643d5b993fbe5a0c`. Geri dönüş noktası `codex/baseline-before-reliability-20261028` dalıdır. İlk durumda yalnızca önceki görüşmede üretilen `PROJE_BAGLAM_PROMPTU.md` izlenmeyen dosyaydı. Eski model ağırlıkları ve eski eğitim kayıtları korunmuştur.

İlk envanter [reports/project_audit.json](reports/project_audit.json) içindedir: 15 PyTorch ağırlığı ile 3 gizlilik dosyası; SHA-256, görev, sınıf isimleri, model YAML'i, parametre sayısı, checkpoint metadata'sı, varsa eğitim ayarları, son epoch ve checkpoint metrikleri kaydedildi. Checkpoint içindeki `epoch=-1`, ağırlığın optimizer bilgisi temizlenmiş olmasından kaynaklanabilir; sıfır eğitim anlamına gelmez. Checkpoint metadata'sı bağımsız kaynak doğrulaması değildir.

| Modül | Kanıtlanan durum | Yerel değerlendirme sınırı |
|---|---|---|
| RDD YOLO12s | Ağırlık yükleniyor, gerçek görsel API'sinde çıkarım üretiyor | Etiketli bağımsız yol hasarı testi yok; metadata mAP'si saha skoru değildir |
| TACO YOLO11s | Ağırlık yükleniyor, birleşik analizde çalışıyor | Yerel atık/zemin bağlam etiketleri yok; nesne tespiti yerde çöp kanıtı değildir |
| COCO YOLO11n | Bağlam için kişi/araç adayları üretiyor | Yerel bağlam benchmarkı yok |
| YuNet / plaka Cascade | Dosyalar mevcut, gerçek video hattı çalışıyor | Türkiye plakaları/yüzleri için etiketli recall testi yok |
| Konteyner sınıflandırma | 224 px, 3 doğrulama görselinde accuracy 0,33333; macro F1 0,22222 | Çok küçük küme; fiziksel doluluk ölçümü değildir |
| Konteyner detection | 4 doğrulama görselinde kutu mAP50 0,79600 | Etiketler tam-görsel kutusu; localization başarısı olarak geçersiz |
| CeyMo pilot | Tek doğrulama görselinde yeniden ölçülen AP sıfır | Görsel eğitimde de bulunuyor; üretim kullanımı kapatıldı |
| Mapillary pilot | 158 görselde düşük kutu ve maske AP | Yerel geliştirme değerlendirmesi; bağımsız saha testi değil |

Konteyner sınıflandırıcısının aynı ağırlığı 320 px ile 2/3 doğru sonuç verdi; eğitimde kullanılan 224 px ile 1/3 doğru verdi. Bu fark yeni model kazanımı değildir; üç örnekte ön işleme duyarlılığını gösterir. İki ölçüm ayrı dosyalarda tutuldu.

Donanım: i7-10750H, yaklaşık 16 GB RAM; `nvidia-smi` GTX 1650 Ti için **4096 MiB** bellek gösterdi. Kurulu PyTorch **2.14.1+cpu** olduğundan CUDA kullanılamadı. Yeni deneyler CPU'da, 2 Torch iş parçacığı ve düşük batch ile yürütüldü. Paket sürümleri [reports/environment.json](reports/environment.json), ilk RAM/donanım gözlemi envanter dosyasındadır.

## B. Veri kalitesi ve yapılan düzeltmeler

| Veri | Bulgular | Sonuç |
|---|---|---|
| CeyMo | 100 train / 1 val; `1.jpg` iki kümede aynı piksel içeriğine sahip | `ceymo_clean_v1`: 99 train / 1 val; train kopyası çıkarıldı |
| Konteyner detection | 7 train / 4 val; 11 kutunun tamamı tam görüntü; 1 birebir küme çakışması | Mevcut localization ağırlığı devre dışı; özgün etiketler korundu |
| Mapillary aktif veri | 629 train / 158 val; 56.739 anotasyon satırının 37.058'i birebir tekrar | `mapillary_seg_dedup_v1`: 19.681 tekil anotasyon; 629/158 görüntü |
| Mapillary birleşik | 673 train / 158 val, toplam 831 görüntü; aktif ağırlık bu YAML ile eğitilmemiş | Otomatik aktif veri/model kabul edilmedi |
| Bina verisi | 629/158 görüntü ve eğitim ayarları mevcut | Başlangıç envanterinde tamamlanmış bina ağırlığı yok; aktif özellik sayılmadı |
| Taşan çöp | Eğitim görüntüsü yok | Eğitim yapılmadı; sahte sonuç üretilmedi |

Mapillary kök nedeni: farklı kaynak sınıfların RGB rengi aynı olabiliyor. Eski dönüştürücü her kaynak sınıf için konturu yeniden çıkarıp aynı hedef sınıf poligonunu tekrar yazıyordu. Yeni dönüştürücü renkleri hedef sınıfa göre birleştirip konturları bir kez çıkarır. Tek renkten iki kaynak sınıfın aynı poligonu iki kez üretmediği regresyon testi vardır.

**Nedensellik sınırı:** Ultralytics bazı birebir tekrarları zaten yüklemede elediğinden, bu anotasyon kusurunun düzeltilmesi tek başına AP artışı olarak sunulmadı. Aynı eski ağırlığın özgün/tekrarsız etiketlerdeki geliştirme skorları çok yakın çıktı. Düşük başarımı yalnızca bu hataya bağlamak doğru değildir.

Otomatik denetçi bozuk görüntü, eksik/boş/yetim etiket, sınıf ID'si, sonlu olmayan koordinat, kutu sınırı/alanı, poligon biçimi/alanı, tam-görsel kutusu, tekrarlanan anotasyon ve piksel içeriğine göre birebir küme çakışması kontrol eder. dHash mesafesi ≤3 olan çapraz küme yakın kopyaları inceleme adayı olarak çıkarır. İlk taramada yakın kopya adayı bulunmaması, video/rota sızıntısının olmadığı anlamına gelmez. Video/gün/kamera kaynak manifestleri bulunmadığından grup bağımsızlığı **bilinmiyor**.

Yeni veri sürümleri özgün etiketleri değiştirmez. Görüntüler mümkünse hardlink ile paylaşılır; bu dosyalar yerinde değiştirilmemelidir. Değişiklikler `repair_manifest.json` dosyalarında, hızlı kontrol [reports/data_repair_summary.json](reports/data_repair_summary.json) içindedir. Yeni sürümlerde etiket biçim hatası ve birebir etiket tekrarı sayısı sıfırdır. Bağımsız test sayısı sıfırdır.

Konteynerler için 10 benzersiz fotoğrafta görsel incelemeye dayalı kutu taslakları hazırlandı; çoklu konteyner görülen örnekte ayrı kutular eklendi. Bunlar **insan incelemesi gereken taslaklardır**. Kapalı/içi görünmeyen örneklerin eski `empty` etiketi doğru kabul edilmedi; durum etiketleri aktarılmadı. `data/container_localization_draft_v1` için eğitim YAML'i üretilmedi. İnceleme görseli `tmp/reliability-baseline/container_bbox_draft_review.jpg` altındadır.

## C. Kod düzeltmeleri ve test kanıtları

| Dosya/alan | Problem ve çözüm | Doğrulama |
|---|---|---|
| `src/http_upload.py` | `cgi.FieldStorage` kaldırıldı; streaming multipart, toplam gövde sınırı, alan/header sınırları, 1 MB bellekten sonra geçici dosya, eksik gövde kontrolü | Limit aşımında gövde okunmadan 413; binary/truncated/duplicate alan testleri |
| `src/demo_server.py` | Tüm model çağrıları ortak `RLock` altında; yerel Host/Origin kontrolü, HttpOnly SameSite oturumu, oturum sahipliği ve Range destekli parça sunumu | HTTP 200/202/400/401/403/404/413/206 yolları ve farklı oturum erişim testleri |
| `src/jobs.py` | Tek worker, en fazla 3 aktif/bekleyen iş; ilerleme, iptal, TTL temizliği; disk kontrolü | Kuyruk/iptal/temizlik ve video sırasında statik sayfa yanıtı testleri |
| `src/video_pipeline.py` | Model yükleme dahil tüm yaşam döngüsü `finally`; bozuk/eksik kare, PTS ve FPS kontrolü; tam çıktı decode | Gerçek ağırlıklarla kısa video; bozuk video ve yükleme öncesi model hatası temizliği |
| `src/privacy.py` | Büyük yüzü reddeden filtre ve toplam %20 alan sınırı kaldırıldı; varsayılan opak maske; güncel kutu da kapsanır; yüz/plaka türleri karışmaz | Büyük/kalabalık kutu, yumuşatma gecikmesi ve geçersiz koordinatta tam maske testleri |
| `src/tracking.py` | ID yeniden kullanımını ve türlerin eşleşmesini önleyen sayaç/eşleştirme; gözlenmeyen eski track olay sayısını artırmaz | 20 gözlemin tek aday olması, iki ayrı nesne ve ID tükenme testleri |
| `src/inference.py` | Her konteyner ayrı kırpıntıdan sınıflandırılır; düşük skorda `unknown`; geçersiz bootstrap devre dışı | İki farklı 40×40 kırpıntı ve düşük güven abstention testi; gerçek veri kalitesi ölçümü değil |
| `src/scene_context.py` | Atık için yol/kaldırım maskesindeki destek alanı ve kişi/araç dışlama ipucu; yanlış boyutlu maskede kesin karar yok | Sentetik yüzey desteği, kişi alanı ve mask alignment testleri |
| `src/gps.py` | Sıralı mutlak/bağıl zamanlar, geçerli koordinatlar, tekrar zaman reddi; kapsam dışında konum yok | Enterpolasyon, timezone, NaN ve tekrar zaman testleri |
| `src/ingest.py` | Kaydedilen karelere önce opak maske; rota alt klasörü, overwrite reddi, `.partial` çıktı ve hata temizliği; PTS ile eşleme | Sentetik video ile kare/JSONL/temizlik/GPS kapsam testi; dedektör doğruluğu mock ile ölçülmedi |
| `web/app.js` | Örnekler ayrı; gerçek olaylar harita/listeye bağlandı; GPS olmayan kayıt işaretsiz; iş ilerleme/iptal/yerel inceleme; metinler `textContent` ile | JavaScript syntax kontrolü geçti; tarayıcıda oynatma tamamlanmadı |
| `src/model_registry.py`, `config/models.json` | Açık ağırlık seçimi ve SHA-256; durum/etkinlik kaydı; eğitim aktif modeli değiştirmez | Taşınma sonrası tüm 7 model kaydının checksum'u eşleşti; yanlış hash testi |
| Veri/eğitim scriptleri | Destructive Mapillary klasör silme kaldırıldı; tam-görsel bootstrap üretimi engellendi; boş/bozuk/sızıntılı veriyle eğitim engeli; yeterli boş disk kontrolü | Etiket denetim testleri ve gerçek veri raporları |

RDD dönüştürücüde temiz, anotasyonsuz sahnelerin atılması da düzeltildi: gerçekten boş VOC anotasyonu artık negatif örnek olarak korunur; desteklenmeyen nesneler içeren görüntü temiz yol etiketi yapılmaz. Geçersiz kutu ve dolu çıktı klasörü reddedilir. Bu davranış sentetik VOC girdisiyle test edildi; yeni bir RDD saha modeli eğitildiği anlamına gelmez.

`cgi` değişikliği Python 3.13'te standart modülün kaldırılmasıyla da ilgilidir; gerçek çalışma testi **Python 3.12.14** üzerinde yapıldı, 3.13 için genel uyumluluk testi iddia edilmez. [Python belgeleri](https://docs.python.org/3/library/cgi.html)

Toplam **36 unit/integration testi geçti**: [reports/regression_tests.log](reports/regression_tests.log). Sahte modeller kullanılan testler geometri/kilit/durum davranışını denetler; gerçek doğruluk kanıtı değildir. Gerçek ağırlıkların API/video duman testi ayrı kayıttır: [reports/real_model_smoke.json](reports/real_model_smoke.json).

Eski/yeni takip yardımcılarının aynı sentetik durumla karşılaştırmasında, tüm track'ler kaybolduktan sonraki ID eski kodda yeniden 1 olurken yeni kodda 2 oldu. Çakışan yüz/plaka türleri eski kodda tek track'e birleşirken yeni kod iki ayrı track tuttu. Kanıt: [reports/tracker_before_after.json](reports/tracker_before_after.json). IDF1/HOTA veya gerçek araç videosu takip başarımı ölçülmedi.

## D. Eğitim ve model karşılaştırması

### Mapillary aday eğitimi

Başlangıç ağırlığı aktif `train_advertisement_fix/best.pt`; veri `mapillary_seg_dedup_v1`. Transfer learning kullanıldı; sıfırdan eğitim değildir. İki epoch, 320 px, batch 2, workers 0, CPU, seed 0, AdamW, LR 0,0002, nbs 16, warmup 0; mosaic/flip/MixUp/Copy-Paste/erasing kapalı, cache kapalı. Eğitim yaklaşık **554,29 saniye** sürdü.

Eski ve yeni ağırlık aynı **158 doğrulama görüntüsü**, aynı veri fingerprint'i, 320 px, batch 1, confidence 0,001 ve aynı kütüphane sürümüyle yeniden ölçüldü. Değerlendirme protokolü [Ultralytics Val belgeleri](https://docs.ultralytics.com/modes/val/) ile uyumludur.

| Metrik | Önce | Aday | Mutlak değişim |
|---|---:|---:|---:|
| Kutu precision | 0,372928 | 0,394276 | +0,021348 |
| Kutu recall | 0,110644 | 0,123414 | +0,012770 |
| Kutu mAP50 | 0,087595 | 0,094394 | +0,006799 |
| Kutu mAP50-95 | 0,042552 | 0,048275 | +0,005723 |
| Maske precision | 0,336719 | 0,354324 | +0,017605 |
| Maske recall | 0,079481 | 0,089237 | +0,009756 |
| Maske mAP50 | 0,056724 | 0,064442 | +0,007717 |
| Maske mAP50-95 | 0,023703 | 0,028210 | +0,004507 |

Sınıf bazlı AP50 (geliştirme kümesi):

| Sınıf | Kutu önce → aday | Maske önce → aday |
|---|---:|---:|
| road | 0,22591 → 0,24106 | 0,17282 → 0,18451 |
| sidewalk | 0,05852 → 0,06058 | 0,04146 → 0,04326 |
| lane_marking | 0,17347 → 0,17982 | 0,06751 → 0,07777 |
| trash_can | 0 → 0,00660 | 0 → 0,00660 |
| billboard | 0,02119 → 0,02595 | 0,01159 → 0,02126 |
| sign | 0,04648 → 0,05235 | 0,04696 → 0,05325 |

Özellikle `trash_can`, `billboard` ve `sign` değerleri güvenilir saha kullanımı için yeterli başarı göstermiyor. Ortalama artış bu zayıf sınıfları gizlememelidir.

Kanıtlar: [eski ağırlık değerlendirmesi](reports/comparison_mapillary_baseline.json), [aday değerlendirmesi](reports/comparison_mapillary_candidate.json), [karşılaştırma ve kabul kararı](reports/model_comparison.json), [eğitim ayarları](reports/mapillary_dedup_candidate_training.json). Sınıf bazlı kutu/maske AP, precision/recall ve confusion matrix değerlendirme dosyalarındadır.

Artış küçüktür; mutlak başarım hâlâ düşüktür. Aday **etkinleştirilmedi**. Bağımsız test, alt grup/gizlilik regresyonları ve tekrar seed ölçümleri yoktur. Ek eğitim ile veri/augmentation değişiklikleri birlikte uygulandığı için artış tek bir değişikliğe atfedilemez. Tam augmentation/model-scale ablation çalışması yapılmadı. Road ve sidewalk maskeleriyle bağlam hesabı hâlâ deneysel ve insan incelemesi gerektirir.

### CeyMo kök neden araştırması

Orijinal doğrulama sıfır AP verdi; etiket biçimi/sınıf ID/yollar geçerli bulundu, tek doğrulama örneğinin eğitim kopyası saptandı ve kutular görüntü üzerinde incelendi. 40 epoch mikro deneyinde varsayılan nominal batch birikimi optimizer güncellemelerini sınırladı. nbs=1 ile eğitim kaybı düştü, fakat doğrulama kararsız kaldı. Normalizasyon ve eğitim/çıkarım geometrisi ayrıca araştırıldı.

Aynı tek görsel, aynı seed, 640 px, batch/nbs 1, sabit BatchNorm istatistikleri, kapalı augmentation ve **80 epoch** ile yalnızca LR değiştirilen kontrol:

| LR | Süre | Aynı görselde precision | Recall | mAP50 |
|---|---:|---:|---:|---:|
| 0,003 | 86,56 sn | 0 | 0 | 0 |
| 0,0001 | 84,74 sn | 0,971417 | 1,0 | 0,995 |

Bu, mikro deneyde aşırı öğrenme hızının başarısızlık üretebildiğini ve hattın uygun ayarla iki nesneyi öğrenebildiğini kanıtlar. **Orijinal 100 görsellik pilotun başarısızlığının yalnızca LR'den kaynaklandığını kanıtlamaz.** Orijinal farklı veri, optimizer, augmentation ve warmup kullanmıştır. Yönlü ok sınıflarında yatay çevirmenin semantik riskine dikkat edilmelidir. Sağlam çok görselli, grup bağımsız validation hâlâ gereklidir.

Bu tabloda train ve val aynı tek görseldir; AP=0,995 saha başarısı değildir. Yeni ağırlık yalnızca teşhis çıktısı olarak saklandı. Kanıtlar: [yüksek LR kontrolü](reports/ceymo_overfit_80_nbs1_bnTrue_lr0.003.json), [tamamlanan düşük LR deneyi](reports/ceymo_overfit_low_lr_completed.json). CeyMo pilotu etkinleştirilmedi.

Yol hasarı, TACO, konteyner durum ve gizlilik için uygun bağımsız etiketli veri bulunmadığından doğruluk artırdığı iddia edilen yeni model üretilmedi. Konteyneri geçersiz kutularla tekrar eğitmek yerine veri kapısı kondu. Scratch/fine-tune saha karşılaştırması yapılmadı; veri ve hesaplama gereksinimi sağlanmadan sonuç uydurulmadı.

## E. Gerçek saha başarısı ve doğrulanamayanlar

**Ölçülemedi:** bağımsız saha precision/recall, gece/yağmur/hareket alt grupları, küçük nesne başarımı, yanlış negatif gizlilik oranı, frame-level face/plate recall, maske coverage ground truth'u, konum hatası, event-level precision/recall, duplicate event rate, IDF1/HOTA ve rota bazlı güven aralıkları.

Sabit eşikte yanlış pozitif/negatif inceleme adaylarını çıkaran `src/review_samples.py` eklendi; CeyMo için yerel hata inceleme JSON'i ve maskeli görsel üretildi. Bu öneriler otomatik eğitim etiketi yapılmaz. Düşük skorlu ve tartışmalı örnekler için insan onayı gerekir. Ground-truth olmadan modelin bir tahminini “yanlış pozitif” diye kesinleştirmek doğru değildir.

Threshold seçimi aracı yalnızca validation girdisini kabul eder; test kümesiyle eşik seçimi engellenir. Eksik pozitif sayısı ayrıca hesaba katılabilir. Henüz yeni sınıf eşikleri, güven kalibrasyonu veya “%95 doğruluk” değeri üretilmedi.

## F. Performans ve kaynak kullanımı

İlk gerçek ağırlıklı duman testi: 320×240, 5 kare, 5 FPS sentetik video. Tüm çıktı yeniden decode edildi; 5/5 kare, aynı çözünürlük ve FPS doğrulandı. İşlem yaklaşık 3,63 saniye sürdü. GPU kullanılmadı. Bu kısa örneğin hızı uzun/1080p/4K video performansına genellenemez. Son tekrarın süreleri `reports/real_model_smoke.json` içindedir.

Kaydedilen süreler: görsel endpoint'leri, video toplam/analiz süresi ve model doğrulamasında preprocess/inference/postprocess ms/görüntü. Tepe RAM/VRAM, ayrıntılı decode/encode profili ve karşılaştırılabilir eski/yeni toplam video benchmarkı ölçülmedi. 1/3/5/10 FPS örnekleme ve adaptif örnekleme saha karşılaştırması yapılmadı.

Çıktı ses aktarmaz; kod yalnızca video writer kullanır. **FFprobe kurulu olmadığından bağımsız codec/ses doğrulaması yapılmadı**; raporda `ffprobe=null` kalır. Destek eklendi: araç varsa stream bilgisi okunur ve ses akışı reddedilir. Tarayıcı otomasyon aracı `failed to write kernel assets` hatası verdi; bu nedenle gerçek tarayıcı playback E2E testi tamamlandı denemez. JavaScript syntax kontrolü geçti.

Disk dolması düşük LR deneyinin ilk çalışmasını checkpoint yazarken durdurdu. Tamamlanmamış koşu başarı olarak kaydedilmedi; daha sonra yeterli alanla ayrı koşu tamamlandı. Yalnızca bu çalışmada oluşturulan üç redundant diagnostic `last.pt` ve başarısız kısmi `best.pt` alan açmak için kaldırıldı; sonuç CSV/YAML'leri ve tamamlanmış best checkpoint'ler korundu. Yeni eğitimler öncesinde disk kontrolü eklendi. Büyük deneyler için bu minimumun üzerinde alan planlanmalıdır.

## G. Kalan riskler ve kabul durumu

| Kabul alanı | Durum |
|---|---|
| Model dosyaları/metadata/sha ve veri biçimi envanteri | Tamamlandı; eğitim kaynaklarının tümü yerelde yok |
| Yanlış/şüpheli anotasyon raporu ve sürümlü düzeltme | Tamamlandı; konteyner kutuları insan incelemesi gereken taslak |
| Bağımsız test verisi | Yokluğu belgelendi; oluşturulamadı |
| CeyMo sıfır skor araştırması | Mikro deneyde LR etkisi kanıtlandı; orijinal pilot için tek neden iddiası yok |
| Mapillary düşük skor araştırması | Tekrar üretim hatası, kısa pilot ve dengesizlik kaydedildi; düşük saha başarısı çözülmüş sayılmıyor |
| Gizlilik filtreleri, takip türleri, hata temizliği | Kod düzeltmeleri ve testler tamamlandı; gerçek recall ölçülmedi |
| Çoklu konteyner/geometri ve unknown | Kod yolu test edildi; gerçek kutu modeli doğrulanmadığı için kullanım kapalı |
| Olay sözleşmesi, track bazlı birleştirme, GPS ve harita | Eklendi; gerçek fiziksel olay doğruluğu ve tekrar-ID hatası ölçülmedi |
| API/kuyruk/iptal/oturum erişimi | Yerel demo düzeyinde test edildi |
| Çıktının tüm karelerini decode | Kısa video testinde geçti; VFR/uzun video taraması sınırlı |
| Model karşılaştırması ve otomatik terfi engeli | Geliştirme kıyası tamamlandı, aday terfisi reddedildi |
| Regresyon testleri | 36 test geçti; model doğruluğu testleriyle karıştırılmadı |
| Üretim güvenliği | Kullanıcı hesabı, kalıcı denetim kaydı, kota ve operasyonel izleme tamamlanmadı |

Yerel oturum çerezi gerçek bir çok kullanıcılı yetkilendirme altyapısı değildir. Onay düğmesi, kullanıcıya ait geçici gizlilik incelemesi kaydıdır. Gizlilik dedektörleri kaçırabilir; opak maske sadece tespit edilen alanları kapsar. Rus plaka Cascade'in yanlış pozitifleri görsel incelemelerde de görülebilir; Türkiye plakalarında etiketsiz veriyle başarı oranı verilemez. Nesne/sahne çıktıları inceleme adayıdır; tabela/kaldırım varlığı ihlal değildir. Piksel boyutu metre veya mevzuata uygunluk kararı değildir.

Proje işlem sırasında C: konumundan E: konumuna taşındı. Yeni klasörde hesabın yazma hakkı yoktu; kullanıcının açık izniyle yalnızca `E:\birsav` altında `BATU\batuhan` hesabına miraslanan **Modify** hakkı eklendi. Veri YAML yolları güncellendi; eski içerikleri [reports/workspace_path_migration.json](reports/workspace_path_migration.json) içinde saklandı. Tarihsel eğitim kayıtlarının eski yolları değiştirilmedi. Taşınma sonrası model checksum'ları ve testler tekrar doğrulandı.

## H. Sonraki iterasyon

1. **En yüksek fayda: bağımsız saha verisi ve etiketler.** Eğitimde olmayan rota/gün/kamera görüntüleri; yüz/plaka kutuları, yol hasarı, yerde çöp, konteyner gerçek kutusu/durumu ve olay kimlikleri. Maliyet etiketleme ve insan incelemesidir; veri sızıntısı en önemli risktir. Kaynak manifesti olmadan rastgele karelerden sahte bağımsız test oluşturulmayacak.
2. **Konteyner taslaklarının insan doğrulaması ve yeni örnekler.** Negatif sokaklar, örtülü/kapalı, uzak, çoklu ve gece konteynerleri. Gerçek bbox onayından sonra detector ve kırpıntı sınıflandırıcı ayrı değerlendirilmelidir. Birkaç saatlik inceleme ilk adımı mümkün kılar; güvenilir saha modeli için daha geniş çeşitlilik gerekir.
3. **Gizlilik etiketli benchmark ve uygun plaka modeli.** Yalnızca konum tespiti; OCR yapılmaz. Yüz/plaka recall'u ve maskelenmeyen nesne oranı ölçülmeden güvenilir anonimlik iddiası verilmez. Lisans/kaynak doğrulaması ve olası GPU kurulumu ayrıca planlanmalıdır; bu çalışmada video harici servise yüklenmedi.
4. **Mapillary/CeyMo veri deneyi.** Sınıf dengesi, temiz poligonlar, doğru augmentation ve rota bazlı validation; tek değişkenli kontrollü kıyas. GPU sürümü ayrı ortamda doğrulanabilirse eğitim süresi kısalabilir. Mevcut CPU süreleri yukarıda kayıtlıdır; CUDA paketi için büyük indirme yapılmadı.
5. **Video/olay benchmarkı.** Gerçek araç hareketinde ByteTrack/SORT/optical-flow kıyası, VFR/PTS/rotasyon/uzun video ve 1/3/5/10 FPS kaçırma oranı. Etiketli video olmadan algoritma değişikliği başarı kanıtı sayılmamalıdır.
6. **Ürün ve operasyon.** Kalıcı kayıt, gerçek kullanıcı yetkilendirmesi, yetkili personel iş akışı, kaynak/iş kotaları ve tarayıcı E2E. Yerel demo temelinin üzerine, doğrulanmış ihtiyaçlarla eklenmelidir.

Bu rapor, tamamlanan uygulama/ölçüm iterasyonunu ve açık işleri ayırır. Projenin yüksek doğruluklu saha sistemi hedefine ulaştığı anlamına gelmez.
